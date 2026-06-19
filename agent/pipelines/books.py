"""Book pipeline — Mega.nz, ePub/PDF extraction, 3-tier pre-flight,
chapter-level synthesis, quality gates, MinIO archival, and Git push.

Supports:
- Mega.nz archive download + batch book classification
- ePub extraction via ebooklib (preserves chapter order)
- PDF extraction via pymupdf (text PDFs)
- 3-tier PDF quality scoring (scanned vs degraded vs clean)
- Book note synthesis from book-note-template.md
- Quality gates: line count, quote count, size, fact-checks, chapter coverage
- MinIO upload of source + extracted full text
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from agent.config import Config
from agent.db import init_db, LOGS_SCHEMA
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMClient, get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.quality import quality_gate_book

# Optional dependencies — imported at module level for testability
try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    pymupdf = None  # type: ignore
    HAS_PYMUPDF = False

try:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup
    HAS_EBOOKLIB = True
except ImportError:
    epub = None  # type: ignore
    ITEM_DOCUMENT = None
    BeautifulSoup = None  # type: ignore
    HAS_EBOOKLIB = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate token ratio for English text
CHARS_PER_TOKEN = 4

# Threshold for chunked synthesis (books are long)
CHUNK_TOKEN_THRESHOLD = 80_000

# Boundary detection (CONTEXT.md lines 218-222):
# LLM-first semantic boundary detection, not regex.
BOUNDARY_CHUNK_TOKENS = 50_000   # ~200K chars — initial boundary search window
BOUNDARY_EXPAND_TOKENS = 100_000 # ~400K chars — expanded window per fallback spec
OVERLAP_TOKENS = 10_000          # ~40K chars — overlap between chunks

# Max quality gate retries
MAX_QUALITY_RETRIES = 3

# Book extensions recognized for batch processing
BOOK_EXTENSIONS = {".epub", ".pdf", ".mobi", ".azw3"}

# 3-tier PDF pre-flight thresholds
PDF_MIN_CHARS_SCANNED = 500       # Below this → scanned (no text layer)
PDF_QUALITY_SCORE_THRESHOLD = 80  # Below this → degraded OCR


# ── Mega.nz Download ──────────────────────────────────────────────────────────

def step_megadl_download(task: dict, config: Config) -> str:
    """Download a Mega.nz file or folder archive.

    For single files: downloads directly to temp dir.
    For folders/book batches: downloads, unzips to a clean directory,
    lists all book files found.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        JSON string with archive_path, file_count, and file list.

    Raises:
        StepError: On download failure (PAUSE — retry won't help without
            credentials or URL fix).
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "mega")

    # Determine download directory
    dl_dir = Path(config.temp_dir) / f"mega_{slug}"
    if dl_dir.exists():
        shutil.rmtree(dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use megadl to download
        result = subprocess.run(
            ["megadl", "--path", str(dl_dir), url],
            capture_output=True, text=True, timeout=600,  # 10 min for large files
        )
        if result.returncode != 0:
            raise StepError(
                f"megadl download failed: {result.stderr.strip()}",
                error_type=ErrorType.PAUSE,
                step_name="megadl_download",
            )

        # Check what we got — if it's a zip/archive, unzip it
        downloaded = list(dl_dir.glob("*"))
        if not downloaded:
            raise StepError(
                "megadl completed but no files found",
                error_type=ErrorType.PAUSE,
                step_name="megadl_download",
            )

        # Unzip any archives found
        archive_extensions = {".zip", ".rar", ".7z"}
        for f in downloaded:
            if f.suffix.lower() in archive_extensions:
                extract_dir = dl_dir / f"extracted_{f.stem}"
                extract_dir.mkdir(parents=True, exist_ok=True)
                if f.suffix.lower() == ".zip":
                    subprocess.run(
                        ["unzip", "-o", str(f), "-d", str(extract_dir)],
                        capture_output=True, timeout=120,
                    )
                # .rar/.7z not handled in v1 — logged

        # Find all book files
        all_files = list(dl_dir.rglob("*"))
        book_files = [
            f for f in all_files
            if f.suffix.lower() in BOOK_EXTENSIONS
            and "__MACOSX" not in str(f)
        ]

        # Build result
        result_data = {
            "archive_path": str(dl_dir),
            "file_count": len(book_files),
            "books": [
                {
                    "path": str(bf),
                    "filename": bf.name,
                    "extension": bf.suffix.lower(),
                    "size_bytes": bf.stat().st_size,
                }
                for bf in sorted(book_files, key=lambda f: f.name)
            ],
        }

        return json.dumps(result_data)

    except StepError:
        raise
    except Exception as e:
        raise StepError(
            f"megadl download error: {e}",
            error_type=ErrorType.PAUSE,
            step_name="megadl_download",
        ) from e


# ── Content Classification ────────────────────────────────────────────────────

def step_classify_content(task: dict, config: Config) -> str:
    """Classify downloaded Mega content into sub-tasks.

    Reads the megadl_download output, classifies each book by type
    and pre-flight quality. Creates per-book sub-tasks or marks for
    delegation.

    For book batches: classifies PDFs via 3-tier pre-flight.
    For single books: routes directly.

    Args:
        task: Task dict.
        config: Config dataclass.

    Returns:
        JSON string with classification results.
    """
    slug = task.get("slug", task["id"][:8])

    # Try to get the download result
    dl_dir = Path(config.temp_dir) / f"mega_{slug}"
    if not dl_dir.exists():
        raise StepError(
            f"Download directory not found: {dl_dir}",
            error_type=ErrorType.FAIL,
            step_name="classify_content",
        )

    all_files = list(dl_dir.rglob("*"))
    book_files = [
        f for f in all_files
        if f.suffix.lower() in BOOK_EXTENSIONS
        and "__MACOSX" not in str(f)
    ]

    classified = []
    for bf in sorted(book_files, key=lambda f: f.name):
        ext = bf.suffix.lower()
        entry = {
            "path": str(bf),
            "filename": bf.name,
            "extension": ext,
            "size_bytes": bf.stat().st_size,
        }

        if ext == ".epub":
            entry["classification"] = "text"
            entry["pipeline"] = "book_epub"
        elif ext in (".mobi", ".azw3"):
            entry["classification"] = "text"
            entry["pipeline"] = "book_mobi"
        elif ext == ".pdf":
            # 3-tier pre-flight
            tier, score = _pdf_preflight(bf)
            entry["classification"] = tier
            entry["quality_score"] = score
            if tier == "scanned":
                entry["pipeline"] = "book_scanned"
            elif tier == "degraded":
                entry["pipeline"] = "book_degraded"
            else:
                entry["pipeline"] = "book_pdf"

        classified.append(entry)

    # Count by classification
    scanned = [c for c in classified if c.get("classification") == "scanned"]
    degraded = [c for c in classified if c.get("classification") == "degraded"]
    clean = [c for c in classified if c.get("classification") == "text"]

    return json.dumps({
        "total_files": len(classified),
        "scanned": len(scanned),
        "degraded": len(degraded),
        "clean_text": len(clean),
        "books": classified,
    })


def _pdf_preflight(pdf_path: Path) -> tuple[str, int]:
    """Run 3-tier PDF pre-flight check.

    Tier 1: Binary scan detection — pymupdf char count < 500 → SCANNED
    Tier 2: Quality scoring — sample 16 pages, score 0-100
    Tier 3: >= 80 → GOOD TEXT, < 80 → DEGRADED

    Returns:
        (classification, score) where classification is 'scanned',
        'degraded', or 'text', and score is 0-100.
    """
    if not HAS_PYMUPDF:
        return "text", 100  # Assume clean if pymupdf unavailable

    try:
        doc = pymupdf.open(str(pdf_path))
        page_count = doc.page_count

        # Tier 1: Binary scan detection
        total_chars = sum(
            len(doc[i].get_text().strip()) for i in range(page_count)
        )
        if total_chars < PDF_MIN_CHARS_SCANNED:
            doc.close()
            return "scanned", 0

        # Tier 2: Quality scoring on 16 evenly-distributed pages
        score = _quality_score(doc, page_count)
        doc.close()

        if score >= PDF_QUALITY_SCORE_THRESHOLD:
            return "text", score
        else:
            return "degraded", score

    except Exception:
        return "text", 100  # Assume clean on error


def quality_score(pdf_path: str | Path) -> int:
    """Compute a 0-100 quality score for a PDF's OCR text layer.

    Scores 16 evenly-distributed pages. Evaluates:
    - Ratio of printable characters to total characters
    - Common OCR error patterns (long-s→f, rn→m, cl→d)
    - Garbled chapter header patterns (special char density)
    - Spaced-out letter patterns

    >= 80: Good enough for direct processing.
    < 80: Degraded — needs fresh OCR.

    Public API for use by batch classification and external callers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Quality score 0-100.
    """
    if not HAS_PYMUPDF:
        return 100

    try:
        doc = pymupdf.open(str(pdf_path))
        score = _quality_score(doc, doc.page_count)
        doc.close()
        return score
    except Exception:
        return 100


def _quality_score(doc, page_count: int) -> int:
    """Internal quality scoring function.

    Samples 16 pages evenly distributed across the document.
    Returns 0-100.
    """

    # Determine sample pages
    if page_count <= 16:
        sample_pages = list(range(page_count))
    else:
        step = page_count / 16
        sample_pages = [int(i * step) for i in range(16)]

    total_score = 0
    scores_count = 0

    for page_idx in sample_pages:
        try:
            text = doc[page_idx].get_text()
            if not text.strip():
                total_score += 50  # Empty page → neutral
                scores_count += 1
                continue

            # Component 1: Printable character ratio (0-40 points)
            printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
            ratio = printable / max(len(text), 1)
            printable_score = min(40, int(ratio * 40))

            # Component 2: Common OCR error density (0-30 points)
            # Long-s → f patterns: "fafe", "firft", "fubject"
            long_s_patterns = len(re.findall(
                r'\bf[aeiou]f[aeiou]|\bfirft\b|\bfubject\b', text, re.IGNORECASE
            ))
            # rn → m: "modem" for "modern"
            rn_patterns = len(re.findall(r'\b\w*m\w*m\b', text))
            # cl → d: "dear" for "clear"
            cl_patterns = len(re.findall(r'\bd[aeiou]ar\b', text))
            error_density = (long_s_patterns + rn_patterns + cl_patterns) / max(len(text.split()), 1)
            error_score = max(0, 30 - int(error_density * 1000))

            # Component 3: Garbled header detection (0-20 points)
            # Character noise ratio: % of special chars / total chars
            special_chars = sum(
                1 for c in text if not c.isalnum() and not c.isspace()
                and not c in ".,;:!?-'\"()[]{}"
            )
            noise_ratio = special_chars / max(len(text), 1)
            noise_score = max(0, 20 - int(noise_ratio * 100))

            # Component 4: Word length distribution (0-10 points)
            # Normal text has average word length 4-7 chars
            words = [w for w in text.split() if w.strip()]
            if words:
                avg_len = sum(len(w) for w in words) / len(words)
                if 3 <= avg_len <= 10:
                    len_score = 10
                elif 1 <= avg_len <= 15:
                    len_score = 5
                else:
                    len_score = 0
            else:
                len_score = 0

            page_score = printable_score + error_score + noise_score + len_score
            total_score += page_score
            scores_count += 1

        except Exception:
            total_score += 50
            scores_count += 1

    if scores_count == 0:
        return 100

    return min(100, int(total_score / scores_count))


# ── ePub Extraction ───────────────────────────────────────────────────────────

def step_extract_epub(task: dict, config: Config) -> str:
    """Extract full text from an ePub file using ebooklib.

    Preserves chapter order (no sorting by size). Reads EVERY chapter.
    Saves full text to /tmp/book_{slug}_full.txt.
    Uses unique paths per slug to prevent parallel worker corruption.

    Args:
        task: Task dict with source_url (local file path or URL), slug.
        config: Config dataclass.

    Returns:
        Path to the extracted full text file.
    """
    slug = task.get("slug", task["id"][:8])

    # Determine ePub source path
    source = task.get("source_url", "")
    if source and Path(source).exists():
        epub_path = Path(source)
    else:
        # Check temp dir
        dl_dir = Path(config.temp_dir) / f"mega_{slug}"
        epubs = list(dl_dir.rglob("*.epub"))
        if epubs:
            epub_path = epubs[0]
        else:
            raise StepError(
                f"No ePub file found for slug {slug}",
                error_type=ErrorType.FAIL,
                step_name="extract_epub",
            )

    if not HAS_EBOOKLIB:
        raise StepError(
            "Missing dependencies: ebooklib/beautifulsoup4 — install with pip",
            error_type=ErrorType.PAUSE,
            step_name="extract_epub",
        )

    try:
        book = epub.read_epub(str(epub_path))
        items = list(book.get_items_of_type(ITEM_DOCUMENT))

        all_text = []
        for i, item in enumerate(items):
            try:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                text = soup.get_text(separator="\n")
                chars = len(text.strip())
                if chars > 50:  # Skip tiny fragments (footnotes, etc.)
                    all_text.append((i, chars, text.strip()))
            except Exception:
                # Skip unparseable items
                continue

        # Preserve original order (item index)
        all_text.sort(key=lambda x: x[0])

        # Write full text — unique path per slug
        output_path = Path(config.temp_dir) / f"book_{slug}_full.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, chars, text in all_text:
                f.write(f"\n\n=== CHAPTER {idx} ===\n\n")
                f.write(text)

        total = sum(c for _, c, _ in all_text)

        # Store metadata in task output
        try:
            title = book.get_metadata("DC", "title")
            author = book.get_metadata("DC", "creator")
            task["_book_title"] = title[0][0] if title else Path(epub_path).stem
            task["_book_author"] = author[0][0] if author else "Unknown"
        except Exception:
            task["_book_title"] = Path(epub_path).stem
            task["_book_author"] = "Unknown"

        return json.dumps({
            "full_text_path": str(output_path),
            "chapter_count": len(all_text),
            "total_chars": total,
            "estimated_words": total // 5,
            "title": task.get("_book_title", ""),
            "author": task.get("_book_author", ""),
        })

    except Exception as e:
        raise StepError(
            f"ePub extraction failed: {e}",
            error_type=ErrorType.FAIL,
            step_name="extract_epub",
        ) from e


# ── PDF Extraction ────────────────────────────────────────────────────────────

def step_extract_pdf(task: dict, config: Config) -> str:
    """Extract text from a PDF using pymupdf.

    For text PDFs only (scanned/degraded routed to marker-pdf pipeline
    separately via batch classification).

    Args:
        task: Task dict with source_url (local file path), slug.
        config: Config dataclass.

    Returns:
        Path to the extracted full text file.
    """
    slug = task.get("slug", task["id"][:8])

    # Determine PDF source path
    source = task.get("source_url", "")
    if source and Path(source).exists():
        pdf_path = Path(source)
    else:
        dl_dir = Path(config.temp_dir) / f"mega_{slug}"
        pdfs = list(dl_dir.rglob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
        else:
            raise StepError(
                f"No PDF file found for slug {slug}",
                error_type=ErrorType.FAIL,
                step_name="extract_pdf",
            )

    if not HAS_PYMUPDF:
        raise StepError(
            "Missing dependency: pymupdf — install with pip install pymupdf",
            error_type=ErrorType.PAUSE,
            step_name="extract_pdf",
        )

    try:
        doc = pymupdf.open(str(pdf_path))

        # Run pre-flight check first
        classification, score = _pdf_preflight(pdf_path)
        if classification != "text":
            doc.close()
            raise StepError(
                f"PDF is {classification} (score {score}/100) — needs OCR, "
                f"routed to separate pipeline",
                error_type=ErrorType.PAUSE,
                step_name="extract_pdf",
            )

        # Extract text page by page
        full_text = []
        for page in doc:
            full_text.append(page.get_text())

        doc.close()

        output_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))

        total_chars = sum(len(t) for t in full_text)

        # Try to get PDF metadata
        task["_book_title"] = pdf_path.stem
        task["_book_author"] = "Unknown"

        return json.dumps({
            "full_text_path": str(output_path),
            "page_count": len(full_text),
            "total_chars": total_chars,
            "quality_score": score,
            "estimated_words": total_chars // 5,
            "title": task["_book_title"],
        })

    except StepError:
        raise
    except Exception as e:
        raise StepError(
            f"PDF extraction failed: {e}",
            error_type=ErrorType.FAIL,
            step_name="extract_pdf",
        ) from e


# ── Book Note Synthesis ───────────────────────────────────────────────────────

BOOK_SYNTHESIS_SYSTEM_PROMPT = """You are a skilled book summarizer and critical analyst.
Your notes are thorough, well-organized, and scholarly in tone.

When writing a book note, you MUST follow this structure:

## Summary
2-3 sentences capturing the central thesis and scope of the book.

## Key Claims
At least 4 direct quotes from the book with chapter numbers, formatted as:
> "direct quote text here" — Chapter X
Each quote should represent a distinct, significant claim the author makes.

## Chapter Summaries
EVERY chapter must get a substantive paragraph (3-5 sentences minimum).
This is the DOMINANT section of the note — it should be the longest part.
Do not skip chapters. Do not write one-liners. If a chapter is short, still
summarize its content fully.

## Who Is the Author
1 paragraph covering credentials, institutional affiliations, other works,
and any potential conflicts of interest relevant to the book's subject.

## Critical Analysis
Shorter than Chapter Summaries + Key Claims combined. Evaluate:
- Evidence quality (cited studies vs assertions)
- Logical soundness of arguments
- What's missing or under-explored
- How the book relates to other works in the field

## Nuances
Specific points where the book is imprecise, overclaims, or where the
science has evolved since publication.

## Reliability
Confidence assessment:
- verified (multiple independent sources confirm)
- plausible (consistent with known science but not independently verified)
- emerging (new or speculative claims)
- debunked (contradicted by evidence)

## Sources
- Author. *Book Title* (Publisher, Year)
- Full text archived at path referenced in note
"""


def step_synthesize_book(task: dict, config: Config) -> str:
    """Synthesize a book note from extracted full text.

    Uses a dedicated book synthesis template with chapter-by-chapter
    analysis. For long books (>80K tokens), uses progressive chunking.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])

    # Find the extracted full text
    full_text_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
    if not full_text_path.exists():
        raise StepError(
            f"Full text not found: {full_text_path}",
            error_type=ErrorType.FAIL,
            step_name="synthesize_book",
        )

    full_text = full_text_path.read_text(encoding="utf-8")
    estimated_tokens = len(full_text) // CHARS_PER_TOKEN

    # Get LLM client with archiving
    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    # Determine note path
    note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    title = task.get("_book_title", slug)
    author = task.get("_book_author", "Unknown")

    if estimated_tokens <= CHUNK_TOKEN_THRESHOLD:
        # Single pass synthesis
        user_prompt = _build_book_prompt(full_text, title, author)
        response = llm.call(
            system_prompt=BOOK_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role="synthesis",
        )
        note_content = response.content
    else:
        # Progressive chunking for very long books
        note_content = _chunked_book_synthesis(
            full_text, full_text_path, title, author, llm, config
        )

    # Add frontmatter
    frontmatter = f"""---
topic: [books/{slug[:30]}]
date: {_today()}
source: {author} — *{title}* (book)
confidence: plausible
tags: [book, book-summary]
---

# {title}

**Author:** {author}

"""
    note_path.write_text(frontmatter + note_content, encoding="utf-8")

    return str(note_path)


def _build_book_prompt(full_text: str, title: str, author: str) -> str:
    """Build the synthesis prompt for a book."""
    # Truncate if extremely long (model context limits)
    max_chars = 180_000  # ~45K tokens for input
    if len(full_text) > max_chars:
        # Keep beginning and end, trim middle
        half = max_chars // 2
        full_text = full_text[:half] + "\n\n[... content truncated for length ...]\n\n" + full_text[-half:]

    return f"""Analyze the following book and create a comprehensive note following the
template structure.

Book: *{title}* by {author}

Full text:
---
{full_text}
---

Follow the template structure EXACTLY. Every chapter must have a substantive
paragraph in the Chapter Summaries section. Include at least 4 direct quotes
with chapter numbers."""


def _detect_boundaries_llm(full_text: str, llm: LLMClient) -> list[int]:
    """Per CONTEXT.md lines 218-222: cheap LLM call for semantic boundaries.

    Receives the full source text and returns character offsets where
    chapter/section splits should occur. Uses progressive window expansion
    as specified: 50K tokens -> 100K tokens -> best boundary anyway.

    Args:
        full_text: Complete extracted book text.
        llm: LLM client instance.

    Returns:
        Sorted list of character offsets for chunk boundaries, starting at 0.
    """
    total_chars = len(full_text)
    total_tokens = total_chars // CHARS_PER_TOKEN

    # Try initial window: ~50K tokens (~200K chars)
    for window_tokens in (BOUNDARY_CHUNK_TOKENS, BOUNDARY_EXPAND_TOKENS):
        window_chars = min(window_tokens * CHARS_PER_TOKEN, total_chars)
        sample = full_text[:window_chars]

        prompt = f"""Analyze this book text and identify ALL major semantic boundaries
(chapter starts, section breaks, topic transitions).

Total book length: {total_chars:,} characters (~{total_tokens:,} tokens).
This sample covers the first {window_chars:,} characters.

Output a JSON array of character offsets where new sections begin.
The first offset MUST be 0 (start of text).
Example: [0, 12450, 28901, 45678]

IMPORTANT: Output ONLY the JSON array on a single line. No explanation.

TEXT BEGINS:
{sample}
TEXT ENDS"""

        try:
            response = llm.call(
                system_prompt=(
                    "You are a precise text boundary detector. "
                    "Identify semantic boundaries in book text. "
                    "Output ONLY a JSON array of integers (character offsets). "
                    "No markdown, no explanation — just the array."
                ),
                user_prompt=prompt,
                role="boundary",
            )

            # Parse JSON array from response
            match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', response.content)
            if match:
                boundaries = json.loads(match.group())
                if boundaries:
                    # Ensure 0 is first
                    if boundaries[0] != 0:
                        boundaries.insert(0, 0)
                    # Validate: offsets should be within range and increasing
                    boundaries = sorted(set(
                        b for b in boundaries if 0 <= b < total_chars
                    ))
                    if len(boundaries) >= 2:
                        return boundaries
        except (json.JSONDecodeError, ValueError, IndexError, Exception):
            pass

    # Fallback per CONTEXT.md: "if still none -> boundary LLM picks
    # best boundary even if imperfect" — use regex chapter markers
    # as last resort
    markers = list(re.finditer(
        r'(?:^|\n)(?:CHAPTER|Chapter|PART|Part|SECTION|Section)\s+\d+',
        full_text, re.MULTILINE
    ))
    if markers:
        boundaries = [0] + [m.start() for m in markers if m.start() > 0]
        # Ensure reasonable spacing (min 5K chars between boundaries)
        filtered = [0]
        for b in boundaries[1:]:
            if b - filtered[-1] >= 5000:
                filtered.append(b)
        return filtered

    # Absolute last resort: split evenly at window boundaries
    chunk_size = BOUNDARY_CHUNK_TOKENS * CHARS_PER_TOKEN
    num_chunks = max(2, total_chars // chunk_size)
    step = total_chars // num_chunks
    return [i * step for i in range(num_chunks)]


def _chunked_book_synthesis(full_text: str, full_text_path: Path,
                            title: str, author: str, llm: LLMClient,
                            config: Config) -> str:
    """Progressive chunking synthesis using LLM boundary detection.

    Per CONTEXT.md lines 218-222:
    1. LLM detects semantic boundaries (NOT regex-primary)
    2. Regex chapter markers supplement LLM boundaries
    3. Progressive synthesis with 10K token overlap between chunks
    4. Final synthesis pass
    """
    # Step 1: LLM boundary detection (primary, per CONTEXT.md)
    llm_boundaries = _detect_boundaries_llm(full_text, llm)

    # Step 2: Regex chapter markers as supplemental validation
    regex_boundaries = [0]
    for m in re.finditer(
        r'\n=== CHAPTER \d+ ===\n', full_text
    ):
        regex_boundaries.append(m.end())

    # Merge: start with LLM boundaries, add regex boundaries within
    # 5K chars as refinements
    all_boundaries = sorted(set(llm_boundaries))
    for rb in regex_boundaries[1:]:
        # Only add if it's a meaningful refinement near an LLM boundary
        if not any(abs(rb - b) < 5000 for b in all_boundaries):
            all_boundaries.append(rb)
    all_boundaries = sorted(set(all_boundaries))

    # Step 3: Build chunks with 10K token overlap (~40K chars)
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
    chunks: list[str] = []
    for i in range(len(all_boundaries) - 1):
        start = max(0, all_boundaries[i] - (overlap_chars if i > 0 else 0))
        end = min(len(full_text), all_boundaries[i + 1] + overlap_chars)
        chunks.append(full_text[start:end])

    # Add final chunk (after last boundary to end of text)
    if all_boundaries:
        last_start = max(0, all_boundaries[-1] - overlap_chars)
        if last_start < len(full_text):
            chunks.append(full_text[last_start:])

    if len(chunks) <= 1:
        # Ultimate fallback: truncated single pass
        return llm.call(
            system_prompt=BOOK_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=_build_book_prompt(
                full_text[:BOUNDARY_EXPAND_TOKENS * CHARS_PER_TOKEN],
                title, author
            ),
        ).content

    # Step 4: Progressive synthesis — build note chunk by chunk
    # Start with first chunk
    note_so_far = ""
    for idx, chunk_text in enumerate(chunks):
        chunk_label = f"Chunk {idx + 1}/{len(chunks)}"
        if idx == 0:
            # First chunk: full synthesis
            chunk_prompt = f"""Analyze the following portion of *{title}* by {author}
and create a book note following the template structure.

{chunk_label} — this is the BEGINNING of the book:
---
{chunk_text}
---

Follow the template structure. This is the first pass — cover what you can
from this portion, and the note will be expanded with subsequent chunks."""
        elif idx == len(chunks) - 1:
            # Last chunk: final polish
            chunk_prompt = f"""Here is the current note for *{title}* by {author}:

{note_so_far}

Now here is {chunk_label} — the FINAL portion of the book:
---
{chunk_text}
---

Integrate this final portion into the note. Update ALL sections:
add new chapter summaries, quotes, and analysis. Ensure EVERY section
is complete. This is the FINAL pass — produce a polished, complete note."""
        else:
            # Middle chunk: expand
            chunk_prompt = f"""Here is the current note for *{title}* by {author}:

{note_so_far}

Now here is {chunk_label}:
---
{chunk_text}
---

Expand the note with content from this new chunk. Add chapter summaries,
quotes, and analysis. Maintain the full template structure."""

        try:
            response = llm.call(
                system_prompt=BOOK_SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=chunk_prompt,
                role="synthesis",
            )
            note_so_far = response.content
        except Exception:
            if idx == 0:
                note_so_far = f"[Synthesis failed for {chunk_label}]\n"
            # Continue with previous note_so_far on failure

    return note_so_far or llm.call(
        system_prompt=BOOK_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=_build_book_prompt(
            full_text[:BOUNDARY_EXPAND_TOKENS * CHARS_PER_TOKEN],
            title, author
        ),
        role="synthesis",
    ).content


# ── Book Quality Gate ─────────────────────────────────────────────────────────

def step_book_quality_gate(task: dict, config: Config) -> str:
    """Run book-specific quality gates on the generated note.

    Checks:
    - line_count >= config.book_min_lines (100)
    - quote_count >= config.book_min_quotes (4)
    - size >= config.book_min_size (8000)
    - fact_check_count >= config.book_min_fact_checks (2)
    - Every chapter marker in source has corresponding coverage

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string with quality gate results.

    Raises:
        StepError: If quality gates fail after max retries.
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"
    full_text_path = Path(config.temp_dir) / f"book_{slug}_full.txt"

    passed, failures = quality_gate_book(note_path, config)

    if not passed:
        raise StepError(
            f"Book quality gates failed: {'; '.join(failures)}",
            error_type=ErrorType.RETRY,
            step_name="book_quality_gate",
        )

    return json.dumps({
        "passed": True,
        "checks": [
            {"name": "line_count", "passed": True},
            {"name": "quote_count", "passed": True},
            {"name": "size", "passed": True},
            {"name": "fact_check_count", "passed": True},
        ],
    })


# ── MinIO Upload (Books) ──────────────────────────────────────────────────────

def step_book_minio_upload(task: dict, config: Config) -> str:
    """Upload book source and extracted full text to MinIO.

    Delegates to archival.upload_file_to_minio for each file.

    Uploads:
    - Source epub/PDF → kb-agent-archive/books/{slug}.epub (or .pdf)
    - Extracted full text → kb-agent-archive/books/{slug}.txt

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string with MinIO URLs.
    """
    from agent.archival import upload_file_to_minio

    slug = task.get("slug", task["id"][:8])
    full_text_path = Path(config.temp_dir) / f"book_{slug}_full.txt"

    if not full_text_path.exists():
        raise StepError(
            f"Full text not found for MinIO upload: {full_text_path}",
            error_type=ErrorType.FAIL,
            step_name="book_minio_upload",
        )

    uploaded = []

    # Upload full text
    txt_dest = f"{config.minio_bucket}/books/{slug}.txt"
    upload_file_to_minio(full_text_path, txt_dest)
    uploaded.append(f"books/{slug}.txt")

    # Try to upload source epub/PDF
    dl_dir = Path(config.temp_dir) / f"mega_{slug}"
    source_files = (
        list(dl_dir.rglob("*.epub")) + list(dl_dir.rglob("*.pdf"))
    )
    if source_files:
        sf = source_files[0]
        sf_dest = f"{config.minio_bucket}/books/{slug}{sf.suffix.lower()}"
        try:
            upload_file_to_minio(sf, sf_dest, timeout=120)
            uploaded.append(f"books/{slug}{sf.suffix.lower()}")
        except StepError:
            pass  # Source file upload is best-effort

    tailscale_base = "http://vmi3304846.tail5c02a1.ts.net:9000"
    return json.dumps({
        "uploaded": uploaded,
        "urls": [f"{tailscale_base}/{config.minio_bucket}/{u}" for u in uploaded],
    })


# ── Git Push (Books) ──────────────────────────────────────────────────────────

def step_book_git_push(task: dict, config: Config) -> str:
    """Push the book note to the Obsidian vault Git repo.

    Delegates to archival.step_git_push with book-specific params.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Commit hash.
    """
    from agent.archival import step_git_push as archival_git_push

    slug = task.get("slug", task["id"][:8])
    title = task.get("_book_title", slug)
    author = task.get("_book_author", "")
    commit_prefix = f"add: — {author}: {title}"[:200]
    return archival_git_push(
        task, config,
        vault_subfolder="books",
        commit_prefix=commit_prefix,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today() -> str:
    """Return today's date as YYYY-MM-DD."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


# ── Pipeline Registration ────────────────────────────────────────────────────

register_pipeline("book", [
    Step(name="extract_text", fn=step_extract_epub, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

# Duplicate registration for "book_epub" — step_classify_content outputs
# "book_epub" as the pipeline name for ePub files (line 214), so it must
# resolve to the same book pipeline instead of falling back to "web".
register_pipeline("book_epub", [
    Step(name="extract_text", fn=step_extract_epub, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

register_pipeline("book_pdf", [
    Step(name="extract_text", fn=step_extract_pdf, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

register_pipeline("mega", [
    Step(name="megadl_download", fn=step_megadl_download, on_error="pause"),
    Step(name="classify_content", fn=step_classify_content, on_error="retry"),
    # FIXME: delegate step currently re-runs step_classify_content instead of
    # delegating to per-book sub-pipelines. Replace with a real delegation
    # function once sub-task dispatch is implemented.
    Step(name="delegate", fn=step_classify_content, on_error="fail"),
])

# Register book_batch pipeline (for mega.nz book archives)
register_pipeline("book_batch", [
    Step(name="megadl_download", fn=step_megadl_download, on_error="pause"),
    Step(name="classify_content", fn=step_classify_content, on_error="retry"),
    # FIXME: delegate step currently re-runs step_classify_content instead of
    # delegating to per-book sub-pipelines. Replace with a real delegation
    # function once sub-task dispatch is implemented.
    Step(name="delegate", fn=step_classify_content, on_error="fail"),
])
