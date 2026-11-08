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

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from agent.archival import append_note_tags
from agent.config import Config
from agent.core.note import write_vault_note
from agent.redact import redact_http_urls_in_text
from agent.db import init_db, LOGS_SCHEMA
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMClient, get_llm_client
from agent.pipelines import PIPELINES, Step, register_pipeline
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

# Pipelines assigned by step_classify_content — every value must be registered.
CLASSIFY_BOOK_PIPELINES = frozenset({
    "book_epub",
    "book_mobi",
    "book_pdf",
    "book_scanned",
    "book_degraded",
})

_log = logging.getLogger(__name__)


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

    # Determine download directory
    dl_dir = Path(config.temp_dir) / f"mega_{slug}"
    if dl_dir.exists():
        shutil.rmtree(dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="megadl_download",
        ) from exc

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
                suffix = f.suffix.lower()
                if suffix == ".zip":
                    result = subprocess.run(
                        ["unzip", "-o", str(f), "-d", str(extract_dir)],
                        capture_output=True, text=True, timeout=120,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout or "unzip failed").strip()
                        raise StepError(
                            f"failed to extract .zip archive {f.name}: {detail[:200]}",
                            error_type=ErrorType.FAIL,
                            step_name="megadl_download",
                        )
                elif suffix == ".rar":
                    result = subprocess.run(
                        ["unar", "-no-directory", "-o", str(extract_dir), str(f)],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout or "unar failed").strip()
                        raise StepError(
                            f"failed to extract .rar archive {f.name}: {detail[:200]}",
                            error_type=ErrorType.FAIL,
                            step_name="megadl_download",
                        )
                elif suffix == ".7z":
                    result = subprocess.run(
                        ["7z", "x", str(f), f"-o{extract_dir}", "-y"],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout or "7z failed").strip()
                        raise StepError(
                            f"failed to extract .7z archive {f.name}: {detail[:200]}",
                            error_type=ErrorType.FAIL,
                            step_name="megadl_download",
                        )

        # Find all book files
        all_files = list(dl_dir.rglob("*"))
        book_files = [
            f for f in all_files
            if f.suffix.lower() in BOOK_EXTENSIONS
            and "__MACOSX" not in str(f)
        ]
        if not book_files:
            raise StepError(
                "megadl download produced no book files after extraction",
                error_type=ErrorType.FAIL,
                step_name="megadl_download",
            )

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


def _book_child_slug(parent_slug: str, file_path: str) -> str:
    """Deterministic child slug; hash suffix avoids truncation collisions."""
    from agent.core.slugify import SLUG_MAX_LEN

    stem = Path(file_path).stem
    base = f"{parent_slug}_{stem}"
    if len(base) <= SLUG_MAX_LEN:
        return base
    digest = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:10]
    prefix_len = SLUG_MAX_LEN - len(digest) - 1
    return f"{base[:prefix_len]}_{digest}"


_CHILD_SLUG_HASH_SUFFIX = re.compile(r"_[0-9a-f]{10}$", re.IGNORECASE)


def _vault_frontmatter_source_url(source_url: str) -> str:
    """Vault-safe source_url: redact local paths and Mega decryption fragments."""
    from agent.core.status import redact_source_url_for_display

    redacted = redact_source_url_for_display(source_url or "")
    if not redacted:
        return redacted
    parsed = urlparse(redacted.strip())
    if parsed.fragment and (
        parsed.scheme in ("http", "https")
        or "mega.nz" in (parsed.netloc or "").lower()
    ):
        return redacted.split("#", 1)[0]
    return redacted


def _resolve_delegate_child_task(
    conn,
    child_slug: str,
    pipeline: str,
    file_path: str,
    steps: list[str],
    tags,
) -> str:
    from agent.db import create_delegate_task

    return create_delegate_task(
        conn,
        file_path,
        pipeline,
        child_slug,
        steps,
        tags=tags,
    )


def step_delegate_books(task: dict, config: Config) -> str:
    """Create one child task per classified book using its registered pipeline.

    Reads the classify_content step output from agent.db and enqueues pending
    tasks for each book whose pipeline is registered (e.g. book_epub, book_pdf).
    """
    from agent.db import AGENT_SCHEMA, init_db
    from agent.pipelines import PIPELINES, get_pipeline_names, import_all_pipelines

    import_all_pipelines()

    task_id = task["id"]
    db_path = config.kb_agent_dir / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    try:
        row = conn.execute(
            "SELECT output FROM task_steps "
            "WHERE task_id = ? AND name = 'classify_content' AND status = 'done' "
            "ORDER BY step_order LIMIT 1",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row or not row[0]:
        raise StepError(
            "classify_content output not found — cannot delegate",
            error_type=ErrorType.FAIL,
            step_name="delegate",
        )

    try:
        classification = json.loads(row[0])
    except json.JSONDecodeError as e:
        raise StepError(
            f"Invalid classify_content JSON: {e}",
            error_type=ErrorType.FAIL,
            step_name="delegate",
        ) from e

    books = classification.get("books") or []
    parent_slug = task.get("slug", task["id"][:8])
    tags = task.get("tags")

    conn = init_db(db_path, AGENT_SCHEMA)
    created: list[dict] = []
    skipped: list[dict] = []
    try:
        for book in books:
            pipeline = book.get("pipeline") or ""
            if pipeline not in PIPELINES:
                skipped.append({
                    "filename": book.get("filename"),
                    "pipeline": pipeline,
                    "reason": "pipeline_not_registered",
                })
                continue

            file_path = book.get("path") or ""
            if not file_path or not Path(file_path).exists():
                skipped.append({
                    "filename": book.get("filename"),
                    "pipeline": pipeline,
                    "reason": "missing_file",
                })
                continue

            child_slug = _book_child_slug(parent_slug, file_path)
            steps = get_pipeline_names(pipeline, "")
            child_id = _resolve_delegate_child_task(
                conn, child_slug, pipeline, file_path, steps, tags,
            )
            created.append({
                "task_id": child_id,
                "slug": child_slug,
                "pipeline": pipeline,
                "path": file_path,
            })
        conn.commit()
    finally:
        conn.close()

    if not created:
        raise StepError(
            "delegate step created no child book tasks",
            error_type=ErrorType.FAIL,
            step_name="delegate",
        )

    return json.dumps({
        "parent_task_id": task_id,
        "child_tasks_created": len(created),
        "skipped": len(skipped),
        "created": created,
        "skipped_books": skipped,
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

    except Exception as exc:
        _log.warning("PDF preflight failed for %s: %s", pdf_path, exc)
        return "degraded", 0


def _resolve_local_book_path(
    task: dict,
    config: Config,
    extensions: set[str],
    step_name: str,
) -> Path:
    """Resolve a local book file for child or batch mega tasks.

    Prefer ``task['source_url']`` when it is an existing local path (delegated
    mega children). Fall back to ``mega_{slug}`` only when that directory exists
    (batch/parent download layout).
    """
    source = (task.get("source_url") or "").strip()
    if source:
        src_path = Path(source)
        if src_path.is_file() and src_path.suffix.lower() in extensions:
            return src_path

    slug = task.get("slug", task["id"][:8])
    dl_dir = Path(config.temp_dir) / f"mega_{slug}"
    if dl_dir.is_dir():
        matches = [
            f for f in dl_dir.rglob("*")
            if f.suffix.lower() in extensions and "__MACOSX" not in str(f)
        ]
        if matches:
            slug_lower = slug.lower()
            exact_stem = [f for f in matches if f.stem.lower() == slug_lower]
            if exact_stem:
                return sorted(exact_stem, key=lambda f: f.name)[0]

            if _CHILD_SLUG_HASH_SUFFIX.search(slug):
                raise StepError(
                    f"No book file ({', '.join(sorted(extensions))}) with exact "
                    f"stem matching hash-suffixed slug {slug}",
                    error_type=ErrorType.FAIL,
                    step_name=step_name,
                )

            slug_matches = [
                f for f in matches if slug_lower in f.name.lower()
            ]
            if len(slug_matches) == 1:
                return slug_matches[0]
            if len(slug_matches) > 1:
                raise StepError(
                    f"Ambiguous book file match for slug {slug} under {dl_dir}",
                    error_type=ErrorType.FAIL,
                    step_name=step_name,
                )
            return sorted(matches, key=lambda f: f.name)[0]

    raise StepError(
        f"No book file ({', '.join(sorted(extensions))}) found for slug {slug}",
        error_type=ErrorType.FAIL,
        step_name=step_name,
    )


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
        return 0


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

    epub_path = _resolve_local_book_path(
        task, config, {".epub"}, "extract_epub",
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
                f.write(redact_http_urls_in_text(text))

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

    pdf_path = _resolve_local_book_path(
        task, config, {".pdf"}, "extract_pdf",
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
            f.write(redact_http_urls_in_text("\n\n".join(full_text)))

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


def _write_pdf_full_text(
    pdf_path: Path,
    slug: str,
    config: Config,
    *,
    quality_score: int | None = None,
    warning: str | None = None,
) -> dict:
    """Extract PDF text to book_{slug}_full.txt and return extract JSON fields."""
    doc = pymupdf.open(str(pdf_path))
    full_text = [page.get_text() for page in doc]
    doc.close()

    output_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(redact_http_urls_in_text("\n\n".join(full_text)))

    total_chars = sum(len(t) for t in full_text)
    payload = {
        "full_text_path": str(output_path),
        "page_count": len(full_text),
        "total_chars": total_chars,
        "estimated_words": total_chars // 5,
        "title": pdf_path.stem,
    }
    if quality_score is not None:
        payload["quality_score"] = quality_score
    if warning:
        payload["degraded_warning"] = warning
    return payload


def step_extract_mobi(task: dict, config: Config) -> str:
    """Convert MOBI/AZW3 to ePub via calibre, then extract like digital ePub."""
    slug = task.get("slug", task["id"][:8])
    mobi_path = _resolve_local_book_path(
        task, config, {".mobi", ".azw3"}, "extract_mobi",
    )

    if not shutil.which("ebook-convert"):
        raise StepError(
            "Missing dependency: calibre ebook-convert — install calibre",
            error_type=ErrorType.PAUSE,
            step_name="extract_mobi",
        )

    epub_path = Path(config.temp_dir) / f"book_{slug}_converted.epub"
    try:
        result = subprocess.run(
            ["ebook-convert", str(mobi_path), str(epub_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise StepError(
                f"ebook-convert failed: {result.stderr.strip() or result.stdout.strip()}",
                error_type=ErrorType.FAIL,
                step_name="extract_mobi",
            )
    except subprocess.TimeoutExpired as exc:
        raise StepError(
            "ebook-convert timed out",
            error_type=ErrorType.FAIL,
            step_name="extract_mobi",
        ) from exc

    child_task = {**task, "source_url": str(epub_path)}
    return step_extract_epub(child_task, config)


def step_extract_pdf_scanned(task: dict, config: Config) -> str:
    """OCR a scanned PDF via marker-pdf and write full text for synthesis."""
    slug = task.get("slug", task["id"][:8])
    pdf_path = _resolve_local_book_path(
        task, config, {".pdf"}, "extract_pdf_scanned",
    )

    marker_bin = shutil.which("marker_single") or shutil.which("marker")
    if not marker_bin:
        raise StepError(
            "Missing dependency: marker-pdf CLI (marker_single) — install marker-pdf",
            error_type=ErrorType.PAUSE,
            step_name="extract_pdf_scanned",
        )

    ocr_dir = Path(config.temp_dir) / f"book_{slug}_ocr"
    if ocr_dir.exists():
        shutil.rmtree(ocr_dir)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [marker_bin, str(pdf_path), str(ocr_dir)],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            raise StepError(
                f"marker-pdf OCR failed: {result.stderr.strip() or result.stdout.strip()}",
                error_type=ErrorType.FAIL,
                step_name="extract_pdf_scanned",
            )
    except subprocess.TimeoutExpired as exc:
        raise StepError(
            "marker-pdf OCR timed out",
            error_type=ErrorType.FAIL,
            step_name="extract_pdf_scanned",
        ) from exc

    md_files = sorted(ocr_dir.rglob("*.md"))
    if not md_files:
        raise StepError(
            f"marker-pdf produced no markdown under {ocr_dir}",
            error_type=ErrorType.FAIL,
            step_name="extract_pdf_scanned",
        )

    combined = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in md_files)
    output_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
    output_path.write_text(redact_http_urls_in_text(combined), encoding="utf-8")

    task["_book_title"] = pdf_path.stem
    task["_book_author"] = "Unknown"

    return json.dumps({
        "full_text_path": str(output_path),
        "page_count": len(md_files),
        "total_chars": len(combined),
        "estimated_words": len(combined) // 5,
        "title": task["_book_title"],
        "ocr_engine": "marker-pdf",
    })


def step_extract_pdf_degraded(task: dict, config: Config) -> str:
    """Extract degraded PDF text — warn and attempt direct pymupdf extraction."""
    slug = task.get("slug", task["id"][:8])
    pdf_path = _resolve_local_book_path(
        task, config, {".pdf"}, "extract_pdf_degraded",
    )

    if not HAS_PYMUPDF:
        raise StepError(
            "Missing dependency: pymupdf — install with pip install pymupdf",
            error_type=ErrorType.PAUSE,
            step_name="extract_pdf_degraded",
        )

    tier, score = _pdf_preflight(pdf_path)
    warning = (
        f"Degraded PDF ingest (preflight={tier!r}, score={score}/100) — "
        "using existing text layer; quality may be poor"
    )
    _log.warning("%s for %s", warning, pdf_path)

    try:
        payload = _write_pdf_full_text(
            pdf_path, slug, config, quality_score=score, warning=warning,
        )
        task["_book_title"] = pdf_path.stem
        task["_book_author"] = "Unknown"
        payload["title"] = task["_book_title"]
        return json.dumps(payload)
    except Exception as e:
        raise StepError(
            f"Degraded PDF extraction failed: {e}",
            error_type=ErrorType.FAIL,
            step_name="extract_pdf_degraded",
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
source_url: {_vault_frontmatter_source_url(task.get("source_url", ""))}
source_files:
  text: <filled by archival step>
confidence: plausible
tags: [book, book-summary]
---

# {title}

**Author:** {author}

"""
    write_vault_note(note_path, frontmatter + note_content)
    append_note_tags(note_path, task.get("tags"))

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
        StepError: If quality gates fail (unrecoverable — FAIL, no retry).
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"

    passed, failures = quality_gate_book(note_path, config)

    if not passed:
        raise StepError(
            f"Book quality gates failed: {'; '.join(failures)}",
            error_type=ErrorType.FAIL,
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

    source_suffixes = {".epub", ".pdf", ".mobi", ".azw3"}
    try:
        sf = _resolve_local_book_path(
            task, config, source_suffixes, "book_minio_upload",
        )
        sf_dest = f"{config.minio_bucket}/books/{slug}{sf.suffix.lower()}"
        try:
            upload_file_to_minio(sf, sf_dest, timeout=120)
            uploaded.append(f"books/{slug}{sf.suffix.lower()}")
        except StepError:
            pass  # Source file upload is best-effort
    except StepError:
        pass  # Source file upload is best-effort


    # Update note frontmatter with source_files URLs
    note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"
    if note_path.exists():
        content = note_path.read_text(encoding="utf-8")
        lines = ["source_files:"]
        for u in uploaded:
            suffix = Path(u).suffix.lower().lstrip(".")
            kind = "epub" if suffix == "epub" else ("pdf" if suffix == "pdf" else "text")
            lines.append(f"  {kind}: {u}")
        block = "\n".join(lines)
        if "source_files:" in content:
            content = re.sub(
                r"source_files:.*?(?=\n\w|\Z)", block, content, flags=re.DOTALL
            )
        else:
            content = content[:3] + "\n" + block + content[3:]
        write_vault_note(note_path, content)

    return json.dumps({
        "uploaded": uploaded,
        "urls": [f"{u}" for u in uploaded],
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

register_pipeline("book_mobi", [
    Step(name="extract_text", fn=step_extract_mobi, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

register_pipeline("book_scanned", [
    Step(name="extract_text", fn=step_extract_pdf_scanned, on_error="pause"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

register_pipeline("book_degraded", [
    Step(name="extract_text", fn=step_extract_pdf_degraded, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_book, on_error="retry"),
    Step(name="quality_gate", fn=step_book_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_book_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_book_git_push, on_error="retry"),
])

register_pipeline("mega", [
    Step(name="megadl_download", fn=step_megadl_download, on_error="pause"),
    Step(name="classify_content", fn=step_classify_content, on_error="retry"),
    Step(name="delegate", fn=step_delegate_books, on_error="fail"),
])

# Register book_batch pipeline (for mega.nz book archives)
register_pipeline("book_batch", [
    Step(name="megadl_download", fn=step_megadl_download, on_error="pause"),
    Step(name="classify_content", fn=step_classify_content, on_error="retry"),
    Step(name="delegate", fn=step_delegate_books, on_error="fail"),
])

# Registers mega_video pipeline (Mega-hosted video files)
import agent.pipelines.mega_video  # noqa: F401, E402

_missing_classify = CLASSIFY_BOOK_PIPELINES - set(PIPELINES.keys())
if _missing_classify:
    raise RuntimeError(
        f"Invariant: classify_content pipelines must be registered: {_missing_classify}"
    )
