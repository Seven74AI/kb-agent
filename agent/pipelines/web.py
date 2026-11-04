"""Web article pipeline — Firecrawl extraction, LLM synthesis, quality gates,
MinIO archival, and Git push.

The simplest pipeline. No cookies, no video. Verifies the entire architecture.
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import get_llm_client, LLMClient
from agent.quality import quality_gate_web, quality_gate_retry_loop
from agent.archival import append_note_tags
from agent.archival import step_minio_upload as archival_minio_upload
from agent.archival import step_git_push as archival_git_push
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also


# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate token ratio for English text
CHARS_PER_TOKEN = 4

# Threshold above which we use progressive chunking
CHUNK_TOKEN_THRESHOLD = 50_000
CHUNK_OVERLAP_TOKENS = 10_000

# Max retry rounds for quality gate
MAX_QUALITY_RETRIES = 3


# ── Substack-Specific Steps ────────────────────────────────────────────────────

# Substack boilerplate patterns to remove from Firecrawl markdown
SUBSTACK_BOILERPLATE = [
    r'^Subscribe\s*$.*?(?=\n##|\n#|\Z)',
    r'^Substack is the home for great culture.*$',
    r'^\*This is .*? of .*?\.\*$',
    r'^Thanks for reading .*?\!.*$',
    r'^\*\*Subscribe\*\*.*$',
    r'^\[Subscribe\].*$',
    r'^### Subscribe to .*$',
    r'^\n---\n\n_Subscribe to get.*$',
    r'^Ready for more\?.*$',
    r'^\nSubstack\s*\n© \d{4}.*$',
    r'^\n\[Share\].*$',
]


def step_substack_cleanup(task: dict, config: Config) -> str:
    """Remove Substack boilerplate from the raw Firecrawl markdown.

    Strips subscribe CTAs, author avatar links, discussion/promo sections,
    and footer cruft. Keeps the article content intact.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        'cleaned' if successful.
    """
    slug = task.get("slug", task["id"][:8])
    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"

    if not raw_path.exists():
        return "no raw file to clean"

    content = raw_path.read_text(encoding="utf-8")

    for pattern in SUBSTACK_BOILERPLATE:
        content = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # Remove consecutive blank lines (max 2)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Remove trailing whitespace on each line
    content = '\n'.join(line.rstrip() for line in content.splitlines())

    raw_path.write_text(content, encoding="utf-8")
    return "cleaned"


def step_substack_video_pause(task: dict, config: Config) -> str:
    """Pause a Substack video task — video download is not yet implemented.

    Per CONTEXT.md: Substack video download is NOT fully solved.
    Mux CDN requires signed JWT token not available via the public API.

    This step always raises StepError(PAUSE) so the consumer stores the
    reason and the operator can handle it manually.

    Raises:
        StepError(PAUSE): Always — video download not implemented.
    """
    from agent.exceptions import StepError, ErrorType
    raise StepError(
        "Substack video download failed — Mux CDN requires signed JWT, "
        "not available via public API. See CONTEXT.md.",
        error_type=ErrorType.PAUSE,
        step_name="substack_video_pause",
    )


def step_substack_image_extract(task: dict, config: Config) -> str:
    """Download the lead image from a Substack image post and run Tesseract OCR.

    Uses curl to fetch the page, extracts the image URL from preloads JSON
    or og:image meta, downloads the image, and runs tesseract with
    multi-language support (eng+fra+spa+deu+ara).

    The OCR output is appended to the raw markdown file under an ## Image section.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        'image_extracted: <ocr_chars> chars' if successful.
    """
    import json as _json

    slug = task.get("slug", task["id"][:8])
    url = task["source_url"]

    # Fetch the page HTML
    result = subprocess.run(
        ["curl", "-s", "--max-time", "15", url],
        capture_output=True, text=True, timeout=20,
    )
    html = result.stdout

    image_url = None

    # Try preloads JSON first
    preloads_match = re.search(
        r'window\.__preloads\s*=\s*(\{.+?\});\s*</script>',
        html, re.DOTALL,
    )
    if preloads_match:
        try:
            preloads = _json.loads(preloads_match.group(1))
            attachments = preloads.get("attachments", [])
            for att in attachments:
                if att.get("type") == "image":
                    image_url = att.get("url") or att.get("src")
                    if image_url:
                        break
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback: og:image meta tag
    if not image_url:
        og_match = re.search(
            r'<meta\s+property="og:image"\s+content="([^"]+)"',
            html, re.IGNORECASE,
        )
        if og_match:
            image_url = og_match.group(1)

    if not image_url:
        return "no image found"

    # Download the image
    img_path = Path(config.temp_dir) / f"{slug}_image"
    dl_result = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "30", "-o", str(img_path), image_url],
        capture_output=True, text=True, timeout=35,
    )

    if dl_result.returncode != 0 or not img_path.exists() or img_path.stat().st_size < 100:
        return f"image download failed: {dl_result.stderr[:100] if dl_result.stderr else 'empty file'}"

    # Run Tesseract OCR
    ocr_result = subprocess.run(
        ["tesseract", str(img_path), "stdout",
         "-l", "eng+fra+spa+deu+ara"],
        capture_output=True, text=True, timeout=60,
    )

    ocr_text = ocr_result.stdout.strip()
    if not ocr_text:
        return "image extracted but OCR produced no text"

    # Append OCR output to raw markdown
    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
    if raw_path.exists():
        existing = raw_path.read_text(encoding="utf-8")
        new_content = existing + f"\n\n## Image\n\nOCR from lead image:\n\n{ocr_text}\n"
        raw_path.write_text(new_content, encoding="utf-8")

    return f"image_extracted: {len(ocr_text)} chars OCR"


# ── Firecrawl Extraction ─────────────────────────────────────────────────────

def step_firecrawl_extract(task: dict, config: Config) -> str:
    """Extract article content via Firecrawl, save raw markdown.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the raw markdown file.

    Raises:
        ValueError: If content is too short or extraction fails.
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])

    result = subprocess.run(
        [
            "curl", "-s", "--max-time", "60",
            config.firecrawl_url,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "url": url,
                "formats": ["markdown"],
                "waitFor": 3000,
            }),
        ],
        capture_output=True, text=True, timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Firecrawl curl failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        markdown = data["data"]["markdown"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Firecrawl response parse error: {e}") from e

    if len(markdown) < 500:
        raise ValueError(
            f"Content too short ({len(markdown)} chars) — "
            f"possible 404 or empty page"
        )

    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
    raw_path.write_text(markdown, encoding="utf-8")

    return str(raw_path)


# ── LLM Synthesis ────────────────────────────────────────────────────────────

def step_synthesize_web(task: dict, config: Config) -> str:
    """LLM synthesis for web articles — reads source, builds prompt, calls LLM,
    writes note to vault.

    Handles chunking transparently: single pass for ≤50K tokens,
    progressive chunking for larger content.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    source_url = task.get("source_url", "")
    source_path = Path(config.temp_dir) / f"{slug}_raw.md"

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_content = source_path.read_text(encoding="utf-8")
    estimated_tokens = _estimate_tokens(source_content)

    # Get the LLM client
    llm = get_llm_client(config)

    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    if estimated_tokens <= CHUNK_TOKEN_THRESHOLD:
        # Single pass
        note_content = _synthesize_single_pass(
            source_content, source_url, slug, config, llm
        )
    else:
        # Progressive chunking
        note_content = _synthesize_chunked(
            source_content, source_url, slug, config, llm
        )

    note_path.write_text(note_content, encoding="utf-8")
    append_note_tags(note_path, task.get("tags"))
    return str(note_path)


def _synthesize_single_pass(
    source: str, source_url: str, slug: str, config: Config, llm: LLMClient,
) -> str:
    """Single-pass synthesis — source fits in one LLM call."""
    system_prompt = _build_synthesis_prompt(source_url, slug, config)
    user_prompt = (
        f"Here is the source article. Read it fully, then write a comprehensive "
        f"note following the template and quality rules in the system prompt.\n\n"
        f"=== SOURCE ===\n{source}\n=== END SOURCE ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
    return response.content


def _synthesize_chunked(
    source: str, source_url: str, slug: str, config: Config, llm: LLMClient,
) -> str:
    """Progressive chunked synthesis for large content.

    1. Find semantic boundaries with a cheap LLM call
    2. Synthesize each chunk progressively, feeding the previous note as context
    3. Final polish pass to unify the note
    """
    boundaries = _find_semantic_boundaries(source, llm)
    if not boundaries:
        # Fallback: single pass if boundary detection fails
        return _synthesize_single_pass(source, source_url, slug, config, llm)

    previous_note = ""
    total = len(boundaries)

    for i, (start, end) in enumerate(boundaries):
        chunk = source[start:end]

        # Add overlap from previous chunk
        if i > 0:
            overlap_start = max(0, boundaries[i - 1][1] - CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN)
            chunk = source[overlap_start:end]

        chunk_note = _synthesize_chunk(
            chunk, previous_note, source_url, slug, config, llm,
            chunk_index=i, total=total,
        )
        previous_note = chunk_note

    # Final polish
    polish_prompt = (
        "You are a knowledge base editor. Review and polish this note to ensure:\n"
        "1. Consistent tone and depth across all sections\n"
        "2. No duplicate content introduced by chunk overlap\n"
        "3. Key Claims are properly numbered and each has a direct quote with '> '\n"
        "4. Section-by-Section Analysis is the longest section\n"
        "5. All template sections are present and populated\n\n"
        "Return the polished note, maintaining YAML frontmatter exactly."
    )
    polish_response = llm.call(
        system_prompt=polish_prompt,
        user_prompt=f"Polish this note:\n\n{previous_note}",
    )
    return polish_response.content


def _synthesize_chunk(
    chunk: str, previous_note: str, source_url: str, slug: str,
    config: Config, llm: LLMClient, chunk_index: int, total: int,
) -> str:
    """Synthesize one chunk, building on the previous partial note."""
    is_first = (chunk_index == 0)
    is_last = (chunk_index == total - 1)

    if is_first:
        system_prompt = _build_synthesis_prompt(source_url, slug, config)
        system_prompt += (
            f"\n\nCHUNK {chunk_index + 1}/{total}: This is the BEGINNING of the article. "
            "Write a note covering this portion. Later chunks will extend it."
        )
    elif is_last:
        system_prompt = (
            "You are a knowledge base curator finishing a note from the final chunk. "
            "Previous note content is provided. Extend it with this chunk's material: "
            "add new Key Claims, extend Section-by-Section, update Critical Analysis "
            "and Nuances with the full picture now available. "
            "Return the COMPLETE updated note (not just the new part)."
        )
    else:
        system_prompt = (
            "You are a knowledge base curator extending a note with a middle chunk. "
            "Previous note content is provided. Extend it with this chunk's material. "
            "Return the COMPLETE updated note (not just the new part)."
        )

    user_prompt = (
        f"PREVIOUS NOTE (so far):\n{previous_note}\n\n"
        f"=== NEW CHUNK {chunk_index + 1}/{total} ===\n{chunk}\n=== END CHUNK ==="
    ) if previous_note else (
        f"=== CHUNK {chunk_index + 1}/{total} ===\n{chunk}\n=== END CHUNK ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
    return response.content


def _build_synthesis_prompt(source_url: str, slug: str, config: Config) -> str:
    """Build the system prompt for web article synthesis.

    Loads the deep-treatment template and injects quality rules.
    """
    template_path = config.kb_agent_dir / "templates" / "deep-treatment-template.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = _fallback_template()

    source_domain = urlparse(source_url).netloc or "unknown"
    date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are a knowledge base curator. Follow the deep treatment standard.

SOURCE: The user will provide the full article text. Read it ENTIRELY before writing.

TEMPLATE (follow this exact structure):
{template}

QUALITY RULES (non-negotiable):
- Key Claims: ≥4 specific claims, each with a direct quote (start line with '> ')
- Section-by-Section Analysis: must be the LONGEST section — longer than Key Claims + Critical Analysis combined
- Target note size: 10,000–20,000 characters
- Every template section must be populated — no empty or placeholder sections
- Read the ENTIRE source before writing. Never push v1 — re-read and refine.
- Use direct quotes from the source extensively. Every claim needs evidence.
- The Summary should capture the essential argument in 3-5 sentences.
- Context must identify who wrote this, why now, and any conflicts of interest.
- Critical Analysis must identify what's new vs restating, what's omitted.
- Nuances must address what the source exaggerates or distorts.

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: {source_domain}
  source_url: {source_url}
  source_files:
    raw: <filled by archival step>
  confidence: plausible
  tags: [tag1, tag2, tag3]
- Then the note sections as shown in the template.
- Leave the source_files: field in frontmatter — it will be filled later.
- Leave the See Also section empty — it will be filled later by a separate step.

DO NOT invent content. Every factual claim must be traceable to the source.
DO NOT use web_search or external knowledge — work only from the provided source.
DO NOT truncate or summarize away important detail from the source."""
    return prompt


def _fallback_template() -> str:
    """Fallback template if the deep-treatment template file is missing."""
    return """---
date: YYYY-MM-DD
source: <platform/author>
source_url: <URL>
source_files:
  raw: <filled by archival step>
confidence: plausible
tags: []
---

# Title

## Summary
3-5 sentences. The essential argument.

## Key Claims
≥4 specific claims with direct quotes (>).

### 1. Claim title
> « direct quote from the source »
Analysis.

## Section-by-Section Analysis
The DOMINANT section — longer than Key Claims + Critical Analysis combined.

### Section 1: Title
What it argues, key evidence, notable quotes.

## Context
Who wrote this, why now, conflicts of interest.

## Critical Analysis
What's new vs restating? What's omitted? Keep shorter than thematic analysis.

## Nuances
What the source omits, exaggerates, or distorts.

## Reliability
Confidence level with justification.

## Sources
- Original source (full citation)
- Raw source archived: minio: URL

## See Also
- (filled by See Also step)
"""


# ── Chunking Helpers ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate for English text (~4 chars/token)."""
    return len(text) // CHARS_PER_TOKEN


def _find_semantic_boundaries(source: str, llm: LLMClient) -> list[tuple[int, int]]:
    """Find semantic split points in large content using a cheap LLM call.

    The LLM identifies natural boundaries (section breaks, topic shifts)
    and returns byte offsets for chunking.

    Args:
        source: Full source text.
        llm: LLM client (can use a cheaper model).

    Returns:
        List of (start_offset, end_offset) tuples for each chunk.
        Returns empty list if detection fails.
    """
    estimated_tokens = _estimate_tokens(source)
    target_chunks = max(2, estimated_tokens // CHUNK_TOKEN_THRESHOLD)

    # Send a sample of the structure for boundary detection
    # The LLM needs to see section headers to find good split points
    structure_sample = source[:3000] + "\n\n...[middle of article]...\n\n" + source[-3000:]

    boundary_prompt = (
        "You are a text structure analyzer. The user will show you the beginning "
        "and end of a long article, plus a list of all section headers in order. "
        "Your job: identify the best places to split the article into "
        f"{target_chunks} chunks of roughly equal size (~{CHUNK_TOKEN_THRESHOLD} tokens each).\n\n"
        "Rules:\n"
        "1. Split at natural boundaries: section headers (##, #), major topic shifts\n"
        "2. Each chunk should be as close to equal size as possible\n"
        "3. Never split in the middle of a paragraph — find the nearest heading\n\n"
        "RESPOND with ONLY a JSON list of byte offset pairs:\n"
        '[[0, 50000], [50000, 100000], ...]\n'
        "The first pair starts at 0. The last pair ends at the article length.\n"
        "If you cannot determine boundaries, respond with []"
    )

    # Also extract section headers to help the LLM
    headers = re.findall(r'^(#{1,3} .+)$', source, re.MULTILINE)
    headers_text = "\n".join(headers[:50])  # Limit to first 50 headers

    user_prompt = (
        f"Article length: {len(source)} chars (~{estimated_tokens} tokens)\n"
        f"Target: {target_chunks} chunks\n\n"
        f"STRUCTURE SAMPLE (beginning + end):\n{structure_sample}\n\n"
        f"ALL SECTION HEADERS (in order):\n{headers_text}\n\n"
        "Return the JSON boundary list now."
    )

    try:
        response = llm.call(system_prompt=boundary_prompt, user_prompt=user_prompt)
        # Extract JSON from response
        content = response.content.strip()
        # Find the JSON array in the response
        json_match = re.search(r'\[\[.*?\]\]', content, re.DOTALL)
        if json_match:
            boundaries = json.loads(json_match.group())
            if boundaries and isinstance(boundaries, list):
                # Validate: all pairs, ascending, cover full range
                valid = all(
                    isinstance(b, list) and len(b) == 2 and
                    isinstance(b[0], int) and isinstance(b[1], int)
                    for b in boundaries
                )
                if valid and boundaries[0][0] == 0:
                    # Ensure last boundary covers the full source
                    boundaries[-1][1] = len(source)
                    return [(int(b[0]), int(b[1])) for b in boundaries]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: equal-sized chunks
    return _equal_chunks(source, target_chunks)


def _equal_chunks(source: str, num_chunks: int) -> list[tuple[int, int]]:
    """Fallback: split source into equal-sized chunks at paragraph boundaries."""
    chunk_size = len(source) // num_chunks
    boundaries = []
    start = 0

    for i in range(num_chunks):
        if i == num_chunks - 1:
            boundaries.append((start, len(source)))
        else:
            end = start + chunk_size
            # Find nearest paragraph break after `end`
            paragraph_break = source.find("\n\n", end)
            if paragraph_break != -1 and paragraph_break - end < chunk_size // 2:
                end = paragraph_break
            boundaries.append((start, end))
            start = end

    return boundaries



# ── Quality Gate Step (wraps quality.py) ──────────────────────────────────────

def step_quality_gate(task: dict, config: Config) -> str:
    """Run quality gates on the synthesized note.

    Wraps quality.quality_gate_web with the retry loop.
    Returns 'passed' or raises StepError(FAIL) with failure details.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        'passed' if all gates pass.

    Raises:
        StepError: If quality gates fail after max retries.
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    llm = get_llm_client(config)

    passed, rounds = quality_gate_retry_loop(
        note_path, config, quality_gate_web, llm, max_rounds=MAX_QUALITY_RETRIES,
    )

    if not passed:
        # Read the failure message written by the retry loop
        content = note_path.read_text(encoding="utf-8")
        raise StepError(
            f"Quality gate failed after {rounds} rounds. "
            f"First line: {content.split(chr(10))[0]}",
            error_type=ErrorType.FAIL,
            step_name="quality_gate",
        )

    return "passed"


# ── MinIO Upload Step (wraps archival.py) ─────────────────────────────────────

def step_minio_upload(task: dict, config: Config) -> str:
    """Upload source content to MinIO and add source_files: field to note frontmatter.

    Wraps archival.step_minio_upload and adds the source_files URL to the note.

    Args:
        task: Task dict with id, slug, content_type, source_url.
        config: Config dataclass.

    Returns:
        MinIO object URL for the archived file.
    """
    slug = task.get("slug", task["id"][:8])

    # Upload the raw source
    minio_url = archival_minio_upload(task, config)

    # Add source_files: field to the note's frontmatter
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    if note_path.exists():
        content = note_path.read_text(encoding="utf-8")
        # Insert or replace source_files.raw in frontmatter
        if content.startswith("---"):
            end_fm = content.find("---", 3)
            if end_fm != -1:
                frontmatter = content[3:end_fm]
                if re.search(r'^source_files:', frontmatter, re.MULTILINE):
                    # Replace the raw: placeholder with the real URL
                    content = re.sub(
                        r'^  raw:.*$', f'  raw: {minio_url}',
                        content, flags=re.MULTILINE
                    )
                else:
                    # No source_files: field — insert after the opening ---
                    content = content[:3] + "\nsource_files:\n  raw: " + minio_url + content[3:]
                note_path.write_text(content, encoding="utf-8")

    return minio_url


# ── Git Push Step (wraps archival.py) ─────────────────────────────────────────

def step_git_push(task: dict, config: Config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Wraps archival.step_git_push.

    Args:
        task: Task dict with id, slug, content_type.
        config: Config dataclass.

    Returns:
        The commit hash of the push.
    """
    return archival_git_push(task, config)


# ── Pipeline Registration ────────────────────────────────────────────────────

register_pipeline("web", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

# Substack pipelines — share the web article template
register_pipeline("substack", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("substack_text", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("substack_video", [
    Step(name="substack_video_pause", fn=step_substack_video_pause, on_error="fail"),
])

register_pipeline("substack_image", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="substack_image_extract", fn=step_substack_image_extract, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
