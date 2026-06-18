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
from agent.llm import get_llm_client, LLMClient
from agent.quality import quality_gate_web, quality_gate_retry_loop
from agent.archival import step_minio_upload as archival_minio_upload
from agent.archival import step_git_push as archival_git_push
from agent.pipelines import Step, register_pipeline


# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate token ratio for English text
CHARS_PER_TOKEN = 4

# Threshold above which we use progressive chunking
CHUNK_TOKEN_THRESHOLD = 50_000
CHUNK_OVERLAP_TOKENS = 10_000

# Max retry rounds for quality gate
MAX_QUALITY_RETRIES = 3


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
  confidence: plausible
  tags: [tag1, tag2, tag3]
- Then the note sections as shown in the template.
- Leave the minio: field in frontmatter — it will be filled later.
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
minio: <filled by archival step>
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


# ── See Also ──────────────────────────────────────────────────────────────────

def step_see_also(task: dict, config: Config) -> str:
    """Search vault for related notes and add wikilinks to the note.

    Searches the vault for notes matching keywords from the generated note,
    then calls a lightweight LLM to select the most relevant ones and add
    a See Also section.

    Only links to notes that actually exist in the vault are added.

    Args:
        task: Task dict with slug, content_type.
        config: Config dataclass.

    Returns:
        Status message describing how many links were added.
    """
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "web")

    folder = "articles"
    note_path = config.vault_path / "Knowledge base" / folder / f"{slug}.md"

    if not note_path.exists():
        return f"No note to add See Also to: {note_path}"

    note_content = note_path.read_text(encoding="utf-8")

    # Skip if See Also already populated
    if "## See Also" in note_content and "[[ " not in note_content:
        # Has the section but no actual links yet — that's OK, we'll fill it
        pass

    # Extract keywords from the note for searching
    keywords = _extract_keywords(note_content)

    # Search vault for matching notes
    candidate_paths = _search_vault(keywords, config.vault_path, exclude_slug=slug)

    if not candidate_paths:
        # No related notes found
        return f"No related notes found in vault for {slug}"

    # Use a cheap LLM to pick the best related notes
    llm = get_llm_client(config)

    candidates_text = "\n".join(
        f"- [[{p.stem}]]: {p.read_text(encoding='utf-8')[:200]}..."
        for p in candidate_paths[:10]
    )

    see_also_prompt = (
        "You are a knowledge base linker. Given a note and a list of candidate "
        "related notes, select the 3-5 most relevant ones. A relevant note:\n"
        "1. Covers overlapping topics or themes\n"
        "2. Provides context that enriches the current note\n"
        "3. Is NOT about a completely different subject\n\n"
        "RESPOND with ONLY a ## See Also section with wikilinks, like:\n"
        "## See Also\n"
        "- [[note-slug-1]] — why it's related\n"
        "- [[note-slug-2]] — why it's related\n\n"
        "If none are truly related, respond with ## See Also\n_No related notes found._"
    )

    response = llm.call(
        system_prompt=see_also_prompt,
        user_prompt=(
            f"CURRENT NOTE:\n{note_content[:2000]}\n\n"
            f"CANDIDATE NOTES:\n{candidates_text}"
        ),
    )

    # Replace or append the See Also section
    see_also_content = response.content.strip()
    if "## See Also" not in see_also_content:
        see_also_content = f"## See Also\n{see_also_content}"

    updated = re.sub(
        r'## See Also.*?(?=\n## |\Z)',
        see_also_content,
        note_content,
        flags=re.DOTALL,
    )

    note_path.write_text(updated, encoding="utf-8")

    # Count links added
    link_count = see_also_content.count("[[")
    return f"Added {link_count} See Also links to {slug}"


def _extract_keywords(note_content: str, max_keywords: int = 8) -> list[str]:
    """Extract meaningful keywords from a note for vault search.

    Uses simple heuristics: section titles, capitalized phrases.
    """
    keywords = set()
    lines = note_content.split("\n")

    # Template headers to exclude
    template_headers = {
        "summary", "key claims", "key claims / points",
        "section-by-section analysis", "section-by-section / thematic analysis",
        "context", "critical analysis", "nuances", "reliability",
        "sources", "see also",
    }

    # Section headers — lines starting with #, ##, or ###
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("### ") and line_stripped[4:].strip()[0].isdigit():
            # Numbered sub-header like "### 1. Claim Title"
            header = line_stripped[4:].strip()
            # Strip leading number (e.g., "1. Claim Title" → "Claim Title")
            if ". " in header[:4]:
                header = header.split(". ", 1)[1]
            keywords.add(header)
        elif line_stripped.startswith("## ") and not line_stripped.startswith("### "):
            header = line_stripped[3:].strip()
            if header.lower() not in template_headers:
                keywords.add(header)

    # Capitalized multi-word phrases (potential names, concepts)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', note_content):
        phrase = match.group(1)
        if len(phrase) > 10 and not phrase.startswith(("The ", "This ", "That ")):
            keywords.add(phrase)

    return list(keywords)[:max_keywords]


def _search_vault(
    keywords: list[str], vault_path: Path, exclude_slug: str,
) -> list[Path]:
    """Search the vault for notes matching keywords.

    Uses grep for keyword matching — fast and works on the filesystem.
    """
    knowledge_dir = vault_path / "Knowledge base"
    if not knowledge_dir.exists():
        return []

    results: set[Path] = set()
    for kw in keywords[:5]:  # Limit keywords to avoid excessive grep calls
        try:
            proc = subprocess.run(
                ["grep", "-rl", "--include=*.md", kw, str(knowledge_dir)],
                capture_output=True, text=True, timeout=30,
            )
            for line in proc.stdout.strip().split("\n"):
                if line:
                    p = Path(line)
                    if p.stem != exclude_slug:
                        results.add(p)
        except subprocess.TimeoutExpired:
            continue

    return list(results)


# ── Quality Gate Step (wraps quality.py) ──────────────────────────────────────

def step_quality_gate(task: dict, config: Config) -> str:
    """Run quality gates on the synthesized note.

    Wraps quality.quality_gate_web with the retry loop.
    Returns 'passed' or raises ValueError with failure details.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        'passed' if all gates pass.

    Raises:
        ValueError: If quality gates fail after max retries.
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
        raise ValueError(
            f"Quality gate failed after {rounds} rounds. "
            f"First line: {content.split(chr(10))[0]}"
        )

    return "passed"


# ── MinIO Upload Step (wraps archival.py) ─────────────────────────────────────

def step_minio_upload(task: dict, config: Config) -> str:
    """Upload source content to MinIO and add minio: field to note frontmatter.

    Wraps archival.step_minio_upload and adds the minio URL to the note.

    Args:
        task: Task dict with id, slug, content_type, source_url.
        config: Config dataclass.

    Returns:
        MinIO object URL for the archived file.
    """
    slug = task.get("slug", task["id"][:8])

    # Upload the raw source
    minio_url = archival_minio_upload(task, config)

    # Add minio: field to the note's frontmatter
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    if note_path.exists():
        content = note_path.read_text(encoding="utf-8")
        # Insert or replace minio: field in frontmatter
        if content.startswith("---"):
            end_fm = content.find("---", 3)
            if end_fm != -1:
                frontmatter = content[3:end_fm]
                if re.search(r'^minio:', frontmatter, re.MULTILINE):
                    # LLM-template includes a minio: placeholder — replace with real URL
                    content = re.sub(
                        r'^minio:.*$', f'minio: {minio_url}',
                        content, flags=re.MULTILINE
                    )
                else:
                    # No minio: field — insert one after the opening ---
                    content = content[:3] + "\nminio: " + minio_url + content[3:]
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
    Step(name="firecrawl_extract", fn=step_firecrawl_extract),
    Step(name="synthesize_note", fn=step_synthesize_web),
    Step(name="quality_gate", fn=step_quality_gate),
    Step(name="minio_upload", fn=step_minio_upload),
    Step(name="see_also", fn=step_see_also),
    Step(name="git_push", fn=step_git_push),
])
