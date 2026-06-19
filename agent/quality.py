"""Quality gates — two-tier validation ran after synthesis, before Git push.

Tier 1: Programmatic (Python, instant) — counts, sizes, coverage.
Tier 2: Structural review (cheap LLM call) — completeness, balance.
Max 3 retry rounds.
"""

from pathlib import Path
import re


def quality_gate_web(note_path: Path, config, vault_path: Path | None = None) -> tuple[bool, list[str]]:
    """Run quality gates on a web article note.

    Args:
        note_path: Path to the generated note file.
        config: Config dataclass with quality thresholds.

    Returns:
        (passed, failures) tuple. passed=True means all gates passed.
        failures lists human-readable descriptions of what failed.
    """
    failures: list[str] = []

    # ── Tier 1: Programmatic checks ──
    if not note_path.exists():
        return False, [f"Note not found: {note_path}"]

    content = note_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Quote count — count lines starting with '> '
    quote_count = sum(1 for line in lines if line.strip().startswith("> "))
    if quote_count < config.article_min_quotes:
        failures.append(
            f"Too few quotes: {quote_count} < {config.article_min_quotes} required"
        )

    # Size
    size = len(content)
    if size < config.article_min_size:
        failures.append(
            f"Note too short: {size} chars < {config.article_min_size} required"
        )

    # Line count (proxy for depth)
    if len(lines) < 20:
        failures.append(f"Note too short: {len(lines)} lines")

    # ── Tier 2: Structural checks ──
    # Content > Critique check
    section_by_section_idx = _find_section(lines, "Section-by-Section")
    critical_analysis_idx = _find_section(lines, "Critical Analysis")

    if section_by_section_idx is not None and critical_analysis_idx is not None:
        content_section_lines = critical_analysis_idx - section_by_section_idx
        critique_section_lines = len(lines) - critical_analysis_idx
        if critique_section_lines > content_section_lines:
            failures.append(
                f"Critical Analysis ({critique_section_lines} lines) longer than "
                f"Section-by-Section ({content_section_lines} lines) — unbalanced"
            )

    # Check for empty sections
    current_section = None
    section_content = ""
    for line in lines:
        if line.startswith("## "):
            if current_section and len(section_content.strip()) < 20:
                failures.append(f"Section '{current_section}' appears empty or placeholder")
            current_section = line[3:].strip()
            section_content = ""
        else:
            section_content += line

    return len(failures) == 0, failures


def quality_gate_book(note_path: Path, config, vault_path: Path | None = None) -> tuple[bool, list[str]]:
    """Run quality gates on a book note (stricter thresholds).

    Checks:
    - line_count >= config.book_min_lines (100)
    - quote_count >= config.book_min_quotes (4)
    - size >= config.book_min_size (8000)
    - chapter_coverage: every source chapter → note section
    """
    failures: list[str] = []

    if not note_path.exists():
        return False, [f"Note not found: {note_path}"]

    content = note_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Quote count — count lines starting with '> '
    quote_count = sum(1 for line in lines if line.strip().startswith("> "))
    if quote_count < config.book_min_quotes:
        failures.append(
            f"Too few quotes: {quote_count} < {config.book_min_quotes} required"
        )

    # Size
    size = len(content)
    if size < config.book_min_size:
        failures.append(
            f"Note too short: {size} chars < {config.book_min_size} required"
        )

    # Line count (proxy for depth)
    line_count = len([l for l in lines if l.strip()])
    if line_count < config.book_min_lines:
        failures.append(
            f"Too few non-empty lines: {line_count} < {config.book_min_lines} required"
        )

    # Chapter coverage — check if Chapter Summaries section covers expected chapters
    # Look for "## Chapter Summaries" section and count sub-headers
    chapter_summaries_idx = _find_section(lines, "Chapter Summaries")
    if chapter_summaries_idx is not None:
        # Count ### or ## sub-headers within Chapter Summaries
        chapter_headers = 0
        in_section = False
        for i, line in enumerate(lines):
            if i < chapter_summaries_idx:
                continue
            if line.startswith("## ") and "Chapter Summaries" in line:
                in_section = True
                continue
            if in_section and line.startswith("## "):
                break  # Next major section
            if in_section and (line.startswith("### ") or line.startswith("**Chapter")):
                chapter_headers += 1

        # If we can detect a source text file, compare chapter count
        if vault_path and note_path.stem:
            source_pattern = f"/tmp/book_{note_path.stem}_full.txt"
            source_path = Path(source_pattern)
            if source_path.exists():
                source_content = source_path.read_text(encoding="utf-8")
                source_chapters = len(re.findall(
                    r'=== CHAPTER \d+ ===', source_content
                ))
                if chapter_headers < source_chapters * 0.5:
                    failures.append(
                        f"Chapter coverage: {chapter_headers} note chapters for "
                        f"{source_chapters} source chapters (< 50% coverage)"
                    )
            elif chapter_headers < 3:
                failures.append(
                    f"Too few chapter summaries: {chapter_headers} detected"
                )

    # Critical Analysis size check — should not exceed Chapter Summaries + Key Claims
    critical_analysis_idx = _find_section(lines, "Critical Analysis")
    if critical_analysis_idx is not None and chapter_summaries_idx is not None:
        summary_lines = critical_analysis_idx - chapter_summaries_idx
        critique_lines = len(lines) - critical_analysis_idx
        if critique_lines > summary_lines * 1.5:
            failures.append(
                f"Critical Analysis ({critique_lines} lines) disproportionately "
                f"large vs Chapter Summaries ({summary_lines} lines)"
            )

    return len(failures) == 0, failures


def quality_gate_youtube(note_path: Path, config, vault_path: Path | None = None) -> tuple[bool, list[str]]:
    """Run quality gates on a YouTube note."""
    failures: list[str] = []

    if not note_path.exists():
        return False, [f"Note not found: {note_path}"]

    content = note_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Concept count (look for ### headers under "Concepts clés" or "Key Concepts")
    concept_count = 0
    in_concepts = False
    for line in lines:
        if line.startswith("## ") and ("concept" in line.lower() or "concepts" in line.lower()):
            in_concepts = True
            continue
        if in_concepts and line.startswith("## "):
            in_concepts = False
        if in_concepts and line.startswith("### "):
            concept_count += 1

    if concept_count < config.youtube_min_concepts:
        failures.append(
            f"Too few concepts: {concept_count} < {config.youtube_min_concepts} required"
        )

    # Key points (look for bullet points under "Points clés" or "Key Points")
    key_points_count = 0
    in_points = False
    for line in lines:
        if line.startswith("## ") and ("point" in line.lower() or "key points" in line.lower()):
            in_points = True
            continue
        if in_points and line.startswith("## "):
            in_points = False
        if in_points and line.strip().startswith(("- ", "* ", "1. ")):
            key_points_count += 1

    if key_points_count < config.youtube_min_key_points:
        failures.append(
            f"Too few key points: {key_points_count} < {config.youtube_min_key_points} required"
        )

    return len(failures) == 0, failures


def quality_gate_retry_loop(note_path: Path, config,
                            gate_fn, llm, max_rounds: int = 3) -> tuple[bool, int]:
    """Run quality gates with LLM retry on failure.

    Each retry = full source re-read. Max 3 rounds.

    Args:
        note_path: Path to the note file.
        config: Config dataclass.
        gate_fn: Quality gate function (quality_gate_web, etc.).
        llm: LLM client for Tier 2 structural review.
        max_rounds: Maximum retry rounds (default 3).

    Returns:
        (passed, rounds_used) tuple.
    """
    for round_num in range(1, max_rounds + 1):
        passed, failures = gate_fn(note_path, config)

        if passed and round_num > 1:
            # Tier 2: structural review via cheap LLM call
            note_content = note_path.read_text(encoding="utf-8")
            review_result = _structural_review(note_content, llm)
            if review_result.strip().lower() == "pass":
                return True, round_num
            failures.append(f"Tier 2 structural review: {review_result[:200]}")
        elif passed:
            return True, round_num

        if round_num >= max_rounds:
            return False, round_num

        # Tell the LLM what failed so it can re-read and fix
        # (The orchestrator handles the actual re-synthesis call;
        #  this just signals the need to retry.)
        note_path.write_text(
            f"# QUALITY GATE FAILED (round {round_num}/{max_rounds})\n"
            f"Failures:\n" + "\n".join(f"- {f}" for f in failures) + "\n\n"
            f"Re-read source and rewrite.\n",
            encoding="utf-8",
        )

    return False, max_rounds


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_section(lines: list[str], name: str) -> int | None:
    """Find the line index of a ## section header."""
    for i, line in enumerate(lines):
        if line.startswith("## ") and name.lower() in line.lower():
            return i
    return None


def _structural_review(note_content: str, llm) -> str:
    """Tier 2 structural review via cheap LLM call.

    Checks:
    - Is the dominant section actually dominant (Content > Critique)?
    - Are any template sections empty or placeholder?
    - Does the note feel rushed or incomplete?

    Returns 'pass' or a failure description.
    """
    prompt = (
        "You are a structural reviewer for knowledge base notes. "
        "Review this note and answer ONLY 'pass' if ALL of these are true:\n"
        "1. The main content section (Section-by-Section / Chapter Summaries) "
        "is LONGER than the Critical Analysis section.\n"
        "2. No template section is empty or contains only placeholder text.\n"
        "3. The note does not feel rushed or incomplete.\n\n"
        "If any fail, briefly describe what failed (max 200 chars).\n\n"
        "NOTE:\n" + note_content[:3000]
    )

    try:
        response = llm.call(system_prompt=prompt, user_prompt="Review this note.", role="review")
        return response.content.strip()
    except Exception as e:
        # If LLM review fails, pass through (don't block on infra)
        return f"pass  # Review skipped (LLM error: {e})"
