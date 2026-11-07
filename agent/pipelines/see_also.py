"""See Also — vault search + LLM-powered wikilinks for all content types.

Extracted from web.py (originally lines 605-762) and generalized to support
all content types: web/articles, books, threads, instagram, youtube/videos.

Every note gets a ## See Also section with wikilinks to related vault notes.
"""

import re
import subprocess
from pathlib import Path

from agent.archival import CONTENT_TYPE_TO_VAULT_FOLDER as CONTENT_TYPE_FOLDERS
from agent.config import Config
from agent.llm import get_llm_client
from agent.pipelines.video_common import _slug_path


def _get_folder(content_type: str) -> str:
    """Map content_type to its vault Knowledge base subfolder."""
    return CONTENT_TYPE_FOLDERS.get(content_type, "articles")


# ── Main step ─────────────────────────────────────────────────────────────────

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

    folder = _get_folder(content_type)
    note_path = config.vault_path / "Knowledge base" / folder / f"{slug}.md"

    # Threads and instagram pipelines sanitize their slugs with _slug_path
    if not note_path.exists():
        safe_slug = _slug_path(slug)
        alt_path = config.vault_path / "Knowledge base" / folder / f"{safe_slug}.md"
        if alt_path.exists():
            note_path = alt_path
        else:
            return f"No note to add See Also to: {note_path}"

    note_content = note_path.read_text(encoding="utf-8")

    # Skip if See Also already has substantive links (3+ wikilinks)
    existing_links = re.findall(r'\[\[([^\]]+)\]\]', note_content)
    see_also_header = "## Voir aussi" if content_type == "bunny_stream" else "## See Also"
    if len(existing_links) >= 3 and see_also_header in note_content:
        return f"See Also already populated with {len(existing_links)} links for {slug}"

    # Extract keywords from the note for searching
    keywords = _extract_keywords(note_content)

    # Search vault for matching notes
    candidate_paths = _search_vault(keywords, config.vault_path, exclude_slug=note_path.stem)

    if not candidate_paths:
        return f"No related notes found in vault for {slug}"

    # Use a cheap LLM to pick the best related notes
    llm = get_llm_client(config)

    candidates_text = "\n".join(
        f"- [[{p.stem}]]: {p.read_text(encoding='utf-8')[:200]}..."
        for p in candidate_paths[:10]
    )

    if content_type == "bunny_stream":
        see_also_prompt = (
            "Tu es un assistant de liaison pour une base de connaissances en français. "
            "À partir de la note courante et d'une liste de notes candidates, "
            "sélectionne les 3 à 5 plus pertinentes.\n\n"
            "RÉPONDS avec UNIQUEMENT une section ## Voir aussi avec des wikilinks :\n"
            "## Voir aussi\n"
            "- [[slug-note]] — pourquoi c'est pertinent\n\n"
            "Si aucune note n'est pertinente : ## Voir aussi\n_Aucune note connexe._"
        )
    else:
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
    if see_also_header not in see_also_content:
        see_also_content = f"{see_also_header}\n{see_also_content}"

    pattern = (
        r'## Voir aussi.*?(?=\n## |\Z)'
        if content_type == "bunny_stream"
        else r'## See Also.*?(?=\n## |\Z)'
    )
    updated = re.sub(
        pattern,
        see_also_content,
        note_content,
        flags=re.DOTALL,
    )

    note_path.write_text(updated, encoding="utf-8")

    # Count links added
    link_count = see_also_content.count("[[")
    return f"Added {link_count} See Also links to {slug}"


# ── Helpers ───────────────────────────────────────────────────────────────────

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
                ["grep", "-rlF", "--include=*.md", "--", kw, str(knowledge_dir)],
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
