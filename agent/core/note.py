"""Read vault notes by slug."""

import re
from pathlib import Path

from agent.archival import CONTENT_TYPE_TO_VAULT_FOLDER as CONTENT_TYPE_FOLDERS


def _vault_folder(content_type: str) -> str:
    return CONTENT_TYPE_FOLDERS.get(content_type, "articles")


def _slug_path(slug: str) -> str:
    return re.sub(r'[<>:"/\\|?*=&]', "_", slug)[:80]


def _note_candidates(slug: str, vault_path: Path, content_type: str | None) -> list[Path]:
    """Build ordered candidate paths for a note slug."""
    knowledge = vault_path / "Knowledge base"
    safe = _slug_path(slug)
    candidates: list[Path] = []

    if content_type:
        folder = _vault_folder(content_type)
        if folder:
            candidates.append(knowledge / folder / f"{safe}.md")
            candidates.append(knowledge / folder / f"{slug}.md")
        else:
            candidates.append(knowledge / f"{safe}.md")
            candidates.append(knowledge / f"{slug}.md")

    for folder in sorted(set(CONTENT_TYPE_FOLDERS.values())):
        if folder:
            p = knowledge / folder / f"{safe}.md"
            if p not in candidates:
                candidates.append(p)

    candidates.append(knowledge / f"{safe}.md")
    candidates.append(knowledge / f"{slug}.md")
    return candidates


def get_note(
    slug: str,
    vault_path: Path,
    *,
    content_type: str | None = None,
) -> dict | None:
    """Load a note from the vault by slug.

    Returns:
        Dict with slug, path (relative to vault), and content, or None if not found.
    """
    slug = (slug or "").strip()
    if not slug:
        return None

    for path in _note_candidates(slug, vault_path, content_type):
        if path.is_file():
            try:
                rel = path.relative_to(vault_path)
            except ValueError:
                rel = path
            return {
                "slug": path.stem,
                "path": str(rel),
                "content": path.read_text(encoding="utf-8"),
            }
    return None
