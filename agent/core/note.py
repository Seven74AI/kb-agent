"""Read vault notes by slug."""

import re
from pathlib import Path

from agent.archival import CONTENT_TYPE_TO_VAULT_FOLDER as CONTENT_TYPE_FOLDERS
from agent.core.slugify import SLUG_MAX_LEN
from agent.redact import (
    prepare_vault_note_for_write,
    redact_http_urls_in_text,
    scrub_vault_note_content,
    write_vault_note,
)

def _vault_folder(content_type: str) -> str:
    return CONTENT_TYPE_FOLDERS.get(content_type, "articles")


def _slug_path(slug: str) -> str:
    return re.sub(r'[<>:"/\\|?*=&]', "_", slug)[:SLUG_MAX_LEN]


def _note_candidates(slug: str, vault_path: Path, content_type: str | None) -> list[Path]:
    """Build ordered candidate paths for a note slug."""
    knowledge = vault_path / "Knowledge base"
    safe = _slug_path(slug)
    candidates: list[Path] = []

    if content_type:
        folder = _vault_folder(content_type)
        if folder:
            candidates.append(knowledge / folder / f"{safe}.md")
        else:
            candidates.append(knowledge / f"{safe}.md")

    for folder in sorted(set(CONTENT_TYPE_FOLDERS.values())):
        if folder:
            p = knowledge / folder / f"{safe}.md"
            if p not in candidates:
                candidates.append(p)

    candidates.append(knowledge / f"{safe}.md")
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
            raw_content = path.read_text(encoding="utf-8")
            return {
                "slug": path.stem,
                "path": str(rel),
                "content": scrub_vault_note_content(
                    redact_http_urls_in_text(raw_content),
                ),
            }
    return None
