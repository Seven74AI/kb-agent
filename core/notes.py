"""Vault note search and read (no Flask/MCP)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from core.types import NoteContent, NoteSearchHit, NoteSearchResult

_SOURCE_URL_RE = re.compile(
    r"^source_url:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def search_notes(
    vault_path: Path,
    query: str,
    *,
    limit: int = 20,
    knowledge_subdir: str = "Knowledge base",
) -> NoteSearchResult:
    """Full-text search markdown notes under the vault knowledge folder.

    Uses grep on the filesystem (same approach as dedup_check and see_also).
    """
    query = query.strip()
    if not query:
        return NoteSearchResult(query=query, hits=[])

    knowledge_dir = vault_path / knowledge_subdir
    if not knowledge_dir.exists():
        return NoteSearchResult(query=query, hits=[])

    try:
        proc = subprocess.run(
            ["grep", "-rl", "--include=*.md", query, str(knowledge_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return NoteSearchResult(query=query, hits=[])

    hits: list[NoteSearchHit] = []
    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        path = Path(line)
        snippet = _grep_snippet(path, query)
        hits.append(
            NoteSearchHit(
                path=str(path.relative_to(vault_path)),
                slug=path.stem,
                snippet=snippet,
            )
        )
        if len(hits) >= limit:
            break

    return NoteSearchResult(query=query, hits=hits)


def get_note(
    vault_path: Path,
    slug_or_path: str,
    *,
    knowledge_subdir: str = "Knowledge base",
) -> NoteContent | None:
    """Load a note by slug or vault-relative path.

    If ``slug_or_path`` has no directory component, searches common folders
    under ``Knowledge base/`` for ``{slug}.md``.
    """
    slug_or_path = slug_or_path.strip().strip("/")
    if not slug_or_path:
        return None

    candidate: Path | None = None
    if "/" in slug_or_path or slug_or_path.endswith(".md"):
        candidate = vault_path / slug_or_path
        if not candidate.suffix:
            candidate = candidate.with_suffix(".md")
    else:
        knowledge_dir = vault_path / knowledge_subdir
        for folder in (
            "articles",
            "videos",
            "books",
            "threads",
            "instagram",
            "",
        ):
            path = knowledge_dir / folder / f"{slug_or_path}.md" if folder else (
                knowledge_dir / f"{slug_or_path}.md"
            )
            if path.exists():
                candidate = path
                break

    if candidate is None or not candidate.exists():
        return None

    body = candidate.read_text(encoding="utf-8")
    source_match = _SOURCE_URL_RE.search(body)
    source_url = source_match.group(1).strip() if source_match else None

    return NoteContent(
        path=str(candidate.relative_to(vault_path)),
        slug=candidate.stem,
        body=body,
        frontmatter_source_url=source_url,
    )


def _grep_snippet(path: Path, query: str, max_len: int = 160) -> str | None:
    try:
        proc = subprocess.run(
            ["grep", "-m1", "-i", query, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = proc.stdout.strip()
        if not line:
            return None
        return line[:max_len]
    except (subprocess.TimeoutExpired, OSError):
        return None
