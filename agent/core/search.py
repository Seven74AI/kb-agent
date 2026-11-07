"""Vault note search."""

import subprocess
from pathlib import Path


def search_notes(query: str, vault_path: Path, *, limit: int = 50) -> list[dict]:
    """Search markdown notes under Knowledge base for a text query.

    Uses grep for filesystem search (same approach as see_also vault search).

    Returns:
        List of dicts with path (relative to vault), slug, and optional snippet.
    """
    query = (query or "").strip()
    if not query:
        return []

    knowledge_dir = vault_path / "Knowledge base"
    if not knowledge_dir.exists():
        return []

    try:
        proc = subprocess.run(
            ["grep", "-rl", "--include=*.md", "--", query, str(knowledge_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    results: list[dict] = []
    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        path = Path(line)
        try:
            rel = path.relative_to(vault_path)
        except ValueError:
            rel = path
        results.append({
            "path": str(rel),
            "slug": path.stem,
        })
        if len(results) >= limit:
            break
    return results
