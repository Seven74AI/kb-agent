"""Pure helpers shared by core operations."""

from __future__ import annotations

import re


def slugify(url: str) -> str:
    """Derive a filesystem-safe slug from a URL."""
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[:100]
