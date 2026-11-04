"""URL slug derivation for task and note naming."""

import re


def slugify(url: str) -> str:
    """Derive a filesystem-safe slug from a URL."""
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[:100]
