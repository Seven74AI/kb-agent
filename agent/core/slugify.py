"""URL slug derivation for task and note naming."""

import re

# Single length cap for every slug-derived path segment (note filename, MinIO
# object key, temp files). 128 chars: well under ext4's 255-byte component
# limit, round power-of-two, enough headroom that long paths don't collide.
SLUG_MAX_LEN = 128


def slugify(url: str) -> str:
    """Derive a filesystem-safe slug from a URL.

    Preserves the YouTube video id (``?v=ID``) — it's the identity, not tracking
    noise. Without it every ``watch?v=`` URL collapses to the same slug and notes
    overwrite each other.
    """
    slug = re.sub(r"^https?://", "", url)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    video_id = f"_v_{m.group(1)}" if m else ""
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[: SLUG_MAX_LEN - len(video_id)] + video_id
