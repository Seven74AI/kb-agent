"""URL slug derivation for task and note naming."""

import re

# Single length cap for every slug-derived path segment (note filename, MinIO
# object key, temp files). 128 chars: well under ext4's 255-byte component
# limit, round power-of-two, enough headroom that long paths don't collide.
SLUG_MAX_LEN = 128

_YOUTUBE_V_PARAM = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
_COMMENT_ID_PARAM = re.compile(r"[?&]comment_id=([A-Za-z0-9_-]+)")
_YOUTUBE_PATH_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:shorts|embed)/)([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def _youtube_video_id(url: str) -> str:
    m = _YOUTUBE_V_PARAM.search(url)
    if m:
        return m.group(1)
    m = _YOUTUBE_PATH_ID.search(url)
    return m.group(1) if m else ""


def slugify(url: str, *, content_type: str | None = None) -> str:
    """Derive a filesystem-safe slug from a URL.

    Preserves the YouTube video id (``?v=ID``) — it's the identity, not tracking
    noise. Without it every ``watch?v=`` URL collapses to the same slug and notes
    overwrite each other.

    For ``instagram_comment``, preserves ``comment_id`` the same way — multiple
    comments on one post must not share a slug.
    """
    slug = re.sub(r"^https?://", "", url)
    identity_suffix = ""
    vid = _youtube_video_id(url)
    if vid:
        identity_suffix = f"_v_{vid}"
    elif content_type == "instagram_comment":
        m = _COMMENT_ID_PARAM.search(url)
        if m:
            identity_suffix = f"_c_{m.group(1)}"
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[: SLUG_MAX_LEN - len(identity_suffix)] + identity_suffix
