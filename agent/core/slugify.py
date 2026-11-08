"""URL slug derivation for task and note naming."""

import re
from urllib.parse import urlparse, urlunparse

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


def _netloc_without_userinfo(parsed) -> str:
    """Hostname and port only — same policy as :func:`agent.redact.url_for_display`."""
    if parsed.hostname:
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            return f"{host}:{parsed.port}"
        return host
    netloc = parsed.netloc or ""
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    return netloc


def _url_for_slug_body(url: str) -> str:
    """Drop http(s) userinfo before slug sanitization; leave non-http URLs unchanged."""
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return urlunparse(
            (
                parsed.scheme,
                _netloc_without_userinfo(parsed),
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    return url


def slugify(url: str, *, content_type: str | None = None) -> str:
    """Derive a filesystem-safe slug from a URL.

    Preserves the YouTube video id (``?v=ID``) — it's the identity, not tracking
    noise. Without it every ``watch?v=`` URL collapses to the same slug and notes
    overwrite each other.

    For ``instagram_comment``, preserves ``comment_id`` the same way — multiple
    comments on one post must not share a slug.

    URL fragments (``#...``) are removed and do not affect the slug. Deferred
    media enqueued as ``post_url#<video_id>`` therefore shares the parent's
    slug; batch staging must key files by ``ingest_queue.id``, not slug.

    Netloc credentials (``user:pass@``) are stripped before slug building so secrets
    never land in ``tasks.slug``, MinIO keys, or vault note filenames.
    """
    slug = re.sub(r"^https?://", "", _url_for_slug_body(url))
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


def batch_standalone_video_storage_slug(
    *,
    ingest_id: str,
    slug: str,
    content_type: str,
    parent_slug: str | None = None,
) -> str:
    """MinIO object prefix and synthesize dispatch slug for batch-collected standalone videos.

    Native Substack deferred URLs use ``post#<video_id>``; :func:`slugify` strips the
    fragment so siblings share one slug. Standalone batch rows (no ``parent_slug``) must
    key artifacts and dedupe on ``ingest_queue.id`` instead. Embedded siblings keep
    the parent slug and are keyed by ingest id only in temp/staging paths (not here).
    """
    if content_type == "substack_video" and not (parent_slug or "").strip():
        return ingest_id
    return slug
