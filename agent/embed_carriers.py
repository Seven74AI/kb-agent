"""Substack embed-carrier detection: one host→handler table, derived handled/exempt sets."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple
from urllib.parse import urlparse

# ── Leaf HTML unescape (stdlib only; no agent.* imports) ─────────────────────


def html_unescape(s: str) -> str:
    return (
        s.replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


# ── Extraction regexes (scoped to extract_embed_media) ───────────────────────

_YOUTUBE_IFRAME_RE = re.compile(
    r"youtube(?:-nocookie)?\.com/embed/([A-Za-z0-9_-]{6,})"
)
_YOUTUBE_VIDEOID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{6,})"')
_YOUTUBE_WATCH_RE = re.compile(
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?(?:[^"\s<>]*&)?v=([A-Za-z0-9_-]{6,})',
    re.IGNORECASE,
)
_YOUTUBE_SHORTS_RE = re.compile(
    r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{6,})",
    re.IGNORECASE,
)
_YOUTUBE_BE_RE = re.compile(
    r"(?:https?://)?youtu\.be/([A-Za-z0-9_-]{6,})",
    re.IGNORECASE,
)

_RUMBLE_URL_RE = re.compile(
    r"https?://rumble\.com/(?:embed/)?(v[a-z0-9]+(?:-[a-z0-9-]+)?)"
)

_BUNNY_EMBED_URL_RE = re.compile(
    r"https?://(?:player\.)?mediadelivery\.net/embed/\d+/[a-f0-9]{8}-[a-f0-9]{4}-"
    r"[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
    re.IGNORECASE,
)

_NATIVE_VIDEO_ID_RE = re.compile(
    r'"videoUpload"\s*:\s*\{\s*"id"\s*:\s*"([a-f0-9-]+)"'
    r"|/api/v1/video/upload/([a-f0-9-]+)/?",
    re.IGNORECASE,
)

# Non-video Substack assets that must not trigger foreign-embed PAUSE in carrier URLs.
_EXTRA_HANDLED_URL_MARKERS = ("substack-post-media.",)

# Known foreign platforms (label only — hosts not in EMBED_HOST_HANDLERS).
_FOREIGN_EMBED_URL_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"instagram\.com", re.IGNORECASE), "instagram"),
    (re.compile(r"threads\.(?:net|com)/", re.IGNORECASE), "threads"),
    (re.compile(r"tiktok\.com", re.IGNORECASE), "tiktok"),
    (re.compile(r"vimeo\.com", re.IGNORECASE), "vimeo"),
    (re.compile(r"(?:player\.)?mediadelivery\.net", re.IGNORECASE), "bunny"),
    (re.compile(r"bunnycdn\.com", re.IGNORECASE), "bunny"),
    (re.compile(r"(?:www\.)?bunny\.net", re.IGNORECASE), "bunny"),
    (re.compile(r"terragrandir\.com", re.IGNORECASE), "bunny"),
    (re.compile(r"mega\.nz", re.IGNORECASE), "mega"),
    (re.compile(r"(?:platform\.)?(?:twitter|x)\.com", re.IGNORECASE), "twitter"),
]

_IFRAME_SRC_RE = re.compile(
    r'<iframe[^>]+src=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_LINK_HREF_RE = re.compile(
    r'<a[^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_EMBED_CARRIER_PATH_RE = re.compile(
    r"(?:/embed(?:/|$|[?#])|/video/embed|player\.|widget)",
    re.IGNORECASE,
)
_OEMBED_PROVIDER_RE = re.compile(
    r'"provider_(?:url|name)"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


def _embed_carrier_class_re(marker: str) -> re.Pattern[str]:
    return re.compile(
        rf'\bclass\s*=\s*["\'][^"\']*\b{re.escape(marker)}\b',
        re.IGNORECASE,
    )


_DATA_INSTGRM_PERMALINK_RE = re.compile(r"data-instgrm-permalink\s*=", re.IGNORECASE)
_DATA_THREADS_POST_PERMALINK_RE = re.compile(
    r"data-text-post-permalink\s*=", re.IGNORECASE
)
_INSTAGRAM_MEDIA_CLASS_RE = _embed_carrier_class_re("instagram-media")
_TIKTOK_EMBED_CLASS_RE = _embed_carrier_class_re("tiktok-embed")
_THREADS_TEXT_POST_CLASS_RE = _embed_carrier_class_re("text-post-media")
_TWITTER_TWEET_CLASS_RE = _embed_carrier_class_re("twitter-tweet")
_TWITTER_TIMELINE_CLASS_RE = _embed_carrier_class_re("twitter-timeline")


@dataclass(frozen=True)
class ExtractedMediaIndex:
    youtube_ids: list[str]
    rumble_urls: list[str]
    bunny_urls: list[str]
    native_video_ids: list[str]


@dataclass(frozen=True)
class EmbedReport:
    foreign_pause: str | None
    handled_unenqueued: str | None


def _match_youtube_canonical(url: str, extracted: ExtractedMediaIndex) -> bool:
    return any(vid in url for vid in extracted.youtube_ids)


def _match_rumble_canonical(url: str, extracted: ExtractedMediaIndex) -> bool:
    for canonical in extracted.rumble_urls:
        slug = canonical.rsplit("/", 1)[-1]
        if slug and slug in url:
            return True
    return False


def _match_bunny_canonical(url: str, extracted: ExtractedMediaIndex) -> bool:
    norm = url.rstrip("/")
    return any(
        norm == b.rstrip("/") or norm.startswith(b.rstrip("/") + "/")
        for b in extracted.bunny_urls
    )


def _match_native_canonical(url: str, extracted: ExtractedMediaIndex) -> bool:
    return any(vid in url for vid in extracted.native_video_ids)


class EmbedHostHandler(NamedTuple):
    """One handled platform: host match, carrier shape, extractor bucket, coverage."""

    host_re: re.Pattern[str]
    carrier_re: re.Pattern[str]
    canonical_matcher: Callable[[str, ExtractedMediaIndex], bool]
    extractor_key: str


EMBED_HOST_HANDLERS: list[EmbedHostHandler] = [
    EmbedHostHandler(
        host_re=re.compile(
            r"youtube(?:-nocookie)?\.com|youtu\.be", re.IGNORECASE
        ),
        carrier_re=re.compile(
            r"youtube(?:-nocookie)?\.com/embed/|youtu\.be/", re.IGNORECASE
        ),
        canonical_matcher=_match_youtube_canonical,
        extractor_key="youtube",
    ),
    EmbedHostHandler(
        host_re=re.compile(r"rumble\.com", re.IGNORECASE),
        carrier_re=re.compile(r"rumble\.com/embed/", re.IGNORECASE),
        canonical_matcher=_match_rumble_canonical,
        extractor_key="rumble",
    ),
    EmbedHostHandler(
        host_re=re.compile(r"(?:player\.)?mediadelivery\.net", re.IGNORECASE),
        carrier_re=re.compile(
            r"(?:player\.)?mediadelivery\.net/embed/", re.IGNORECASE
        ),
        canonical_matcher=_match_bunny_canonical,
        extractor_key="bunny",
    ),
    EmbedHostHandler(
        host_re=re.compile(r"substack\.com/api/v1/video", re.IGNORECASE),
        carrier_re=re.compile(r"substack\.com/api/v1/video", re.IGNORECASE),
        canonical_matcher=_match_native_canonical,
        extractor_key="native",
    ),
]


def handled_embed_host_re() -> re.Pattern[str]:
    """OR of every ``host_re`` in :data:`EMBED_HOST_HANDLERS` plus image CDN marker."""
    parts = [h.host_re.pattern for h in EMBED_HOST_HANDLERS]
    parts.append(re.escape(_EXTRA_HANDLED_URL_MARKERS[0].rstrip(".")) + r"\.")
    return re.compile(r"(?:{})".format("|".join(f"(?:{p})" for p in parts)), re.IGNORECASE)


def handled_embed_carrier_re() -> re.Pattern[str]:
    """OR of every ``carrier_re`` — no standalone table elsewhere."""
    parts = [h.carrier_re.pattern for h in EMBED_HOST_HANDLERS]
    return re.compile(r"(?:{})".format("|".join(f"(?:{p})" for p in parts)), re.IGNORECASE)


_HANDLED_EMBED_URL_RE = handled_embed_host_re()


def _handler_for_url(url: str) -> EmbedHostHandler | None:
    for handler in EMBED_HOST_HANDLERS:
        if handler.host_re.search(url):
            return handler
    return None


def _is_handled_embed_url(url: str) -> bool:
    if not url:
        return False
    if _HANDLED_EMBED_URL_RE.search(url):
        return True
    return any(marker in url for marker in _EXTRA_HANDLED_URL_MARKERS)


def _foreign_platform_label(url: str) -> str | None:
    if not url or _is_handled_embed_url(url):
        return None
    for pattern, platform in _FOREIGN_EMBED_URL_RULES:
        if pattern.search(url):
            return platform
    return None


def collect_embed_carrier_urls(html: str) -> list[str]:
    """Iframe src, oEmbed provider, embed-path anchor hrefs — order preserved, deduped."""
    decoded = html_unescape(html)
    out: list[str] = []
    seen: set[str] = set()
    for pattern in (_IFRAME_SRC_RE, _OEMBED_PROVIDER_RE):
        for m in pattern.finditer(decoded):
            url = m.group(1)
            if url not in seen:
                seen.add(url)
                out.append(url)
    for m in _LINK_HREF_RE.finditer(decoded):
        url = m.group(1)
        if _EMBED_CARRIER_PATH_RE.search(url) and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _extract_youtube_ids(html: str) -> list[str]:
    decoded = html_unescape(html)
    ids: list[str] = []
    for pattern in (
        _YOUTUBE_IFRAME_RE,
        _YOUTUBE_VIDEOID_RE,
        _YOUTUBE_WATCH_RE,
        _YOUTUBE_SHORTS_RE,
        _YOUTUBE_BE_RE,
    ):
        for m in pattern.finditer(decoded):
            if m.group(1) not in ids:
                ids.append(m.group(1))
    return ids


def _extract_rumble_urls(html: str) -> list[str]:
    decoded = html_unescape(html)
    out: list[str] = []
    for m in _RUMBLE_URL_RE.finditer(decoded):
        url = f"https://rumble.com/{m.group(1)}"
        if url not in out:
            out.append(url)
    return out


def _extract_bunny_embed_urls(html: str) -> list[str]:
    decoded = html_unescape(html)
    out: list[str] = []
    for m in _BUNNY_EMBED_URL_RE.finditer(decoded):
        url = m.group(0)
        if url not in out:
            out.append(url)
    return out


def _extract_native_video_ids(html: str) -> list[str]:
    decoded = html_unescape(html)
    out: list[str] = []
    for m in _NATIVE_VIDEO_ID_RE.finditer(decoded):
        vid = m.group(1) or m.group(2)
        if vid and vid not in out:
            out.append(vid)
    return out


def extract_embed_media(html: str) -> ExtractedMediaIndex:
    return ExtractedMediaIndex(
        youtube_ids=_extract_youtube_ids(html),
        rumble_urls=_extract_rumble_urls(html),
        bunny_urls=_extract_bunny_embed_urls(html),
        native_video_ids=_extract_native_video_ids(html),
    )


def extract_native_video_ids(html: str) -> list[str]:
    return _extract_native_video_ids(html)


def _embed_carrier_pause_label(url: str) -> str | None:
    if not url or _is_handled_embed_url(url):
        return None
    platform = _foreign_platform_label(url)
    if platform:
        return platform
    host = urlparse(url).hostname
    return host if host else "unknown"


def _detect_unsupported_foreign_embed(html: str) -> str | None:
    decoded = html_unescape(html)
    if _DATA_INSTGRM_PERMALINK_RE.search(decoded) or _INSTAGRAM_MEDIA_CLASS_RE.search(
        decoded
    ):
        return "instagram"
    if _TIKTOK_EMBED_CLASS_RE.search(decoded):
        return "tiktok"
    if _DATA_THREADS_POST_PERMALINK_RE.search(decoded) or _THREADS_TEXT_POST_CLASS_RE.search(
        decoded
    ):
        return "threads"
    if _TWITTER_TWEET_CLASS_RE.search(decoded) or _TWITTER_TIMELINE_CLASS_RE.search(
        decoded
    ):
        return "twitter"
    for url in collect_embed_carrier_urls(html):
        label = _embed_carrier_pause_label(url)
        if label:
            return label
    return None


def _detect_unenqueued_handled_embed(
    html: str, extracted: ExtractedMediaIndex
) -> str | None:
    for url in collect_embed_carrier_urls(html):
        handler = _handler_for_url(url)
        if handler is None or not handler.carrier_re.search(url):
            continue
        if not handler.canonical_matcher(url, extracted):
            return url
    return None


def classify_page_embeds(html: str, extracted: ExtractedMediaIndex) -> EmbedReport:
    return EmbedReport(
        foreign_pause=_detect_unsupported_foreign_embed(html),
        handled_unenqueued=_detect_unenqueued_handled_embed(html, extracted),
    )


def classify_carrier_url_bucket(
    url: str, extracted: ExtractedMediaIndex
) -> str:
    """Property-test helper: bucket name for one collected carrier URL."""
    handler = _handler_for_url(url)
    if handler is not None and handler.carrier_re.search(url):
        if handler.canonical_matcher(url, extracted):
            return "covered"
        return "handled_unenqueued"
    label = _embed_carrier_pause_label(url)
    if label:
        return "foreign"
    if handler is not None:
        return "handled_non_carrier"
    return "neutral"
