"""Web article pipeline — Firecrawl extraction, LLM synthesis, quality gates,
MinIO archival, and Git push.

The simplest pipeline. No cookies, no video. Verifies the entire architecture.
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from agent.config import Config
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.exceptions import ErrorType, StepError
from agent.llm import get_llm_client, LLMClient
from agent.quality import quality_gate_web, quality_gate_retry_loop
from agent.archival import append_note_tags
from agent.archival import step_minio_upload as archival_minio_upload
from agent.archival import step_git_push as archival_git_push
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import ensure_h264
from agent.pipelines.youtube import (
    step_cleanup_temp as _yt_cleanup_temp,
    step_diarize as _yt_diarize,
    step_extract_audio as _yt_extract_audio,
    step_fetch_transcript as _yt_fetch_transcript,
    step_git_push as _yt_git_push,
    step_merge_and_identify as _yt_merge_and_identify,
    step_minio_upload as _yt_minio_upload,
    step_quality_gate as _yt_quality_gate,
    step_synthesize_youtube as _yt_synthesize,
    step_transcribe as _yt_transcribe,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate token ratio for English text
CHARS_PER_TOKEN = 4

# Threshold above which we use progressive chunking
CHUNK_TOKEN_THRESHOLD = 50_000
CHUNK_OVERLAP_TOKENS = 10_000

# Max retry rounds for quality gate
MAX_QUALITY_RETRIES = 3


# ── Substack-Specific Steps ────────────────────────────────────────────────────

# Substack boilerplate patterns to remove from Firecrawl markdown
SUBSTACK_BOILERPLATE = [
    r'^Subscribe\s*$.*?(?=\n##|\n#|\Z)',
    r'^Substack is the home for great culture.*$',
    r'^\*This is .*? of .*?\.\*$',
    r'^Thanks for reading .*?\!.*$',
    r'^\*\*Subscribe\*\*.*$',
    r'^\[Subscribe\].*$',
    r'^### Subscribe to .*$',
    r'^\n---\n\n_Subscribe to get.*$',
    r'^Ready for more\?.*$',
    r'^\nSubstack\s*\n© \d{4}.*$',
    r'^\n\[Share\].*$',
]


def step_substack_cleanup(task: dict, config: Config) -> str:
    """Remove Substack boilerplate from the raw Firecrawl markdown.

    Strips subscribe CTAs, author avatar links, discussion/promo sections,
    and footer cruft. Keeps the article content intact.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        'cleaned' if successful.
    """
    slug = task.get("slug", task["id"][:8])
    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"

    if not raw_path.exists():
        return "no raw file to clean"

    content = raw_path.read_text(encoding="utf-8")

    for pattern in SUBSTACK_BOILERPLATE:
        content = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # Remove consecutive blank lines (max 2)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Remove trailing whitespace on each line
    content = '\n'.join(line.rstrip() for line in content.splitlines())

    raw_path.write_text(content, encoding="utf-8")
    return "cleaned"


def _ingest_download_stem(task: dict) -> str:
    """Filesystem stem for per-ingest download artifacts (queue/task id, not slug)."""
    return task["id"]


def step_substack_video_extract_metadata(task: dict, config: Config) -> str:
    """Find the native ``videoUpload.id`` and write ``{id}_metadata.json``.

    Native Substack video is NOT Mux/signed-JWT-gated — the public endpoint
    ``/api/v1/video/upload/<id>/src?type=mp4`` 307-redirects to a CloudFront-
    signed URL that Substack's server mints. The video id lives in the post JSON
    as ``videoUpload.id``. The Substack cookie is only needed for PAYWALLED
    posts (to see the id at all).
    """
    stem = _ingest_download_stem(task)
    url = task["source_url"]
    _assert_fetch_url_allowed(url)
    domain = urlparse(url).netloc

    cookie = Path(config.cookie_dir) / "substack_cookies.txt"
    cmd = ["curl", "-s", "-L", "--max-time", "30", url]
    if cookie.exists():
        cmd += ["-b", str(cookie)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    html = result.stdout or ""

    fragment_id = urlparse(url).fragment
    if fragment_id and re.fullmatch(r"[a-f0-9-]+", fragment_id, re.IGNORECASE):
        vid_from_frag = fragment_id
    else:
        vid_from_frag = None

    vid = re.search(r'"videoUpload"\s*:\s*\{\s*"id"\s*:\s*"([a-f0-9-]+)"', html)
    if vid_from_frag:
        video_id = vid_from_frag
    elif vid:
        video_id = vid.group(1)
    else:
        video_id = None

    if not video_id:
        raise StepError(
            "No native video id (videoUpload.id) found — paywalled or non-video post",
            error_type=ErrorType.PAUSE, step_name="substack_video_metadata",
        )
    title_m = re.search(r"<title>([^<]+)</title>", html)
    meta = {
        "title": title_m.group(1).strip() if title_m else "",
        "video_id": video_id,
        "domain": domain,
    }
    Path(config.temp_dir, f"{stem}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8",
    )
    return f"video_id={video_id}"


def step_substack_video_download(task: dict, config: Config) -> str:
    """Download native video via the public ``/api/v1/video/upload/<id>/src`` endpoint."""
    stem = _ingest_download_stem(task)
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    if not meta_path.exists():
        raise StepError(
            "metadata missing — run extract_metadata first",
            error_type=ErrorType.RETRY, step_name="substack_video_download",
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vid, domain = meta["video_id"], meta["domain"]
    output_path = Path(config.temp_dir) / f"{stem}.mp4"
    endpoint = f"https://{domain}/api/v1/video/upload/{vid}/src?type=mp4"
    _assert_fetch_url_allowed(endpoint)

    cookie = Path(config.cookie_dir) / "substack_cookies.txt"
    cmd = ["curl", "-s", "-L", "--max-time", "600", "-o", str(output_path), endpoint]
    if cookie.exists():
        cmd += ["-b", str(cookie)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=650)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
        raise StepError(
            f"Substack video download failed: {result.stderr.strip()[:200] or 'empty file'}",
            error_type=ErrorType.RETRY, step_name="substack_video_download",
        )

    output_path = Path(ensure_h264(output_path))
    return str(output_path)


# Body-image dimension below this (both sides) is an avatar/icon, not content.
AVATAR_MAX_SIDE = 400

# Embedded YouTube: iframe src OR data-attrs videoId (HTML-escaped in SSR).
_YOUTUBE_IFRAME_RE = re.compile(r'youtube(?:-nocookie)?\.com/embed/([A-Za-z0-9_-]{6,})')
_YOUTUBE_VIDEOID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{6,})"')
_YOUTUBE_WATCH_RE = re.compile(
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?(?:[^"\s<>]*&)?v=([A-Za-z0-9_-]{6,})',
    re.IGNORECASE,
)
_YOUTUBE_SHORTS_RE = re.compile(
    r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{6,})',
    re.IGNORECASE,
)
_YOUTUBE_BE_RE = re.compile(
    r'(?:https?://)?youtu\.be/([A-Za-z0-9_-]{6,})',
    re.IGNORECASE,
)
_IMAGE_KEY_RE = re.compile(
    r'substack-post-media\.s3\.amazonaws\.com(?:%2F|/)public(?:%2F|/)images(?:%2F|/)'
    r'([a-z0-9-]+_\d+x\d+\.\w+)'
)

# Real image formats the VLM can read. Anything else (`.bin`, etc.) is a
# non-image asset that Substack serves under the same `images/` prefix.
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "heic", "avif"}


def _html_unescape(s: str) -> str:
    return (
        s.replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


_NATIVE_VIDEO_ID_RE = re.compile(
    r'"videoUpload"\s*:\s*\{\s*"id"\s*:\s*"([a-f0-9-]+)"'
    r'|/api/v1/video/upload/([a-f0-9-]+)/',
    re.IGNORECASE,
)

_PAYWALL_MARKERS = (
    "subscribe to continue",
    "data-testid=\"paywall\"",
    "data-testid='paywall'",
    "class=\"paywall",
    "this post is for paid subscribers",
)


def _substack_cookie_configured(config: Config) -> bool:
    cookie = Path(config.cookie_dir) / "substack_cookies.txt"
    if not cookie.exists():
        return False
    try:
        return cookie.stat().st_size > 0
    except OSError:
        return False


def _is_substack_unauth_paywall_shell(html: str) -> bool:
    """Heuristic: login/paywall shell with no article embeds to extract."""
    if not html.strip():
        return False
    lower = html.lower()
    if any(marker in lower for marker in _PAYWALL_MARKERS):
        return True
    # Paywalled shell often lacks body images and article markup
    if "paywall" in lower and not _IMAGE_KEY_RE.search(html):
        if "article" not in lower and "post-body" not in lower:
            return True
    return False


def _extract_native_video_ids(html: str) -> list[str]:
    """All native Substack ``videoUpload.id`` values, deduped, order-preserved."""
    decoded = _html_unescape(html)
    out: list[str] = []
    for m in _NATIVE_VIDEO_ID_RE.finditer(decoded):
        vid = m.group(1) or m.group(2)
        if vid and vid not in out:
            out.append(vid)
    return out


def _fetch_substack_html(url: str, config: Config | None = None) -> str:
    _assert_fetch_url_allowed(url)
    cmd = ["curl", "-s", "-L", "--max-time", "20", url]
    if config is not None:
        cookie = Path(config.cookie_dir) / "substack_cookies.txt"
        if cookie.exists():
            cmd += ["-b", str(cookie)]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout or ""


def _is_content_image(key: str) -> bool:
    """True if the image is article content, not author/channel chrome.

    Substack pages carry three kinds of non-content image that must be skipped:
    small avatars (``_144x144``/``_372x372``), square headshots/logos (author
    photo, channel logo), and extreme banners (channel header). Content images
    (charts, screenshots, documents) are typically rectangular with an aspect
    ratio in the ~0.8–3.0 band. Non-image assets (``.bin``) served under the
    same ``images/`` prefix are also dropped.
    """
    ext = re.search(r'\.(\w+)$', key)
    if not ext or ext.group(1).lower() not in IMAGE_EXTS:
        return False  # .bin / non-image asset
    dim = re.search(r'_(\d+)x(\d+)\.\w+$', key)
    if not dim:
        return True  # no dimensions → assume content
    w, h = int(dim.group(1)), int(dim.group(2))
    if w < AVATAR_MAX_SIDE and h < AVATAR_MAX_SIDE:
        return False  # small avatar/icon
    aspect = w / h
    if 0.8 <= aspect <= 1.25:
        return False  # square → author headshot / channel logo
    if aspect > 3.0 or aspect < 1 / 3.0:
        return False  # extreme banner
    return True


def _extract_body_images(html: str) -> list[str]:
    """All article body images, chrome filtered, order-preserved, both encodings.

    Substack emits images in two forms — the plain `data-attrs` ``src`` URL
    (``https://substack-post-media.s3.amazonaws.com/public/images/<uuid>_WxH.ext``)
    and the URL-encoded CDN form (``substackcdn.com/image/fetch/…https%3A%2F%2F
    substack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F<uuid>_WxH.ext``).
    Both carry the same ``<uuid>_WxH.ext`` key; we match that key regardless of
    plain vs ``%2F`` slashes and reconstruct the canonical high-res URL, then
    drop author/channel chrome (avatars, square headshots/logos, banners) via
    :func:`_is_content_image`.
    """
    decoded = _html_unescape(html)
    seen: set[str] = set()
    out: list[str] = []
    for m in _IMAGE_KEY_RE.finditer(decoded):
        key = m.group(1)
        if key in seen:
            continue
        seen.add(key)
        if not _is_content_image(key):
            continue
        out.append(f"https://substack-post-media.s3.amazonaws.com/public/images/{key}")
    return out


def _extract_youtube_ids(html: str) -> list[str]:
    """Embedded YouTube video IDs, deduped (iframe, videoId JSON, prose links)."""
    decoded = _html_unescape(html)
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


_RUMBLE_URL_RE = re.compile(
    r'https?://rumble\.com/(?:embed/)?(v[a-z0-9]+(?:-[a-z0-9-]+)?)'
)


def _extract_rumble_urls(html: str) -> list[str]:
    """Embedded Rumble URLs, deduped, order-preserved."""
    decoded = _html_unescape(html)
    out: list[str] = []
    for m in _RUMBLE_URL_RE.finditer(decoded):
        url = f"https://rumble.com/{m.group(1)}"
        if url not in out:
            out.append(url)
    return out


def _download_and_ocr(img_url: str, slug: str, idx: int, config: Config) -> str:
    """Download an image and run Tesseract OCR; return the text (or '' on failure)."""
    _assert_fetch_url_allowed(img_url)
    img_path = Path(config.temp_dir) / f"{slug}_img{idx}"
    dl = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "30", "-o", str(img_path), img_url],
        capture_output=True, text=True, timeout=35,
    )
    if dl.returncode != 0 or not img_path.exists() or img_path.stat().st_size < 100:
        return ""
    ocr = subprocess.run(
        ["tesseract", str(img_path), "stdout", "-l", "eng+fra+spa+deu+ara"],
        capture_output=True, text=True, timeout=60,
    )
    return ocr.stdout.strip()


def _enqueue_embedded_media(parent_slug: str, images: list[str], youtube_ids: list[str],
                            rumble_urls: list[str], native_video_ids: list[str],
                            post_url: str, config: Config, tags) -> int:
    """Enqueue embedded images + videos into the nightly batch, linked to the parent.

    Each media item carries ``parent_slug`` + ``media_kind`` so the nightly batch
    appends its reading/transcript to the parent's raw markdown and fires the
    parent's deferred synthesis once every sibling is done. Deliberately does NOT
    download or OCR here — media is batch-processed at night (scripts/batch_ingest.py).
    Lazy-imports to avoid a circular import with agent.pipelines.

    Returns the number of ingest_queue rows for ``parent_slug`` (after dedup).
    """
    if not images and not youtube_ids and not rumble_urls and not native_video_ids:
        return 0
    from agent.core.submit import submit_ingest
    from agent.db import AGENT_SCHEMA, init_db

    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    try:
        for img_url in images:
            submit_ingest(
                img_url, config, conn, content_type="substack_image", tags=tags,
                parent_slug=parent_slug, media_kind="image",
            )
        for vid in youtube_ids:
            submit_ingest(
                f"https://www.youtube.com/watch?v={vid}", config, conn, tags=tags,
                parent_slug=parent_slug, media_kind="video",
            )
        for url in rumble_urls:
            submit_ingest(
                url, config, conn, content_type="rumble", tags=tags,
                parent_slug=parent_slug, media_kind="video",
            )
        for vid in native_video_ids:
            submit_ingest(
                f"{post_url}#{vid}", config, conn, content_type="substack_video", tags=tags,
                parent_slug=parent_slug, media_kind="video",
            )
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (parent_slug,),
        ).fetchone()
        return row["n"] if hasattr(row, "keys") else row[0]
    finally:
        conn.close()


MEDIA_DEFERRED_MARKER_PREFIX = "media_deferred:"


def _persist_parent_media_deferral(conn, task_id: str, slug: str) -> None:
    """Mark the parent task so a later zero-embed re-fetch cannot skip the media gate.

    Written to ``substack_cleanup`` step output (not ``substack_embedded_media``) so
    ``retry_task`` clearing that step onward does not erase the deferral signal.
    """
    if not task_id:
        return
    marker = f"{MEDIA_DEFERRED_MARKER_PREFIX}{slug}"
    with conn:
        conn.execute(
            """
            UPDATE task_steps SET output = CASE
                WHEN output IS NULL OR output = '' THEN ?
                WHEN instr(COALESCE(output, ''), ?) > 0 THEN output
                ELSE COALESCE(output, '') || char(10) || ?
            END
            WHERE task_id = ? AND name = 'substack_cleanup'
            """,
            (marker, MEDIA_DEFERRED_MARKER_PREFIX, marker, task_id),
        )


def _parent_media_was_deferred(conn, task_id: str, slug: str) -> bool:
    if task_id:
        row = conn.execute(
            "SELECT output FROM task_steps WHERE task_id = ? AND name = 'substack_cleanup'",
            (task_id,),
        ).fetchone()
        if row and row[0] and f"{MEDIA_DEFERRED_MARKER_PREFIX}{slug}" in row[0]:
            return True
        row = conn.execute(
            "SELECT output FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
            (task_id,),
        ).fetchone()
        if row and row[0]:
            out = row[0].lower()
            if "awaiting media" in out or MEDIA_DEFERRED_MARKER_PREFIX in out:
                return True
    return False


def _deferred_media_batch_finished(conn, config: Config, slug: str) -> bool:
    """True when deferred media work is done and batch synthesis has taken over."""
    from agent.db import count_pending_media, has_synthesize_task_signal

    if count_pending_media(conn, slug) > 0:
        return False
    return has_synthesize_task_signal(conn, slug)


def step_substack_embedded_media(task: dict, config: Config) -> str:
    """Extract embedded media and DEFER synthesis to the nightly batch.

    Finds every body image (chart/screenshot) and embedded video (YouTube/Rumble)
    and enqueues each into ingest_queue linked back to this article
    (``parent_slug`` + ``media_kind``). Nothing is OCR'd or downloaded inline —
    the nightly batch reads the images (VLM on Modal) and transcribes the videos,
    appending both into this article's raw markdown, then dispatches the deferred
    synthesis once all media has landed.

    Raises StepError(PAUSE) when media was found (the deferral signal); returns
    normally when the article has no media (plain text synthesizes inline).
    """
    slug = task.get("slug", task["id"][:8])
    url = task["source_url"]

    html = _fetch_substack_html(url, config)
    if not html:
        raise StepError(
            "no html fetched",
            error_type=ErrorType.RETRY,
            step_name="substack_embedded_media",
        )

    images = _extract_body_images(html)
    youtube_ids = _extract_youtube_ids(html)
    rumble_urls = _extract_rumble_urls(html)
    native_video_ids = _extract_native_video_ids(html)
    extracted_total = (
        len(images) + len(youtube_ids) + len(rumble_urls) + len(native_video_ids)
    )
    if extracted_total == 0:
        if not _substack_cookie_configured(config) and _is_substack_unauth_paywall_shell(html):
            raise StepError(
                "awaiting substack cookies — paywalled/login HTML without article body",
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )
        manifest_path = Path(config.temp_dir, f"{slug}_media.json")
        had_manifest = manifest_path.exists()
        from agent.db import AGENT_SCHEMA, count_pending_media, init_db

        task_id = task.get("id", "")
        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        try:
            pending = count_pending_media(conn, slug)
            prior_deferral = _parent_media_was_deferred(conn, task_id, slug)
            batch_finished = _deferred_media_batch_finished(conn, config, slug)
            if (had_manifest or pending > 0) and not batch_finished:
                _persist_parent_media_deferral(conn, task_id, slug)
        finally:
            conn.close()
        if pending > 0:
            raise StepError(
                "awaiting media — re-fetch found no embeds but siblings pending",
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )
        if had_manifest and not batch_finished:
            raise StepError(
                "awaiting media — re-fetch found no embeds but siblings pending",
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )
        if prior_deferral and not batch_finished:
            raise StepError(
                "awaiting media — re-fetch found no embeds; prior deferral with no batch completion",
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )
        if batch_finished:
            raise StepError(
                "batch synthesis dispatched or completed — parent must not re-enter synthesis",
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )
        return "no embedded media"

    manifest_path = Path(config.temp_dir, f"{slug}_media.json")
    had_manifest = manifest_path.exists()

    enqueued_total = _enqueue_embedded_media(
        slug, images, youtube_ids, rumble_urls, native_video_ids, url, config, task.get("tags"),
    )

    gate_total = enqueued_total
    if had_manifest:
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_total = prior.get("total")
            if prior_total is not None:
                gate_total = max(int(prior_total), enqueued_total)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    manifest = {
        "url": url,
        "content_type": task.get("content_type", "substack_text"),
        "total": gate_total,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8",
    )

    n_videos = len(youtube_ids) + len(rumble_urls) + len(native_video_ids)
    from agent.db import AGENT_SCHEMA, count_pending_media, init_db

    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    try:
        pending = count_pending_media(conn, slug)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (slug,),
        ).fetchone()
        queue_count = row["n"] if hasattr(row, "keys") else row[0]

        if had_manifest and pending == 0:
            if gate_total != queue_count:
                raise StepError(
                    f"media gate: ingest rows {queue_count} != manifest total {gate_total}",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            return "media ready — deferred to batch"

        _persist_parent_media_deferral(conn, task.get("id", ""), slug)
    finally:
        conn.close()

    raise StepError(
        f"awaiting media: {len(images)} images + {n_videos} videos enqueued",
        error_type=ErrorType.PAUSE, step_name="substack_embedded_media",
    )


# ── Firecrawl Extraction ─────────────────────────────────────────────────────

def step_firecrawl_extract(task: dict, config: Config) -> str:
    """Extract article content via Firecrawl, save raw markdown.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the raw markdown file.

    Raises:
        ValueError: If content is too short or extraction fails.
    """
    url = task["source_url"]
    _assert_fetch_url_allowed(url)
    slug = task.get("slug", task["id"][:8])

    result = subprocess.run(
        [
            "curl", "-s", "--max-time", "60",
            config.firecrawl_url,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "url": url,
                "formats": ["markdown"],
                "waitFor": 3000,
            }),
        ],
        capture_output=True, text=True, timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Firecrawl curl failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        markdown = data["data"]["markdown"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Firecrawl response parse error: {e}") from e

    if len(markdown) < 500:
        raise ValueError(
            f"Content too short ({len(markdown)} chars) — "
            f"possible 404 or empty page"
        )

    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
    raw_path.write_text(markdown, encoding="utf-8")

    return str(raw_path)


# ── LLM Synthesis ────────────────────────────────────────────────────────────

_SUBSTACK_PARENT_CONTENT_TYPES = frozenset({"substack_text", "substack_image"})


def _parent_article_deferred_to_batch(task: dict, config: Config, conn) -> bool:
    """True when this parent slug has a durable deferral manifest or marker."""
    slug = task.get("slug", task["id"][:8])
    if (Path(config.temp_dir) / f"{slug}_media.json").is_file():
        return True
    if _parent_media_was_deferred(conn, task.get("id", ""), slug):
        return True
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
        (slug,),
    ).fetchone()
    sibling_count = row["n"] if hasattr(row, "keys") else row[0]
    return int(sibling_count) > 0


def _supersede_deferred_parent_task(conn, parent_slug: str) -> None:
    """Mark the original Substack parent ingestion task terminal after batch synthesis."""
    reason = "deferred synthesis dispatched; parent ingestion superseded"
    with conn:
        conn.execute(
            """
            UPDATE tasks SET status = 'completed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type NOT GLOB '*_synthesize'
            AND status IN ('paused', 'pending', 'in_progress')
            """,
            (reason, parent_slug),
        )


def _guard_parent_deferred_batch_synthesis(task: dict, config: Config) -> bool:
    """Parent Substack tasks must not inline-synthesize when batch owns the slug.

    Returns True when synthesis already finished and the caller must skip the LLM.
    """
    content_type = task.get("content_type", "")
    if content_type not in _SUBSTACK_PARENT_CONTENT_TYPES:
        return False
    from agent.db import AGENT_SCHEMA, has_completed_parent_synthesis, init_db

    slug = task.get("slug", task["id"][:8])
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    try:
        if has_completed_parent_synthesis(conn, slug, vault_path=config.vault_path):
            _supersede_deferred_parent_task(conn, slug)
            return True
        if not _parent_article_deferred_to_batch(task, config, conn):
            return False
        raise StepError(
            "deferred to batch synthesis — parent must not inline synthesize",
            error_type=ErrorType.PAUSE,
            step_name="synthesize_note",
        )
    finally:
        conn.close()


def step_synthesize_web(task: dict, config: Config) -> str:
    """LLM synthesis for web articles — reads source, builds prompt, calls LLM,
    writes note to vault.

    Handles chunking transparently: single pass for ≤50K tokens,
    progressive chunking for larger content.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    if _guard_parent_deferred_batch_synthesis(task, config):
        return "synthesis already completed"
    source_url = task.get("source_url", "")
    source_path = Path(config.temp_dir) / f"{slug}_raw.md"

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_content = source_path.read_text(encoding="utf-8")
    estimated_tokens = _estimate_tokens(source_content)

    # Get the LLM client
    llm = get_llm_client(config)

    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    if estimated_tokens <= CHUNK_TOKEN_THRESHOLD:
        # Single pass
        note_content = _synthesize_single_pass(
            source_content, source_url, slug, config, llm
        )
    else:
        # Progressive chunking
        note_content = _synthesize_chunked(
            source_content, source_url, slug, config, llm
        )

    note_path.write_text(note_content, encoding="utf-8")
    append_note_tags(note_path, task.get("tags"))
    return str(note_path)


def _synthesize_single_pass(
    source: str, source_url: str, slug: str, config: Config, llm: LLMClient,
) -> str:
    """Single-pass synthesis — source fits in one LLM call."""
    system_prompt = _build_synthesis_prompt(source_url, slug, config)
    user_prompt = (
        f"Here is the source article. Read it fully, then write a comprehensive "
        f"note following the template and quality rules in the system prompt.\n\n"
        f"=== SOURCE ===\n{source}\n=== END SOURCE ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
    return response.content


def _synthesize_chunked(
    source: str, source_url: str, slug: str, config: Config, llm: LLMClient,
) -> str:
    """Progressive chunked synthesis for large content.

    1. Find semantic boundaries with a cheap LLM call
    2. Synthesize each chunk progressively, feeding the previous note as context
    3. Final polish pass to unify the note
    """
    boundaries = _find_semantic_boundaries(source, llm)
    if not boundaries:
        # Fallback: single pass if boundary detection fails
        return _synthesize_single_pass(source, source_url, slug, config, llm)

    previous_note = ""
    total = len(boundaries)

    for i, (start, end) in enumerate(boundaries):
        chunk = source[start:end]

        # Add overlap from previous chunk
        if i > 0:
            overlap_start = max(0, boundaries[i - 1][1] - CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN)
            chunk = source[overlap_start:end]

        chunk_note = _synthesize_chunk(
            chunk, previous_note, source_url, slug, config, llm,
            chunk_index=i, total=total,
        )
        previous_note = chunk_note

    # Final polish
    polish_prompt = (
        "You are a knowledge base editor. Review and polish this note to ensure:\n"
        "1. Consistent tone and depth across all sections\n"
        "2. No duplicate content introduced by chunk overlap\n"
        "3. Key Claims are properly numbered and each has a direct quote with '> '\n"
        "4. Section-by-Section Analysis is the longest section\n"
        "5. All template sections are present and populated\n\n"
        "Return the polished note, maintaining YAML frontmatter exactly."
    )
    polish_response = llm.call(
        system_prompt=polish_prompt,
        user_prompt=f"Polish this note:\n\n{previous_note}",
    )
    return polish_response.content


def _synthesize_chunk(
    chunk: str, previous_note: str, source_url: str, slug: str,
    config: Config, llm: LLMClient, chunk_index: int, total: int,
) -> str:
    """Synthesize one chunk, building on the previous partial note."""
    is_first = (chunk_index == 0)
    is_last = (chunk_index == total - 1)

    if is_first:
        system_prompt = _build_synthesis_prompt(source_url, slug, config)
        system_prompt += (
            f"\n\nCHUNK {chunk_index + 1}/{total}: This is the BEGINNING of the article. "
            "Write a note covering this portion. Later chunks will extend it."
        )
    elif is_last:
        system_prompt = (
            "You are a knowledge base curator finishing a note from the final chunk. "
            "Previous note content is provided. Extend it with this chunk's material: "
            "add new Key Claims, extend Section-by-Section, update Critical Analysis "
            "and Nuances with the full picture now available. "
            "Return the COMPLETE updated note (not just the new part)."
        )
    else:
        system_prompt = (
            "You are a knowledge base curator extending a note with a middle chunk. "
            "Previous note content is provided. Extend it with this chunk's material. "
            "Return the COMPLETE updated note (not just the new part)."
        )

    user_prompt = (
        f"PREVIOUS NOTE (so far):\n{previous_note}\n\n"
        f"=== NEW CHUNK {chunk_index + 1}/{total} ===\n{chunk}\n=== END CHUNK ==="
    ) if previous_note else (
        f"=== CHUNK {chunk_index + 1}/{total} ===\n{chunk}\n=== END CHUNK ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
    return response.content


def _build_synthesis_prompt(source_url: str, slug: str, config: Config) -> str:
    """Build the system prompt for web article synthesis.

    Loads the deep-treatment template and injects quality rules.
    """
    template_path = config.kb_agent_dir / "templates" / "deep-treatment-template.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = _fallback_template()

    source_domain = urlparse(source_url).netloc or "unknown"
    date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are a knowledge base curator. Follow the deep treatment standard.

SOURCE: The user will provide the full article text. Read it ENTIRELY before writing.

TEMPLATE (follow this exact structure):
{template}

QUALITY RULES (non-negotiable):
- Key Claims: ≥4 specific claims, each with a direct quote (start line with '> ')
- Section-by-Section Analysis: must be the LONGEST section — longer than Key Claims + Critical Analysis combined
- Target note size: 10,000–20,000 characters
- Every template section must be populated — no empty or placeholder sections
- Read the ENTIRE source before writing. Never push v1 — re-read and refine.
- Use direct quotes from the source extensively. Every claim needs evidence.
- The Summary should capture the essential argument in 3-5 sentences.
- Context must identify who wrote this, why now, and any conflicts of interest.
- Critical Analysis must identify what's new vs restating, what's omitted.
- Nuances must address what the source exaggerates or distorts.

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: {source_domain}
  source_url: {source_url}
  source_files:
    raw: <filled by archival step>
  confidence: plausible
  tags: [tag1, tag2, tag3]
- Then the note sections as shown in the template.
- Leave the source_files: field in frontmatter — it will be filled later.
- Leave the See Also section empty — it will be filled later by a separate step.

DO NOT invent content. Every factual claim must be traceable to the source.
DO NOT use web_search or external knowledge — work only from the provided source.
DO NOT truncate or summarize away important detail from the source."""
    return prompt


def _fallback_template() -> str:
    """Fallback template if the deep-treatment template file is missing."""
    return """---
date: YYYY-MM-DD
source: <platform/author>
source_url: <URL>
source_files:
  raw: <filled by archival step>
confidence: plausible
tags: []
---

# Title

## Summary
3-5 sentences. The essential argument.

## Key Claims
≥4 specific claims with direct quotes (>).

### 1. Claim title
> « direct quote from the source »
Analysis.

## Section-by-Section Analysis
The DOMINANT section — longer than Key Claims + Critical Analysis combined.

### Section 1: Title
What it argues, key evidence, notable quotes.

## Context
Who wrote this, why now, conflicts of interest.

## Critical Analysis
What's new vs restating? What's omitted? Keep shorter than thematic analysis.

## Nuances
What the source omits, exaggerates, or distorts.

## Reliability
Confidence level with justification.

## Sources
- Original source (full citation)

## See Also
- (filled by See Also step)
"""


# ── Chunking Helpers ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate for English text (~4 chars/token)."""
    return len(text) // CHARS_PER_TOKEN


def _find_semantic_boundaries(source: str, llm: LLMClient) -> list[tuple[int, int]]:
    """Find semantic split points in large content using a cheap LLM call.

    The LLM identifies natural boundaries (section breaks, topic shifts)
    and returns byte offsets for chunking.

    Args:
        source: Full source text.
        llm: LLM client (can use a cheaper model).

    Returns:
        List of (start_offset, end_offset) tuples for each chunk.
        Returns empty list if detection fails.
    """
    estimated_tokens = _estimate_tokens(source)
    target_chunks = max(2, estimated_tokens // CHUNK_TOKEN_THRESHOLD)

    # Send a sample of the structure for boundary detection
    # The LLM needs to see section headers to find good split points
    structure_sample = source[:3000] + "\n\n...[middle of article]...\n\n" + source[-3000:]

    boundary_prompt = (
        "You are a text structure analyzer. The user will show you the beginning "
        "and end of a long article, plus a list of all section headers in order. "
        "Your job: identify the best places to split the article into "
        f"{target_chunks} chunks of roughly equal size (~{CHUNK_TOKEN_THRESHOLD} tokens each).\n\n"
        "Rules:\n"
        "1. Split at natural boundaries: section headers (##, #), major topic shifts\n"
        "2. Each chunk should be as close to equal size as possible\n"
        "3. Never split in the middle of a paragraph — find the nearest heading\n\n"
        "RESPOND with ONLY a JSON list of byte offset pairs:\n"
        '[[0, 50000], [50000, 100000], ...]\n'
        "The first pair starts at 0. The last pair ends at the article length.\n"
        "If you cannot determine boundaries, respond with []"
    )

    # Also extract section headers to help the LLM
    headers = re.findall(r'^(#{1,3} .+)$', source, re.MULTILINE)
    headers_text = "\n".join(headers[:50])  # Limit to first 50 headers

    user_prompt = (
        f"Article length: {len(source)} chars (~{estimated_tokens} tokens)\n"
        f"Target: {target_chunks} chunks\n\n"
        f"STRUCTURE SAMPLE (beginning + end):\n{structure_sample}\n\n"
        f"ALL SECTION HEADERS (in order):\n{headers_text}\n\n"
        "Return the JSON boundary list now."
    )

    try:
        response = llm.call(system_prompt=boundary_prompt, user_prompt=user_prompt)
        # Extract JSON from response
        content = response.content.strip()
        # Find the JSON array in the response
        json_match = re.search(r'\[\[.*?\]\]', content, re.DOTALL)
        if json_match:
            boundaries = json.loads(json_match.group())
            if boundaries and isinstance(boundaries, list):
                # Validate: all pairs, ascending, cover full range
                valid = all(
                    isinstance(b, list) and len(b) == 2 and
                    isinstance(b[0], int) and isinstance(b[1], int)
                    for b in boundaries
                )
                if valid and boundaries[0][0] == 0:
                    # Ensure last boundary covers the full source
                    boundaries[-1][1] = len(source)
                    return [(int(b[0]), int(b[1])) for b in boundaries]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: equal-sized chunks
    return _equal_chunks(source, target_chunks)


def _equal_chunks(source: str, num_chunks: int) -> list[tuple[int, int]]:
    """Fallback: split source into equal-sized chunks at paragraph boundaries."""
    chunk_size = len(source) // num_chunks
    boundaries = []
    start = 0

    for i in range(num_chunks):
        if i == num_chunks - 1:
            boundaries.append((start, len(source)))
        else:
            end = start + chunk_size
            # Find nearest paragraph break after `end`
            paragraph_break = source.find("\n\n", end)
            if paragraph_break != -1 and paragraph_break - end < chunk_size // 2:
                end = paragraph_break
            boundaries.append((start, end))
            start = end

    return boundaries



# ── Quality Gate Step (wraps quality.py) ──────────────────────────────────────

def step_quality_gate(task: dict, config: Config) -> str:
    """Run quality gates on the synthesized note.

    Wraps quality.quality_gate_web with the retry loop.
    Returns 'passed' or raises StepError(FAIL) with failure details.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        'passed' if all gates pass.

    Raises:
        StepError: If quality gates fail after max retries.
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    llm = get_llm_client(config)

    passed, rounds = quality_gate_retry_loop(
        note_path, config, quality_gate_web, llm, max_rounds=MAX_QUALITY_RETRIES,
    )

    if not passed:
        # Read the failure message written by the retry loop
        content = note_path.read_text(encoding="utf-8")
        raise StepError(
            f"Quality gate failed after {rounds} rounds. "
            f"First line: {content.split(chr(10))[0]}",
            error_type=ErrorType.FAIL,
            step_name="quality_gate",
        )

    return "passed"


# ── MinIO Upload Step (wraps archival.py) ─────────────────────────────────────

def step_minio_upload(task: dict, config: Config) -> str:
    """Upload source content to MinIO and add source_files: field to note frontmatter.

    Wraps archival.step_minio_upload and adds the source_files URL to the note.

    Args:
        task: Task dict with id, slug, content_type, source_url.
        config: Config dataclass.

    Returns:
        MinIO object URL for the archived file.
    """
    slug = task.get("slug", task["id"][:8])

    # Upload the raw source
    minio_url = archival_minio_upload(task, config)

    # Add source_files: field to the note's frontmatter
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    if note_path.exists():
        content = note_path.read_text(encoding="utf-8")
        # Insert or replace source_files.raw in frontmatter
        if content.startswith("---"):
            end_fm = content.find("---", 3)
            if end_fm != -1:
                frontmatter = content[3:end_fm]
                if re.search(r'^source_files:', frontmatter, re.MULTILINE):
                    # Replace the raw: placeholder with the real URL
                    content = re.sub(
                        r'^  raw:.*$', f'  raw: {minio_url}',
                        content, flags=re.MULTILINE
                    )
                else:
                    # No source_files: field — insert after the opening ---
                    content = content[:3] + "\nsource_files:\n  raw: " + minio_url + content[3:]
                note_path.write_text(content, encoding="utf-8")

        # Write the real archive URL into the Sources section (before See Also).
        # If the note lacks a See Also header, skip — the frontmatter raw: field
        # already carries the URL.
        content = re.sub(
            r'^## See Also\s*$',
            f'- Raw source archived: `{minio_url}`\n\n## See Also',
            content, flags=re.MULTILINE, count=1,
        )
        note_path.write_text(content, encoding="utf-8")

    return minio_url


# ── Git Push Step (wraps archival.py) ─────────────────────────────────────────

def step_git_push(task: dict, config: Config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Wraps archival.step_git_push.

    Args:
        task: Task dict with id, slug, content_type.
        config: Config dataclass.

    Returns:
        The commit hash of the push.
    """
    return archival_git_push(task, config)


# ── Pipeline Registration ────────────────────────────────────────────────────

register_pipeline("web", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

# Substack pipelines — share the web article template
register_pipeline("substack", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="substack_embedded_media", fn=step_substack_embedded_media, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("substack_text", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="substack_embedded_media", fn=step_substack_embedded_media, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("substack_video", [
    Step(name="extract_metadata", fn=step_substack_video_extract_metadata, on_error="pause"),
    Step(name="download_video", fn=step_substack_video_download, on_error="retry"),
    Step(name="extract_audio", fn=_yt_extract_audio, on_error="retry"),
    Step(name="diarize", fn=_yt_diarize, on_error="retry"),
    Step(name="transcribe", fn=_yt_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=_yt_merge_and_identify, on_error="retry"),
    Step(name="cleanup_temp", fn=_yt_cleanup_temp, on_error="fail"),
    Step(name="synthesize_youtube", fn=_yt_synthesize, on_error="retry"),
    Step(name="quality_gate", fn=_yt_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=_yt_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=_yt_git_push, on_error="retry"),
])

register_pipeline("substack_video_synthesize", [
    Step(name="fetch_transcript", fn=_yt_fetch_transcript, on_error="retry"),
    Step(name="synthesize_youtube", fn=_yt_synthesize, on_error="retry"),
    Step(name="quality_gate", fn=_yt_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=_yt_git_push, on_error="retry"),
])

register_pipeline("substack_image", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract, on_error="retry"),
    Step(name="substack_cleanup", fn=step_substack_cleanup, on_error="retry"),
    Step(name="substack_embedded_media", fn=step_substack_embedded_media, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

# Deferred-synthesis variants: the nightly batch reads the article's media
# (images VLM-read, videos transcribed) into raw.md, then dispatches one of
# these to synthesize the note with everything in it.
register_pipeline("substack_text_synthesize", [
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("substack_image_synthesize", [
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
