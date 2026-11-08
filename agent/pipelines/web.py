"""Web article pipeline — Firecrawl extraction, LLM synthesis, quality gates,
MinIO archival, and Git push.

The simplest pipeline. No cookies, no video. Verifies the entire architecture.
"""

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin, urlparse, urlunparse

from agent.config import Config
from agent.detection import (
    _curl_netscape_cookie_config,
    _safe_url_for_log,
    _unlink_curl_netscape_sidecar,
)
from agent.core.slugify import _netloc_without_userinfo
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.embed_carriers import (
    ExtractedMediaIndex,
    classify_page_embeds,
    extract_embed_media,
    extract_native_video_ids as _extract_native_video_ids,
    html_unescape as _html_unescape,
)
from agent.exceptions import ErrorType, StepError
from agent.notify import sanitize_visible_message
from agent.redact import redact_http_urls_in_text, url_for_display, write_vault_note

logger = logging.getLogger(__name__)
from agent.llm import get_llm_client, LLMClient
from agent.quality import quality_gate_web, quality_gate_retry_loop
from agent.archival import append_note_tags
from agent.archival import step_minio_upload as archival_minio_upload
from agent.archival import step_git_push as archival_git_push
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import _resolve_slug, ensure_h264
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

    from agent.persist import persist_durable_text

    persist_durable_text(raw_path, content, policy="boundary4")
    return "cleaned"


def _ingest_download_stem(task: dict) -> str:
    """Temp-dir stem for substack_video download/metadata (matches later pipeline steps).

    Consumer path names artifacts by slug so ``_find_video_file`` / extract_audio find them;
    ingest id only when ``_batch_ingest`` (embedded siblings share a parent slug).
    """
    if task.get("_batch_ingest"):
        return task["id"]
    return _resolve_slug(task)


# SSRF curl policy (web.py): ``--max-redirs 0`` on first-party Substack fetches whose
# URL is validated up-front and must not chain off-origin (post HTML for metadata /
# embedded-media discovery). The ``/api/v1/video/upload/<id>/src?type=mp4`` endpoint is
# the sole exception: it 307-redirects once to a signed CDN URL (CloudFront / Substack CDN).
_SUBSTACK_VIDEO_CDN_HOST_SUFFIXES = (".cloudfront.net", ".substackcdn.com")
_SUBSTACK_VIDEO_CDN_EXACT_HOSTS = ("substack-post-media.s3.amazonaws.com",)


def _assert_substack_video_cdn_url_allowed(url: str) -> None:
    """Reject video CDN URLs outside Substack's CloudFront / CDN hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise StepError(
            "Substack video CDN URL must use http or https",
            error_type=ErrorType.FAIL,
            step_name="substack_video_download",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise StepError(
            "Substack video CDN URL must include a host",
            error_type=ErrorType.FAIL,
            step_name="substack_video_download",
        )
    allowed = host in _SUBSTACK_VIDEO_CDN_EXACT_HOSTS or host == "substackcdn.com" or any(
        host.endswith(suffix) for suffix in _SUBSTACK_VIDEO_CDN_HOST_SUFFIXES
    )
    if not allowed:
        raise StepError(
            f"Substack video CDN host not allowlisted: {host}",
            error_type=ErrorType.FAIL,
            step_name="substack_video_download",
        )
    _assert_fetch_url_allowed(url)


def _parse_curl_location_header(headers: str) -> str | None:
    for line in headers.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def _resolve_substack_video_mp4_redirect(endpoint: str, cookie_args: list[str]) -> str:
    """Follow at most one validated hop from the public ``.../src?type=mp4`` endpoint."""
    head_cmd = [
        "curl", "-s", "-I", "--max-redirs", "0",
        "--max-time", "30", *cookie_args, endpoint,
    ]
    head = subprocess.run(head_cmd, capture_output=True, text=True, timeout=35)
    if head.returncode != 0:
        raise StepError(
            f"Substack video redirect probe failed: {head.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="substack_video_download",
        )
    location = _parse_curl_location_header(head.stdout or "")
    if not location:
        raise StepError(
            "Substack video src endpoint did not return a redirect Location",
            error_type=ErrorType.RETRY,
            step_name="substack_video_download",
        )
    mp4_url = urljoin(endpoint, location)
    _assert_substack_video_cdn_url_allowed(mp4_url)
    return mp4_url


def step_substack_video_extract_metadata(task: dict, config: Config) -> str:
    """Find the native ``videoUpload.id`` and write ``{stem}_metadata.json``.

    Native Substack video is NOT Mux/signed-JWT-gated — the public endpoint
    ``/api/v1/video/upload/<id>/src?type=mp4`` 307-redirects to a CloudFront-
    signed URL that Substack's server mints. The video id lives in the post JSON
    as ``videoUpload.id``. The Substack cookie is only needed for PAYWALLED
    posts (to see the id at all).
    """
    stem = _ingest_download_stem(task)
    url = task["source_url"]
    _assert_fetch_url_allowed(url)
    domain = _netloc_without_userinfo(urlparse(url))

    cookie = Path(config.cookie_dir) / "substack_cookies.txt"
    conf_path: Path | None = None
    try:
        cookie_args, conf_path = _curl_netscape_cookie_config(cookie)
        cmd = [
            "curl", "-s", "-L", "--max-redirs", "0",
            "--max-time", "30", *cookie_args, url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        html = result.stdout or ""
    finally:
        _unlink_curl_netscape_sidecar(conf_path)

    video_id = _resolve_native_video_id(html, url)

    if not video_id:
        raise StepError(
            "No native video id (videoUpload.id) found — paywalled or non-video post",
            error_type=ErrorType.PAUSE, step_name="substack_video_metadata",
        )
    title_m = re.search(r"<title>([^<]+)</title>", html)
    meta = {
        "title": redact_http_urls_in_text(title_m.group(1).strip() if title_m else ""),
        "video_id": video_id,
        "domain": domain,
    }
    from agent.persist import persist_durable_json

    persist_durable_json(
        Path(config.temp_dir, f"{stem}_metadata.json"),
        meta,
        policy="boundary4",
        indent=None,
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
    conf_path: Path | None = None
    try:
        cookie_args, conf_path = _curl_netscape_cookie_config(cookie)
        mp4_url = _resolve_substack_video_mp4_redirect(endpoint, cookie_args)
        cmd = [
            "curl", "-s", "--max-redirs", "0", "--max-time", "600", "-o", str(output_path),
            *cookie_args, mp4_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=650)
        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
            raise StepError(
                f"Substack video download failed: {result.stderr.strip()[:200] or 'empty file'}",
                error_type=ErrorType.RETRY, step_name="substack_video_download",
            )

        output_path = Path(ensure_h264(output_path))
        return str(output_path)
    finally:
        _unlink_curl_netscape_sidecar(conf_path)


# Body-image dimension below this (both sides) is an avatar/icon, not content.
AVATAR_MAX_SIDE = 400

_IMAGE_KEY_RE = re.compile(
    r'substack-post-media\.s3\.amazonaws\.com(?:%2F|/)public(?:%2F|/)images(?:%2F|/)'
    r'([a-z0-9-]+_\d+x\d+\.\w+)'
)

# Real image formats the VLM can read. Anything else (`.bin`, etc.) is a
# non-image asset that Substack serves under the same `images/` prefix.
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "heic", "avif"}


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


def _resolve_native_video_id(html: str, source_url: str) -> str | None:
    """One native Substack video id: URL ``#fragment`` wins, then extracted HTML ids."""
    fragment_id = urlparse(source_url).fragment
    if fragment_id and re.fullmatch(r"[a-f0-9-]+", fragment_id, re.IGNORECASE):
        return fragment_id
    ids = _extract_native_video_ids(html)
    return ids[0] if ids else None


def _fetch_substack_html(url: str, config: Config | None = None) -> str:
    _assert_fetch_url_allowed(url)
    conf_path: Path | None = None
    try:
        cookie_args: list[str] = []
        if config is not None:
            cookie_args, conf_path = _curl_netscape_cookie_config(
                Path(config.cookie_dir) / "substack_cookies.txt"
            )
        cmd = [
            "curl", "-s", "-L", "--max-redirs", "0",
            "--max-time", "20", *cookie_args, url,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout or ""
    finally:
        _unlink_curl_netscape_sidecar(conf_path)


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


def _embedded_media_index(
    youtube_ids: list[str],
    rumble_urls: list[str],
    bunny_urls: list[str],
    native_video_ids: list[str],
) -> ExtractedMediaIndex:
    return ExtractedMediaIndex(
        youtube_ids=youtube_ids,
        rumble_urls=rumble_urls,
        bunny_urls=bunny_urls,
        native_video_ids=native_video_ids,
    )


def _extract_youtube_ids(html: str) -> list[str]:
    return extract_embed_media(html).youtube_ids


def _extract_rumble_urls(html: str) -> list[str]:
    return extract_embed_media(html).rumble_urls


def _extract_bunny_embed_urls(html: str) -> list[str]:
    return extract_embed_media(html).bunny_urls


def _detect_unenqueued_handled_embed(
    html: str,
    youtube_ids: list[str],
    rumble_urls: list[str],
    bunny_urls: list[str],
    native_video_ids: list[str],
) -> str | None:
    extracted = _embedded_media_index(
        youtube_ids, rumble_urls, bunny_urls, native_video_ids,
    )
    return classify_page_embeds(html, extracted).handled_unenqueued


def _pause_on_unenqueued_handled_embed(
    html: str,
    youtube_ids: list[str],
    rumble_urls: list[str],
    bunny_urls: list[str],
    native_video_ids: list[str],
) -> None:
    """PAUSE when a handled embed carrier is exempt from foreign-embed PAUSE but not enqueued."""
    orphan = _detect_unenqueued_handled_embed(
        html, youtube_ids, rumble_urls, bunny_urls, native_video_ids,
    )
    if orphan:
        raise StepError(
            f"media gate: unenqueued handled embed ({orphan})",
            error_type=ErrorType.PAUSE,
            step_name="substack_embedded_media",
        )


def _detect_unsupported_foreign_embed(html: str) -> str | None:
    """Thin wrapper — classification lives in :mod:`agent.embed_carriers`."""
    return classify_page_embeds(html, extract_embed_media(html)).foreign_pause


class EmbeddedMediaEnqueueStats(NamedTuple):
    """Per-pass enqueue outcomes for the parent media deferral gate."""

    success_count: int
    queue_count: int
    failure_lines: tuple[str, ...] = ()


def _submit_embed_succeeded(result: dict) -> bool:
    """Embed-success reflects actual ingest row state — never a permanently failed duplicate."""
    status = result.get("status")
    if status in ("error", "failed"):
        return False
    if not result.get("ingest_id"):
        return False
    if result.get("created_new"):
        return True
    if status in ("queued", "pending"):
        return True
    if status == "dispatched":
        return True
    return False


def _bump_embed_success_or_warn(
    parent_slug: str,
    media_kind: str,
    result: dict,
    success_count: int,
    failure_lines: list[str],
) -> int:
    if _submit_embed_succeeded(result):
        return success_count + 1
    status = result.get("status")
    logger.warning(
        "embedded media enqueue not accepted (%s) parent=%s status=%s",
        media_kind,
        parent_slug,
        sanitize_visible_message(str(status) if status is not None else "missing"),
    )
    failure_lines.append(
        sanitize_visible_message(
            f"embedded media enqueue not accepted ({media_kind}): "
            f"{status if status is not None else 'missing'}",
        ),
    )
    return success_count


def _enqueue_exception_line(media_kind: str, exc: Exception) -> str:
    return sanitize_visible_message(
        f"embedded media enqueue failed ({media_kind}): {type(exc).__name__}: {exc}",
    )


def _enqueue_failure_block(failure_lines: tuple[str, ...]) -> str:
    if not failure_lines:
        return ""
    return "enqueue failures:\n" + "\n".join(failure_lines)


def _message_with_enqueue_failures(message: str, stats: EmbeddedMediaEnqueueStats) -> str:
    block = _enqueue_failure_block(stats.failure_lines)
    if not block:
        return message
    return f"{message}\n{block}" if message else block


def _enqueue_embedded_media(parent_slug: str, images: list[str], youtube_ids: list[str],
                            rumble_urls: list[str], native_video_ids: list[str],
                            bunny_urls: list[str], post_url: str, config: Config,
                            tags) -> EmbeddedMediaEnqueueStats:
    """Enqueue embedded images + videos into the nightly batch, linked to the parent.

    Each media item carries ``parent_slug`` + ``media_kind`` so the nightly batch
    appends its reading/transcript to the parent's raw markdown and fires the
    parent's deferred synthesis once every sibling is done. Deliberately does NOT
    download or OCR here — media is batch-processed at night (scripts/batch_ingest.py).
    Lazy-imports to avoid a circular import with agent.pipelines.

    Returns successful submit count and ingest_queue row count for ``parent_slug``.
    """
    if (
        not images
        and not youtube_ids
        and not rumble_urls
        and not native_video_ids
        and not bunny_urls
    ):
        return EmbeddedMediaEnqueueStats(0, 0)
    from agent.core.submit import submit_ingest
    from agent.db import AGENT_SCHEMA, init_db

    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    success_count = 0
    failure_lines: list[str] = []
    try:
        for img_url in images:
            try:
                result = submit_ingest(
                    img_url, config, conn, content_type="substack_image", tags=tags,
                    parent_slug=parent_slug, media_kind="image",
                )
            except Exception as exc:
                logger.warning(
                    "embedded media enqueue failed (image) parent=%s url=%s",
                    parent_slug,
                    _safe_url_for_log(img_url),
                    exc_info=True,
                )
                failure_lines.append(_enqueue_exception_line("image", exc))
                continue
            success_count = _bump_embed_success_or_warn(
                parent_slug, "image", result, success_count, failure_lines,
            )
        for vid in youtube_ids:
            try:
                result = submit_ingest(
                    f"https://www.youtube.com/watch?v={vid}", config, conn, tags=tags,
                    parent_slug=parent_slug, media_kind="video",
                )
            except Exception as exc:
                logger.warning(
                    "embedded media enqueue failed (youtube) parent=%s id=%s",
                    parent_slug,
                    vid,
                    exc_info=True,
                )
                failure_lines.append(_enqueue_exception_line("youtube", exc))
                continue
            success_count = _bump_embed_success_or_warn(
                parent_slug, "youtube", result, success_count, failure_lines,
            )
        for url in rumble_urls:
            try:
                result = submit_ingest(
                    url, config, conn, content_type="rumble", tags=tags,
                    parent_slug=parent_slug, media_kind="video",
                )
            except Exception as exc:
                logger.warning(
                    "embedded media enqueue failed (rumble) parent=%s url=%s",
                    parent_slug,
                    _safe_url_for_log(url),
                    exc_info=True,
                )
                failure_lines.append(_enqueue_exception_line("rumble", exc))
                continue
            success_count = _bump_embed_success_or_warn(
                parent_slug, "rumble", result, success_count, failure_lines,
            )
        for vid in native_video_ids:
            try:
                result = submit_ingest(
                    f"{post_url}#{vid}", config, conn, content_type="substack_video", tags=tags,
                    parent_slug=parent_slug, media_kind="video",
                )
            except Exception as exc:
                logger.warning(
                    "embedded media enqueue failed (native video) parent=%s id=%s",
                    parent_slug,
                    vid,
                    exc_info=True,
                )
                failure_lines.append(_enqueue_exception_line("native_video", exc))
                continue
            success_count = _bump_embed_success_or_warn(
                parent_slug, "native_video", result, success_count, failure_lines,
            )
        for bunny_url in bunny_urls:
            try:
                result = submit_ingest(
                    bunny_url, config, conn, content_type="bunny_stream", tags=tags,
                    parent_slug=parent_slug, media_kind="video",
                )
            except Exception as exc:
                logger.warning(
                    "embedded media enqueue failed (bunny) parent=%s url=%s",
                    parent_slug,
                    _safe_url_for_log(bunny_url),
                    exc_info=True,
                )
                failure_lines.append(_enqueue_exception_line("bunny", exc))
                continue
            success_count = _bump_embed_success_or_warn(
                parent_slug, "bunny", result, success_count, failure_lines,
            )
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (parent_slug,),
        ).fetchone()
        queue_count = row["n"] if hasattr(row, "keys") else row[0]
        return EmbeddedMediaEnqueueStats(
            success_count, int(queue_count), tuple(failure_lines),
        )
    finally:
        conn.close()


MEDIA_DEFERRED_MARKER_PREFIX = "media_deferred:"


def _persist_parent_media_deferral(conn, task_id: str, slug: str) -> None:
    """Mark the parent task so a later zero-embed re-fetch cannot skip the media gate.

    Written to ``substack_cleanup`` step output (not ``substack_embedded_media``) so
    ``retry_task`` clearing that step onward does not erase the deferral signal.
    """
    from agent.db import update_step

    if not task_id:
        return
    marker = f"{MEDIA_DEFERRED_MARKER_PREFIX}{slug}"
    row = conn.execute(
        "SELECT id, status, output FROM task_steps "
        "WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    ).fetchone()
    if not row:
        return
    step_id = row["id"] if hasattr(row, "keys") else row[0]
    status = row["status"] if hasattr(row, "keys") else row[1]
    current = (row["output"] if hasattr(row, "keys") else row[2]) or ""
    if MEDIA_DEFERRED_MARKER_PREFIX in current:
        merged = current
    elif not current:
        merged = marker
    else:
        merged = f"{current}\n{marker}"
    update_step(conn, step_id, status, merged)


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
            from agent.deferral_fsm import is_deferred_media_pause

            if is_deferred_media_pause(row[0]):
                return True
    return False


def _deferred_media_batch_finished(conn, config: Config, slug: str) -> bool:
    """True when deferred media work is done and batch synthesis has taken over."""
    from agent.db import count_pending_media, has_synthesize_task_signal

    if count_pending_media(conn, slug) > 0:
        return False
    return has_synthesize_task_signal(conn, slug)


def _pause_on_unsupported_foreign_embed(html: str) -> None:
    """PAUSE after handled siblings are enqueued — never block their ingest_queue rows."""
    unsupported = _detect_unsupported_foreign_embed(html)
    if unsupported:
        raise StepError(
            f"unsupported embed: {unsupported}",
            error_type=ErrorType.PAUSE,
            step_name="substack_embedded_media",
        )


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
    bunny_urls = _extract_bunny_embed_urls(html)
    extracted_total = (
        len(images)
        + len(youtube_ids)
        + len(rumble_urls)
        + len(native_video_ids)
        + len(bunny_urls)
    )
    if extracted_total == 0:
        _pause_on_unenqueued_handled_embed(
            html, youtube_ids, rumble_urls, bunny_urls, native_video_ids,
        )
        _pause_on_unsupported_foreign_embed(html)
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
            conn.execute("BEGIN IMMEDIATE")
            pending = count_pending_media(conn, slug)
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
                (slug,),
            ).fetchone()
            # Ingest sibling row count (re-read atomically with pending before success).
            int(row["n"] if hasattr(row, "keys") else row[0])
            prior_deferral = _parent_media_was_deferred(conn, task_id, slug)
            batch_finished = _deferred_media_batch_finished(conn, config, slug)
            if (had_manifest or pending > 0) and not batch_finished:
                _persist_parent_media_deferral(conn, task_id, slug)
            if pending > 0:
                conn.rollback()
                raise StepError(
                    "awaiting media — re-fetch found no embeds but siblings pending",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            if had_manifest and not batch_finished:
                conn.rollback()
                raise StepError(
                    "awaiting media — re-fetch found no embeds but siblings pending",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            if prior_deferral and not batch_finished:
                conn.rollback()
                raise StepError(
                    "awaiting media — re-fetch found no embeds; prior deferral with no batch completion",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            if batch_finished:
                conn.rollback()
                raise StepError(
                    "batch synthesis dispatched or completed — parent must not re-enter synthesis",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            conn.commit()
        except StepError:
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return "no embedded media"

    manifest_path = Path(config.temp_dir, f"{slug}_media.json")
    had_manifest = manifest_path.exists()

    enqueue_stats = _enqueue_embedded_media(
        slug,
        images,
        youtube_ids,
        rumble_urls,
        native_video_ids,
        bunny_urls,
        url,
        config,
        task.get("tags"),
    )
    _pause_on_unenqueued_handled_embed(
        html, youtube_ids, rumble_urls, bunny_urls, native_video_ids,
    )
    enqueued_success = enqueue_stats.success_count

    prior_manifest_total = 0
    prior_manifest_media_changed = False
    if had_manifest:
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_total = prior.get("total")
            if prior_total is not None:
                prior_manifest_total = int(prior_total)
            if prior.get("media_changed"):
                prior_manifest_media_changed = True
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    gate_total = max(extracted_total, enqueued_success, prior_manifest_total)

    from agent.db import normalize_deferral_parent_content_type

    manifest = {
        "content_type": normalize_deferral_parent_content_type(
            task.get("content_type", "substack_text"),
        ),
        "total": gate_total,
    }
    from agent.persist import persist_durable_json

    persist_durable_json(manifest_path, manifest, policy="boundary4", indent=None)

    from agent.db import (
        AGENT_SCHEMA,
        count_pending_media,
        ensure_parent_media_deferral,
        init_db,
        parent_media_complete,
    )
    from agent.notify import sanitize_visible_message

    if enqueued_success < extracted_total:
        _pause_on_unsupported_foreign_embed(html)
        from agent.persist import scrub_durable_text

        shortfall = scrub_durable_text(
            f"media gate: enqueue shortfall — extracted {extracted_total}, "
            f"enqueued {enqueued_success}",
            policy="boundary4",
        )
        raise StepError(
            _message_with_enqueue_failures(shortfall, enqueue_stats),
            error_type=ErrorType.PAUSE,
            step_name="substack_embedded_media",
        )

    n_videos = (
        len(youtube_ids) + len(rumble_urls) + len(native_video_ids) + len(bunny_urls)
    )

    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    media_ready = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (slug,),
        ).fetchone()
        queue_count = int(row["n"] if hasattr(row, "keys") else row[0])
        if queue_count != extracted_total:
            conn.rollback()
            raise StepError(
                _message_with_enqueue_failures(
                    f"media gate: ingest rows {queue_count} != extracted total {extracted_total}",
                    enqueue_stats,
                ),
                error_type=ErrorType.PAUSE,
                step_name="substack_embedded_media",
            )

        deferral_kw: dict = {}
        if prior_manifest_media_changed:
            deferral_kw["media_changed"] = 1
        ensure_parent_media_deferral(
            conn,
            slug,
            url,
            task.get("content_type", "substack_text"),
            gate_total,
            **deferral_kw,
        )
        from scripts.batch_ingest import refresh_parent_media_manifest

        refresh_parent_media_manifest(conn, config, slug)
        try:
            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            refreshed = {}
        if refreshed.get("media_changed"):
            from agent.db import mark_parent_media_changed

            mark_parent_media_changed(conn, slug)

        pending = count_pending_media(conn, slug)

        if pending == 0:
            if gate_total != queue_count:
                conn.rollback()
                raise StepError(
                    f"media gate: ingest rows {queue_count} != manifest total {gate_total}",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
            if not parent_media_complete(conn, slug, raw_path=raw_path):
                conn.rollback()
                raise StepError(
                    "media gate: parent media not complete — raw markers or bodies missing",
                    error_type=ErrorType.PAUSE,
                    step_name="substack_embedded_media",
                )
            media_ready = True
        else:
            _persist_parent_media_deferral(conn, task.get("id", ""), slug)
        conn.commit()
    except StepError:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if media_ready:
        _pause_on_unsupported_foreign_embed(html)
        return _message_with_enqueue_failures(
            "media ready — deferred to batch", enqueue_stats,
        )

    _pause_on_unsupported_foreign_embed(html)
    raise StepError(
        _message_with_enqueue_failures(
            f"awaiting media: {len(images)} images + {n_videos} videos enqueued",
            enqueue_stats,
        ),
        error_type=ErrorType.PAUSE, step_name="substack_embedded_media",
    )


# ── Firecrawl Extraction ─────────────────────────────────────────────────────

_WIKISPOOKS_BYPASS_PATH = "/w/welcome_back.php"


def _firecrawl_outbound_url(url: str) -> str:
    """Sanitize scrape target URLs sent to Firecrawl (no userinfo / tracking params).

    Wikispooks DDOS bypass URLs keep only the ``goto=`` query (required for fetch).
    """
    parsed = urlparse(url.strip())
    if (
        parsed.scheme in ("http", "https")
        and parsed.path.rstrip("/") == _WIKISPOOKS_BYPASS_PATH
        and parsed.query.startswith("goto=")
    ):
        host = _netloc_without_userinfo(parsed)
        return urlunparse((parsed.scheme, host, parsed.path, "", parsed.query, ""))
    return url_for_display(url)


def step_firecrawl_extract(task: dict, config: Config) -> str:
    """Extract article content via Firecrawl, save raw markdown.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the raw markdown file.

    Raises:
        StepError: On curl/parse failures (RETRY) or content too short (FAIL).
    """
    url = task["source_url"]
    _assert_fetch_url_allowed(url)
    slug = task.get("slug", task["id"][:8])
    outbound_url = _firecrawl_outbound_url(url)

    result = subprocess.run(
        [
            "curl", "-s", "--max-redirs", "0", "--max-time", "60",
            config.firecrawl_url,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "url": outbound_url,
                "formats": ["markdown"],
                "waitFor": 3000,
            }),
        ],
        capture_output=True, text=True, timeout=90,
    )

    from agent.persist import scrub_durable_text

    if result.returncode != 0:
        raise StepError(
            scrub_durable_text(
                f"Firecrawl curl failed: {result.stderr.strip()[:200]}",
                policy="boundary4",
            ),
            error_type=ErrorType.RETRY,
            step_name="firecrawl_extract",
        )

    try:
        data = json.loads(result.stdout)
        markdown = data["data"]["markdown"]
    except (json.JSONDecodeError, KeyError) as e:
        raise StepError(
            scrub_durable_text(f"Firecrawl response parse error: {e}", policy="boundary4"),
            error_type=ErrorType.RETRY,
            step_name="firecrawl_extract",
        ) from e

    if len(markdown) < 500:
        raise StepError(
            scrub_durable_text(
                f"Content too short ({len(markdown)} chars) — "
                f"possible 404 or empty page",
                policy="boundary4",
            ),
            error_type=ErrorType.FAIL,
            step_name="firecrawl_extract",
        )

    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
    from agent.persist import persist_durable_text

    persist_durable_text(raw_path, markdown, policy="boundary4")

    return str(raw_path)


# ── LLM Synthesis ────────────────────────────────────────────────────────────

_SUBSTACK_PARENT_CONTENT_TYPES = frozenset({"substack_text", "substack_image"})


def _parent_article_deferred_to_batch(task: dict, config: Config, conn) -> bool:
    """True when the parent slug is batch-deferred (any durable deferral signal).

    Invariant: ``parent_media_deferral`` is authoritative — a surviving deferral row
    must block parent inline synthesis even when temp manifest, siblings, and step
    markers were wiped (partial cleanup, ``remove_ingest``, temp_dir reset).
    """
    from agent.db import get_parent_media_deferral

    slug = task.get("slug", task["id"][:8])
    if get_parent_media_deferral(conn, slug) is not None:
        return True
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


def _guard_parent_deferred_batch_synthesis(task: dict, config: Config) -> bool:
    """Parent Substack tasks must not inline-synthesize when batch owns the slug.

    Returns True when synthesis already finished and the caller must skip the LLM.
    """
    from agent.db import normalize_deferral_parent_content_type

    content_type = normalize_deferral_parent_content_type(task.get("content_type", ""))
    if content_type not in _SUBSTACK_PARENT_CONTENT_TYPES:
        return False
    from agent.db import (
        AGENT_SCHEMA,
        has_completed_parent_synthesis,
        init_db,
        supersede_deferred_parent_task,
    )

    slug = task.get("slug", task["id"][:8])
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    try:
        if has_completed_parent_synthesis(
            conn,
            slug,
            vault_path=config.vault_path,
            temp_dir=Path(config.temp_dir),
        ):
            supersede_deferred_parent_task(conn, slug)
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


def _require_parent_media_complete_at_execute(
    task: dict, config: Config, conn, raw_path: Path,
) -> None:
    """Fail closed when a ``*_synthesize`` task runs before sibling media landed.

    Uses ``raw_path`` so the DB gate can bootstrap a missing deferral row from
    ``{temp_dir}/{slug}_media.json`` before applying the sibling-dispatch checks.
    """
    from agent.db import is_synthesize_content_type, parent_deferral_synthesize_execute_blocked

    if not is_synthesize_content_type(task.get("content_type", "")):
        return
    slug = task.get("slug", task["id"][:8])
    reason = parent_deferral_synthesize_execute_blocked(conn, slug, raw_path=raw_path)
    if reason:
        raise StepError(reason, error_type=ErrorType.PAUSE, step_name="synthesize_note")


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

    from agent.db import AGENT_SCHEMA, init_db

    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    try:
        _require_parent_media_complete_at_execute(task, config, conn, source_path)
    finally:
        conn.close()

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

    write_vault_note(note_path, note_content)
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

    source_domain = _netloc_without_userinfo(urlparse(source_url)) or "unknown"
    display_source_url = url_for_display(source_url)
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
  source_url: {display_source_url}
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
                write_vault_note(note_path, content)

        # Write the real archive URL into the Sources section (before See Also).
        # If the note lacks a See Also header, skip — the frontmatter raw: field
        # already carries the URL.
        content = re.sub(
            r'^## See Also\s*$',
            f'- Raw source archived: `{minio_url}`\n\n## See Also',
            content, flags=re.MULTILINE, count=1,
        )
        write_vault_note(note_path, content)

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
