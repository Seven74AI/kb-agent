"""Rumble video pipeline — Flaresolverr Cloudflare bypass + CDN download.

Rumble sits behind Cloudflare's managed "Just a moment" challenge, and a
``cf_clearance`` cookie is IP-bound, so a cookie exported elsewhere will NOT
clear the 403 from the server. Flaresolverr (localhost:8191) solves the
challenge server-side and returns the real page; from it we resolve the actual
video id (the URL slug is a redirect alias — e.g. ``v2y5lfu-…`` resolves to
``v2vk68c``) and then the un-gated CDN mp4 URLs on ``*.cdn.rumble.cloud``, which
download directly with no cookie.

Reuses the shared YouTube downstream steps (audio → diarize → transcribe →
merge → synthesize → gate → archive → see-also → push) — a Rumble video is a
transcript-backed video note, same shape as YouTube.
"""

import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from agent.config import Config
from agent.constants import FLARESOLVERR_URL
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.exceptions import ErrorType, StepError
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import ensure_h264
from agent.pipelines.youtube import (
    step_cleanup_temp,
    step_diarize,
    step_extract_audio,
    step_fetch_transcript,
    step_git_push,
    step_merge_and_identify,
    step_minio_upload,
    step_quality_gate,
    step_synthesize_youtube,
    step_transcribe,
)

MAX_HEIGHT = 720
# CDN mp4 URLs expire; only reuse metadata written during the current ingest.
RUMBLE_METADATA_MAX_AGE_SEC = 6 * 3600

_RUMBLE_CDN_HOST_SUFFIXES = (".cdn.rumble.cloud", ".rumble.com")


def _download_artifact_stem(task: dict) -> str:
    """Temp-dir stem for download-phase artifacts (id-keyed in nightly batch)."""
    if task.get("_batch_ingest"):
        return task["id"]
    return task.get("slug", task["id"][:8])


def _assert_rumble_cdn_url_allowed(url: str) -> None:
    """Reject CDN URLs whose host is outside Rumble's CDN / origin domains."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise StepError(
            "Rumble CDN URL must use http or https",
            error_type=ErrorType.FAIL,
            step_name="download_video",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise StepError(
            "Rumble CDN URL must include a host",
            error_type=ErrorType.FAIL,
            step_name="download_video",
        )
    allowed = host in ("rumble.com", "cdn.rumble.cloud") or any(
        host.endswith(suffix) for suffix in _RUMBLE_CDN_HOST_SUFFIXES
    )
    if not allowed:
        raise StepError(
            f"Rumble CDN host not allowlisted: {host}",
            error_type=ErrorType.FAIL,
            step_name="download_video",
        )
    _assert_fetch_url_allowed(url)


def _flaresolverr_get(url: str) -> str:
    """Fetch a URL through Flaresolverr; return the cleared response body."""
    _assert_fetch_url_allowed(url)
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST", FLARESOLVERR_URL,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"cmd": "request.get", "url": url, "maxTimeout": 60000}),
            "--max-time", "70",
        ],
        capture_output=True, text=True, timeout=75,
    )
    if result.returncode != 0:
        raise StepError(
            f"Flaresolverr request failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise StepError(
            f"Flaresolverr bad response: {exc}",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )
    if data.get("status") != "ok":
        raise StepError(
            f"Flaresolverr error: {data.get('message')}",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )
    return data["solution"].get("response", "")


def resolve_rumble(url: str) -> tuple[str, list[dict]]:
    """Resolve a Rumble URL → ``(title, [{"url", "height"}, …])`` ascending.

    Two Flaresolverr hops: the page (for the real video id) then the embedJS/u3
    API (for the mp4 CDN URLs). Raises StepError(RETRY) on any failure.
    """
    page = _flaresolverr_get(url)

    mobj = re.search(r'rumble\.com/embed/(v[a-z0-9]+)/', page)
    if not mobj:
        mobj = re.search(r'"video"\s*:\s*"(v[a-z0-9]+)"', page)
    if not mobj:
        raise StepError(
            "Rumble video id not found in page",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )
    video_id = mobj.group(1)

    body = _flaresolverr_get(
        f"https://rumble.com/embedJS/u3/?request=video&ver=2&v={video_id}"
    )
    pm = re.search(r"<pre>(.*)</pre>", body, re.DOTALL)  # embedJS wraps JSON in <pre>
    if pm:
        body = pm.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise StepError(
            f"embedJS JSON parse failed: {exc}",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )

    title = data.get("title", "")
    mp4 = data.get("ua", {}).get("mp4", {})
    formats: list[dict] = []
    for height, info in mp4.items():
        if not isinstance(info, dict) or not info.get("url"):
            continue
        try:
            h = int(info.get("meta", {}).get("h", height))
        except (TypeError, ValueError):
            h = 0
        formats.append({"url": info["url"], "height": h})
    formats.sort(key=lambda f: f["height"])
    if not formats:
        raise StepError(
            "No mp4 formats in Rumble embedJS response",
            error_type=ErrorType.RETRY, step_name="rumble_resolve",
        )
    return title, formats


def _load_rumble_formats(task: dict, config: Config) -> list[dict]:
    """Use cached ``{stem}_metadata.json`` from extract_metadata when still fresh."""
    stem = _download_artifact_stem(task)
    source_url = task["source_url"]
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    if meta_path.exists():
        age_sec = time.time() - meta_path.stat().st_mtime
        if age_sec <= RUMBLE_METADATA_MAX_AGE_SEC:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = None
            if meta:
                formats = meta.get("formats")
                url_ok = meta.get("source_url", source_url) == source_url
                if url_ok and isinstance(formats, list) and formats:
                    return formats
    _, formats = resolve_rumble(source_url)
    return formats


def step_rumble_extract_metadata(task: dict, config: Config) -> str:
    """Resolve title + formats and write ``{stem}_metadata.json``."""
    stem = _download_artifact_stem(task)
    title, formats = resolve_rumble(task["source_url"])
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    meta_path.write_text(
        json.dumps(
            {"title": title, "formats": formats, "source_url": task["source_url"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return f"title={title[:60]} formats={len(formats)}"


def step_rumble_download_video(task: dict, config: Config) -> str:
    """Resolve + download the ≤720p mp4 from the un-gated CDN, normalize to H.264."""
    stem = _download_artifact_stem(task)
    output_path = Path(config.temp_dir) / f"{stem}.mp4"

    formats = _load_rumble_formats(task, config)
    pick = None
    for f in formats:
        if f["height"] <= MAX_HEIGHT:
            pick = f
    if pick is None:
        pick = formats[-1]  # smallest (ascending) when all exceed 720p

    _assert_rumble_cdn_url_allowed(pick["url"])
    result = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "600", "-o", str(output_path), pick["url"]],
        capture_output=True, text=True, timeout=650,
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
        raise StepError(
            f"Rumble download failed: {result.stderr.strip()[:200] or 'empty file'}",
            error_type=ErrorType.RETRY, step_name="download_video",
        )

    output_path = Path(ensure_h264(output_path))
    return str(output_path)


register_pipeline("rumble", [
    Step(name="extract_metadata", fn=step_rumble_extract_metadata, on_error="pause"),
    Step(name="download_video", fn=step_rumble_download_video, on_error="retry"),
    Step(name="extract_audio", fn=step_extract_audio, on_error="retry"),
    Step(name="diarize", fn=step_diarize, on_error="retry"),
    Step(name="transcribe", fn=step_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=step_merge_and_identify, on_error="retry"),
    Step(name="cleanup_temp", fn=step_cleanup_temp, on_error="fail"),
    Step(name="synthesize_youtube", fn=step_synthesize_youtube, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("rumble_synthesize", [
    Step(name="fetch_transcript", fn=step_fetch_transcript, on_error="retry"),
    Step(name="synthesize_youtube", fn=step_synthesize_youtube, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
