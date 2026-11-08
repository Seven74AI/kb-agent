"""Mega.nz video pipeline — download via megadl, then shared video processing.

Mirrors the YouTube pipeline (diarize → transcribe → YouTube note template)
but sources video from Mega.nz instead of yt-dlp.
"""

import json
import shutil
import subprocess
from pathlib import Path

from agent.config import Config
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.exceptions import ErrorType, StepError
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import _artifact_stem
from agent.pipelines.youtube import (
    step_cleanup_temp,
    step_diarize,
    step_extract_audio,
    step_git_push,
    step_merge_and_identify,
    step_minio_upload,
    step_quality_gate,
    step_synthesize_youtube,
    step_transcribe,
)

MEGA_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".flv"}


def step_mega_extract_metadata(task: dict, config: Config) -> str:
    """Build minimal metadata from megadl filename listing."""
    url = task["source_url"]
    stem = _artifact_stem(task)
    display_slug = task.get("slug", task["id"][:8])

    title = display_slug
    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="extract_metadata",
        ) from exc
    try:
        result = subprocess.run(
            ["megadl", "--print-names", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if names:
                title = Path(names[0]).stem
    except (subprocess.TimeoutExpired, OSError):
        pass

    metadata = {
        "title": title,
        "uploader": "Mega.nz",
        "duration": 0.0,
        "view_count": 0,
        "upload_date": "",
        "description": "",
        "chapters": [],
    }

    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    slug_path = Path(config.temp_dir) / f"{stem}_slug.txt"
    slug_path.write_text(display_slug)

    return json.dumps(metadata)


def _ffprobe_duration(video_path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _update_metadata_duration(config: Config, slug: str, duration: float) -> None:
    """Patch duration into the metadata JSON written by extract_metadata."""
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    if not meta_path.exists():
        return
    metadata = json.loads(meta_path.read_text())
    metadata["duration"] = duration
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def step_mega_download_video(task: dict, config: Config) -> str:
    """Download a Mega.nz video and normalize to {slug}.webm for downstream steps."""
    url = task["source_url"]
    stem = _artifact_stem(task)
    output_path = Path(config.temp_dir) / f"{stem}.webm"

    dl_dir = Path(config.temp_dir) / f"mega_{stem}"
    if dl_dir.exists():
        shutil.rmtree(dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="download_video",
        ) from exc

    result = subprocess.run(
        ["megadl", "--path", str(dl_dir), url],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise StepError(
            f"megadl download failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.PAUSE,
            step_name="download_video",
        )

    video_files = [
        f for f in dl_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in MEGA_VIDEO_EXTENSIONS
        and "__MACOSX" not in str(f)
    ]
    if not video_files:
        raise StepError(
            "megadl completed but no video file found",
            error_type=ErrorType.PAUSE,
            step_name="download_video",
        )
    if len(video_files) > 1:
        video_files.sort(key=lambda p: p.stat().st_size, reverse=True)

    source = video_files[0]

    if source.suffix.lower() == ".webm" and source.resolve() != output_path.resolve():
        shutil.copy2(source, output_path)
    elif source.suffix.lower() != ".webm" or not output_path.exists():
        convert = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-c:v", "libvpx-vp9", "-b:v", "1M",
                "-c:a", "libopus",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if convert.returncode != 0:
            raise StepError(
                f"ffmpeg webm conversion failed: {convert.stderr.strip()[:200]}",
                error_type=ErrorType.RETRY,
                step_name="download_video",
            )

    if not output_path.exists():
        raise StepError(
            f"Video download claimed success but file missing: {output_path}",
            error_type=ErrorType.RETRY,
            step_name="download_video",
        )

    duration = _ffprobe_duration(output_path)
    if duration > 0:
        _update_metadata_duration(config, stem, duration)

    return str(output_path)


register_pipeline("mega_video", [
    Step(name="extract_metadata", fn=step_mega_extract_metadata, on_error="pause"),
    Step(name="download_video", fn=step_mega_download_video, on_error="retry"),
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
