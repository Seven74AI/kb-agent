"""Shared helpers for video pipelines' synthesize-only (batch) path.

The batch flow transcribes all videos in ONE Modal run and archives the merged
speaker-attributed transcript to MinIO. A synthesize-only task then fetches that
transcript and writes the note — no download/diarize/transcribe steps.
"""

import json
import subprocess
from pathlib import Path

from agent.config import Config
from agent.exceptions import ErrorType, StepError


def _duration_from_merged(merged: dict) -> float:
    """Infer media duration (seconds) from a merged transcript JSON payload."""
    duration = merged.get("duration")
    if isinstance(duration, (int, float)):
        return float(duration)

    nested = merged.get("transcript")
    if isinstance(nested, dict):
        inner_duration = nested.get("duration")
        if isinstance(inner_duration, (int, float)):
            return float(inner_duration)
        segments = nested.get("segments") or []
    else:
        segments = merged.get("segments") or []

    max_end = 0.0
    for seg in segments:
        if isinstance(seg, dict):
            end = seg.get("end", 0) or 0
            try:
                max_end = max(max_end, float(end))
            except (TypeError, ValueError):
                continue
    return max_end


def step_fetch_transcript(task: dict, config: Config) -> str:
    """Fetch the pre-uploaded merged transcript from MinIO (synthesize-only path).

    Reads ``{minio_bucket}/transcripts/{slug}.json`` and writes it to
    ``{temp_dir}/{slug}_merged.json`` so the content-type synthesize step can
    consume it exactly as if it had just merged locally.
    """
    slug = task.get("slug", task["id"][:8])
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    dest = f"{config.minio_bucket}/transcripts/{slug}.json"
    result = subprocess.run(
        ["mc", "cp", f"minio/{dest}", str(merged_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise StepError(
            f"Transcript fetch failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="fetch_transcript",
        )

    # Synthesize-only path skips extract_metadata; write minimal metadata so
    # downstream density checks (e.g. YouTube _check_density) can run.
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    if not meta_path.exists():
        duration = 0.0
        try:
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            duration = _duration_from_merged(merged)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            duration = 0.0
        meta_path.write_text(
            json.dumps({"duration": duration}),
            encoding="utf-8",
        )

    return str(merged_path)
