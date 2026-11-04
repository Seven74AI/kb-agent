"""Shared helpers for video pipelines' synthesize-only (batch) path.

The batch flow transcribes all videos in ONE Modal run and archives the merged
speaker-attributed transcript to MinIO. A synthesize-only task then fetches that
transcript and writes the note — no download/diarize/transcribe steps.
"""

import subprocess
from pathlib import Path

from agent.config import Config
from agent.exceptions import ErrorType, StepError


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
    return str(merged_path)
