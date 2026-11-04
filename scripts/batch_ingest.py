#!/usr/bin/env python3
"""Nightly batch collector — download + batch-transcribe videos, dispatch links.

Reads the durable ingest_queue (status='queued'):
- Non-video (web/books/substack/...) -> full pipeline task (submit_url).
- Video -> runs each pipeline's download-phase steps, then ONE batch Modal
  transcription run (scripts/batch_transcribe.py), uploads video + transcript
  to MinIO, and dispatches a synthesize-only task.

Runs as a --no-agent cron. Idempotent: a link is marked 'dispatched' only after
its task is created; a crash re-runs cleanly next night.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/root/kb-agent")

from agent.archival import upload_file_to_minio  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.db import (  # noqa: E402
    AGENT_SCHEMA,
    init_db,
    list_ingest_by_status,
    update_ingest_status,
)
from agent.core.submit import dispatch_synthesize, submit_url  # noqa: E402

# content_type -> download-phase step names (everything before the ffmpeg audio
# extraction, which batch_transcribe.py does itself).
VIDEO_DOWNLOAD_STEPS = {
    "youtube": ["extract_metadata", "download_video"],
    "bunny_stream": ["extract_metadata", "download_video"],
    "instagram_reel": ["ig_reel_validate", "ig_reel_download"],
    "threads_video": ["threads_detect", "threads_extract_video"],
}

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


def _download_video(q: dict, config) -> Path | None:
    """Run one link's download-phase steps and return the produced video file."""
    from agent.pipelines import get_pipeline

    pipeline = get_pipeline(q["content_type"])
    step_fns = {s.name: s.fn for s in pipeline}
    task = {
        "id": q["id"],
        "slug": q["slug"],
        "source_url": q["url"],
        "content_type": q["content_type"],
    }
    for name in VIDEO_DOWNLOAD_STEPS[q["content_type"]]:
        fn = step_fns.get(name)
        if fn is None:
            print(f"  [warn] {q['slug']}: download step '{name}' not found")
            return None
        try:
            fn(task, config)
        except Exception as exc:  # noqa: BLE001 — download is best-effort per link
            print(f"  [skip] {q['slug']}: {name} failed ({exc})")
            return None

    slug = q["slug"]
    tmp = Path(config.temp_dir)
    for p in tmp.iterdir():
        if p.name.startswith(slug) and p.suffix.lower() in VIDEO_EXTS:
            return p
    print(f"  [warn] {q['slug']}: no video file found after download")
    return None


def _process_videos(video_links: list[dict], config, conn) -> None:
    batch_dir = Path(config.temp_dir) / "batch_videos"
    out_dir = Path(config.temp_dir) / "batch_out"
    batch_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. download all videos (per-type download steps)
    staged = []
    for q in video_links:
        vp = _download_video(q, config)
        if vp is None:
            continue
        dest = batch_dir / f"{q['slug']}{vp.suffix.lower()}"
        shutil.copy(vp, dest)
        staged.append((q, dest))

    if not staged:
        print("No videos downloaded; nothing to transcribe.")
        return

    print(f"Batch-transcribing {len(staged)} video(s) in one Modal run...")
    # 2. ONE batch Modal transcription
    r = subprocess.run(
        [sys.executable, "/root/kb-agent/scripts/batch_transcribe.py",
         str(batch_dir), "--out-dir", str(out_dir)],
        capture_output=True, text=True, timeout=7200,
    )
    print((r.stdout or r.stderr)[-1500:])

    # 3. upload video + transcript, then dispatch synthesize-only
    for q, vdest in staged:
        merged = out_dir / f"{q['slug']}.json"
        if not merged.exists():
            print(f"  [warn] {q['slug']}: no merged transcript; leaving queued")
            continue
        try:
            upload_file_to_minio(vdest, f"{config.minio_bucket}/videos/{q['slug']}{vdest.suffix.lower()}")
            upload_file_to_minio(merged, f"{config.minio_bucket}/transcripts/{q['slug']}.json")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {q['slug']}: upload failed ({exc}); leaving queued")
            continue
        rr = dispatch_synthesize(
            q["url"], config, conn,
            content_type=f"{q['content_type']}_synthesize", slug=q["slug"],
        )
        update_ingest_status(conn, q["id"], "dispatched")
        print(f"  [dispatch] {q['slug']} -> {rr.get('task_id', rr.get('status'))}")


def main() -> None:
    config = load_config()
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)

    queued = list_ingest_by_status(conn, "queued")
    if not queued:
        print("No queued ingest links.")
        return

    print(f"Queued: {len(queued)} link(s)")
    video = [q for q in queued if q["content_type"] in VIDEO_DOWNLOAD_STEPS]
    non_video = [q for q in queued if q["content_type"] not in VIDEO_DOWNLOAD_STEPS]

    for q in non_video:
        r = submit_url(q["url"], config, conn, ignore_dedup=True)
        status = r.get("status")
        if status in ("pending", "paused", "skipped"):
            update_ingest_status(conn, q["id"], "dispatched")
            print(f"  [dispatch] {q['slug']} ({q['content_type']}) -> {r.get('task_id', status)}")
        else:
            update_ingest_status(conn, q["id"], "failed", error=str(r.get("error") or status))
            print(f"  [failed] {q['slug']} -> {status}")

    if video:
        _process_videos(video, config, conn)

    conn.close()
    print(f"DONE: {len(non_video)} non-video + {len(video)} video processed")


if __name__ == "__main__":
    main()
