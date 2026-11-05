#!/usr/bin/env python3
"""Nightly batch collector — download + batch-transcribe videos, dispatch links.

Reads the durable ingest_queue (status='queued'):
- Non-video (web/books/substack/...) -> full pipeline task (submit_url).
- Video -> runs each pipeline's download-phase steps, then ONE batch Modal
  transcription run per content type (scripts/batch_transcribe.py), uploads
  video + transcript to MinIO, and dispatches a synthesize-only task.

Runs as a --no-agent cron. Idempotent: a link is marked 'dispatched' only after
its task is created; a crash re-runs cleanly next night.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Isolate from the Hermes runtime's py3.14 PYTHONPATH — it cross-contaminates
# py3.11 subprocesses (yt-dlp cffi mismatch) when this runs under `hermes cron`.
os.environ.pop("PYTHONPATH", None)

_bootstrap_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bootstrap_root))
# The cron runs a COPY of this script under ~/.hermes/scripts/, so parents[1]
# points at ~/.hermes, not the repo. Add the known install dir so `agent.*`
# imports resolve there too; load_config().kb_agent_dir below is authoritative
# for non-default installs.
sys.path.insert(0, "/root/kb-agent")

from agent.archival import upload_file_to_minio  # noqa: E402
from agent.config import load_config  # noqa: E402

sys.path.insert(0, str(load_config().kb_agent_dir))
from agent.db import (  # noqa: E402
    AGENT_SCHEMA,
    claim_ingest_items,
    finish_batch_run,
    init_db,
    record_batch_run,
    recover_stale_ingest,
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

# content_type -> extra batch_transcribe.py args (language + domain context).
# bunny_stream is the French TDAH summit; other types auto-detect language.
TRANSCRIBE_ARGS = {
    "bunny_stream": [
        "--language", "fr",
        "--initial-prompt",
        "Conférence en français sur le TDAH (trouble du déficit de l'attention avec ou sans "
        "hyperactivité) chez l'adulte : diagnostic, dopamine, neurodéveloppement, hormones, sommeil.",
        "--hotwords",
        "TDAH,TDA,hyperactivité,impulsivité,inattention,neurodéveloppement,neuroatypie,"
        "neurodivergence,dopamine,noradrénaline,méthylphénidate,Ritaline,diagnostic,comorbidité,"
        "autisme,TSA,HPI,psychiatre,psychologue,neuropsychologie,attention,concentration,"
        "sommeil,procrastination,hormones,œstrogène,progestérone,ménopause,émotions",
    ],
}


def _download_video(q: dict, config) -> tuple[Path | None, str | None]:
    """Run one link's download-phase steps. Returns (video_path, error)."""
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
            return None, f"download step '{name}' not found"
        try:
            fn(task, config)
        except Exception as exc:  # noqa: BLE001 — download is best-effort per link
            return None, f"{name} failed: {exc}"

    slug = q["slug"]
    tmp = Path(config.temp_dir)
    for p in tmp.iterdir():
        if p.name.startswith(slug) and p.suffix.lower() in VIDEO_EXTS:
            return p, None
    return None, "no video file found after download"


def _process_videos(video_links: list[dict], config, conn, run_id: int) -> tuple[int, int, list[str]]:
    """Download + batch-transcribe + dispatch video links.

    Returns (failed, dispatched, causes) for the batch-run summary.
    """
    failed = 0
    dispatched = 0
    causes: list[str] = []
    base = Path(config.temp_dir)
    run_tag = f"run_{run_id}"

    # 1. download all videos, staged per content type (per-run dir so a
    #    concurrent batch's staged videos don't get picked up)
    staged: dict[str, list[tuple[dict, Path]]] = {}
    for q in video_links:
        vp, err = _download_video(q, config)
        if vp is None:
            update_ingest_status(conn, q["id"], "queued", error=err or "download failed")
            failed += 1
            causes.append(f"{q['slug']}: {err}")
            print(f"  [failed] {q['slug']}: {err}")
            continue
        ctype = q["content_type"]
        bdir = base / "batch_videos" / run_tag / ctype
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / f"{q['slug']}{vp.suffix.lower()}"
        shutil.copy(vp, dest)
        update_ingest_status(conn, q["id"], "staged")
        staged.setdefault(ctype, []).append((q, dest))

    if not staged:
        print("No videos downloaded; nothing to transcribe.")
        return failed, dispatched, causes

    # 2. one Modal run per content type (per-type language/hotwords)
    for ctype, items in staged.items():
        bdir = base / "batch_videos" / run_tag / ctype
        out_dir = base / "batch_out" / run_tag / ctype
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch-transcribing {len(items)} {ctype} video(s) in one Modal run...")
        for q, _ in items:
            update_ingest_status(conn, q["id"], "transcribing")
        transcribe_script = Path(config.kb_agent_dir) / "scripts" / "batch_transcribe.py"
        r = subprocess.run(
            [sys.executable, str(transcribe_script),
             str(bdir), "--out-dir", str(out_dir)] + TRANSCRIBE_ARGS.get(ctype, []),
            capture_output=True, text=True, timeout=7200,
        )
        print((r.stdout or r.stderr)[-1500:])
        if r.returncode != 0:
            for q, _ in items:
                update_ingest_status(conn, q["id"], "queued", error=f"batch_transcribe failed (exit {r.returncode})")
            failed += len(items)
            causes.append(f"{ctype}: batch_transcribe exit {r.returncode}")
            print(f"  [error] batch_transcribe failed ({r.returncode}); leaving {ctype} queued")
            continue

        # 3. upload video + transcript, then dispatch synthesize-only
        for q, vdest in items:
            merged = out_dir / f"{q['slug']}.json"
            if not merged.exists():
                update_ingest_status(conn, q["id"], "queued", error="no merged transcript produced")
                failed += 1
                causes.append(f"{q['slug']}: no merged transcript")
                print(f"  [warn] {q['slug']}: no merged transcript; leaving queued")
                continue
            try:
                upload_file_to_minio(vdest, f"{config.minio_bucket}/videos/{q['slug']}{vdest.suffix.lower()}")
                upload_file_to_minio(merged, f"{config.minio_bucket}/transcripts/{q['slug']}.json")
            except Exception as exc:  # noqa: BLE001
                update_ingest_status(conn, q["id"], "queued", error=f"upload failed: {exc}")
                failed += 1
                causes.append(f"{q['slug']}: upload failed")
                print(f"  [warn] {q['slug']}: upload failed ({exc}); leaving queued")
                continue
            update_ingest_status(conn, q["id"], "dispatching")
            rr = dispatch_synthesize(
                q["url"], config, conn,
                content_type=f"{q['content_type']}_synthesize", slug=q["slug"],
                tags=q.get("tags"),
            )
            update_ingest_status(conn, q["id"], "dispatched")
            dispatched += 1
            print(f"  [dispatch] {q['slug']} -> {rr.get('task_id', rr.get('status'))}")

    return failed, dispatched, causes


def main() -> None:
    config = load_config()
    conn = None
    run_id = None
    total = 0
    failed = 0
    dispatched = 0
    causes: list[str] = []
    try:
        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        recover_stale_ingest(conn)
        run_id = record_batch_run(conn)
        claimed = claim_ingest_items(conn)
        if not claimed:
            print("No queued ingest links.")
            finish_batch_run(conn, run_id, "ok", 0, 0, 0, "no queued items")
            return

        total = len(claimed)
        print(f"Claimed: {total} link(s)")
        video = [q for q in claimed if q["content_type"] in VIDEO_DOWNLOAD_STEPS]
        non_video = [q for q in claimed if q["content_type"] not in VIDEO_DOWNLOAD_STEPS]

        for q in non_video:
            r = submit_url(q["url"], config, conn, ignore_dedup=True, tags=q.get("tags"))
            status = r.get("status")
            if status in ("pending", "paused", "skipped"):
                update_ingest_status(conn, q["id"], "dispatched")
                dispatched += 1
                print(f"  [dispatch] {q['slug']} ({q['content_type']}) -> {r.get('task_id', status)}")
            else:
                update_ingest_status(conn, q["id"], "failed", error=str(r.get("error") or status))
                failed += 1
                causes.append(f"{q['slug']} ({q['content_type']}): {r.get('error') or status}")
                print(f"  [failed] {q['slug']} -> {status}")

        if video:
            vf, vd, vcauses = _process_videos(video, config, conn, run_id)
            failed += vf
            dispatched += vd
            causes += vcauses

        summary = "; ".join(dict.fromkeys(causes[:5])) or None
        finish_batch_run(conn, run_id, "failed" if failed else "ok", total, failed, dispatched, summary)
        print(f"DONE: {total} processed ({failed} failed, {dispatched} dispatched)")
    except Exception as exc:  # noqa: BLE001 — record the crash, then re-raise for the cron
        if conn is not None and run_id is not None:
            finish_batch_run(conn, run_id, "crashed", total, failed, dispatched, str(exc))
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
