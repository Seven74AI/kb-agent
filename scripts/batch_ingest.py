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
import json
import os
import shutil
import sqlite3
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
    bootstrap_parent_media_deferral_from_manifest,
    claim_ingest_items,
    count_pending_media,
    ensure_parent_media_deferral,
    finish_batch_run,
    get_parent_media_deferral,
    has_completed_parent_synthesis,
    supersede_stale_deferral_synthesize_tasks,
    init_db,
    append_parent_media_atomic,
    parent_media_complete,
    normalize_tags,
    record_batch_run,
    recover_failed_media,
    recover_stale_ingest,
    update_ingest_status,
)
from agent.notify import sanitize_visible_message  # noqa: E402
from agent.core.batch_lock import acquire, release  # noqa: E402
from agent.core.submit import dispatch_synthesize, submit_url  # noqa: E402
from agent.fetch_guard import _assert_fetch_url_allowed  # noqa: E402

# Reject staging huge images to Modal (GPU OOM / pixel bombs).
MAX_STAGED_IMAGE_BYTES = 25 * 1024 * 1024
OCR_BATCH_MAX_FAILURE_RATE = 0.20

# Parent slugs whose deferred synthesize task was already dispatched this run.
_deferred_synthesize_dispatched: set[str] = set()

# content_type -> download-phase step names (everything before the ffmpeg audio
# extraction, which batch_transcribe.py does itself).
VIDEO_DOWNLOAD_STEPS = {
    "youtube": ["extract_metadata", "download_video"],
    "bunny_stream": ["extract_metadata", "download_video"],
    "rumble": ["extract_metadata", "download_video"],
    "substack_video": ["extract_metadata", "download_video"],
    "instagram_reel": ["ig_reel_validate", "ig_reel_download"],
    "threads_video": ["threads_detect", "threads_extract_video"],
}

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")

# content_type values that denote embedded still images when media_kind is missing.
_IMAGE_CONTENT_TYPES = frozenset({"substack_image"})

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


def _media_kind_from_content_type(content_type: str) -> str | None:
    if content_type in _IMAGE_CONTENT_TYPES or content_type.endswith("_image"):
        return "image"
    if content_type in VIDEO_DOWNLOAD_STEPS:
        return "video"
    return None


def _resolve_media_kind(q: dict) -> tuple[str | None, str | None]:
    """Map an embedded ingest row to ``image`` or ``video``. Returns (kind, error)."""
    raw = q.get("media_kind")
    if raw is not None:
        mk = str(raw).strip().lower()
        if mk in ("image", "video"):
            return mk, None
        inferred = _media_kind_from_content_type(q.get("content_type") or "")
        if inferred:
            return inferred, None
        return None, f"unsupported media_kind: {raw!r}"
    inferred = _media_kind_from_content_type(q.get("content_type") or "")
    if inferred:
        return inferred, None
    ct = q.get("content_type") or ""
    return None, f"missing or unknown media_kind (content_type={ct!r})"


def _read_json_file(path: Path, label: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid {label} JSON: {exc}"


def _ingest_download_stem(task: dict) -> str:
    """Filesystem stem for per-ingest download artifacts (queue/task id, not slug)."""
    return task["id"]


def _download_video(q: dict, config) -> tuple[Path | None, str | None]:
    """Run one link's download-phase steps. Returns (video_path, error)."""
    from agent.pipelines import get_pipeline

    steps = VIDEO_DOWNLOAD_STEPS.get(q["content_type"])
    if not steps:
        return None, f"unsupported embedded content_type: {q['content_type']}"
    pipeline = get_pipeline(q["content_type"])
    step_fns = {s.name: s.fn for s in pipeline}
    task = {
        "id": q["id"],
        "slug": q["slug"],
        "source_url": q["url"],
        "content_type": q["content_type"],
        "_batch_ingest": True,
    }
    for name in steps:
        fn = step_fns.get(name)
        if fn is None:
            return None, f"download step '{name}' not found"
        try:
            fn(task, config)
        except Exception as exc:  # noqa: BLE001 — download is best-effort per link
            return None, f"{name} failed: {exc}"

    tmp = Path(config.temp_dir)
    stem = _ingest_download_stem(task)
    for ext in VIDEO_EXTS:
        candidate = tmp / f"{stem}{ext}"
        if candidate.is_file():
            return candidate, None
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
            update_ingest_status(conn, q["id"], "failed", error=err or "download failed")
            failed += 1
            causes.append(f"{q['slug']}: {err}")
            print(f"  [failed] {q['slug']}: {err}")
            continue
        ctype = q["content_type"]
        bdir = base / "batch_videos" / run_tag / ctype
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / f"{q['id']}{vp.suffix.lower()}"
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
                update_ingest_status(conn, q["id"], "failed", error=f"batch_transcribe failed (exit {r.returncode})")
            failed += len(items)
            causes.append(f"{ctype}: batch_transcribe exit {r.returncode}")
            print(f"  [error] batch_transcribe failed ({r.returncode}); marking {ctype} failed")
            continue

        # 3. upload video + transcript, then dispatch synthesize-only
        for q, vdest in items:
            merged = out_dir / f"{q['id']}.json"
            if not merged.exists():
                update_ingest_status(conn, q["id"], "failed", error="no merged transcript produced")
                failed += 1
                causes.append(f"{q['slug']}: no merged transcript")
                print(f"  [warn] {q['slug']}: no merged transcript; marked failed")
                continue
            try:
                upload_file_to_minio(vdest, f"{config.minio_bucket}/videos/{q['slug']}{vdest.suffix.lower()}")
                upload_file_to_minio(merged, f"{config.minio_bucket}/transcripts/{q['slug']}.json")
            except Exception as exc:  # noqa: BLE001
                update_ingest_status(conn, q["id"], "failed", error=f"upload failed: {exc}")
                failed += 1
                causes.append(f"{q['slug']}: upload failed")
                print(f"  [warn] {q['slug']}: upload failed ({exc}); marked failed")
                continue
            update_ingest_status(conn, q["id"], "dispatching")
            rr = dispatch_synthesize(
                q["url"], config, conn,
                content_type=f"{q['content_type']}_synthesize", slug=q["slug"],
                tags=q.get("tags"),
            )
            if rr.get("status") == "error":
                err = rr.get("error") or "dispatch_synthesize failed"
                update_ingest_status(conn, q["id"], "failed", error=err)
                failed += 1
                causes.append(f"{q['slug']}: {err}")
                print(f"  [warn] {q['slug']}: dispatch failed ({err})")
                continue
            update_ingest_status(conn, q["id"], "dispatched")
            dispatched += 1
            print(f"  [dispatch] {q['slug']} -> {rr.get('task_id', rr.get('status'))}")

    return failed, dispatched, causes


def _append_to_parent(
    config,
    parent_slug: str,
    section: str,
    body: str,
    *,
    conn=None,
    ingest_id: str | None = None,
) -> bool:
    """Append a media reading/transcript to the parent article's raw markdown."""
    raw_path = Path(config.temp_dir) / f"{parent_slug}_raw.md"
    if conn is not None and ingest_id:
        return append_parent_media_atomic(conn, ingest_id, raw_path, section, body)
    if not raw_path.exists():
        return False
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8") + f"\n\n## {section}\n\n{body.strip()}\n",
        encoding="utf-8",
    )
    return True


def _parent_tags(conn, parent_slug: str):
    """Tags for the parent article (task row, else any sibling ingest row)."""
    row = conn.execute(
        "SELECT tags FROM tasks WHERE slug = ? LIMIT 1",
        (parent_slug,),
    ).fetchone()
    if row and row[0]:
        return normalize_tags(row[0])
    row = conn.execute(
        "SELECT tags FROM ingest_queue WHERE parent_slug = ? AND tags IS NOT NULL LIMIT 1",
        (parent_slug,),
    ).fetchone()
    if row and row[0]:
        return normalize_tags(row[0])
    return None


def _supersede_deferred_parent_task(conn, parent_slug: str, *, synthesize_task_id: str | None = None) -> None:
    """Mark the original Substack parent ingestion task terminal after batch synthesis."""
    detail = "deferred synthesis dispatched"
    if synthesize_task_id:
        detail = f"{detail} (task {synthesize_task_id})"
    reason = f"{detail}; parent ingestion superseded"
    with conn:
        conn.execute(
            """
            UPDATE tasks SET status = 'completed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type NOT GLOB '*_synthesize'
            AND status IN ('paused', 'pending', 'in_progress')
            """,
            (reason, parent_slug),
        )


def refresh_parent_media_manifest(conn, config, parent_slug: str) -> None:
    """Rewrite on-disk manifest from the DB gate (keeps authoritative expected_total)."""
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        bootstrap_parent_media_deferral_from_manifest(conn, parent_slug, config.temp_dir)
        deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        return
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.temp_dir) / f"{parent_slug}_media.json"
    payload = {
        "url": deferral["source_url"],
        "content_type": deferral["content_type"],
        "total": deferral["expected_total"],
        "media_changed": True,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _maybe_dispatch_parent_synthesize(conn, config, parent_slug: str) -> bool:
    """Fire the parent's deferred synthesis once every sibling media item is done."""
    if parent_slug in _deferred_synthesize_dispatched:
        return False
    if has_completed_parent_synthesis(
        conn,
        parent_slug,
        vault_path=config.vault_path,
        temp_dir=config.temp_dir,
    ):
        print(f"  [skip] {parent_slug}: parent synthesis already completed")
        return False
    manifest_path = Path(config.temp_dir) / f"{parent_slug}_media.json"
    if manifest_path.exists():
        manifest, manifest_err = _read_json_file(manifest_path, "manifest")
        if manifest_err:
            print(f"  [skip] {parent_slug}: {manifest_err}")
            return False
        total = manifest.get("total")
        url = manifest.get("url")
        content_type = manifest.get("content_type")
        if total is not None and url and content_type:
            ensure_parent_media_deferral(
                conn, parent_slug, url, content_type, int(total),
            )
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        bootstrap_parent_media_deferral_from_manifest(conn, parent_slug, config.temp_dir)
        deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        print(f"  [skip] {parent_slug}: deferral record missing; fail closed")
        return False
    raw_path = Path(config.temp_dir) / f"{parent_slug}_raw.md"
    if not parent_media_complete(conn, parent_slug, raw_path=raw_path):
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (parent_slug,),
        ).fetchone()
        actual = row["n"] if hasattr(row, "keys") else row[0]
        expected = deferral["expected_total"]
        print(
            f"  [skip] {parent_slug}: parent media incomplete "
            f"(rows {actual}/{expected}, raw={raw_path.name})"
        )
        return False
    if int(deferral.get("media_changed") or 0):
        supersede_stale_deferral_synthesize_tasks(conn, parent_slug, deferral)
    rr = dispatch_synthesize(
        deferral["source_url"], config, conn,
        content_type=f"{deferral['content_type']}_synthesize", slug=parent_slug,
        tags=_parent_tags(conn, parent_slug),
    )
    if rr.get("duplicate"):
        _deferred_synthesize_dispatched.add(parent_slug)
        _supersede_deferred_parent_task(conn, parent_slug, synthesize_task_id=rr.get("task_id"))
        return False
    _deferred_synthesize_dispatched.add(parent_slug)
    _supersede_deferred_parent_task(conn, parent_slug, synthesize_task_id=rr.get("task_id"))
    print(f"  [deferred] {parent_slug}: all media done -> synthesize")
    return True


def _reconcile_deferred_parent_syntheses(conn, config) -> None:
    """Dispatch deferred synthesis for parents whose media finished in a prior run."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT parent_slug FROM ingest_queue WHERE COALESCE(parent_slug, '') != ''",
    ).fetchall()
    for row in rows:
        parent_slug = row["parent_slug"]
        bootstrap_parent_media_deferral_from_manifest(conn, parent_slug, config.temp_dir)
        raw_path = Path(config.temp_dir) / f"{parent_slug}_raw.md"
        if not parent_media_complete(conn, parent_slug, raw_path=raw_path):
            continue
        _maybe_dispatch_parent_synthesize(conn, config, parent_slug)


def _process_embedded_images(image_items: list[dict], config, conn, run_id: int) -> None:
    """VLM-read embedded images in ONE Modal batch, append readings to parents."""
    base = Path(config.temp_dir)
    stage_dir = base / "batch_images" / f"run_{run_id}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base / "batch_imgout" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[dict, Path]] = []
    for q in image_items:
        try:
            _assert_fetch_url_allowed(q["url"])
        except ValueError as exc:
            update_ingest_status(conn, q["id"], "failed", error=f"blocked fetch URL: {exc}")
            continue
        ext = Path(q["url"].split("?")[0]).suffix.lower() or ".png"
        dest = stage_dir / f"{q['id']}{ext}"
        r = subprocess.run(
            [
                "curl", "-s", "-L", "--max-redirs", "0",
                "--max-time", "60", "-o", str(dest), q["url"],
            ],
            capture_output=True, text=True, timeout=70,
        )
        if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 100:
            update_ingest_status(conn, q["id"], "failed", error="image download failed")
            continue
        size = dest.stat().st_size
        if size > MAX_STAGED_IMAGE_BYTES:
            dest.unlink(missing_ok=True)
            update_ingest_status(
                conn,
                q["id"],
                "failed",
                error=f"image too large ({size} bytes; max {MAX_STAGED_IMAGE_BYTES})",
            )
            continue
        update_ingest_status(conn, q["id"], "staged")
        staged.append((q, dest))
    if not staged:
        return

    r = subprocess.run(
        [sys.executable, str(Path(config.kb_agent_dir) / "scripts" / "batch_ocr.py"),
         str(stage_dir), "--out-dir", str(out_dir)],
        capture_output=True, text=True, timeout=3600,
    )
    print((r.stdout or r.stderr)[-800:])
    if r.returncode != 0:
        print(f"  [warn] batch_ocr exit {r.returncode}; processing available outputs")

    ocr_results: list[tuple[dict, str | None, str | None]] = []
    for q, dest in staged:
        ocr_json = out_dir / f"{dest.name}.ocr.json"
        if not ocr_json.exists():
            ocr_results.append((q, None, "no OCR output"))
            continue
        ocr_data, ocr_err = _read_json_file(ocr_json, "OCR")
        if ocr_err:
            ocr_results.append((q, None, ocr_err))
            continue
        if ocr_data.get("error"):
            ocr_results.append((q, None, f"OCR failed: {ocr_data['error']}"))
            continue
        if ocr_data.get("truncated"):
            ocr_results.append((q, None, "OCR output truncated at token cap"))
            continue
        reading = ocr_data.get("reading", "")
        if not reading.strip():
            ocr_results.append((q, None, "empty OCR reading"))
            continue
        ocr_results.append((q, reading, None))

    failure_count = sum(1 for _q, _reading, err in ocr_results if err)
    batch_failed = bool(staged) and failure_count / len(staged) > OCR_BATCH_MAX_FAILURE_RATE
    if batch_failed:
        print(
            f"  [warn] OCR batch failure rate {failure_count}/{len(staged)} "
            f"exceeds {OCR_BATCH_MAX_FAILURE_RATE:.0%}; failing batch"
        )

    for q, reading, err in ocr_results:
        if batch_failed:
            msg = err if err else "ocr batch failure rate exceeded"
            update_ingest_status(conn, q["id"], "failed", error=msg)
            continue
        if err:
            update_ingest_status(conn, q["id"], "failed", error=err)
            continue
        if not _append_to_parent(
            config, q["parent_slug"], "Image", reading, conn=conn, ingest_id=q["id"],
        ):
            update_ingest_status(
                conn, q["id"], "failed", error="parent raw markdown missing",
            )
            continue
        update_ingest_status(conn, q["id"], "dispatched")
        _maybe_dispatch_parent_synthesize(conn, config, q["parent_slug"])


def _process_embedded_videos(video_items: list[dict], config, conn, run_id: int) -> None:
    """Download + batch-transcribe embedded videos, append transcripts to parents."""
    base = Path(config.temp_dir)
    run_tag = f"run_{run_id}"
    staged: dict[str, list[tuple[dict, Path]]] = {}
    for q in video_items:
        vp, err = _download_video(q, config)
        if vp is None:
            update_ingest_status(conn, q["id"], "failed", error=err or "download failed")
            continue
        ctype = q["content_type"]
        bdir = base / "batch_videos" / run_tag / ctype
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / f"{q['id']}{vp.suffix.lower()}"
        shutil.copy(vp, dest)
        update_ingest_status(conn, q["id"], "staged")
        staged.setdefault(ctype, []).append((q, dest))
    if not staged:
        return

    for ctype, items in staged.items():
        bdir = base / "batch_videos" / run_tag / ctype
        out_dir = base / "batch_out" / run_tag / ctype
        out_dir.mkdir(parents=True, exist_ok=True)
        for q, _ in items:
            update_ingest_status(conn, q["id"], "transcribing")
        transcribe_script = Path(config.kb_agent_dir) / "scripts" / "batch_transcribe.py"
        r = subprocess.run(
            [sys.executable, str(transcribe_script),
             str(bdir), "--out-dir", str(out_dir)] + TRANSCRIBE_ARGS.get(ctype, []),
            capture_output=True, text=True, timeout=7200,
        )
        if r.returncode != 0:
            for q, _ in items:
                update_ingest_status(conn, q["id"], "failed", error=f"batch_transcribe failed ({r.returncode})")
            continue
        for q, vdest in items:
            merged = out_dir / f"{q['id']}.json"
            if not merged.exists():
                update_ingest_status(conn, q["id"], "failed", error="no merged transcript")
                continue
            merged_data, merged_err = _read_json_file(merged, "transcript")
            if merged_err:
                update_ingest_status(conn, q["id"], "failed", error=merged_err)
                continue
            segs = merged_data.get("segments", [])
            transcript = " ".join(s.get("text", "") for s in segs).strip()
            if not transcript:
                update_ingest_status(conn, q["id"], "failed", error="empty transcript")
                continue
            if not _append_to_parent(
                config,
                q["parent_slug"],
                "Video Transcript",
                transcript,
                conn=conn,
                ingest_id=q["id"],
            ):
                update_ingest_status(
                    conn, q["id"], "failed", error="parent raw markdown missing",
                )
                continue
            update_ingest_status(conn, q["id"], "dispatched")
            _maybe_dispatch_parent_synthesize(conn, config, q["parent_slug"])


def main() -> None:
    config = load_config()
    conn = None
    run_id = None
    total = 0
    failed = 0
    dispatched = 0
    causes: list[str] = []
    lock_held = False
    try:
        if not acquire(config):
            print("Another batch run is in progress; exiting.")
            return
        lock_held = True
        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        recover_stale_ingest(conn)
        # Requeue failed rows before claim; duplicate ``queue_ingest`` revive
        # accounts for its own attempt increment on embedded siblings.
        n_retry = recover_failed_media(conn)
        if n_retry:
            print(f"Requeued {n_retry} failed media item(s) for retry.")
        run_id = record_batch_run(conn)
        claimed = claim_ingest_items(conn)
        if not claimed:
            print("No queued ingest links.")
            _reconcile_deferred_parent_syntheses(conn, config)
            finish_batch_run(conn, run_id, "ok", 0, 0, 0, "no queued items")
            return

        total = len(claimed)
        print(f"Claimed: {total} link(s)")
        media = [q for q in claimed if q.get("parent_slug")]
        video = [q for q in claimed if q["content_type"] in VIDEO_DOWNLOAD_STEPS and not q.get("parent_slug")]
        non_video = [q for q in claimed if q["content_type"] not in VIDEO_DOWNLOAD_STEPS and not q.get("parent_slug")]

        # 1. embedded media (linked to a parent article) → append to parent's raw.md
        img_items: list[dict] = []
        vid_items: list[dict] = []
        for q in media:
            kind, kind_err = _resolve_media_kind(q)
            if kind_err:
                update_ingest_status(conn, q["id"], "failed", error=kind_err)
                failed += 1
                causes.append(f"{q['slug']}: {kind_err}")
                print(f"  [failed] {q['slug']}: {kind_err}")
                continue
            if kind == "image":
                img_items.append(q)
            else:
                vid_items.append(q)
        if img_items:
            print(f"Reading {len(img_items)} embedded image(s) via Modal VLM...")
            _process_embedded_images(img_items, config, conn, run_id)
        if vid_items:
            print(f"Processing {len(vid_items)} embedded video(s)...")
            _process_embedded_videos(vid_items, config, conn, run_id)

        # 2. standalone videos (existing flow)
        if video:
            vf, vd, vcauses = _process_videos(video, config, conn, run_id)
            failed += vf
            dispatched += vd
            causes += vcauses

        # 3. standalone non-video (existing flow)
        for q in non_video:
            r = submit_url(q["url"], config, conn, ignore_dedup=True, tags=q.get("tags"))
            status = r.get("status")
            if status == "pending" and r.get("task_id"):
                update_ingest_status(conn, q["id"], "dispatched")
                dispatched += 1
                print(f"  [dispatch] {q['slug']} ({q['content_type']}) -> {r.get('task_id')}")
            elif status in ("skipped", "paused"):
                print(f"  [skip-ingest] {q['slug']} ({q['content_type']}): {status} (left queued)")
            else:
                update_ingest_status(conn, q["id"], "failed", error=str(r.get("error") or status))
                failed += 1
                causes.append(f"{q['slug']} ({q['content_type']}): {r.get('error') or status}")
                print(f"  [failed] {q['slug']} -> {status}")

        _reconcile_deferred_parent_syntheses(conn, config)

        summary_raw = "; ".join(dict.fromkeys(causes[:5])) or None
        summary = sanitize_visible_message(summary_raw) if summary_raw else None
        finish_batch_run(conn, run_id, "failed" if failed else "ok", total, failed, dispatched, summary)
        print(f"DONE: {total} processed ({failed} failed, {dispatched} dispatched)")
    except Exception as exc:  # noqa: BLE001 — record the crash, then re-raise for the cron
        if conn is not None and run_id is not None:
            finish_batch_run(
                conn, run_id, "crashed", total, failed, dispatched,
                sanitize_visible_message(str(exc)),
            )
        raise
    finally:
        if conn is not None:
            conn.close()
        if lock_held:
            release(config)


if __name__ == "__main__":
    main()
