"""Merged-transcript redaction on batch/cron script paths."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import scripts.batch_transcribe as bt

_REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZED = "https://cdn.example.com/spoken.mp3?token=secret&expires=999"


def _load_tdah_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "tdah_daily_check",
        _REPO_ROOT / "scripts" / "tdah_daily_check.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_batch_transcribe_merged_write_redacts_spoken_urls(tmp_path, monkeypatch):
    stage = tmp_path / "in"
    stage.mkdir()
    (stage / "clip.wav").write_bytes(b"RIFF")
    out = tmp_path / "out"

    def fake_sh(cmd, **kw):
        if "kb_transcribe.py" in cmd:
            out.mkdir(parents=True, exist_ok=True)
            diar = {
                "segments": [{"start": 0, "end": 10, "speaker": "SPEAKER_00"}],
            }
            trans = {
                "segments": [
                    {"start": 0, "end": 10, "text": f"see {TOKENIZED} now"},
                ],
            }
            (out / "clip.diarization.json").write_text(
                json.dumps(diar),
                encoding="utf-8",
            )
            (out / "clip.transcript.json").write_text(
                json.dumps(trans),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(bt, "sh", fake_sh)
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["batch_transcribe.py", str(stage), "--out-dir", str(out)])
        rc = bt.main()

    assert rc == 0
    merged = json.loads((out / "clip.json").read_text(encoding="utf-8"))
    text = merged["segments"][0]["text"]
    assert "token=secret" not in text
    assert "https://cdn.example.com/spoken.mp3" in text


def test_batch_ingest_upload_redacts_merged_transcript(
    tmp_path, monkeypatch,
):
    import scripts.batch_ingest as batch_ingest
    from agent.config import Config
    from agent.db import AGENT_SCHEMA, init_db, queue_ingest

    config = Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        minio_bucket="kb-agent-archive",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.kb_agent_dir / "scripts").mkdir(parents=True, exist_ok=True)
    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    ingest_id = queue_ingest(
        conn,
        "https://www.youtube.com/watch?v=redacttest",
        "youtube",
        "redact-slug",
        media_kind="video",
    )[0]
    q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
    vdest = tmp_path / f"{ingest_id}.mp4"
    vdest.write_bytes(b"x" * 200)

    monkeypatch.setattr(
        batch_ingest,
        "load_config",
        lambda: config,
    )
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _c: (vdest, None))

    uploaded_bodies: list[str] = []

    def track_upload(src, dest_key):
        if "transcripts/" in dest_key:
            uploaded_bodies.append(Path(src).read_text(encoding="utf-8"))

    monkeypatch.setattr(batch_ingest, "upload_file_to_minio", track_upload)
    monkeypatch.setattr(
        batch_ingest,
        "dispatch_synthesize",
        lambda *_a, **_k: {"status": "pending", "task_id": "t1"},
    )

    def fake_run(cmd, **kw):
        if len(cmd) > 1 and "batch_transcribe.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            raw = {
                "segments": [
                    {"time": 0.0, "speaker": "SPEAKER_00", "text": f"link {TOKENIZED}"},
                ],
            }
            (out_dir / f"{ingest_id}.json").write_text(
                json.dumps(raw),
                encoding="utf-8",
            )
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
    failed, dispatched, _ = batch_ingest._process_videos([q], config, conn, 1)
    assert failed == 0
    assert dispatched == 1
    assert uploaded_bodies
    uploaded = json.loads(uploaded_bodies[0])
    assert "token=secret" not in uploaded["segments"][0]["text"]
    assert "https://cdn.example.com/spoken.mp3" in uploaded["segments"][0]["text"]


def test_tdah_mc_cp_transcript_redacts_spoken_urls(monkeypatch, tmp_path):
    mod = _load_tdah_module()
    ns = "tdah-adulte-2026-j1-redacttest"
    slug_html = '<a href="/tdah-adulte-j1-redacttest/">x</a>'
    mc_transcript_paths: list[str] = []

    monkeypatch.setattr(mod, "TMP", str(tmp_path))
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(kb_agent_dir=tmp_path))
    monkeypatch.setattr(
        mod,
        "init_db",
        lambda *a, **k: MagicMock(close=lambda: None),
    )
    monkeypatch.setattr(
        mod,
        "fetch",
        lambda url: (
            '<a href="/tdah-adulte-j1-redacttest/">x</a>'
            if url == mod.MAIN
            else slug_html
        ),
    )
    monkeypatch.setattr(mod, "bunny_uuid", lambda _s: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    diar = {
        "segments": [{"start": 0, "end": 10, "speaker": "SPEAKER_00"}],
    }
    trans = {
        "segments": [{"start": 0, "end": 10, "text": f"voir {TOKENIZED}"}],
    }
    (out_dir / f"{ns}.diarization.json").write_text(json.dumps(diar), encoding="utf-8")
    (out_dir / f"{ns}.transcript.json").write_text(json.dumps(trans), encoding="utf-8")

    def fake_sh(cmd, **kw):
        cmd0 = cmd[0] if cmd else ""
        if cmd0 == "mc" and len(cmd) >= 3 and cmd[1] == "cp" and str(cmd[-1]).endswith(".json"):
            mc_transcript_paths.append(cmd[2])
            return MagicMock(stdout="", stderr="", returncode=0)
        if cmd0 == "mc" and cmd[1] == "ls":
            return MagicMock(stdout="", stderr="", returncode=0)
        if "kb_transcribe.py" in cmd or (cmd0.endswith("modal")):
            return MagicMock(stdout="ok", stderr="", returncode=0)
        if cmd0 == "curl":
            out_mp4 = cmd[cmd.index("-o") + 1]
            Path(out_mp4).write_bytes(b"\x00" * 200_000)
            return MagicMock(stdout="", stderr="", returncode=0)
        if cmd0 == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"RIFF")
            return MagicMock(stdout="", stderr="", returncode=0)
        return MagicMock(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mod, "sh", fake_sh)
    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "pending", "task_id": "t1"},
    )

    mod.main()
    assert mc_transcript_paths, "expected mc cp for transcript JSON"
    parsed = json.loads(Path(mc_transcript_paths[0]).read_text(encoding="utf-8"))
    text = parsed["segments"][0]["text"]
    assert "token=secret" not in text
    assert "https://cdn.example.com/spoken.mp3" in text
