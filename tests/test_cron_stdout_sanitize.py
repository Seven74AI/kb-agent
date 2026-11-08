"""Boundary 4: cron/thread stdout must not emit raw secrets from exceptions."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_tdah_modal_output_tail_sanitized(monkeypatch, capsys, tmp_path):
    mod = _load_script_module("tdah_daily_check", "scripts/tdah_daily_check.py")
    raw_tail = (
        "stderr from /root/modal/kb_transcribe.py\n"
        "ERROR: token=modal-secret\n"
    )
    ns = "tdah-adulte-2026-j9-modaltest"
    slug_html = f'<a href="/tdah-adulte-j9-modaltest/">x</a>'

    monkeypatch.setattr(mod, "TMP", str(tmp_path))
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(kb_agent_dir=tmp_path))
    monkeypatch.setattr(
        mod,
        "init_db",
        lambda *a, **k: MagicMock(close=lambda: None),
    )
    monkeypatch.setattr(mod, "fetch", lambda url: slug_html if "terragrandir" in url else "")
    monkeypatch.setattr(mod, "bunny_uuid", lambda _s: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    def fake_sh(cmd, **kw):
        cmd0 = cmd[0] if cmd else ""
        if cmd0.endswith("modal") or (len(cmd) > 1 and "modal" in cmd[0]):
            return MagicMock(stdout="", stderr=raw_tail, returncode=0)
        if cmd0 == "mc":
            return MagicMock(stdout="", stderr="", returncode=0)
        if cmd0 == "curl":
            out_mp4 = cmd[cmd.index("-o") + 1] if "-o" in cmd else str(tmp_path / "x.mp4")
            with open(out_mp4, "wb") as fh:
                fh.write(b"\x00" * 200_000)
            return MagicMock(stdout="", stderr="", returncode=0)
        if cmd0 == "ffmpeg":
            return MagicMock(stdout="", stderr="", returncode=0)
        return MagicMock(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mod, "sh", fake_sh)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    diar = out_dir / f"{ns}.diarization.json"
    trans = out_dir / f"{ns}.transcript.json"
    diar.write_text('{"segments": []}', encoding="utf-8")
    trans.write_text('{"segments": []}', encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "merge_speakers_from_paths",
        lambda *_a, **_k: {"segments": []},
    )
    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "pending", "task_id": "t1"},
    )

    mod.main()
    out = capsys.readouterr().out
    assert "modal-secret" not in out
    assert "token=" not in out
    assert "/root/modal" not in out
    assert "[path]" in out
    assert "Modal output tail:" in out


def test_tdah_finalize_dispatch_sanitizes_error_stdout(monkeypatch, capsys):
    mod = _load_script_module("tdah_daily_check", "scripts/tdah_daily_check.py")
    secret = (
        "dispatch failed api_key=deadbeef\n"
        "Traceback (most recent call last):\n  File '/root/kb-agent/secret/foo.py'"
    )
    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "error", "error": secret},
    )
    mod._finalize_video_dispatch("https://example.com/x", MagicMock(), MagicMock(), "slug-x")
    out = capsys.readouterr().out
    assert "deadbeef" not in out
    assert "api_key=" not in out
    assert "/root/kb-agent" not in out
    assert "dispatch failed" in out


def test_run_consumer_sanitizes_thread_exception(monkeypatch, capsys):
    import run as run_mod

    loop = asyncio.new_event_loop()

    class BoomConsumer:
        async def startup(self):
            return None

        async def run(self):
            raise RuntimeError(
                "token=leaked\nTraceback (most recent call last):\n"
                "  File '/root/kb-agent/run.py'"
            )

        async def shutdown(self):
            return None

    monkeypatch.setattr(run_mod, "Consumer", lambda cfg: BoomConsumer())
    run_mod._run_consumer(MagicMock(), loop)
    err = capsys.readouterr().err
    assert "leaked" not in err
    assert "token=" not in err
    assert "/root/kb-agent" not in err
    assert "Consumer error:" in err


@pytest.mark.asyncio
async def test_watch_scheduler_sanitizes_poll_exception(monkeypatch, capsys):
    from agent.config import Config
    from agent.core.watch_source import run_watch_scheduler

    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise OSError("cookie=/root/secret/cookies.txt")

    monkeypatch.setattr(
        "agent.core.watch_source._poll_due_sources_in_thread",
        boom,
    )

    async def fast_sleep(_):
        return None

    monkeypatch.setattr("asyncio.sleep", fast_sleep)

    stopping = lambda: calls["n"] >= 1
    cfg = Config(kb_agent_dir=Path("/tmp"), vault_path=Path("/tmp/v"))
    await run_watch_scheduler(cfg, poll_interval_seconds=0.01, stopping=stopping)
    out = capsys.readouterr().out
    assert "/root/secret" not in out
    assert "[path]" in out
    assert "poll error:" in out


def test_rotate_logs_missing_db_scrubs_path_in_stdout(monkeypatch, capsys, tmp_path):
    mod = _load_script_module("rotate_logs", "scripts/rotate_logs.py")
    missing = tmp_path / "secret" / "logs.db"
    monkeypatch.setattr(mod, "LOGS_DB", missing)
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert str(missing) not in out
    assert "[path]" in out
    assert "No logs.db at" in out


def test_rotate_logs_gzip_failure_scrubs_stderr(monkeypatch, capsys, tmp_path):
    import sqlite3

    from agent.db import LOGS_SCHEMA, init_db

    mod = _load_script_module("rotate_logs", "scripts/rotate_logs.py")
    db_path = tmp_path / "logs.db"
    conn = init_db(db_path, LOGS_SCHEMA)
    old = "2020-01-01 00:00:00"
    conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content, logged_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("t1", "download", "stderr", "x", old),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(mod, "LOGS_DB", db_path)
    monkeypatch.setattr(
        "gzip.open",
        MagicMock(
            side_effect=OSError(
                "Permission denied for /root/kb-agent/data/logs.db token=gzfail"
            )
        ),
    )
    assert mod.main() == 1
    err = capsys.readouterr().err
    assert "gzfail" not in err
    assert "/root/kb-agent" not in err
    assert "[path]" in err
    assert "ERROR during dump/gzip" in err


def test_rotate_logs_upload_failure_scrubs_stderr(monkeypatch, capsys, tmp_path):
    import sqlite3
    from unittest.mock import MagicMock

    from agent.db import LOGS_SCHEMA, init_db

    mod = _load_script_module("rotate_logs", "scripts/rotate_logs.py")
    db_path = tmp_path / "logs.db"
    conn = init_db(db_path, LOGS_SCHEMA)
    old = "2020-01-01 00:00:00"
    conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content, logged_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("t1", "download", "stderr", "x", old),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(mod, "LOGS_DB", db_path)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: MagicMock(
            returncode=1,
            stderr="mc: ERROR at /root/kb-agent/cookies/x.txt api_key=mcleak",
        ),
    )
    with monkeypatch.context() as mp:
        mp.setattr("pathlib.Path.unlink", lambda *a, **k: None)
        assert mod.main() == 1
    err = capsys.readouterr().err
    assert "mcleak" not in err
    assert "/root/kb-agent" not in err
    assert "[path]" in err
    assert "MinIO upload FAILED" in err


def test_kb_transcribe_visible_stdout_routes_through_sanitize_emit():
    from agent.notify import sanitize_visible_message

    text = (_REPO_ROOT / "modal" / "kb_transcribe.py").read_text(encoding="utf-8")
    assert "def _sanitize_emit" in text
    assert "sanitize_visible_message" in text
    assert "_sanitize_emit(p)" in text
    assert 'print(f"wrote {_sanitize_emit(p)}")' in text

    raw = "wrote /root/kb-agent/tmp/out.json?token=txsecret"
    clean = sanitize_visible_message(raw)
    assert "/root/kb-agent" not in clean
    assert "txsecret" not in clean
    assert "token=" not in clean
    assert "[path]" in clean


def test_transcribe_script_sanitizes_audio_path_stdout(capsys, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import runpy

    out_json = tmp_path / "out.json"
    leaking = "/root/kb-agent/tmp/clip.wav?token=TRANSCRIBE_LEAK"
    repo = _REPO_ROOT
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    fake_info = MagicMock(language="en", duration=1.0)

    class FakeModel:
        def transcribe(self, *_a, **_k):
            return iter([]), fake_info

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        MagicMock(WhisperModel=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setattr(sys, "argv", ["transcribe.py", leaking, str(out_json), "60"])
    with monkeypatch.context() as m:
        m.chdir(repo)
        runpy.run_path(str(repo / "scripts" / "transcribe.py"), run_name="__main__")

    out = capsys.readouterr().out
    assert "TRANSCRIBE_LEAK" not in out
    assert "token=" not in out
    assert "/root/kb-agent" not in out
    assert "[path]" in out
    assert "Transcribing" in out


def test_diarize_script_sanitizes_audio_path_stdout(capsys, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import runpy

    out_json = tmp_path / "out.json"
    leaking = "/root/kb-agent/tmp/clip_8k.wav?token=DIARIZE_LEAK"
    repo = _REPO_ROOT
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    class FakeDiarization:
        def itertracks(self, yield_label=True):
            return iter([])

    class FakePipeline:
        def __call__(self, *_a, **_k):
            return MagicMock(speaker_diarization=FakeDiarization())

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        MagicMock(Pipeline=MagicMock(from_pretrained=lambda *a, **k: FakePipeline())),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token-not-printed")
    monkeypatch.setattr(sys, "argv", ["diarize.py", leaking, str(out_json)])
    with monkeypatch.context() as m:
        m.chdir(repo)
        runpy.run_path(str(repo / "scripts" / "diarize.py"), run_name="__main__")

    out = capsys.readouterr().out
    assert "DIARIZE_LEAK" not in out
    assert "token=" not in out
    assert "/root/kb-agent" not in out
    assert "[path]" in out
    assert "Diarizing" in out


def test_transcribe_and_diarize_progress_prints_route_sanitize():
    for rel in ("scripts/transcribe.py", "scripts/diarize.py"):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "sanitize_visible_message" in text
        assert "from agent.notify import sanitize_visible_message" in text


def test_kb_ocr_visible_stdout_routes_through_sanitize_emit():
    from agent.notify import sanitize_visible_message

    text = (_REPO_ROOT / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
    assert "def _sanitize_emit" in text
    assert "_sanitize_modal_log" not in text
    assert "sanitize_visible_message" in text
    assert '_sanitize_emit(exc)' in text
    assert '_sanitize_emit(image_dir)' in text
    assert '_sanitize_emit(p)' in text

    raw = "fail path=/home/ubuntu/secret/img.png api_key=ocrstdoutleak"
    clean = sanitize_visible_message(raw)
    assert "/home/ubuntu" not in clean
    assert "ocrstdoutleak" not in clean
    assert "[path]" in clean
    assert "[credential redacted]" in clean


def test_db_has_no_ingest_media_appended_reader():
    text = (_REPO_ROOT / "agent" / "db.py").read_text(encoding="utf-8")
    assert "def ingest_media_appended" not in text
    assert "def set_ingest_media_appended" in text


def test_delegate_books_documented_as_implemented():
    doc = (_REPO_ROOT / "docs" / "core-extraction-inventory-consumer.md").read_text(
        encoding="utf-8"
    )
    assert "step_delegate_books" in doc
    assert "not implemented yet" not in doc
    assert "Implemented" in doc


def test_watch_sources_poll_stderr_sanitizes_url_and_error(monkeypatch, capsys):
    mod = _load_script_module("watch_sources_poll", "scripts/watch_sources_poll.py")
    token_url = "https://example.com/watch?token=supersecret&x=1"
    raw_err = "fetch failed: password=bad\n/home/ubuntu/.ssh/id_rsa"

    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(kb_agent_dir=Path("/tmp")))
    monkeypatch.setattr(mod, "init_db", lambda *_a, **_k: MagicMock(close=lambda: None))
    monkeypatch.setattr(
        mod,
        "poll_all_due_sources",
        lambda *a, **k: [{"page_url": token_url, "error": raw_err}],
    )

    assert mod.main() == 0
    err = capsys.readouterr().err
    assert "supersecret" not in err
    assert "token=" not in err
    assert "id_rsa" not in err
    assert "https://example.com/watch" in err
    assert "ERROR" in err
