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
