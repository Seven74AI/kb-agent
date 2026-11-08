"""Invariant: run_batch child output is visible-scrubbed before durable log append."""

from __future__ import annotations

import subprocess
import sys
import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from agent.db import AGENT_SCHEMA, init_db, queue_ingest
from agent.mcp.server import agent_relative_path, build_mcp_server
from agent.persist import append_subprocess_log_line, persist_subprocess_log_stream


def test_append_subprocess_log_line_scrubs_urls_and_paths(tmp_path: Path):
    log = tmp_path / "data" / "batch_manual.log"
    raw = "failed at /root/kb-agent/data/x https://evil.test/v?q=SECRET_TOKEN\n"
    append_subprocess_log_line(log, raw)
    text = log.read_text(encoding="utf-8")
    assert "SECRET_TOKEN" not in text
    assert "/root/kb-agent" not in text
    assert "[path]" in text
    assert "failed at" in text


def test_persist_subprocess_log_stream_drains_scrubbed_child_output(tmp_path: Path):
    log = tmp_path / "batch_manual.log"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.write('line1 /root/kb-agent/leak\\n'); "
                "sys.stdout.write('https://example.com/watch?v=x&token=QUERY_LEAK\\n')"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert child.stdout is not None
    thread = persist_subprocess_log_stream(log, child.stdout)
    assert child.wait(timeout=30) == 0
    thread.join(timeout=5)
    text = log.read_text(encoding="utf-8")
    assert "QUERY_LEAK" not in text
    assert "/root/kb-agent" not in text
    assert "token=" not in text
    assert "[path]" in text
    assert "line1" in text


def test_run_batch_relative_log_path_unchanged(tmp_path: Path):
    log_path = tmp_path / "data" / "batch_manual.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    visible = agent_relative_path(log_path, tmp_path)
    assert visible == "data/batch_manual.log"
    assert not Path(visible).is_absolute()


@pytest.mark.asyncio
async def test_mcp_run_batch_uses_pipe_not_raw_log_fd(monkeypatch, tmp_path):
    import agent.mcp.server as mcp_server

    config = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")
    config.vault_path.mkdir(parents=True, exist_ok=True)
    conn = init_db(tmp_path / "data" / "agent.db", AGENT_SCHEMA)
    queue_ingest(conn, "https://example.com/v.mp4", "youtube", "vid_scrub_batch")
    conn.close()

    monkeypatch.setattr(mcp_server, "is_locked", lambda _cfg: False)
    popen_kwargs: list[dict] = []
    real_popen = mcp_server.subprocess.Popen

    def fake_popen(*_a, **kw):
        popen_kwargs.append(kw)
        return real_popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    monkeypatch.setattr(mcp_server.subprocess, "Popen", fake_popen)
    mcp = build_mcp_server(config)
    run_batch = (await mcp.get_tool("run_batch")).fn
    result = run_batch()
    assert result["status"] == "started"
    assert result["log"] == "data/batch_manual.log"
    assert popen_kwargs
    assert popen_kwargs[0]["stdout"] == subprocess.PIPE
    assert popen_kwargs[0]["stderr"] == subprocess.STDOUT
    log_path = config.kb_agent_dir / result["log"]
    assert log_path.is_file()
    header = log_path.read_text(encoding="utf-8")
    assert "--- run_batch" in header
