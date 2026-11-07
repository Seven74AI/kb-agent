"""Tests for subprocess streaming into step_logs."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.db import LOGS_SCHEMA, init_db
from agent.streaming import log_progress, stream_to_logs


@pytest.fixture
def logs_db(tmp_path):
    return init_db(tmp_path / "logs.db", LOGS_SCHEMA)


def test_log_progress_scrubs_sensitive_paths(logs_db):
    """Progress lines persisted to step_logs must not contain raw filesystem paths."""
    raw = "yt-dlp: cookiefile /root/kb-agent/cookies/yt_cookies.txt"
    log_progress(logs_db, "task-1", "download_video", raw, stream="stderr")

    row = logs_db.execute(
        "SELECT content FROM step_logs WHERE task_id = ?",
        ("task-1",),
    ).fetchone()
    assert row is not None
    assert "/root/kb-agent" not in row["content"]
    assert "[path]" in row["content"]
    assert "yt-dlp" in row["content"]


@pytest.mark.asyncio
async def test_stream_to_logs_scrubs_subprocess_stdout(logs_db):
    """Subprocess stdout lines must be sanitized before INSERT into step_logs."""
    leaky = b"ERROR: Unable to open cookie file /root/kb-agent/cookies/yt_cookies.txt\n"

    reader = asyncio.StreamReader()
    reader.feed_data(leaky)
    reader.feed_eof()

    proc = MagicMock()
    proc.stdout = reader
    proc.stderr = None

    await stream_to_logs(proc, logs_db, "task-2", "extract_metadata", "stdout")

    row = logs_db.execute(
        "SELECT content, stream FROM step_logs WHERE task_id = ?",
        ("task-2",),
    ).fetchone()
    assert row is not None
    assert row["stream"] == "stdout"
    assert "/root/kb-agent" not in row["content"]
    assert "[path]" in row["content"]
