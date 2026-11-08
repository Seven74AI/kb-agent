"""Regression tests for scripts/send_morning_digest.py abnormal-run invariant."""

import importlib
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.db import AGENT_SCHEMA, init_db

send_morning_digest = importlib.import_module("scripts.send_morning_digest")


def _insert_run(
    conn: sqlite3.Connection,
    *,
    started_at: str,
    finished_at: str | None,
    status: str,
    total: int = 0,
    failed: int = 0,
    dispatched: int = 0,
    summary: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO batch_runs "
        "(started_at, finished_at, total, failed, dispatched, status, summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (started_at, finished_at, total, failed, dispatched, status, summary),
    )
    conn.commit()


@pytest.fixture
def digest_db(tmp_path):
    db_path = tmp_path / "data" / "agent.db"
    db_path.parent.mkdir(parents=True)
    return init_db(db_path, AGENT_SCHEMA)


def test_failed_run_before_ok_still_notifies(digest_db, tmp_path):
    """Earlier failed run must alert even when a later run finished ok."""
    cutoff = "2026-11-07 08:00:00"
    _insert_run(
        digest_db,
        started_at="2026-11-07 22:00:00",
        finished_at="2026-11-07 23:00:00",
        status="failed",
        total=3,
        failed=2,
        dispatched=1,
        summary="yt-dlp timeout",
    )
    _insert_run(
        digest_db,
        started_at="2026-11-08 01:00:00",
        finished_at="2026-11-08 02:00:00",
        status="ok",
        total=5,
        failed=0,
        dispatched=5,
        summary="no queued items",
    )

    cfg = Config(kb_agent_dir=tmp_path)
    fixed_now = datetime(2026, 11, 8, 8, 30, tzinfo=timezone.utc)

    with (
        patch.object(send_morning_digest, "load_config", return_value=cfg),
        patch.object(send_morning_digest, "init_db", return_value=digest_db),
        patch.object(send_morning_digest, "_since_morning", return_value=cutoff),
        patch.object(send_morning_digest, "_utc_now", return_value=fixed_now),
        patch.object(send_morning_digest, "notify_batch") as notify,
    ):
        send_morning_digest.main()

    notify.assert_called_once()
    _cfg, kwargs = notify.call_args[0][0], notify.call_args[1]
    assert kwargs["failed"] == 2
    assert kwargs["total"] == 3
    assert kwargs["dispatched"] == 1
    assert "yt-dlp timeout" in kwargs["causes"][0]


def test_stale_running_run_triggers_digest(digest_db, tmp_path):
    """batch_runs left in running past the stale window must alert."""
    cutoff = "2026-11-07 08:00:00"
    _insert_run(
        digest_db,
        started_at="2026-11-07 20:00:00",
        finished_at=None,
        status="running",
    )

    cfg = Config(kb_agent_dir=tmp_path)
    fixed_now = datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)

    with (
        patch.object(send_morning_digest, "load_config", return_value=cfg),
        patch.object(send_morning_digest, "init_db", return_value=digest_db),
        patch.object(send_morning_digest, "_since_morning", return_value=cutoff),
        patch.object(send_morning_digest, "_utc_now", return_value=fixed_now),
        patch.object(send_morning_digest, "notify_batch") as notify,
    ):
        send_morning_digest.main()

    notify.assert_called_once()
    causes = notify.call_args[1]["causes"]
    assert any("still running" in c for c in causes)


def test_all_ok_runs_stay_silent(digest_db, tmp_path):
    """Only successful runs since cutoff must not send Telegram."""
    cutoff = "2026-11-07 08:00:00"
    _insert_run(
        digest_db,
        started_at="2026-11-07 22:00:00",
        finished_at="2026-11-07 23:00:00",
        status="ok",
        total=4,
        failed=0,
        dispatched=4,
    )
    _insert_run(
        digest_db,
        started_at="2026-11-08 01:00:00",
        finished_at="2026-11-08 02:00:00",
        status="ok",
        total=1,
        failed=0,
        dispatched=1,
    )

    cfg = Config(kb_agent_dir=tmp_path)
    fixed_now = datetime(2026, 11, 8, 8, 30, tzinfo=timezone.utc)

    with (
        patch.object(send_morning_digest, "load_config", return_value=cfg),
        patch.object(send_morning_digest, "init_db", return_value=digest_db),
        patch.object(send_morning_digest, "_since_morning", return_value=cutoff),
        patch.object(send_morning_digest, "_utc_now", return_value=fixed_now),
        patch.object(send_morning_digest, "notify_batch") as notify,
    ):
        send_morning_digest.main()

    notify.assert_not_called()


def test_recent_running_not_stale(digest_db, tmp_path):
    """A run still in progress but under the stale threshold stays silent."""
    cutoff = "2026-11-07 08:00:00"
    _insert_run(
        digest_db,
        started_at="2026-11-08 07:30:00",
        finished_at=None,
        status="running",
    )

    cfg = Config(kb_agent_dir=tmp_path)
    fixed_now = datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)

    with (
        patch.object(send_morning_digest, "load_config", return_value=cfg),
        patch.object(send_morning_digest, "init_db", return_value=digest_db),
        patch.object(send_morning_digest, "_since_morning", return_value=cutoff),
        patch.object(send_morning_digest, "_utc_now", return_value=fixed_now),
        patch.object(send_morning_digest, "notify_batch") as notify,
    ):
        send_morning_digest.main()

    notify.assert_not_called()


def test_collect_abnormal_runs_unit():
    """Helper aggregates failed, crashed, and stale-running rows only."""
    now = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    stale_hours = 6
    rows = [
        {"id": 1, "status": "ok", "started_at": "2026-11-08 08:00:00", "summary": None},
        {"id": 2, "status": "failed", "started_at": "2026-11-08 09:00:00", "summary": "boom"},
        {
            "id": 3,
            "status": "running",
            "started_at": (now - timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"),
            "summary": None,
        },
        {
            "id": 4,
            "status": "running",
            "started_at": (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "summary": None,
        },
    ]
    abnormal = send_morning_digest._collect_abnormal_runs(rows, now, stale_hours)
    ids = [r["id"] for r in abnormal]
    assert ids == [2, 3]
