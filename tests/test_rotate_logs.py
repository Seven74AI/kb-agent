"""Tests for log rotation — archiving step_logs + llm_archives to MinIO."""

import gzip
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add kb-agent to path
_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from agent.db import LOGS_SCHEMA, init_db
from scripts.rotate_logs import (
    BATCH_SIZE,
    RETENTION_DAYS,
    delete_batched,
    dump_rows,
    main,
    rebuild_fts,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def logs_db_path(tmp_path):
    """Create a test logs.db with schema."""
    db_path = tmp_path / "logs.db"
    conn = init_db(db_path, LOGS_SCHEMA)
    conn.close()
    return db_path


@pytest.fixture
def logs_conn(logs_db_path):
    """Open a connection to the test logs.db."""
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _insert_step_log(conn, task_id, step_name, content, logged_at, stream="stdout"):
    conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content, logged_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, step_name, stream, content, logged_at),
    )


def _insert_llm_archive(conn, task_id, role, prompt, response, model, created_at,
                        tokens_in=100, tokens_out=200):
    conn.execute(
        "INSERT INTO llm_archives (task_id, role, prompt, response, model, "
        "tokens_in, tokens_out, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, role, prompt, response, model, tokens_in, tokens_out, created_at),
    )


def _row_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _old_date(days_ago):
    """Return an ISO datetime string `days_ago` days in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ── Unit tests: dump_rows ─────────────────────────────────────────────────────


def test_dump_rows_returns_empty_for_no_rows(logs_conn):
    cutoff = _old_date(30)
    records, ids = dump_rows(
        logs_conn, "step_logs",
        columns=["task_id", "step_name", "stream", "content", "logged_at"],
        date_col="logged_at", cutoff=cutoff,
    )
    assert records == []
    assert ids == []


def test_dump_rows_only_returns_old_rows(logs_conn):
    old = _old_date(60)
    recent = _old_date(5)
    cutoff = _old_date(30)

    _insert_step_log(logs_conn, "t1", "download", "old log", old)
    _insert_step_log(logs_conn, "t2", "transcribe", "recent log", recent)
    logs_conn.commit()

    records, ids = dump_rows(
        logs_conn, "step_logs",
        columns=["task_id", "step_name", "stream", "content", "logged_at"],
        date_col="logged_at", cutoff=cutoff,
    )

    assert len(records) == 1
    assert records[0]["_table"] == "step_logs"
    assert records[0]["task_id"] == "t1"
    assert records[0]["content"] == "old log"
    assert len(ids) == 1


def test_dump_rows_llm_archives(logs_conn):
    old = _old_date(60)
    recent = _old_date(5)
    cutoff = _old_date(30)

    _insert_llm_archive(logs_conn, "t1", "synthesis", "prompt1", "resp1", "deepseek", old)
    _insert_llm_archive(logs_conn, "t2", "review", "prompt2", "resp2", "deepseek", recent)
    logs_conn.commit()

    records, ids = dump_rows(
        logs_conn, "llm_archives",
        columns=["task_id", "role", "prompt", "response", "model",
                 "tokens_in", "tokens_out", "created_at"],
        date_col="created_at", cutoff=cutoff,
    )

    assert len(records) == 1
    assert records[0]["_table"] == "llm_archives"
    assert records[0]["task_id"] == "t1"
    assert records[0]["prompt"] == "prompt1"
    assert records[0]["response"] == "resp1"
    assert len(ids) == 1


def test_dump_rows_includes_all_columns(logs_conn):
    old = _old_date(60)
    cutoff = _old_date(30)

    _insert_step_log(logs_conn, "t-abc", "synthesize", "log content", old,
                     stream="stderr")
    logs_conn.commit()

    records, _ = dump_rows(
        logs_conn, "step_logs",
        columns=["task_id", "step_name", "stream", "content", "logged_at"],
        date_col="logged_at", cutoff=cutoff,
    )

    r = records[0]
    assert r["task_id"] == "t-abc"
    assert r["step_name"] == "synthesize"
    assert r["stream"] == "stderr"
    assert r["content"] == "log content"
    assert r["logged_at"] == old


# ── Unit tests: delete_batched ────────────────────────────────────────────────


def test_delete_batched_empty(logs_conn):
    count = delete_batched(logs_conn, "step_logs", [])
    assert count == 0


def test_delete_batched_single(logs_conn):
    _insert_step_log(logs_conn, "t1", "step", "content", _old_date(60))
    logs_conn.commit()
    row_id = logs_conn.execute("SELECT id FROM step_logs").fetchone()["id"]

    count = delete_batched(logs_conn, "step_logs", [row_id])
    logs_conn.commit()

    assert count == 1
    assert _row_count(logs_conn, "step_logs") == 0


def test_delete_batched_multiple_batches(logs_conn):
    ids = []
    for i in range(BATCH_SIZE + 10):
        _insert_step_log(logs_conn, f"t{i}", "step", f"content {i}", _old_date(60))
        logs_conn.commit()
        row_id = logs_conn.execute(
            "SELECT id FROM step_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        ids.append(row_id)

    count = delete_batched(logs_conn, "step_logs", ids)
    logs_conn.commit()

    assert count == BATCH_SIZE + 10
    assert _row_count(logs_conn, "step_logs") == 0


def test_delete_batched_only_deletes_specified(logs_conn):
    _insert_step_log(logs_conn, "keep", "step", "keep me", _old_date(60))
    _insert_step_log(logs_conn, "del", "step", "delete me", _old_date(60))
    logs_conn.commit()

    del_id = logs_conn.execute(
        "SELECT id FROM step_logs WHERE task_id='del'"
    ).fetchone()["id"]

    count = delete_batched(logs_conn, "step_logs", [del_id])
    logs_conn.commit()

    assert count == 1
    assert _row_count(logs_conn, "step_logs") == 1


# ── Unit tests: rebuild_fts ───────────────────────────────────────────────────


def test_rebuild_fts_does_not_raise(logs_conn):
    """rebuild_fts should not raise even with empty FTS table."""
    rebuild_fts(logs_conn)  # Should not raise
    logs_conn.commit()


def test_rebuild_fts_after_deletion(logs_conn):
    """After deleting rows, FTS rebuild should succeed."""
    _insert_step_log(logs_conn, "t1", "step", "test content for FTS", _old_date(60))
    logs_conn.commit()

    # Delete the row
    row_id = logs_conn.execute("SELECT id FROM step_logs").fetchone()["id"]
    delete_batched(logs_conn, "step_logs", [row_id])
    logs_conn.commit()

    # Rebuild should succeed
    rebuild_fts(logs_conn)
    logs_conn.commit()


# ── Integration tests: main() ─────────────────────────────────────────────────


@patch("scripts.rotate_logs.LOGS_DB")
@patch("scripts.rotate_logs.subprocess.run")
def test_main_no_db_returns_0(mock_run, mock_logs_db, tmp_path):
    """When logs.db doesn't exist, main() returns 0 without error."""
    mock_logs_db.exists.return_value = False
    mock_logs_db.__bool__.return_value = False
    mock_logs_db.__truediv__ = lambda self, other: tmp_path / "nonexistent" / str(other)

    # Override LOGS_DB to a nonexistent path
    with patch("scripts.rotate_logs.LOGS_DB", tmp_path / "nonexistent" / "logs.db"):
        result = main()
    assert result == 0
    mock_run.assert_not_called()


@patch("scripts.rotate_logs.subprocess.run")
def test_main_nothing_to_rotate(mock_run, logs_db_path):
    """When all rows are recent, nothing is uploaded or deleted."""
    # Insert a recent row
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    _insert_step_log(conn, "t1", "step", "recent", _old_date(5))
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        result = main()
    assert result == 0
    mock_run.assert_not_called()  # No upload because nothing to archive


@patch("scripts.rotate_logs.subprocess.run")
def test_main_archives_and_deletes_old_rows(mock_run, logs_db_path, tmp_path):
    """Old rows from both tables are archived, uploaded, and deleted."""
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    old = _old_date(60)
    recent = _old_date(5)

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row

    # Old rows (should be archived)
    _insert_step_log(conn, "old-task", "download", "old log 1", old)
    _insert_step_log(conn, "old-task", "transcribe", "old log 2", old)
    _insert_llm_archive(conn, "old-task", "synthesis", "prompt", "response",
                        "deepseek", old)

    # Recent rows (should stay)
    _insert_step_log(conn, "recent-task", "download", "recent log", recent)
    _insert_llm_archive(conn, "recent-task", "review", "prompt2", "response2",
                        "deepseek", recent)
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        with patch("pathlib.Path.unlink", return_value=None):  # Don't delete temp files
            result = main()

    assert result == 0

    # Verify upload was called
    assert mock_run.call_count == 1
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "mc"
    assert call_args[1] == "cp"
    assert "minio/kb-agent-archive/logs/" in call_args[3]  # 4th arg: minio path

    # Verify uploaded file contains both table types
    tmp_gz = Path(call_args[2])
    with gzip.open(tmp_gz, "rb") as f:
        content = f.read().decode("utf-8")
    lines = [json.loads(line) for line in content.strip().split("\n") if line]
    tables = {line["_table"] for line in lines}
    assert "step_logs" in tables
    assert "llm_archives" in tables
    assert len(lines) == 3  # 2 old step_logs + 1 old llm_archive

    # Verify old rows deleted, recent rows preserved
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "step_logs") == 1  # Only the recent one
    assert _row_count(conn, "llm_archives") == 1  # Only the recent one
    remaining = conn.execute(
        "SELECT task_id FROM step_logs"
    ).fetchone()
    assert remaining["task_id"] == "recent-task"
    conn.close()


@patch("scripts.rotate_logs.subprocess.run")
def test_main_upload_failure_preserves_rows(mock_run, logs_db_path):
    """When MinIO upload fails, rows are NOT deleted (retry next run)."""
    mock_run.return_value = MagicMock(
        returncode=1,
        stderr="mc: <ERROR> Unable to connect to MinIO",
    )

    old = _old_date(60)
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    _insert_step_log(conn, "old-task", "download", "should survive", old)
    _insert_llm_archive(conn, "old-task", "synthesis", "prompt", "response",
                        "deepseek", old)
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        result = main()

    assert result == 1  # Upload failure

    # Rows must still be there
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "step_logs") == 1
    assert _row_count(conn, "llm_archives") == 1
    conn.close()


@patch("scripts.rotate_logs.subprocess.run")
def test_main_handles_both_tables_independently(mock_run, logs_db_path):
    """Old llm_archives with no old step_logs still get archived."""
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    old = _old_date(60)
    recent = _old_date(5)

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row

    # Only old llm_archives, recent step_logs
    _insert_llm_archive(conn, "t1", "synthesis", "p", "r", "deepseek", old)
    _insert_step_log(conn, "t2", "download", "recent log", recent)
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        result = main()

    assert result == 0

    # llm_archives old row deleted, step_logs recent row preserved
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "llm_archives") == 0
    assert _row_count(conn, "step_logs") == 1
    conn.close()


@patch("scripts.rotate_logs.subprocess.run")
def test_main_handles_exact_boundary(mock_run, logs_db_path):
    """A row exactly at the 30-day boundary should NOT be archived."""
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    # Row exactly 29 days old — should stay
    boundary = (datetime.now(timezone.utc) - timedelta(days=29)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    _insert_step_log(conn, "t1", "step", "boundary", boundary)
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        result = main()

    assert result == 0
    mock_run.assert_not_called()  # Nothing to archive

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "step_logs") == 1  # Boundary row preserved
    conn.close()


@patch("scripts.rotate_logs.subprocess.run")
def test_main_handles_gzip_error_gracefully(mock_run, logs_db_path, tmp_path):
    """If gzip fails, return 1 and don't touch the DB."""
    old = _old_date(60)

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    _insert_step_log(conn, "t1", "step", "content", old)
    conn.commit()
    conn.close()

    # Make /tmp unwritable for the gzip step
    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        with patch("gzip.open", side_effect=OSError("Permission denied")):
            result = main()

    assert result == 1
    mock_run.assert_not_called()

    # Rows preserved
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "step_logs") == 1
    conn.close()


# ── Edge case: large batch handling ───────────────────────────────────────────


@patch("scripts.rotate_logs.subprocess.run")
def test_main_large_batch_splits_deletes(mock_run, logs_db_path):
    """When there are more rows than BATCH_SIZE, deletes are batched correctly."""
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    old = _old_date(60)
    num_rows = BATCH_SIZE + 150  # Across two batch boundaries

    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    for i in range(num_rows):
        _insert_step_log(conn, f"task-{i}", "step", f"content {i}", old)
    conn.commit()
    conn.close()

    with patch("scripts.rotate_logs.LOGS_DB", logs_db_path):
        result = main()

    assert result == 0
    assert mock_run.call_count == 1

    # All old rows deleted
    conn = sqlite3.connect(str(logs_db_path))
    conn.row_factory = sqlite3.Row
    assert _row_count(conn, "step_logs") == 0
    conn.close()
