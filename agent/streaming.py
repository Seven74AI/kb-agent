"""Streaming — real-time subprocess output to logs.db.

Reads subprocess stdout line-by-line and writes each line to the logs
database. Used for monitoring long-running steps (diarize, transcribe, yt-dlp).
"""

import asyncio
import sqlite3
from pathlib import Path


async def stream_to_logs(
    proc: asyncio.subprocess.Process,
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
    stream: str = "stdout",
):
    """Stream subprocess stdout/stderr line-by-line into logs.db.

    Reads lines as they appear, writing each to the step_logs table.
    This enables real-time progress visibility on the dashboard.

    Args:
        proc: The asyncio subprocess to monitor.
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Name of the current pipeline step.
        stream: 'stdout' or 'stderr'.
    """
    stream_obj = proc.stdout if stream == "stdout" else proc.stderr
    if stream_obj is None:
        return

    while True:
        line = await stream_obj.readline()
        if not line:
            break

        text = line.decode("utf-8", errors="replace").rstrip("\n")
        if not text.strip():
            continue

        with logs_conn:
            logs_conn.execute(
                "INSERT INTO step_logs (task_id, step_name, stream, content) VALUES (?, ?, ?, ?)",
                (task_id, step_name, stream, text),
            )


async def stream_both(
    proc: asyncio.subprocess.Process,
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
):
    """Stream both stdout and stderr simultaneously.

    Args:
        proc: The asyncio subprocess.
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Step name for log entries.
    """
    tasks = []
    if proc.stdout:
        tasks.append(stream_to_logs(proc, logs_conn, task_id, step_name, "stdout"))
    if proc.stderr:
        tasks.append(stream_to_logs(proc, logs_conn, task_id, step_name, "stderr"))

    if tasks:
        await asyncio.gather(*tasks)


def read_logs(logs_path: Path, task_id: str, step_name: str | None = None,
              limit: int = 100) -> list[dict]:
    """Read recent log entries for a task step.

    Args:
        logs_path: Path to logs.db.
        task_id: Task UUID to filter by.
        step_name: Optional step name filter.
        limit: Max entries to return.

    Returns:
        List of log entry dicts with step_name, stream, content, logged_at.
    """
    conn = sqlite3.connect(str(logs_path))
    conn.row_factory = sqlite3.Row

    if step_name:
        rows = conn.execute(
            "SELECT step_name, stream, content, logged_at FROM step_logs "
            "WHERE task_id = ? AND step_name = ? "
            "ORDER BY logged_at DESC LIMIT ?",
            (task_id, step_name, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT step_name, stream, content, logged_at FROM step_logs "
            "WHERE task_id = ? "
            "ORDER BY logged_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]
