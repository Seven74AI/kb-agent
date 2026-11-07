"""Task status queries and dashboard timing helpers."""

import sqlite3
from datetime import UTC, datetime

from agent.db import get_all_tasks, get_last_log_line, get_task_steps
from agent.notify import sanitize_visible_message


def format_elapsed(created_at: str, now: datetime) -> str:
    """Format elapsed time from ISO timestamp to human-readable."""
    try:
        if "T" in created_at:
            created_dt = datetime.fromisoformat(created_at)
        else:
            created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        delta = now - created_dt

        if delta.total_seconds() < 0:
            return "just now"

        total_secs = int(delta.total_seconds())
        if total_secs < 60:
            return f"{total_secs}s"
        if total_secs < 3600:
            return f"{total_secs // 60}m"
        if total_secs < 86400:
            return f"{total_secs // 3600}h {int((total_secs % 3600) / 60)}m"
        return f"{total_secs // 86400}d {int((total_secs % 86400) / 3600)}h"
    except (ValueError, TypeError):
        return "?"


def compute_staleness(
    task: dict,
    logs_conn: sqlite3.Connection | None,
    now: datetime,
) -> tuple[str | None, bool]:
    """Compute staleness from the most recent log timestamp.

    Returns (staleness_str, is_stale) where is_stale=True if >2h since last update.
    """
    if logs_conn is None:
        return None, False

    row = logs_conn.execute(
        "SELECT logged_at FROM step_logs WHERE task_id = ? "
        "ORDER BY logged_at DESC LIMIT 1",
        (task["id"],),
    ).fetchone()

    if not row:
        return None, False

    try:
        logged_at = row[0]
        if "T" in logged_at:
            log_dt = datetime.fromisoformat(logged_at)
        else:
            log_dt = datetime.strptime(logged_at, "%Y-%m-%d %H:%M:%S")
        delta = now - log_dt
        total_secs = int(delta.total_seconds())

        if total_secs < 0:
            return "just now", False
        if total_secs < 60:
            return f"{total_secs}s ago", False
        if total_secs < 3600:
            return f"{total_secs // 60}m ago", False
        if total_secs < 7200:
            return f"{total_secs // 3600}h ago", False
        if total_secs < 86400:
            return f"{total_secs // 3600}h ago", True
        return f"{total_secs // 86400}d ago", True
    except (ValueError, TypeError):
        return None, False


def _task_counts(tasks: list[dict]) -> dict[str, int]:
    counts = {"pending": 0, "in_progress": 0, "done": 0, "paused": 0, "failed": 0, "total": 0}
    counts["total"] = len(tasks)
    for t in tasks:
        status = t["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def _enrich_task(
    task_row: dict,
    agent_conn: sqlite3.Connection,
    logs_conn: sqlite3.Connection | None,
    now: datetime,
) -> dict:
    steps = get_task_steps(agent_conn, task_row["id"])
    raw_last = get_last_log_line(logs_conn, task_row["id"]) if logs_conn else None
    last_log = sanitize_visible_message(raw_last) if raw_last else None
    elapsed = format_elapsed(task_row["created_at"], now)
    staleness_str, is_stale = compute_staleness(task_row, logs_conn, now)
    return {
        "id": task_row["id"],
        "source_url": task_row["source_url"],
        "content_type": task_row["content_type"],
        "status": task_row["status"],
        "error": task_row["error"],
        "slug": task_row.get("slug"),
        "retries": task_row.get("retries", 0),
        "created_at": task_row["created_at"],
        "updated_at": task_row.get("updated_at"),
        "steps": steps,
        "elapsed": elapsed,
        "staleness": staleness_str,
        "is_stale": is_stale,
        "last_log": last_log,
    }


def list_tasks(
    agent_conn: sqlite3.Connection,
    *,
    logs_conn: sqlite3.Connection | None = None,
    limit: int = 100,
    now: datetime | None = None,
) -> dict:
    """List tasks with step detail and aggregate status counts."""
    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    raw = get_all_tasks(agent_conn, limit=limit)
    tasks = [_enrich_task(t, agent_conn, logs_conn, now) for t in raw]
    return {"tasks": tasks, "counts": _task_counts(raw)}


def get_task(
    agent_conn: sqlite3.Connection,
    task_id: str,
    *,
    logs_conn: sqlite3.Connection | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Get a single task with steps and log staleness, or None if missing."""
    row = agent_conn.execute(
        "SELECT id, source_url, content_type, status, error, slug, "
        "retries, created_at, updated_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    task_row = {k: row[k] for k in row.keys()}
    return _enrich_task(task_row, agent_conn, logs_conn, now)
