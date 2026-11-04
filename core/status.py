"""Task queue reads and dashboard-oriented presentation helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from core.types import TaskListResult, TaskRecord, TaskStepRecord


def _row_to_task(row: sqlite3.Row, steps: list[TaskStepRecord]) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        source_url=row["source_url"],
        content_type=row["content_type"],
        status=row["status"],
        slug=row["slug"],
        error=row["error"],
        retries=row["retries"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        steps=steps,
    )


def _fetch_steps(conn: sqlite3.Connection, task_id: str) -> list[TaskStepRecord]:
    rows = conn.execute(
        "SELECT id, step_order, name, status, output, started_at, finished_at "
        "FROM task_steps WHERE task_id = ? ORDER BY step_order",
        (task_id,),
    ).fetchall()
    return [
        TaskStepRecord(
            id=r["id"],
            step_order=r["step_order"],
            name=r["name"],
            status=r["status"],
            output=r["output"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )
        for r in rows
    ]


def list_tasks(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    logs_conn: sqlite3.Connection | None = None,
    now: datetime | None = None,
) -> TaskListResult:
    """List tasks newest-first with optional log enrichment for dashboards."""
    rows = conn.execute(
        "SELECT id, source_url, content_type, status, error, slug, "
        "retries, created_at, updated_at "
        "FROM tasks ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

    counts: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "done": 0,
        "paused": 0,
        "failed": 0,
        "total": len(rows),
    }
    tasks: list[TaskRecord] = []
    reference_now = now or datetime.utcnow()

    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
        steps = _fetch_steps(conn, row["id"])
        last_log = get_last_log_line(logs_conn, row["id"]) if logs_conn else None
        elapsed = format_elapsed(row["created_at"], reference_now)
        staleness_str, is_stale = (
            compute_staleness(row["id"], logs_conn, reference_now)
            if logs_conn
            else (None, False)
        )
        tasks.append(
            TaskRecord(
                id=row["id"],
                source_url=row["source_url"],
                content_type=row["content_type"],
                status=status,
                slug=row["slug"],
                error=row["error"],
                retries=row["retries"] or 0,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                steps=steps,
                last_log=last_log,
                elapsed=elapsed,
                staleness=staleness_str,
                is_stale=is_stale,
            )
        )

    return TaskListResult(tasks=tasks, counts=counts)


def get_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    logs_conn: sqlite3.Connection | None = None,
    now: datetime | None = None,
) -> TaskRecord | None:
    """Fetch one task with steps; optional log enrichment."""
    row = conn.execute(
        "SELECT id, source_url, content_type, status, error, slug, "
        "retries, created_at, updated_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    reference_now = now or datetime.utcnow()
    steps = _fetch_steps(conn, task_id)
    last_log = get_last_log_line(logs_conn, task_id) if logs_conn else None
    elapsed = format_elapsed(row["created_at"], reference_now)
    staleness_str, is_stale = (
        compute_staleness(task_id, logs_conn, reference_now)
        if logs_conn
        else (None, False)
    )
    return TaskRecord(
        id=row["id"],
        source_url=row["source_url"],
        content_type=row["content_type"],
        status=row["status"],
        slug=row["slug"],
        error=row["error"],
        retries=row["retries"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        steps=steps,
        last_log=last_log,
        elapsed=elapsed,
        staleness=staleness_str,
        is_stale=is_stale,
    )


def get_last_log_line(logs_conn: sqlite3.Connection | None, task_id: str) -> str | None:
    """Most recent step_logs line for a task."""
    if logs_conn is None:
        return None
    row = logs_conn.execute(
        "SELECT content FROM step_logs WHERE task_id = ? "
        "ORDER BY logged_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row[0] if row else None


def format_elapsed(created_at: str, now: datetime) -> str:
    """Human-readable elapsed time since task creation (dashboard helper)."""
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
    task_id: str,
    logs_conn: sqlite3.Connection | None,
    now: datetime,
) -> tuple[str | None, bool]:
    """Staleness from latest step_logs row (>2h → stale)."""
    if logs_conn is None:
        return None, False

    row = logs_conn.execute(
        "SELECT logged_at FROM step_logs WHERE task_id = ? "
        "ORDER BY logged_at DESC LIMIT 1",
        (task_id,),
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
