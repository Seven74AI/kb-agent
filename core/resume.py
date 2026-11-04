"""Resume and retry paused or failed tasks."""

from __future__ import annotations

import sqlite3

from core.types import ResumeResult, RetryResult


def normalize_resume_reason_pattern(reason_param: str) -> str:
    """Convert dashboard/MCP reason input to SQL LIKE pattern.

    Matches Flask ``POST /resume`` behavior:
    - Empty → ``%`` (all paused)
    - Bare keyword ``cookies`` → ``%cookies%``
    - Patterns that already contain ``%`` are passed through.
    """
    reason_param = (reason_param or "%").strip()
    if not reason_param:
        reason_param = "%"
    if reason_param != "%" and "%" not in reason_param:
        reason_param = f"%{reason_param}%"
    return reason_param


def resume_task(conn: sqlite3.Connection, reason_pattern: str) -> ResumeResult:
    """Resume paused tasks whose error field matches a LIKE pattern.

    Resets ``retries`` to 0 and clears ``error``. Same semantics as
    ``agent.db.resume_tasks_by_reason``.
    """
    pattern = normalize_resume_reason_pattern(reason_pattern)
    with conn:
        cursor = conn.execute(
            "UPDATE tasks SET status = 'pending', retries = 0, "
            "error = NULL, updated_at = datetime('now') "
            "WHERE status = 'paused' AND error LIKE ?",
            (pattern,),
        )
        count = cursor.rowcount
    return ResumeResult(resumed=count, reason_pattern=pattern)


def retry_task(conn: sqlite3.Connection, task_id: str) -> RetryResult:
    """Re-queue a single paused or failed task (MCP ``retry_task``).

    Resets task status to ``pending``, clears error, and sets failed/running
    steps back to ``pending`` so the consumer can re-execute from the first
    incomplete step.
    """
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return RetryResult(task_id=task_id, retried=False, previous_status=None)

    previous_status = row[0]
    if previous_status not in ("paused", "failed"):
        return RetryResult(
            task_id=task_id,
            retried=False,
            previous_status=previous_status,
        )

    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'pending', retries = 0, error = NULL, "
            "updated_at = datetime('now') WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_steps SET status = 'pending', output = NULL, "
            "finished_at = NULL WHERE task_id = ? AND status IN ('failed', 'paused', 'running')",
            (task_id,),
        )
    return RetryResult(
        task_id=task_id,
        retried=True,
        previous_status=previous_status,
    )
