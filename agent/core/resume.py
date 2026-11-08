"""Resume paused tasks and retry failed tasks."""

import sqlite3

from agent.db import (
    _is_superseded_parent_error,
    has_active_synthesize_task,
    reset_task_steps_for_retry,
    resume_tasks_by_reason,
)


class ResumeReasonError(ValueError):
    """Resume reason is missing or would match every paused task."""


def _is_catch_all_like_pattern(pattern: str) -> bool:
    """True when a SQL LIKE pattern has no literal characters (matches everything)."""
    return not pattern.replace("%", "").replace("_", "")


def normalize_reason_pattern(reason_param: str) -> str:
    """Convert a resume reason query param to a concrete SQL LIKE pattern."""
    reason_param = (reason_param or "").strip()
    if not reason_param or _is_catch_all_like_pattern(reason_param):
        raise ResumeReasonError(
            "reason required (refusing to resume all paused tasks)",
        )
    if "%" not in reason_param and "_" not in reason_param:
        return f"%{reason_param}%"
    return reason_param


def resume_tasks(agent_conn: sqlite3.Connection, reason: str) -> int:
    """Resume all paused tasks whose error matches reason (SQL LIKE)."""
    pattern = normalize_reason_pattern(reason)
    return resume_tasks_by_reason(agent_conn, pattern)


def retry_task(agent_conn: sqlite3.Connection, task_id: str) -> bool:
    """Re-queue a failed or paused task for the consumer.

    Resets failed/paused/running steps to pending; leaves completed steps intact.
    In a deferred-media cycle (``task_in_deferred_media_cycle``), rewinds only
    steps after ``substack_embedded_media`` when that step must be preserved for
    an open deferral; otherwise the embed step is reset to pending like other
    steps.

    Returns:
        True if the task was retried, False if not found or wrong status.
    """
    row = agent_conn.execute(
        "SELECT status, error, slug FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row[0] not in ("failed", "paused"):
        return False
    # Terminal superseded rows must not be resurrected (mirrors complete_task CAS).
    if _is_superseded_parent_error(row[1]):
        return False
    if row[2] and has_active_synthesize_task(agent_conn, row[2]):
        return False

    with agent_conn:
        reset_task_steps_for_retry(
            agent_conn,
            task_id,
            slug=row[2],
            error=row[1],
        )
        agent_conn.execute(
            """
            UPDATE tasks
            SET status = 'pending', error = NULL, retries = 0, updated_at = datetime('now')
            WHERE id = ?
            """,
            (task_id,),
        )
    return True
