"""Resume paused tasks and retry failed tasks."""

import sqlite3

from agent.db import resume_tasks_by_reason


def normalize_reason_pattern(reason_param: str) -> str:
    """Convert a resume reason query param to a SQL LIKE pattern."""
    reason_param = (reason_param or "").strip()
    if not reason_param:
        return "%"
    if reason_param != "%" and "%" not in reason_param:
        return f"%{reason_param}%"
    return reason_param


def resume_tasks(agent_conn: sqlite3.Connection, reason: str) -> int:
    """Resume all paused tasks whose error matches reason (SQL LIKE)."""
    pattern = normalize_reason_pattern(reason)
    return resume_tasks_by_reason(agent_conn, pattern)


def _is_awaiting_media_pause(error: str | None) -> bool:
    return bool(error and "awaiting media" in error.lower())


def retry_task(agent_conn: sqlite3.Connection, task_id: str) -> bool:
    """Re-queue a failed or paused task for the consumer.

    Resets failed/paused/running steps to pending; leaves completed steps intact.
    For deferred media pauses, ensures ``substack_embedded_media`` is pending so
    the consumer cannot skip ahead to synthesis while media is still in flight.

    Returns:
        True if the task was retried, False if not found or wrong status.
    """
    row = agent_conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row[0] not in ("failed", "paused"):
        return False

    with agent_conn:
        if row[0] == "paused" and _is_awaiting_media_pause(row[1]):
            agent_conn.execute(
                """
                UPDATE task_steps
                SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
                WHERE task_id = ? AND name = 'substack_embedded_media' AND status = 'paused'
                """,
                (task_id,),
            )
        agent_conn.execute(
            """
            UPDATE task_steps
            SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
            WHERE task_id = ? AND status IN ('failed', 'paused', 'running')
            """,
            (task_id,),
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
