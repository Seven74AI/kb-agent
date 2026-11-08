"""Resume paused tasks and retry failed tasks."""

import sqlite3

from agent.db import (
    has_active_synthesize_task,
    resume_tasks_by_reason,
    task_deferred_retry_preserves_embed_step,
    task_in_deferred_media_cycle,
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
    For deferred media pauses, ensures ``substack_embedded_media`` is pending so
    the consumer cannot skip ahead to synthesis while media is still in flight.

    Returns:
        True if the task was retried, False if not found or wrong status.
    """
    row = agent_conn.execute(
        "SELECT status, error, slug FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row[0] not in ("failed", "paused"):
        return False
    if row[2] and has_active_synthesize_task(agent_conn, row[2]):
        return False

    with agent_conn:
        slug = row[2]
        in_cycle = task_in_deferred_media_cycle(
            agent_conn, task_id, slug=slug, error=row[1],
        )
        if row[0] == "paused" and in_cycle:
            embed = agent_conn.execute(
                """
                SELECT step_order FROM task_steps
                WHERE task_id = ? AND name = 'substack_embedded_media'
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if embed is not None:
                preserve_embed = task_deferred_retry_preserves_embed_step(
                    agent_conn, task_id, slug,
                )
                if preserve_embed:
                    step_filter = "step_order > ?"
                else:
                    step_filter = "step_order >= ?"
                agent_conn.execute(
                    f"""
                    UPDATE task_steps
                    SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
                    WHERE task_id = ? AND {step_filter}
                    AND status IN ('failed', 'paused', 'running')
                    """,
                    (task_id, embed[0]),
                )
        else:
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
