"""Enqueue URL for processing (submit_url)."""

import sqlite3

from agent.config import Config
from agent.core.pipelines import steps_for_content_type
from agent.core.slugify import slugify
from agent.db import create_task
from agent.detection import dedup_check, detect_content_subtype, detect_platform, preflight_health


def plan_submission(url: str, config: Config, *, ignore_dedup: bool = False) -> dict:
    """Classify a URL and decide the submission outcome — no DB access.

    Args:
        url: The URL to classify.
        config: Typed configuration (used for vault dedup path).
        ignore_dedup: When True, skip the vault dedup check so an already-known
            URL can be re-enqueued (used by the benchmark ``run`` CLI). Webhook
            callers keep the default (False) and dedup as before.

    Returns either a terminal result (``error`` or ``skipped``) or a plan
    carrying everything needed to enqueue (``paused`` or ``pending``).
    Callers that need to touch the DB only for the enqueue step can run this
    first and skip opening a connection entirely on dedup hits.
    """
    url = (url or "").strip()
    if not url:
        return {"status": "error", "http_status": 400, "error": "url required"}

    platform = detect_platform(url)

    if not ignore_dedup and dedup_check(url, config.vault_path):
        return {
            "status": "skipped",
            "http_status": 200,
            "reason": "Already in vault",
            "platform": platform,
        }

    ok, reason = preflight_health(platform)
    slug = slugify(url)
    subtype = detect_content_subtype(url, platform)
    steps = steps_for_content_type(subtype, platform)

    return {
        "status": "paused" if not ok else "pending",
        "http_status": 200 if not ok else 201,
        "url": url,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
        "steps": steps,
        "reason": reason if not ok else None,
    }


def submit_plan(agent_conn: sqlite3.Connection, plan: dict) -> dict:
    """Enqueue a task from a plan produced by :func:`plan_submission`."""
    if plan["status"] == "paused":
        task_id = create_task(
            agent_conn, plan["url"], plan["platform"], plan["slug"], plan["steps"],
            status="paused",
        )
        agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (plan["reason"], task_id))
        agent_conn.commit()
        return {
            "status": "paused",
            "http_status": 200,
            "reason": plan["reason"],
            "platform": plan["platform"],
            "task_id": task_id,
        }

    task_id = create_task(agent_conn, plan["url"], plan["platform"], plan["slug"], plan["steps"])
    return {
        "status": "pending",
        "http_status": 201,
        "task_id": task_id,
        "platform": plan["platform"],
        "content_type": plan["content_type"],
        "slug": plan["slug"],
    }


def submit_url(
    url: str,
    config: Config,
    agent_conn: sqlite3.Connection,
    *,
    ignore_dedup: bool = False,
) -> dict:
    """Submit a URL to the task queue.

    Pure domain logic — no HTTP layer. Caller maps http_status to Flask responses.
    Convenience wrapper for callers that already hold a connection: plans the
    submission and enqueues it in one call. ``agent_conn`` is only touched when a
    task must be created (dedup skips and empty URLs never use it).

    Returns:
        Result dict with status, http_status, and optional task_id, platform,
        content_type, slug, reason, or error.
    """
    plan = plan_submission(url, config, ignore_dedup=ignore_dedup)
    if plan["status"] in ("error", "skipped"):
        return plan
    return submit_plan(agent_conn, plan)
