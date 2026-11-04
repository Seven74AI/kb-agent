"""Enqueue URL for processing (submit_url)."""

import sqlite3

from agent.config import Config
from agent.core.pipelines import steps_for_content_type
from agent.core.slugify import slugify
from agent.db import create_task
from agent.detection import dedup_check, detect_content_subtype, detect_platform, preflight_health


def submit_url(url: str, config: Config, agent_conn: sqlite3.Connection) -> dict:
    """Submit a URL to the task queue.

    Pure domain logic — no HTTP layer. Caller maps http_status to Flask responses.

    Returns:
        Result dict with status, http_status, and optional task_id, platform,
        content_type, slug, reason, or error.
    """
    url = (url or "").strip()
    if not url:
        return {"status": "error", "http_status": 400, "error": "url required"}

    platform = detect_platform(url)

    if dedup_check(url, config.vault_path):
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

    if not ok:
        task_id = create_task(agent_conn, url, platform, slug, steps, status="paused")
        agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (reason, task_id))
        agent_conn.commit()
        return {
            "status": "paused",
            "http_status": 200,
            "reason": reason,
            "platform": platform,
            "task_id": task_id,
        }

    task_id = create_task(agent_conn, url, platform, slug, steps)
    return {
        "status": "pending",
        "http_status": 201,
        "task_id": task_id,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
    }
