"""Enqueue URL for processing (submit_url)."""

import sqlite3

from agent.config import Config
from agent.pipelines import get_pipeline_names
from agent.core.slugify import slugify
from agent.db import create_task, find_active_task, queue_ingest
from agent.detection import (
    DedupCheckError,
    dedup_check,
    detect_content_subtype,
    detect_platform,
    normalize_url,
    preflight_health,
)


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

    if not ignore_dedup:
        try:
            in_vault = dedup_check(url, config.vault_path)
        except DedupCheckError as exc:
            return {
                "status": "error",
                "http_status": 500,
                "error": f"dedup check failed: {exc}",
                "platform": platform,
            }
        if in_vault:
            return {
                "status": "skipped",
                "http_status": 200,
                "reason": "Already in vault",
                "platform": platform,
            }

    ok, reason = preflight_health(platform)
    slug = slugify(url)
    subtype = detect_content_subtype(url, platform)
    steps = get_pipeline_names(subtype, platform)

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


def submit_plan(agent_conn: sqlite3.Connection, plan: dict, tags=None) -> dict:
    """Enqueue a task from a plan produced by :func:`plan_submission`."""
    if plan["status"] == "paused":
        task_id = create_task(
            agent_conn, plan["url"], plan["content_type"], plan["slug"], plan["steps"],
            status="paused", tags=tags,
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

    task_id = create_task(agent_conn, plan["url"], plan["content_type"], plan["slug"], plan["steps"], tags=tags)
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
    tags=None,
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
    return submit_plan(agent_conn, plan, tags=tags)


def submit_ingest(url: str, config: Config, agent_conn: sqlite3.Connection, *,
                  content_type: str | None = None, tags=None,
                  parent_slug: str | None = None, media_kind: str | None = None) -> dict:
    """Queue a URL for deferred batch processing (durable ingest inbox).

    Detects the content type, stores the link durably, and returns. Nothing is
    processed yet — a nightly batch collector dispatches it to ``tasks`` with the
    content-type-appropriate pipeline (including the synthesize-only variant for
    batch-transcribed video).

    ``parent_slug`` + ``media_kind`` link an embedded media item (image/video)
    back to the Substack article that spawned it, for deferred synthesis.
    """
    url = normalize_url((url or "").strip())
    if not url:
        return {"status": "error", "http_status": 400, "error": "url required"}

    platform = detect_platform(url)
    subtype = content_type or detect_content_subtype(url, platform)
    slug = slugify(url)

    ingest_id, created = queue_ingest(
        agent_conn, url, subtype, slug, tags=tags,
        parent_slug=parent_slug, media_kind=media_kind,
    )

    return {
        "status": "queued",
        "http_status": 202,
        "ingest_id": ingest_id,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
        "duplicate": not created,
    }


def dispatch_synthesize(url: str, config: Config, agent_conn: sqlite3.Connection, *,
                        content_type: str, slug: str, tags=None) -> dict:
    """Dispatch a synthesize-only task with an explicit content type + slug.

    Used by batch collectors that have already downloaded, transcribed, and
    archived the source (video + transcript in MinIO). The task starts at the
    synthesize step, using the content type to select the right note template.
    """
    url = (url or "").strip()
    if not url:
        return {"status": "error", "http_status": 400, "error": "url required"}

    platform = detect_platform(url)
    # Blocks in-flight work and prior successful synthesis; allows retry after failed.
    existing_id = find_active_task(agent_conn, slug, content_type)
    if existing_id:
        return {
            "status": "pending",
            "http_status": 200,
            "task_id": existing_id,
            "platform": platform,
            "content_type": content_type,
            "slug": slug,
            "duplicate": True,
        }

    steps = get_pipeline_names(content_type, platform)
    task_id = create_task(agent_conn, url, content_type, slug, steps, tags=tags)

    return {
        "status": "pending",
        "http_status": 201,
        "task_id": task_id,
        "platform": platform,
        "content_type": content_type,
        "slug": slug,
    }
