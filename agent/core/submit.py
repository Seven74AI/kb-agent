"""Enqueue URL for processing (submit_url)."""

import sqlite3

from agent.config import Config
from agent.pipelines import get_pipeline_names
from agent.core.slugify import slugify
from agent.db import (
    _persist_visible_error,
    create_synthesize_task,
    create_task,
    queue_ingest,
)
from agent.detection import (
    DedupCheckError,
    dedup_check,
    detect_content_subtype,
    detect_content_subtype_with_signals,
    detect_platform,
    normalize_url,
    preflight_health,
)
from agent.redact import redact_outward_url_field, url_for_display


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
    raw = (url or "").strip()
    if not raw:
        return {"status": "error", "http_status": 400, "error": "url required"}
    try:
        url = normalize_url(raw)
    except ValueError as exc:
        return {"status": "error", "http_status": 400, "error": str(exc)}

    platform = detect_platform(url)

    if not ignore_dedup:
        try:
            in_vault = dedup_check(url, config.vault_path)
        except DedupCheckError as exc:
            return {
                "status": "error",
                "http_status": 500,
                "error": _persist_visible_error(f"dedup check failed: {exc}"),
                "platform": platform,
            }
        if in_vault:
            return {
                "status": "skipped",
                "http_status": 200,
                "reason": "Already in vault",
                "platform": platform,
            }

    detection = detect_content_subtype_with_signals(url, platform)
    subtype = detection.subtype
    if subtype == "unknown":
        return {
            "status": "error",
            "http_status": 400,
            "error": "unknown content_type",
            "platform": platform,
            "url": url_for_display(url),
        }

    steps = get_pipeline_names(subtype, platform)
    if steps is None:
        return {
            "status": "error",
            "http_status": 400,
            "error": f"unknown content_type: {subtype}",
            "platform": platform,
            "url": url_for_display(url),
        }

    ok, reason = preflight_health(platform)
    if platform == "threads" and detection.threads_subtype_fetch_failed:
        ok = False
        reason = reason or "Threads subtype fetch failed — retry when cookies or network recover"
    slug = slugify(url, content_type=subtype)
    pause_reason = _persist_visible_error(reason) if (not ok and reason) else None

    return {
        "status": "paused" if not ok else "pending",
        "http_status": 200 if not ok else 201,
        "url": url,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
        "steps": steps,
        "reason": pause_reason,
    }


def submit_plan(agent_conn: sqlite3.Connection, plan: dict, tags=None) -> dict:
    """Enqueue a task from a plan produced by :func:`plan_submission`."""
    if plan["status"] == "paused":
        task_id = create_task(
            agent_conn, plan["url"], plan["content_type"], plan["slug"], plan["steps"],
            status="paused", tags=tags,
        )
        pause_reason = _persist_visible_error(plan["reason"] or "")
        agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (pause_reason, task_id))
        agent_conn.commit()
        return {
            "status": "paused",
            "http_status": 200,
            "reason": pause_reason,
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
        return redact_outward_url_field(plan)
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
    raw = (url or "").strip()
    if not raw:
        return {"status": "error", "http_status": 400, "error": "url required"}
    try:
        url = normalize_url(raw)
    except ValueError as exc:
        return {"status": "error", "http_status": 400, "error": str(exc)}

    platform = detect_platform(url)
    subtype = content_type or detect_content_subtype(url, platform)
    slug = slugify(url, content_type=subtype)

    ingest_id, created = queue_ingest(
        agent_conn, url, subtype, slug, tags=tags,
        parent_slug=parent_slug, media_kind=media_kind,
    )

    row = agent_conn.execute(
        "SELECT status FROM ingest_queue WHERE id = ?", (ingest_id,),
    ).fetchone()
    ingest_status = row[0] if row else "queued"
    http_status = 202 if ingest_status == "queued" else 200

    return {
        "status": ingest_status,
        "http_status": http_status,
        "ingest_id": ingest_id,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
        "created_new": created,
        "duplicate": not created,
    }


def dispatch_synthesize(url: str, config: Config, agent_conn: sqlite3.Connection, *,
                        content_type: str, slug: str, tags=None) -> dict:
    """Dispatch a synthesize-only task with an explicit content type + slug.

    Used by batch collectors that have already downloaded, transcribed, and
    archived the source (video + transcript in MinIO). The task starts at the
    synthesize step, using the content type to select the right note template.
    """
    raw = (url or "").strip()
    if not raw:
        return {"status": "error", "http_status": 400, "error": "url required"}
    try:
        url = normalize_url(raw)
    except ValueError as exc:
        return {"status": "error", "http_status": 400, "error": str(exc)}

    platform = detect_platform(url)

    steps = get_pipeline_names(content_type, platform)
    if steps is None:
        return {
            "status": "error",
            "http_status": 400,
            "error": f"unknown content_type: {content_type}",
            "platform": platform,
        }

    # Blocks in-flight work and prior successful synthesis; allows retry after failed.
    task_id, created = create_synthesize_task(
        agent_conn, url, content_type, slug, steps, tags=tags,
    )
    result = {
        "status": "pending",
        "http_status": 201 if created else 200,
        "task_id": task_id,
        "platform": platform,
        "content_type": content_type,
        "slug": slug,
    }
    if not created:
        result["duplicate"] = True
    return result
