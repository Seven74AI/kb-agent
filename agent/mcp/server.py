"""FastMCP 4.x server — stateless streamable HTTP tools over agent.core."""

from __future__ import annotations

import functools
import logging
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import AuthorizationError, ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext

from agent.config import Config, load_config
from agent.core.bulk import bulk_import as core_bulk_import
from agent.core.bulk import ingest_playlist as core_ingest_playlist
from agent.core.detect import detect_type as core_detect_type
from agent.core.note import get_note as core_get_note
from agent.core.resume import resume_tasks as core_resume_tasks
from agent.core.resume import retry_task as core_retry_task
from agent.core.search import search_notes as core_search_notes
from agent.core.status import get_task as core_get_task
from agent.core.status import list_tasks as core_list_tasks
from agent.core.submit import submit_url as core_submit_url
from agent.core.submit import submit_ingest as core_submit_ingest
from agent.core.batch_lock import is_locked
from agent.core.watch_source import register_watch_source, submit_routed
from agent.db import _persist_visible_error, list_ingest_by_status
from agent.mcp.db import open_agent_conn, open_logs_conn
from agent.webhook import (
    _TOKEN_NOT_CONFIGURED_MSG,
    extract_request_token,
    resolve_api_token,
)

logger = logging.getLogger(__name__)


class KbApiTokenMiddleware(Middleware):
    """Reject tool calls unless the request presents the shared API token."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ):
        configured = resolve_api_token()
        if not configured:
            raise ToolError(_TOKEN_NOT_CONFIGURED_MSG)
        try:
            http_request = get_http_request()
        except RuntimeError:
            raise AuthorizationError("Unauthorized") from None
        provided = extract_request_token(http_request)
        if not provided or not secrets.compare_digest(provided, configured):
            raise AuthorizationError("Unauthorized")
        return await call_next(context)

_MAX_CLIENT_ERROR_LEN = 300


def agent_relative_path(path: Path, agent_dir: Path) -> str:
    """Return a client-visible path under ``agent_dir`` (never an absolute host path)."""
    resolved = path.resolve()
    root = agent_dir.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _sanitize_exception_message(message: str) -> str:
    """Scrub paths and credential-shaped text before returning tool errors."""
    scrubbed = _persist_visible_error(message)
    if len(scrubbed) > _MAX_CLIENT_ERROR_LEN:
        scrubbed = scrubbed[: _MAX_CLIENT_ERROR_LEN - 3] + "..."
    return scrubbed


def _safe_tool(fn):
    """Wrap a tool so unexpected exceptions return a uniform error envelope."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — uniform error shape for the caller
            detail = _sanitize_exception_message(str(exc))
            logger.error(
                "MCP tool %s failed: %s: %s",
                fn.__name__,
                type(exc).__name__,
                detail,
            )
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {detail}",
            }
    return wrapper


def build_mcp_server(config: Config | None = None) -> FastMCP:
    """Create a FastMCP server exposing the full kb-agent tool surface."""
    cfg = config or load_config()
    mcp = FastMCP(
        name="kb-agent",
        instructions=(
            "KB Agent MCP — submit URLs, inspect the task queue, search the vault, "
            "and bulk-ingest playlists. Stateless: each HTTP request is independent."
        ),
    )

    @mcp.tool
    @_safe_tool
    def detect_type(url: str) -> dict[str, str]:
        """Detect platform and content subtype for a URL (no side effects)."""
        return core_detect_type(url)

    @mcp.tool
    @_safe_tool
    def submit_url(url: str, tags: str = "") -> dict[str, Any]:
        """Enqueue a URL for pipeline processing. `tags` (comma-separated) are added to the note."""
        conn = open_agent_conn(cfg)
        try:
            return core_submit_url(url, cfg, conn, tags=tags)
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def ingest(url: str, tags: str = "") -> dict[str, Any]:
        """Queue a URL for deferred batch processing. `tags` (comma-separated) are added to the note."""
        conn = open_agent_conn(cfg)
        try:
            return core_submit_ingest(url, cfg, conn, tags=tags)
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def list_tasks(limit: int = 100) -> dict[str, Any]:
        """List tasks in the SQLite queue with step detail and status counts."""
        conn = open_agent_conn(cfg)
        logs = open_logs_conn(cfg)
        try:
            return core_list_tasks(conn, logs_conn=logs, limit=limit)
        finally:
            logs.close()
            conn.close()

    @mcp.tool
    @_safe_tool
    def get_task(task_id: str) -> dict[str, Any]:
        """Get one task by id, or an error payload if not found."""
        conn = open_agent_conn(cfg)
        logs = open_logs_conn(cfg)
        try:
            task = core_get_task(conn, task_id, logs_conn=logs)
            if task is None:
                return {"found": False, "task_id": task_id}
            return {"found": True, "task": task}
        finally:
            logs.close()
            conn.close()

    @mcp.tool
    @_safe_tool
    def resume_task(reason: str = "") -> dict[str, Any]:
        """Resume paused tasks whose error message matches reason (SQL LIKE)."""
        if not reason or not reason.strip():
            return {"status": "error", "error": "reason required (refusing to resume all paused tasks)"}
        conn = open_agent_conn(cfg)
        try:
            count = core_resume_tasks(conn, reason)
            return {"resumed": count}
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def retry_task(task_id: str) -> dict[str, Any]:
        """Re-queue a single failed or paused task."""
        conn = open_agent_conn(cfg)
        try:
            ok = core_retry_task(conn, task_id)
            return {"ok": ok, "task_id": task_id}
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def search_notes(query: str, limit: int = 50) -> list[dict[str, str]]:
        """Search markdown notes under the vault Knowledge base."""
        return core_search_notes(query, cfg.vault_path, limit=limit)

    @mcp.tool
    @_safe_tool
    def get_note(slug: str, content_type: str | None = None) -> dict[str, Any]:
        """Load a vault note by slug; optional content_type narrows folder lookup."""
        note = core_get_note(slug, cfg.vault_path, content_type=content_type)
        if note is None:
            return {"found": False, "slug": slug}
        return {"found": True, "note": note}

    @mcp.tool
    @_safe_tool
    def ingest_playlist(playlist_url: str, tags: str = "") -> dict[str, Any]:
        """Expand a YouTube playlist and queue each video (deferred batch). `tags` (comma-separated) added to each note."""
        conn = open_agent_conn(cfg)
        try:
            return core_ingest_playlist(
                playlist_url, cfg, conn,
                submit_fn=lambda u, c, conn_: submit_routed(u, c, conn_, tags=tags),
            )
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def bulk_import(urls: list[str], tags: str = "") -> dict[str, Any]:
        """Submit many URLs in one call (video → deferred batch, else immediate). `tags` (comma-separated) added to each note."""
        conn = open_agent_conn(cfg)
        try:
            return core_bulk_import(
                urls, cfg, conn,
                submit_fn=lambda u, c, conn_: submit_routed(u, c, conn_, tags=tags),
            )
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def watch_source(
        page_url: str,
        item_pattern: str,
        item_url_template: str,
        cadence_seconds: int = 86400,
        baseline: bool = False,
    ) -> dict[str, Any]:
        """Register a listing page for recurring auto-ingest when new items appear."""
        conn = open_agent_conn(cfg)
        try:
            return register_watch_source(
                conn,
                page_url,
                item_pattern=item_pattern,
                item_url_template=item_url_template,
                cadence_seconds=cadence_seconds,
                fetch_html=None,
                baseline=baseline,
            )
        finally:
            conn.close()

    @mcp.tool
    @_safe_tool
    def run_batch() -> dict[str, Any]:
        """Trigger the batch collector now: download + batch-transcribe + dispatch queued links.

        Runs detached in the background (Modal transcription can take a while);
        returns immediately with the queued count and the log path.
        """
        conn = open_agent_conn(cfg)
        try:
            queued = list_ingest_by_status(conn, "queued")
        finally:
            conn.close()

        if not queued:
            return {"status": "noop", "queued": 0, "message": "No queued ingest links."}

        if is_locked(cfg):
            return {"status": "skipped", "queued": len(queued), "message": "Another batch run is in progress."}

        script = Path(cfg.kb_agent_dir) / "scripts" / "batch_ingest.py"
        log_path = Path(cfg.kb_agent_dir) / "data" / "batch_manual.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log:
            log.write(f"\n--- run_batch {datetime.now(timezone.utc).isoformat()} ---\n")
            subprocess.Popen(
                [sys.executable, str(script)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return {
            "status": "started",
            "queued": len(queued),
            "log": agent_relative_path(log_path, cfg.kb_agent_dir),
            "message": "Batch running in background; results appended to the log.",
        }

    mcp.add_middleware(KbApiTokenMiddleware())
    return mcp


def run_stateless_streamable_http(
    config: Config | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
) -> None:
    """Run MCP on streamable HTTP in stateless mode (MCP spec 2026-07-28 style)."""
    cfg = config or load_config()
    from agent.mcp.db import ensure_databases

    ensure_databases(cfg)
    server = build_mcp_server(cfg)
    server.run(
        transport="streamable-http",
        host=host or cfg.mcp_host,
        port=port if port is not None else cfg.mcp_port,
        path=path or cfg.mcp_path,
        stateless=True,
    )
