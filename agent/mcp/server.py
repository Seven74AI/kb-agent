"""FastMCP 4.x server — stateless streamable HTTP tools over agent.core."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

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
from agent.core.watch_source import register_watch_source
from agent.mcp.db import open_agent_conn, open_logs_conn


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
    def detect_type(url: str) -> dict[str, str]:
        """Detect platform and content subtype for a URL (no side effects)."""
        return core_detect_type(url)

    @mcp.tool
    def submit_url(url: str) -> dict[str, Any]:
        """Enqueue a URL for pipeline processing; returns task status and handle."""
        conn = open_agent_conn(cfg)
        try:
            return core_submit_url(url, cfg, conn)
        finally:
            conn.close()

    @mcp.tool
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
    def resume_task(reason: str = "") -> dict[str, Any]:
        """Resume all paused tasks whose error message matches reason (SQL LIKE)."""
        conn = open_agent_conn(cfg)
        try:
            count = core_resume_tasks(conn, reason)
            return {"resumed": count}
        finally:
            conn.close()

    @mcp.tool
    def retry_task(task_id: str) -> dict[str, Any]:
        """Re-queue a single failed or paused task."""
        conn = open_agent_conn(cfg)
        try:
            ok = core_retry_task(conn, task_id)
            return {"ok": ok, "task_id": task_id}
        finally:
            conn.close()

    @mcp.tool
    def search_notes(query: str, limit: int = 50) -> list[dict[str, str]]:
        """Search markdown notes under the vault Knowledge base."""
        return core_search_notes(query, cfg.vault_path, limit=limit)

    @mcp.tool
    def get_note(slug: str, content_type: str | None = None) -> dict[str, Any]:
        """Load a vault note by slug; optional content_type narrows folder lookup."""
        note = core_get_note(slug, cfg.vault_path, content_type=content_type)
        if note is None:
            return {"found": False, "slug": slug}
        return {"found": True, "note": note}

    @mcp.tool
    def ingest_playlist(playlist_url: str) -> dict[str, Any]:
        """Expand a YouTube playlist and submit each video URL."""
        conn = open_agent_conn(cfg)
        try:
            return core_ingest_playlist(playlist_url, cfg, conn)
        finally:
            conn.close()

    @mcp.tool
    def bulk_import(urls: list[str]) -> dict[str, Any]:
        """Submit many URLs in one call (each routes through submit_url)."""
        conn = open_agent_conn(cfg)
        try:
            return core_bulk_import(urls, cfg, conn)
        finally:
            conn.close()

    @mcp.tool
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
