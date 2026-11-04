"""FastMCP 4.x server — stateless streamable HTTP tools over agent.core."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from agent.config import Config, load_config
from agent.core.detect import detect_type as core_detect_type
from agent.core.submit import submit_url as core_submit_url
from agent.mcp.db import open_agent_conn


def build_mcp_server(config: Config | None = None) -> FastMCP:
    """Create a FastMCP server exposing submit_url and detect_type."""
    cfg = config or load_config()
    mcp = FastMCP(
        name="kb-agent-tracer",
        instructions=(
            "KB Agent MCP tracer — submit URLs for ingestion and detect content "
            "types without enqueueing. Stateless: each HTTP request is independent."
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
