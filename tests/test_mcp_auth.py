"""MCP API token auth — unauthenticated tool calls are rejected."""

from __future__ import annotations

import pytest

from agent.config import Config
from agent.mcp.server import build_mcp_server
from fastmcp.exceptions import AuthorizationError, ToolError
from fastmcp.utilities.tests import asgi_server


@pytest.fixture
def mcp_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


@pytest.fixture
def mcp_server(mcp_config):
    return build_mcp_server(mcp_config)


@pytest.mark.asyncio
async def test_mcp_rejects_unauthenticated_tool_call(mcp_server, monkeypatch):
    monkeypatch.setenv("KB_API_TOKEN", "secret-mcp-token")
    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            with pytest.raises((AuthorizationError, ToolError)):
                await client.call_tool("detect_type", {"url": "https://example.com/x"})


@pytest.mark.asyncio
async def test_mcp_allows_authenticated_tool_call(mcp_server, monkeypatch):
    monkeypatch.setenv("KB_API_TOKEN", "secret-mcp-token")
    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        headers = {"Authorization": "Bearer secret-mcp-token"}
        async with running.client(headers=headers) as client:
            result = await client.call_tool(
                "detect_type", {"url": "https://example.com/x"}
            )
            assert result.data["platform"] == "web"
