"""MCP tracer — stateless streamable HTTP tools (submit_url, detect_type)."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.core import get_note
from agent.db import AGENT_SCHEMA, init_db
from agent.llm import LLMResponse
from agent.mcp.server import build_mcp_server
from agent.pipelines.web import step_synthesize_web
from fastmcp.utilities.tests import asgi_server


CORE_ROOT = Path(__file__).resolve().parents[1] / "agent" / "core"
FORBIDDEN_IMPORT_ROOTS = {"flask", "mcp", "fastmcp"}


def _module_import_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(CORE_ROOT.glob("*.py")), ids=lambda p: p.name)
def test_agent_core_has_no_flask_or_mcp_imports(path: Path):
    roots = _module_import_roots(path.read_text(encoding="utf-8"))
    assert roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS), (
        f"{path.name} imports forbidden roots: {roots & FORBIDDEN_IMPORT_ROOTS}"
    )


@pytest.fixture
def mcp_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


@pytest.fixture
def mcp_server(mcp_config):
    return build_mcp_server(mcp_config)


async def _with_stateless_client(mcp_server):
    return asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    )


@pytest.mark.asyncio
async def test_mcp_lists_submit_url_and_detect_type(mcp_server):
    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            assert names == {"submit_url", "detect_type"}


@pytest.mark.asyncio
async def test_mcp_detect_type_tool(mcp_server, monkeypatch):
    monkeypatch.setattr(
        "agent.detection.detect_content_subtype",
        lambda url, platform: "web",
    )
    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            result = await client.call_tool(
                "detect_type", {"url": "https://example.com/article"}
            )
            payload = result.data
            assert payload["platform"] == "web"
            assert payload["content_type"] == "web"


@pytest.mark.asyncio
async def test_mcp_submit_url_enqueues_task(mcp_server, mcp_config, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    init_db(db_path, AGENT_SCHEMA)

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            result = await client.call_tool(
                "submit_url", {"url": "https://example.com/mcp-submit-test"}
            )
            body = result.data
            assert body["status"] == "pending"
            assert body["task_id"]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status, source_url FROM tasks WHERE id = ?",
        (body["task_id"],),
    ).fetchone()
    conn.close()
    assert row["status"] == "pending"
    assert row["source_url"] == "https://example.com/mcp-submit-test"


@pytest.mark.asyncio
async def test_mcp_submit_to_note_via_web_pipeline(mcp_server, mcp_config, monkeypatch):
    """MCP submit_url → pending task → web synthesize step → vault note."""
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    init_db(db_path, AGENT_SCHEMA)

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            result = await client.call_tool(
                "submit_url", {"url": "https://example.com/mcp-e2e-note"}
            )
            body = result.data

    slug = body["slug"]
    source_path = mcp_config.temp_dir / f"{slug}_raw.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "# MCP E2E\n\n> \"Quote one\" — Author\n\nBody paragraph.\n\n"
        "> \"Quote two\" — Other\n\nMore analysis text here.\n",
        encoding="utf-8",
    )

    task_dict = {
        "id": body["task_id"],
        "source_url": "https://example.com/mcp-e2e-note",
        "content_type": "web",
        "slug": slug,
    }

    mock_response = LLMResponse(
        content=(
            "---\n"
            "source_url: https://example.com/mcp-e2e-note\n"
            "content_type: web\n"
            "---\n\n"
            "# MCP E2E Note\n\n"
            "Synthesized note body with enough length for testing purposes.\n"
        ),
        model="test",
    )
    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        step_synthesize_web(task_dict, mcp_config)

    note = get_note(slug, mcp_config.vault_path, content_type="web")
    assert note is not None
    assert "MCP E2E Note" in note["content"]


def test_http_app_stateless_streamable_transport(mcp_server):
    app = mcp_server.http_app(
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    )
    assert app is not None
