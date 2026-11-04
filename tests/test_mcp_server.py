"""MCP server — stateless streamable HTTP tools (full kb-agent surface)."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.core import get_note
from agent.db import AGENT_SCHEMA, create_task, init_db
from agent.llm import LLMResponse
from agent.mcp.server import build_mcp_server
from agent.pipelines.web import step_synthesize_web
from fastmcp.utilities.tests import asgi_server


CORE_ROOT = Path(__file__).resolve().parents[1] / "agent" / "core"
FORBIDDEN_IMPORT_ROOTS = {"flask", "mcp", "fastmcp"}

EXPECTED_MCP_TOOLS = frozenset({
    "submit_url",
    "ingest",
    "detect_type",
    "list_tasks",
    "get_task",
    "resume_task",
    "retry_task",
    "search_notes",
    "get_note",
    "ingest_playlist",
    "bulk_import",
    "watch_source",
    "run_batch",
})


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
async def test_mcp_exposes_eleven_tools(mcp_server):
    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            assert names == EXPECTED_MCP_TOOLS


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


@pytest.mark.asyncio
async def test_mcp_list_and_get_task(mcp_server, mcp_config, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://example.com/list-me",
        "web",
        "list_me",
        ["extract"],
    )
    conn.close()

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            listed = await client.call_tool("list_tasks", {"limit": 10})
            body = listed.data
            assert body["counts"]["total"] >= 1
            got = await client.call_tool("get_task", {"task_id": task_id})
            payload = got.data
            assert payload["found"] is True
            assert payload["task"]["id"] == task_id


@pytest.mark.asyncio
async def test_mcp_resume_and_retry_task(mcp_server, mcp_config):
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(conn, "https://x.com", "web", "x", ["s1"], status="paused")
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", ("cookies bad", task_id))
    conn.commit()
    conn.close()

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            resumed = await client.call_tool("resume_task", {"reason": "cookies"})
            assert resumed.data["resumed"] == 1

            conn = sqlite3.connect(str(db_path))
            conn.execute("UPDATE tasks SET status = 'failed' WHERE id = ?", (task_id,))
            conn.commit()
            conn.close()

            retried = await client.call_tool("retry_task", {"task_id": task_id})
            assert retried.data["ok"] is True


@pytest.mark.asyncio
async def test_mcp_search_notes_and_get_note(mcp_server, mcp_config):
    articles = mcp_config.vault_path / "Knowledge base" / "articles"
    (articles / "mcp-search.md").write_text("# MCP\nquantum field notes\n", encoding="utf-8")

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            hits = await client.call_tool("search_notes", {"query": "quantum"})
            assert any("mcp-search" in h["slug"] for h in hits.data)
            note = await client.call_tool(
                "get_note", {"slug": "mcp-search", "content_type": "web"}
            )
            assert note.data["found"] is True
            assert "quantum" in note.data["note"]["content"]


@pytest.mark.asyncio
async def test_mcp_bulk_import_calls_submit_per_url(mcp_server, mcp_config, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    init_db(db_path, AGENT_SCHEMA)
    calls: list[str] = []

    import agent.mcp.server as mcp_server_mod

    real_routed = mcp_server_mod.submit_routed

    def track_routed(url, config, conn, *, tags=None):
        calls.append(url)
        return real_routed(url, config, conn, tags=tags)

    monkeypatch.setattr(mcp_server_mod, "submit_routed", track_routed)

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            result = await client.call_tool(
                "bulk_import",
                {"urls": ["https://a.com/1", "https://a.com/2"]},
            )
            body = result.data
            assert body["count"] == 2
            assert len(body["results"]) == 2
    assert calls == ["https://a.com/1", "https://a.com/2"]


@pytest.mark.asyncio
async def test_mcp_ingest_playlist_expands_to_submit(mcp_server, mcp_config, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    db_path = mcp_config.kb_agent_dir / "data" / "agent.db"
    init_db(db_path, AGENT_SCHEMA)

    def fake_expand(_playlist: str) -> list[str]:
        return [
            "https://youtube.com/watch?v=aaa",
            "https://youtube.com/watch?v=bbb",
        ]

    monkeypatch.setattr("agent.core.bulk.expand_youtube_playlist", fake_expand)
    calls: list[str] = []

    import agent.mcp.server as mcp_server_mod

    real_routed = mcp_server_mod.submit_routed

    def track_routed(url, config, conn, *, tags=None):
        calls.append(url)
        return real_routed(url, config, conn, tags=tags)

    monkeypatch.setattr(mcp_server_mod, "submit_routed", track_routed)

    async with asgi_server(
        mcp_server,
        transport="streamable-http",
        path="/mcp",
        stateless_http=True,
    ) as running:
        async with running.client() as client:
            result = await client.call_tool(
                "ingest_playlist",
                {"playlist_url": "https://youtube.com/playlist?list=PLtest"},
            )
            body = result.data
            assert body["status"] == "ok"
            assert body["count"] == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_mcp_watch_source_registers(mcp_server, mcp_config):
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
                "watch_source",
                {
                    "page_url": "https://example.com/listing",
                    "item_pattern": r"video/(\d+)",
                    "item_url_template": "https://example.com/watch/{item}",
                    "baseline": False,
                },
            )
            row = result.data
            assert row["page_url"] == "https://example.com/listing"
            assert "{item}" in row["item_url_template"] or "watch" in row["item_url_template"]
