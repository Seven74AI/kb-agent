"""Tests for the core package — API contract and import boundaries."""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from agent.db import AGENT_SCHEMA, LOGS_SCHEMA, create_task, init_db, pause_task
from agent.detection import detect_platform as agent_detect_platform
from core import (
    detect_platform,
    detect_type,
    get_note,
    get_task,
    list_tasks,
    normalize_resume_reason_pattern,
    resume_task,
    retry_task,
    search_notes,
    slugify,
    submit_url,
    SubmitDependencies,
    SubmitOutcome,
)
from core.pipelines import PIPELINE_STEPS, resolve_pipeline_steps


CORE_ROOT = Path(__file__).resolve().parents[1] / "core"
FORBIDDEN_IMPORT_ROOTS = {"flask", "mcp", "fastmcp"}


def _iter_core_py_files() -> list[Path]:
    return sorted(CORE_ROOT.glob("*.py"))


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


@pytest.mark.parametrize("path", _iter_core_py_files(), ids=lambda p: p.name)
def test_core_module_has_no_flask_or_mcp_imports(path: Path):
    roots = _module_import_roots(path.read_text(encoding="utf-8"))
    assert roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS), (
        f"{path.name} imports forbidden roots: {roots & FORBIDDEN_IMPORT_ROOTS}"
    )


def test_detect_platform_matches_agent():
    urls = [
        "https://youtube.com/watch?v=abc",
        "https://threads.net/@user/post/1",
        "https://example.com/article",
        "https://mega.nz/file/abc",
    ]
    for url in urls:
        assert detect_platform(url) == agent_detect_platform(url)


def test_detect_type_platform_only():
    result = detect_type("https://youtube.com/watch?v=x")
    assert result.platform == "youtube"
    assert result.content_type == "youtube"


def test_slugify_matches_webhook_behavior():
    from agent.webhook import slugify as webhook_slugify

    url = "https://example.com/path?q=1#frag"
    assert slugify(url) == webhook_slugify(url)


def test_pipeline_steps_web_default():
    steps = resolve_pipeline_steps("unknown_subtype", "web")
    assert steps == PIPELINE_STEPS["web"]
    assert "firecrawl_extract" in steps


def test_normalize_resume_reason_pattern():
    assert normalize_resume_reason_pattern("") == "%"
    assert normalize_resume_reason_pattern("cookies") == "%cookies%"
    assert normalize_resume_reason_pattern("%fire%") == "%fire%"


@pytest.fixture
def agent_conn(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    yield conn
    conn.close()


@pytest.fixture
def logs_conn(tmp_path):
    db_path = tmp_path / "logs.db"
    conn = init_db(db_path, LOGS_SCHEMA)
    yield conn
    conn.close()


def test_submit_url_skipped_when_dedup(agent_conn, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    result = submit_url(
        agent_conn,
        "https://example.com/a",
        vault_path=vault,
        deps=SubmitDependencies(
            dedup_check=lambda url, vp: True,
            preflight_health=lambda p: (True, ""),
            detect_content_subtype=lambda url, platform: platform,
            create_task=create_task,
        ),
    )
    assert result.outcome == SubmitOutcome.SKIPPED


def test_submit_url_creates_task(agent_conn, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    result = submit_url(
        agent_conn,
        "https://example.com/article",
        vault_path=vault,
        deps=SubmitDependencies(
            dedup_check=lambda url, vp: False,
            preflight_health=lambda p: (True, ""),
            detect_content_subtype=lambda url, platform: "web",
            create_task=create_task,
        ),
    )
    assert result.outcome == SubmitOutcome.CREATED
    assert result.task_id
    row = agent_conn.execute(
        "SELECT status, content_type FROM tasks WHERE id = ?",
        (result.task_id,),
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == "web"


def test_resume_and_retry_task(agent_conn):
    task_id = create_task(
        agent_conn,
        "https://example.com",
        "web",
        "slug",
        ["s1"],
        status="paused",
    )
    agent_conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("cookies invalid", task_id),
    )
    agent_conn.commit()

    resumed = resume_task(agent_conn, "cookies")
    assert resumed.resumed == 1

    pause_task(agent_conn, task_id, "manual pause")
    retry = retry_task(agent_conn, task_id)
    assert retry.retried is True
    row = agent_conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_list_tasks_and_get_task(agent_conn, logs_conn):
    tid = create_task(agent_conn, "https://a.com", "web", "a", ["extract"])
    now = datetime(2026, 6, 19, 12, 0, 0)
    listing = list_tasks(agent_conn, logs_conn=logs_conn, now=now)
    assert listing.counts["total"] == 1
    one = get_task(agent_conn, tid, logs_conn=logs_conn, now=now)
    assert one is not None
    assert one.source_url == "https://a.com"
    assert len(one.steps) == 1


def test_search_and_get_note(tmp_path):
    vault = tmp_path / "vault"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    note_path = note_dir / "my-note.md"
    note_path.write_text(
        "source_url: https://example.com/x\n\nUniquePhraseForSearch123",
        encoding="utf-8",
    )

    hits = search_notes(vault, "UniquePhraseForSearch123")
    assert len(hits.hits) == 1
    assert hits.hits[0].slug == "my-note"

    content = get_note(vault, "my-note")
    assert content is not None
    assert "UniquePhraseForSearch123" in content.body
    assert content.frontmatter_source_url == "https://example.com/x"
