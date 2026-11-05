"""Tests for agent.core — pure domain layer (no Flask)."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.config import Config
from agent.core import (
    bulk_import,
    compute_staleness,
    detect_type,
    format_elapsed,
    get_note,
    get_task,
    ingest_playlist,
    list_tasks,
    normalize_reason_pattern,
    resume_tasks,
    retry_task,
    search_notes,
    slugify,
    submit_url,
)
from agent.db import AGENT_SCHEMA, LOGS_SCHEMA, create_task, init_db, pause_task


@pytest.fixture
def core_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


@pytest.fixture
def agent_conn(tmp_path):
    db_path = tmp_path / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def logs_conn(tmp_path):
    db_path = tmp_path / "data" / "logs.db"
    conn = init_db(db_path, LOGS_SCHEMA)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_slugify_basic():
    assert slugify("https://example.com/foo/bar") == "example.com_foo_bar"


def test_slugify_youtube_watch_v_param():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert slugify(url).endswith("_v_dQw4w9WgXcQ")


@pytest.mark.parametrize(
    "url,vid",
    [
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_slugify_youtube_path_ids(url, vid):
    assert slugify(url).endswith(f"_v_{vid}")


def test_steps_for_mega_video_not_mega_delegate():
    from agent.pipelines import get_pipeline_names

    steps = get_pipeline_names("mega_video", "mega")
    assert steps[0] == "extract_metadata"
    assert "megadl_download" not in steps
    assert "download_video" in steps
    assert steps[-1] == "git_push"


def test_bulk_import_passes_tags_when_submit_accepts(core_config, agent_conn):
    received: list = []

    def submit_with_tags(url, config, conn, tags=None):
        received.append(tags)
        return {"status": "ok", "url": url}

    bulk_import(
        ["https://example.com/a"],
        core_config,
        agent_conn,
        tags="foo, bar",
        submit_fn=submit_with_tags,
    )
    assert received == ["foo, bar"]


def test_bulk_import_omits_tags_when_submit_does_not_accept(core_config, agent_conn):
    received: list[str] = []

    def submit_plain(url, config, conn):
        received.append(url)
        return {"status": "ok"}

    bulk_import(
        ["https://example.com/a"],
        core_config,
        agent_conn,
        tags="ignored",
        submit_fn=submit_plain,
    )
    assert received == ["https://example.com/a"]


def test_detect_type_web():
    result = detect_type("https://example.com/page")
    assert result["platform"] == "web"
    assert result["content_type"] == "web"


def test_submit_url_empty(core_config, agent_conn):
    result = submit_url("", core_config, agent_conn)
    assert result["status"] == "error"
    assert result["http_status"] == 400


def test_submit_url_creates_task(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    result = submit_url("https://example.com/article", core_config, agent_conn)
    assert result["status"] == "pending"
    assert result["http_status"] == 201
    assert result["task_id"]
    assert result["platform"] == "web"


def test_submit_url_dedup_skips(core_config, agent_conn, monkeypatch):
    note = core_config.vault_path / "Knowledge base" / "articles" / "existing.md"
    note.write_text("---\nsource_url: https://example.com/dup\n---\n", encoding="utf-8")
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    result = submit_url("https://example.com/dup", core_config, agent_conn)
    assert result["status"] == "skipped"
    assert result["http_status"] == 200


def test_bulk_import_submits_each_url(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    seen: list[str] = []

    def track(url, config, conn):
        seen.append(url)
        return submit_url(url, config, conn)

    result = bulk_import(
        ["https://example.com/a", "https://example.com/b"],
        core_config,
        agent_conn,
        submit_fn=track,
    )
    assert result["count"] == 2
    assert len(seen) == 2


def test_ingest_playlist_expands(monkeypatch, core_config, agent_conn):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    monkeypatch.setattr(
        "agent.core.bulk.expand_youtube_playlist",
        lambda _url: ["https://youtube.com/watch?v=1", "https://youtube.com/watch?v=2"],
    )
    result = ingest_playlist("https://youtube.com/playlist?list=PLx", core_config, agent_conn)
    assert result["status"] == "ok"
    assert result["count"] == 2


def test_submit_url_health_pause(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr(
        "agent.core.submit.preflight_health",
        lambda p: (False, "cookies invalid"),
    )
    result = submit_url("https://youtube.com/watch?v=abc", core_config, agent_conn)
    assert result["status"] == "paused"
    assert result["http_status"] == 200
    assert result["task_id"]


def test_list_tasks_and_get_task(core_config, agent_conn, logs_conn):
    task_id = create_task(agent_conn, "https://a.com", "web", "a_slug", ["extract"])
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    assert listed["counts"]["total"] == 1
    assert len(listed["tasks"]) == 1

    one = get_task(agent_conn, task_id, logs_conn=logs_conn)
    assert one is not None
    assert one["id"] == task_id
    assert one["steps"][0]["name"] == "extract"


def test_get_task_missing(agent_conn):
    assert get_task(agent_conn, "nonexistent-id") is None


def test_normalize_reason_pattern():
    assert normalize_reason_pattern("") == "%"
    assert normalize_reason_pattern("cookies") == "%cookies%"
    assert normalize_reason_pattern("%fire%") == "%fire%"


def test_resume_tasks(agent_conn):
    task_id = create_task(agent_conn, "https://x.com", "web", "x", ["s1"], status="paused")
    agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?", ("cookies bad", task_id))
    agent_conn.commit()
    count = resume_tasks(agent_conn, "cookies")
    assert count == 1
    row = agent_conn.execute("SELECT status, error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row[0] == "pending"
    assert row[1] is None


def test_retry_task_failed(agent_conn):
    task_id = create_task(agent_conn, "https://x.com", "web", "x", ["s1", "s2"])
    fail_task_id = agent_conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 's2'",
        (task_id,),
    ).fetchone()[0]
    agent_conn.execute("UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 's1'", (task_id,))
    agent_conn.execute(
        "UPDATE task_steps SET status = 'failed', output = 'boom' WHERE id = ?",
        (fail_task_id,),
    )
    agent_conn.execute("UPDATE tasks SET status = 'failed', error = 'boom' WHERE id = ?", (task_id,))
    agent_conn.commit()

    assert retry_task(agent_conn, task_id) is True
    row = agent_conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row[0] == "pending"
    step2 = agent_conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (fail_task_id,),
    ).fetchone()
    assert step2[0] == "pending"
    assert step2[1] is None


def test_retry_task_not_failed(agent_conn):
    task_id = create_task(agent_conn, "https://x.com", "web", "x", ["s1"])
    assert retry_task(agent_conn, task_id) is False


def test_search_notes(core_config):
    articles = core_config.vault_path / "Knowledge base" / "articles"
    (articles / "alpha.md").write_text("# Alpha\nquantum computing overview\n", encoding="utf-8")
    (articles / "beta.md").write_text("# Beta\nunrelated topic\n", encoding="utf-8")
    hits = search_notes("quantum", core_config.vault_path)
    paths = {h["path"] for h in hits}
    assert any("alpha.md" in p for p in paths)
    assert not any("beta.md" in p for p in paths)


def test_get_note_by_slug(core_config):
    articles = core_config.vault_path / "Knowledge base" / "articles"
    (articles / "my-note.md").write_text("# Title\nBody text\n", encoding="utf-8")
    note = get_note("my-note", core_config.vault_path, content_type="web")
    assert note is not None
    assert "Body text" in note["content"]
    assert note["slug"] == "my-note"


def test_format_elapsed_seconds():
    created = "2026-06-19T10:00:00"
    now = datetime(2026, 6, 19, 10, 0, 30)
    assert format_elapsed(created, now) == "30s"


def test_compute_staleness_old(logs_conn):
    now = datetime.now(timezone.utc)
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, content, logged_at) VALUES (?, ?, ?, ?)",
        ("task-2", "extract", "done", (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")),
    )
    logs_conn.commit()
    staleness, is_stale = compute_staleness({"id": "task-2"}, logs_conn, now.replace(tzinfo=None))
    assert staleness is not None
    assert is_stale
