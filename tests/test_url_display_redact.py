"""Regression: outward surfaces never emit http(s) URLs with query or fragment."""

import sqlite3

import pytest

from agent.core.redact import url_for_display
from agent.core.status import get_task, list_tasks
from agent.core.submit import plan_submission
from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest
from agent.webhook import app

MEGA_WITH_KEY = "https://mega.nz/file/abc123#decryptionkey"
TOKENIZED = "https://cdn.example.com/video.mp4?token=secret&expires=999"
PLAIN = "https://example.com/article/path"


def test_url_for_display_strips_mega_fragment():
    assert url_for_display(MEGA_WITH_KEY) == "https://mega.nz/file/abc123"
    assert "#" not in url_for_display(MEGA_WITH_KEY)


def test_url_for_display_strips_query_params():
    assert url_for_display(TOKENIZED) == "https://cdn.example.com/video.mp4"
    assert "?" not in url_for_display(TOKENIZED)
    assert "token" not in url_for_display(TOKENIZED)


def test_url_for_display_preserves_plain_path():
    assert url_for_display(PLAIN) == PLAIN


def test_status_and_list_tasks_redact_http_url(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(conn, MEGA_WITH_KEY, "mega_video", "mega_test", ["download"])
    conn.commit()
    conn.close()

    conn = init_db(db_path, AGENT_SCHEMA)
    one = get_task(conn, task_id)
    many = list_tasks(conn, limit=10)
    conn.close()

    assert one is not None
    assert one["source_url"] == "https://mega.nz/file/abc123"
    assert one["source_url_href"] == "https://mega.nz/file/abc123"
    assert many["tasks"][0]["source_url"] == one["source_url"]

    conn = init_db(db_path, AGENT_SCHEMA)
    row = conn.execute("SELECT source_url FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    assert row[0] == MEGA_WITH_KEY


def test_plan_submission_error_plan_redacts_url(monkeypatch, tmp_path):
    from agent.config import load_config

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KB_VAULT_PATH", str(vault))
    core_config = load_config()
    monkeypatch.setattr(
        "agent.core.submit.detect_content_subtype",
        lambda url, platform: "unknown",
    )
    result = plan_submission(MEGA_WITH_KEY, core_config)
    assert result["status"] == "error"
    assert result["url"] == "https://mega.nz/file/abc123"
    assert "#" not in result["url"]


@pytest.fixture
def submit_client(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    db_path = tmp_path / "data" / "agent.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path, AGENT_SCHEMA)
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_AUTHORIZATION"] = "Bearer test-api-token"
        yield c


def test_dashboard_ingest_and_task_urls_redacted(submit_client, monkeypatch, tmp_path):
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    db_path = tmp_path / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    queue_ingest(conn, TOKENIZED, "web", "cdn_vid")
    create_task(conn, MEGA_WITH_KEY, "mega_video", "mega_dash", ["download"])
    conn.commit()
    conn.close()

    resp = submit_client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "decryptionkey" not in html
    assert "token=secret" not in html
    assert "https://mega.nz/file/abc123" in html
    assert "https://cdn.example.com/video.mp4" in html


def test_submit_json_redacts_url_on_error(submit_client, monkeypatch):
    monkeypatch.setattr(
        "agent.webhook.plan_submission",
        lambda url, cfg: {
            "status": "error",
            "http_status": 400,
            "error": "unknown content_type",
            "url": url,
        },
    )
    resp = submit_client.post("/submit", data={"url": TOKENIZED})
    data = resp.get_json()
    assert data["url"] == "https://cdn.example.com/video.mp4"
