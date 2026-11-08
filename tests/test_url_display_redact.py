"""Regression: outward surfaces never emit http(s) URLs with query or fragment."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from agent.config import Config
from agent.core.bulk import ingest_playlist
from agent.core.redact import url_for_display
from agent.redact import redact_http_urls_in_text, scrub_vault_note_content
from agent.core.status import get_task, list_tasks
from agent.core.submit import plan_submission
from agent.detection import ContentSubtypeDetection
from agent.core.watch_source import get_watch_source, poll_watch_source, register_watch_source
from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest
from agent.webhook import app

MEGA_WITH_KEY = "https://mega.nz/file/abc123#decryptionkey"
TOKENIZED = "https://cdn.example.com/video.mp4?token=secret&expires=999"
PLAIN = "https://example.com/article/path"


@pytest.fixture
def agent_conn(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def core_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


def test_url_for_display_strips_mega_fragment():
    assert url_for_display(MEGA_WITH_KEY) == "https://mega.nz/file/abc123"
    assert "#" not in url_for_display(MEGA_WITH_KEY)


def test_url_for_display_strips_query_params():
    assert url_for_display(TOKENIZED) == "https://cdn.example.com/video.mp4"
    assert "?" not in url_for_display(TOKENIZED)
    assert "token" not in url_for_display(TOKENIZED)


def test_url_for_display_preserves_plain_path():
    assert url_for_display(PLAIN) == PLAIN


def test_redact_http_urls_in_text_strips_query():
    text = f"SOURCE URL: {TOKENIZED}\nbody"
    redacted = redact_http_urls_in_text(text)
    assert "token=secret" not in redacted
    assert "https://cdn.example.com/video.mp4" in redacted


def test_scrub_vault_note_content_redacts_frontmatter_only():
    body = f"""---
source_url: {TOKENIZED}
tags: [web]
---

source_url: {TOKENIZED}
"""
    scrubbed = scrub_vault_note_content(body)
    frontmatter = scrubbed.split("---")[1]
    assert "token=secret" not in frontmatter
    assert TOKENIZED in scrubbed.split("---", 2)[2]


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
        "agent.core.submit.detect_content_subtype_with_signals",
        lambda url, platform: ContentSubtypeDetection(subtype="unknown"),
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


PLAYLIST_WITH_LIST = "https://youtube.com/playlist?list=PLsecret&session=abc"
WATCH_PAGE = "https://example.com/listing?tracker=1"
WATCH_TEMPLATE = "https://cdn.example.com/watch/{item}?token=secret"


def test_ingest_playlist_redacts_playlist_url_in_ok(monkeypatch, core_config, agent_conn):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    monkeypatch.setattr(
        "agent.core.bulk.expand_youtube_playlist",
        lambda _url: ["https://youtube.com/watch?v=1"],
    )
    result = ingest_playlist(PLAYLIST_WITH_LIST, core_config, agent_conn)
    assert result["status"] == "ok"
    assert result["playlist_url"] == "https://youtube.com/playlist"
    assert "?" not in result["playlist_url"]
    assert "list=PLsecret" not in result["playlist_url"]


def test_ingest_playlist_redacts_playlist_url_when_empty(monkeypatch, core_config, agent_conn):
    monkeypatch.setattr(
        "agent.core.bulk.expand_youtube_playlist",
        lambda _url: [],
    )
    result = ingest_playlist(PLAYLIST_WITH_LIST, core_config, agent_conn)
    assert result["status"] == "error"
    assert result["playlist_url"] == "https://youtube.com/playlist"
    assert "?" not in result["playlist_url"]


def test_register_watch_source_duplicate_error_redacts_page_url(agent_conn):
    page_token = "https://example.com/listing?tracker=supersecret"
    register_watch_source(
        agent_conn,
        page_token,
        item_pattern=r"id-(\d+)",
        item_url_template="https://example.com/i/{item}",
        cadence_seconds=3600,
        fetch_html=None,
        baseline=False,
    )
    with pytest.raises(ValueError, match="already registered") as exc_info:
        register_watch_source(
            agent_conn,
            page_token,
            item_pattern=r"id-(\d+)",
            item_url_template="https://example.com/i/{item}",
            cadence_seconds=3600,
            fetch_html=None,
            baseline=False,
        )
    msg = str(exc_info.value)
    assert "supersecret" not in msg
    assert "tracker=" not in msg
    assert "https://example.com/listing" in msg


def test_poll_watch_source_not_registered_error_redacts_page_url(agent_conn):
    token_page = "https://example.com/missing?sig=hidden"
    with pytest.raises(ValueError, match="not registered") as exc_info:
        poll_watch_source(
            agent_conn,
            token_page,
            MagicMock(),
            fetch_html=lambda _u: "",
            force=True,
        )
    msg = str(exc_info.value)
    assert "hidden" not in msg
    assert "sig=" not in msg
    assert "https://example.com/missing" in msg


def test_watch_source_register_redacts_urls_in_response(agent_conn):
    row = register_watch_source(
        agent_conn,
        WATCH_PAGE,
        item_pattern=r"id-(\d+)",
        item_url_template=WATCH_TEMPLATE,
        cadence_seconds=3600,
        fetch_html=None,
        baseline=False,
    )
    assert row["page_url"] == "https://example.com/listing"
    assert row["item_url_template"] == "https://cdn.example.com/watch/{item}"
    assert "?" not in row["page_url"]
    assert "token" not in row["item_url_template"]

    stored = get_watch_source(agent_conn, WATCH_PAGE)
    assert stored is not None
    assert stored["page_url"] == WATCH_PAGE
    assert stored["item_url_template"] == WATCH_TEMPLATE


def test_poll_watch_source_redacts_page_url_in_outcome(core_config, agent_conn, monkeypatch):
    page = "https://example.com/poll-me?sig=hidden"
    register_watch_source(
        agent_conn,
        page,
        item_pattern=r"x",
        item_url_template="https://example.com/i/{item}",
        cadence_seconds=1,
        fetch_html=lambda _u: "",
        baseline=False,
    )

    def _noop_submit(_url, _cfg, _conn):
        return {"status": "ok"}

    monkeypatch.setattr(
        "agent.core.watch_source.extract_items_available",
        lambda _html, _pat, now=None: frozenset(),
    )
    result = poll_watch_source(
        agent_conn,
        page,
        core_config,
        fetch_html=lambda _u: "",
        submit_fn=_noop_submit,
        force=True,
    )
    assert result["page_url"] == "https://example.com/poll-me"
    assert "?" not in result["page_url"]


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
