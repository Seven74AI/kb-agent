"""Boundary 4: exception JSON, cron stdout, and read-path errors strip URL query/userinfo."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.config import Config
from agent.core.watch_source import poll_watch_source, register_watch_source
from agent.db import AGENT_SCHEMA, init_db

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEAKY_URL = "https://cdn.example.com/v?token=SECRET&sig=abc"
_RAW_ERR = f"upstream refused {_LEAKY_URL}"

SAMPLE_HTML = """
<html><body>
  <a href="/tdah-adulte-j1-intro/">Intro</a>
  <a href="/tdah-adulte-j2-keynote/">Keynote</a>
</body></html>
"""

SAMPLE_HTML_NEW = """
<html><body>
  <a href="/tdah-adulte-j1-intro/">Intro</a>
  <a href="/tdah-adulte-j2-keynote/">Keynote</a>
  <a href="/tdah-adulte-j3-panel/">Panel</a>
</body></html>
"""
TDAH_PAGE = "https://www.terragrandir.com/conferences-tdah-adulte-2026/"
TDAH_PATTERN = r"/tdah-adulte-(j\d+-[a-z0-9-]+)/"
TDAH_TEMPLATE = "https://www.terragrandir.com/tdah-adulte-{item}/"


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


@pytest.fixture(autouse=True)
def _kb_api_token(monkeypatch):
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")


@pytest.fixture
def client():
    from agent.webhook import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_AUTHORIZATION"] = "Bearer test-api-token"
        yield c


def _load_script_module(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_mcp_sanitize_exception_message_strips_url_query():
    from agent.mcp.server import _sanitize_exception_message

    out = _sanitize_exception_message(_RAW_ERR)
    assert "token=SECRET" not in out
    assert "SECRET" not in out
    assert "sig=abc" not in out
    assert "upstream refused" in out


def test_ingest_playlist_expand_error_strips_url_query(monkeypatch, core_config, agent_conn):
    from agent.core.bulk import PlaylistExpandError, ingest_playlist

    def fail(_url):
        raise PlaylistExpandError(_RAW_ERR)

    monkeypatch.setattr("agent.core.bulk.expand_youtube_playlist", fail)
    result = ingest_playlist("https://youtube.com/playlist?list=PLx", core_config, agent_conn)
    assert result["status"] == "error"
    assert "token=SECRET" not in result["error"]
    assert "upstream refused" in result["error"]


def test_plan_submission_dedup_failure_strips_url_query(monkeypatch, core_config):
    from agent.core.submit import plan_submission
    from agent.detection import DedupCheckError

    def boom(*_a, **_k):
        raise DedupCheckError(_RAW_ERR)

    monkeypatch.setattr("agent.core.submit.dedup_check", boom)
    plan = plan_submission("https://example.com/post", core_config)
    assert plan["status"] == "error"
    assert "token=SECRET" not in plan["error"]
    assert "dedup check failed" in plan["error"]


def test_poll_watch_source_item_error_strips_url_query(agent_conn, core_config, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def boom(_url, _cfg, _conn):
        raise RuntimeError(_RAW_ERR)

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    outcome = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=boom,
        now=now,
    )
    err = outcome["submissions"][0]["error"]
    assert "token=SECRET" not in err
    assert "RuntimeError" in err


def test_dashboard_read_path_strips_url_in_task_error(client, monkeypatch, tmp_path):
    from agent.db import AGENT_SCHEMA, create_task, init_db

    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)
    create_task(conn, "https://example.com/fail", "web", "fail_slug", ["extract"], status="failed")
    conn.execute(
        "UPDATE tasks SET error = ? WHERE slug = ?",
        (_RAW_ERR, "fail_slug"),
    )
    conn.commit()
    conn.close()

    resp = client.get("/dashboard")
    html = resp.data.decode()
    assert "token=SECRET" not in html
    assert "upstream refused" in html


def test_batch_ingest_scrub_for_log_strips_url_query():
    import scripts.batch_ingest as batch_ingest

    clean = batch_ingest._scrub_for_log(_RAW_ERR)
    assert "token=SECRET" not in clean
    assert "upstream refused" in clean


def test_batch_ingest_child_tail_strips_url_query(capsys):
    import scripts.batch_ingest as batch_ingest

    result = MagicMock(stdout=_RAW_ERR + "\n", stderr="", returncode=1)
    batch_ingest._print_scrubbed_child_tail(result, tail_chars=500)
    out = capsys.readouterr().out
    assert "token=SECRET" not in out
    assert "upstream refused" in out


def test_record_parent_deferral_scrubs_combined_prior_url(tmp_path):
    import scripts.batch_ingest as batch_ingest
    from agent.db import AGENT_SCHEMA, create_task, init_db

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    slug = "parent_url_scrub"
    create_task(
        conn,
        "https://example.com/p",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE slug = ?",
        (f"paused awaiting {_LEAKY_URL}", slug),
    )
    conn.commit()

    batch_ingest._record_parent_deferral_dispatch_failure(conn, slug, "dispatch timeout")
    conn.commit()
    row = conn.execute("SELECT error FROM tasks WHERE slug = ?", (slug,)).fetchone()
    error = row[0] or ""
    assert "token=SECRET" not in error
    assert "deferred synthesis dispatch failed" in error


def test_batch_ocr_child_tail_strips_url_query(capsys):
    mod = _load_script_module("batch_ocr", "scripts/batch_ocr.py")
    result = MagicMock(stdout="", stderr=_RAW_ERR, returncode=1)
    mod._print_scrubbed_child_tail(result, tail_chars=500)
    out = capsys.readouterr().out
    assert "token=SECRET" not in out


def test_batch_transcribe_child_tail_strips_url_query(capsys):
    mod = _load_script_module("batch_transcribe", "scripts/batch_transcribe.py")
    result = MagicMock(stdout=_RAW_ERR, stderr="", returncode=1)
    mod._print_scrubbed_child_tail(result, tail_chars=500)
    out = capsys.readouterr().out
    assert "token=SECRET" not in out


def test_tdah_dispatch_and_modal_tail_strip_url_query(monkeypatch, capsys, tmp_path):
    mod = _load_script_module("tdah_daily_check", "scripts/tdah_daily_check.py")
    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "error", "error": _RAW_ERR},
    )
    mod._finalize_video_dispatch("https://example.com/x", MagicMock(), MagicMock(), "slug-x")
    out = capsys.readouterr().out
    assert "token=SECRET" not in out

    tail = f"modal stderr: fetch {_LEAKY_URL}\n"
    from agent.notify import scrub_outbound_visible_message

    printed = scrub_outbound_visible_message(tail[-1500:])
    assert "token=SECRET" not in printed


def test_rotate_logs_archive_record_strips_url_in_step_content():
    mod = _load_script_module("rotate_logs", "scripts/rotate_logs.py")
    record = {"_table": "step_logs", "content": _RAW_ERR}
    clean = mod.sanitize_archive_record(record)
    assert "token=SECRET" not in clean["content"]
    assert "upstream refused" in clean["content"]
