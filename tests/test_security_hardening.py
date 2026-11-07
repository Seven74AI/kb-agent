"""Regression tests for security hardening (format strings, resume, cookies, grep -F)."""

import json
import os
import stat
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.webhook import app
from agent.core.watch_source import items_to_urls
from agent.db import AGENT_SCHEMA, create_task, init_db
from agent.pipelines.see_also import _search_vault


@pytest.fixture(autouse=True)
def _kb_api_token(monkeypatch):
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_AUTHORIZATION"] = "Bearer test-api-token"
        yield c


def test_items_to_urls_does_not_expand_format_attributes():
    """Malicious templates must not use str.format attribute traversal."""
    template = "https://evil.example/{item.__class__.__init__.__globals__}"
    urls = items_to_urls(["secret-id"], template)
    # Only the literal ``{item}`` token is substituted; no attribute expansion.
    assert urls == [template]
    assert "SECRET_MARKER" not in urls[0]


def test_post_resume_without_reason_returns_400(client, monkeypatch, tmp_path):
    """POST /resume without reason must not resume all paused tasks."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)

    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com/1", "web", "slug_1", ["s1"], status="paused"
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", ("error 1", task_id))
    conn.commit()
    conn.close()

    resp = client.post("/resume")
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "reason required" in data["error"].lower()


def test_post_resume_empty_reason_returns_400(client):
    resp = client.post("/resume?reason=")
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "reason required" in data["error"].lower()


def test_post_resume_wildcard_reason_returns_400(client):
    """reason=% must not resume every paused task."""
    resp = client.post("/resume?reason=%")
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "reason required" in data["error"].lower()


def test_upload_cookies_saves_mode_600(client, monkeypatch, tmp_path):
    """Cookie files must be owner read/write only (chmod 600)."""
    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    cookie_content = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tLOGIN_INFO\tabc\n"
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(cookie_content.encode())
    tmp.close()

    try:
        with open(tmp.name, "rb") as f:
            resp = client.post(
                "/upload-cookies",
                data={
                    "cookie_type": "yt_cookies.txt",
                    "cookie_file": (f, "yt_cookies.txt"),
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 201
        data = json.loads(resp.data)
        assert data["permissions"] == "600"
        dest = cookie_dir / "yt_cookies.txt"
        assert oct(dest.stat().st_mode)[-3:] == "600"
        assert dest.stat().st_mode & stat.S_IRWXG == 0
        assert dest.stat().st_mode & stat.S_IRWXO == 0
    finally:
        os.unlink(tmp.name)


def test_search_notes_grep_uses_fixed_string_flag(tmp_path):
    from agent.core.search import search_notes

    vault = tmp_path / "vault"
    (vault / "Knowledge base").mkdir(parents=True)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        search_notes("foo.bar", vault)
        argv = mock_run.call_args[0][0]
        assert argv[0] == "grep"
        assert "-rlF" in argv or (argv[1] == "-rlF")


def test_see_also_search_vault_grep_uses_fixed_string_flag(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Knowledge base").mkdir(parents=True)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _search_vault(["Quantum Physics"], vault, "exclude-me")
        argv = mock_run.call_args[0][0]
        assert argv[0] == "grep"
        assert argv[1] == "-rlF"
