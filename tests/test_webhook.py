"""Tests for webhook.py — Flask routes and dashboard rendering."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

# Add kb-agent to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.core import slugify
from agent.webhook import app, _format_elapsed, _compute_staleness
from agent.db import (
    AGENT_SCHEMA, LOGS_SCHEMA,
    create_task, init_db, resume_tasks_by_reason,
    get_task_steps, get_all_tasks, get_last_log_line,
    pause_task, complete_task, fail_task, queue_ingest,
)


@pytest.fixture(autouse=True)
def _kb_api_token(monkeypatch):
    """Most webhook routes require a configured API token."""
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")


@pytest.fixture
def client():
    """Flask test client (sends bearer token on every request)."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_AUTHORIZATION"] = "Bearer test-api-token"
        yield c


@pytest.fixture
def temp_dbs(tmp_path):
    """Create temporary agent.db and logs.db for testing."""
    agent_db = tmp_path / "data" / "agent.db"
    logs_db = tmp_path / "data" / "logs.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)

    agent_conn = init_db(agent_db, AGENT_SCHEMA)
    agent_conn.row_factory = sqlite3.Row
    logs_conn = init_db(logs_db, LOGS_SCHEMA)
    logs_conn.row_factory = sqlite3.Row

    yield agent_conn, logs_conn

    agent_conn.close()
    logs_conn.close()


# ── slugify tests ─────────────────────────────────────────────────────────────

def test_slugify_basic():
    assert slugify("https://example.com/foo/bar") == "example.com_foo_bar"


def test_slugify_removes_query():
    s = slugify("https://example.com/page?utm=tracking#frag")
    assert "?" not in s
    assert "#" not in s


def test_slugify_truncates():
    long_url = "https://example.com/" + "a" * 200
    result = slugify(long_url)
    assert len(result) <= 128


# ── _format_elapsed tests ─────────────────────────────────────────────────────

def test_format_elapsed_seconds():
    from datetime import datetime, timezone
    created = "2026-06-19T10:00:00"
    now = datetime(2026, 6, 19, 10, 0, 30)
    assert _format_elapsed(created, now) == "30s"


def test_format_elapsed_minutes():
    from datetime import datetime
    created = "2026-06-19T10:00:00"
    now = datetime(2026, 6, 19, 10, 5, 0)
    assert _format_elapsed(created, now) == "5m"


def test_format_elapsed_hours():
    from datetime import datetime
    created = "2026-06-19T10:00:00"
    now = datetime(2026, 6, 19, 12, 30, 0)
    assert "2h" in _format_elapsed(created, now)
    assert "30m" in _format_elapsed(created, now)


def test_format_elapsed_days():
    from datetime import datetime
    created = "2026-06-17T10:00:00"
    now = datetime(2026, 6, 19, 10, 0, 0)
    assert "2d" in _format_elapsed(created, now)


# ── DB helper tests ───────────────────────────────────────────────────────────

def test_resume_tasks_by_reason(temp_dbs):
    agent_conn, _ = temp_dbs

    # Create a paused task
    task_id = create_task(agent_conn, "https://youtube.com/watch?v=abc", "youtube",
                          "test_slug", ["step1", "step2"], status="paused")
    agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?",
                       ("cookies invalid — LOGIN_INFO missing", task_id))
    agent_conn.commit()

    step_id = agent_conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? ORDER BY step_order LIMIT 1",
        (task_id,),
    ).fetchone()[0]
    agent_conn.execute(
        "UPDATE task_steps SET status = 'paused', output = 'cookies invalid' WHERE id = ?",
        (step_id,),
    )
    agent_conn.commit()

    # Resume by cookies reason
    count = resume_tasks_by_reason(agent_conn, "%cookies%")
    assert count == 1

    # Verify task is now pending
    row = agent_conn.execute("SELECT status, error, retries FROM tasks WHERE id = ?",
                             (task_id,)).fetchone()
    assert row[0] == "pending"
    assert row[1] is None
    assert row[2] == 0
    step_row = agent_conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?", (step_id,),
    ).fetchone()
    assert step_row[0] == "pending"
    assert step_row[1] is None


def test_resume_tasks_by_reason_no_match(temp_dbs):
    agent_conn, _ = temp_dbs

    task_id = create_task(agent_conn, "https://example.com", "web",
                          "test_slug", ["step1"], status="paused")
    agent_conn.execute("UPDATE tasks SET error = ? WHERE id = ?",
                       ("cookies invalid", task_id))
    agent_conn.commit()

    # Resume by non-matching reason
    count = resume_tasks_by_reason(agent_conn, "%firecrawl%")
    assert count == 0

    # Task still paused
    row = agent_conn.execute("SELECT status FROM tasks WHERE id = ?",
                             (task_id,)).fetchone()
    assert row[0] == "paused"


def test_get_task_steps(temp_dbs):
    agent_conn, _ = temp_dbs

    task_id = create_task(agent_conn, "https://example.com", "web",
                          "test_slug", ["extract", "synthesize", "quality"])
    steps = get_task_steps(agent_conn, task_id)
    assert len(steps) == 3
    assert steps[0]["name"] == "extract"
    assert steps[1]["name"] == "synthesize"
    assert steps[2]["name"] == "quality"
    assert all(s["status"] == "pending" for s in steps)


def test_get_all_tasks(temp_dbs):
    agent_conn, _ = temp_dbs

    create_task(agent_conn, "https://a.com", "web", "a", ["s1"])
    create_task(agent_conn, "https://b.com", "youtube", "b", ["s1", "s2"])
    create_task(agent_conn, "https://c.com", "substack", "c", ["s1"])

    tasks = get_all_tasks(agent_conn, limit=10)
    assert len(tasks) == 3
    # Verify all three URLs are present (order depends on identical timestamps)
    urls = {t["source_url"] for t in tasks}
    assert urls == {"https://a.com", "https://b.com", "https://c.com"}


def test_get_last_log_line(temp_dbs):
    _, logs_conn = temp_dbs

    # Insert a log line
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content) VALUES (?, ?, ?, ?)",
        ("test-task-123", "firecrawl_extract", "stdout", "Extracted 5000 chars"),
    )
    logs_conn.commit()

    line = get_last_log_line(logs_conn, "test-task-123")
    assert line == "Extracted 5000 chars"


def test_get_last_log_line_none(temp_dbs):
    _, logs_conn = temp_dbs
    line = get_last_log_line(logs_conn, "nonexistent-task")
    assert line is None


# ── Webhook route tests ───────────────────────────────────────────────────────

def test_get_index_redirects(client):
    """GET / should redirect to /dashboard."""
    resp = client.get("/")
    assert resp.status_code in (302, 301)
    assert "/dashboard" in resp.headers.get("Location", "")


def test_post_submit_no_url(client):
    """POST /submit without URL returns 400."""
    resp = client.post("/submit", data={})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data


def test_post_submit_web_url(client, monkeypatch, tmp_path):
    """POST /submit with a web URL returns 201 with task_id."""
    # Override config paths for testing
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    # Ensure vault path doesn't exist (no dedup match)
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)

    # Mock preflight_health — Firecrawl not reachable in the test env
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    resp = client.post("/submit", data={"url": "https://example.com/article"})
    assert resp.status_code == 201
    data = json.loads(resp.data)
    assert data["status"] == "pending"
    assert "task_id" in data
    assert data["platform"] == "web"


def test_post_submit_youtube_no_cookies(client, monkeypatch, tmp_path):
    """POST /submit with YouTube URL and no cookies returns paused (200)."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)

    # Mock preflight_health to return unhealthy for youtube
    def mock_preflight(platform):
        if platform == "youtube":
            return False, "cookies invalid — no yt_cookies.txt found"
        return True, ""
    monkeypatch.setattr("agent.core.submit.preflight_health", mock_preflight)

    resp = client.post("/submit", data={"url": "https://youtube.com/watch?v=abc123"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "paused"
    assert "cookies" in data["reason"].lower()


def test_post_submit_with_cookies(client, monkeypatch, tmp_path):
    """POST /submit with YouTube URL and valid cookies returns 201."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)

    # Mock preflight_health to return healthy
    monkeypatch.setattr("agent.core.submit.preflight_health",
                        lambda p: (True, ""))

    resp = client.post("/submit", data={"url": "https://youtube.com/watch?v=abc123"})
    assert resp.status_code == 201
    data = json.loads(resp.data)
    assert data["status"] == "pending"
    assert data["platform"] == "youtube"


def test_post_submit_dedup_skip_opens_no_connection(client, monkeypatch, tmp_path):
    """Dedup skip must not open a DB connection (regression fix for PR #35)."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    vault = tmp_path / "vault"
    (vault / "Knowledge base").mkdir(parents=True)
    (vault / "Knowledge base" / "dup.md").write_text(
        "---\nsource_url: https://example.com/dup\n---\n", encoding="utf-8"
    )
    monkeypatch.setattr("agent.webhook.config.vault_path", vault)

    def boom():
        raise AssertionError("_get_agent_conn() must not be called on dedup skip")
    monkeypatch.setattr("agent.webhook._get_agent_conn", boom)

    resp = client.post("/submit", data={"url": "https://example.com/dup"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "skipped"
    assert data["reason"] == "Already in vault"


def test_post_submit_closes_connection(client, monkeypatch, tmp_path):
    """/submit must close the agent connection it opens."""
    import agent.webhook as webhook
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    original_get_conn = webhook._get_agent_conn

    class TrackingConn:
        def __init__(self, real):
            self._real = real
            self.closed = False
        def __getattr__(self, name):
            return getattr(self._real, name)
        def __enter__(self):
            self._real.__enter__()
            return self
        def __exit__(self, *args):
            return self._real.__exit__(*args)
        def close(self):
            self.closed = True
            self._real.close()

    conns = []
    def spy_get_conn():
        conn = TrackingConn(original_get_conn())
        conns.append(conn)
        return conn
    monkeypatch.setattr("agent.webhook._get_agent_conn", spy_get_conn)

    resp = client.post("/submit", data={"url": "https://example.com/close-me"})
    assert resp.status_code == 201
    assert len(conns) == 1
    assert conns[0].closed is True


def test_post_resume_by_reason(client, monkeypatch, tmp_path):
    """POST /resume?reason=cookies resumes matching paused tasks."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)

    # Create a paused task with cookies error
    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)

    task_id = create_task(conn, "https://youtube.com/watch?v=xyz", "youtube",
                          "test_slug", ["step1"], status="paused")
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?",
                 ("cookies invalid", task_id))
    conn.commit()
    conn.close()

    # Resume
    resp = client.post("/resume?reason=cookies")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"
    assert data["resumed"] == 1




def test_upload_cookies_success(client, monkeypatch, tmp_path):
    """POST /upload-cookies saves file with 600 permissions."""
    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    cookie_content = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tLOGIN_INFO\tabc\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tSAPISID\tdef\n"
    )
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
        assert data["status"] == "ok"
        assert data["file"] == "yt_cookies.txt"
        assert data["size_bytes"] == len(cookie_content)
        assert data["permissions"] == "600"

        # Verify file exists on disk with correct content and permissions
        dest = cookie_dir / "yt_cookies.txt"
        assert dest.exists()
        assert dest.read_text() == cookie_content
        assert oct(dest.stat().st_mode)[-3:] == "600"
    finally:
        os.unlink(tmp.name)


def test_upload_cookies_invalid_type(client):
    """POST /upload-cookies with invalid type returns 400."""
    resp = client.post("/upload-cookies", data={"cookie_type": "invalid.txt"})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data


def test_upload_cookies_rejects_empty_file(client, monkeypatch, tmp_path):
    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
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
        assert resp.status_code == 400
        assert "empty" in json.loads(resp.data)["error"].lower()
        assert not (cookie_dir / "yt_cookies.txt").exists()
    finally:
        os.unlink(tmp.name)


def test_upload_cookies_rejects_non_netscape_content(client, monkeypatch, tmp_path):
    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(b"LOGIN_INFO\nnot-a-netscape-file\n")
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
        assert resp.status_code == 400
        assert "Netscape" in json.loads(resp.data)["error"]
    finally:
        os.unlink(tmp.name)


def test_upload_cookies_rejects_unsafe_cookie_value(client, monkeypatch, tmp_path):
    """Cookie values with quotes or newlines must fail closed at upload."""
    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    cookie_content = (
        "# Netscape HTTP Cookie File\n"
        ".instagram.com\tTRUE\t/\tTRUE\t0\tsessionid\tbad\"inject\n"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(cookie_content.encode())
    tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            resp = client.post(
                "/upload-cookies",
                data={
                    "cookie_type": "ig_cookies.txt",
                    "cookie_file": (f, "ig_cookies.txt"),
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        assert "disallowed" in json.loads(resp.data)["error"].lower()
        assert not (cookie_dir / "ig_cookies.txt").exists()
    finally:
        os.unlink(tmp.name)


def test_upload_cookies_rejects_oversized_payload(client, monkeypatch, tmp_path):
    from agent.webhook import MAX_COOKIE_UPLOAD_BYTES

    cookie_dir = tmp_path / "cookies"
    monkeypatch.setattr("agent.webhook.config.cookie_dir", cookie_dir)

    header = b"# Netscape HTTP Cookie File\n"
    payload = header + (b"x" * (MAX_COOKIE_UPLOAD_BYTES - len(header) + 1))
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(payload)
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
        assert resp.status_code == 400
        assert "byte limit" in json.loads(resp.data)["error"].lower()
    finally:
        os.unlink(tmp.name)


def test_dashboard_renders_html(client, monkeypatch, tmp_path):
    """GET /dashboard returns HTML with status table."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)

    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)

    # Create a few tasks
    create_task(conn, "https://example.com/1", "web", "slug_1", ["extract", "synthesize"])
    create_task(conn, "https://youtube.com/watch?v=vid", "youtube", "vid_slug",
                ["cookie_validate", "yt_dlp_download"], status="paused")
    conn.execute("UPDATE tasks SET error = ? WHERE source_url = ?",
                 ("cookies missing", "https://youtube.com/watch?v=vid"))
    conn.commit()
    conn.close()

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "KB Agent Dashboard" in html
    assert "example.com" in html
    assert "youtube" in html


def test_dashboard_scrubs_task_error(client, monkeypatch, tmp_path):
    """Task errors on the dashboard omit host filesystem paths."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)

    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)
    secret = "/root/kb-agent/secret/task.py"
    create_task(conn, "https://example.com/fail", "web", "fail_slug", ["extract"], status="failed")
    conn.execute(
        "UPDATE tasks SET error = ? WHERE slug = ?",
        (f'extract died at "{secret}"', "fail_slug"),
    )
    conn.commit()
    conn.close()

    resp = client.get("/dashboard")
    html = resp.data.decode()
    assert secret not in html
    assert "extract died" in html


def test_dashboard_scrubs_ingest_error(client, monkeypatch, tmp_path):
    """Ingest queue errors shown on the dashboard omit host filesystem paths."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)

    agent_db = tmp_path / "data" / "agent.db"
    agent_db.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(agent_db, AGENT_SCHEMA)
    secret = "/root/kb-agent/secret/foo.py"
    ingest_id, _ = queue_ingest(conn, "https://example.com/img.png", "substack_image", "img1")
    conn.execute(
        "UPDATE ingest_queue SET status='failed', error=? WHERE id=?",
        (f'batch failed at "{secret}"', ingest_id),
    )
    conn.commit()
    conn.close()

    resp = client.get("/dashboard")
    html = resp.data.decode()
    assert secret not in html
    assert "batch failed" in html


def test_dashboard_empty(client, monkeypatch, tmp_path):
    """GET /dashboard with no tasks shows empty state."""
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    # Don't create any DB — or create empty
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "No tasks yet" in html or "Total: 0" in html


# ── Staleness computation tests ───────────────────────────────────────────────

def test_compute_staleness_recent(temp_dbs):
    """Recent log line should show 'just now' or seconds ago."""
    from datetime import datetime, timezone, timedelta
    agent_conn, logs_conn = temp_dbs

    # Insert a very recent log
    now = datetime.now(timezone.utc)
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, content, logged_at) VALUES (?, ?, ?, ?)",
        ("task-1", "extract", "done",
         (now - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")),
    )
    logs_conn.commit()

    staleness, is_stale = _compute_staleness(
        {"id": "task-1"}, logs_conn, now.replace(tzinfo=None)
    )
    assert staleness is not None
    assert not is_stale  # Recent = not stale


def test_compute_staleness_old(temp_dbs):
    """Old log line should be stale."""
    from datetime import datetime, timezone, timedelta
    agent_conn, logs_conn = temp_dbs

    now = datetime.now(timezone.utc)
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, content, logged_at) VALUES (?, ?, ?, ?)",
        ("task-2", "extract", "done",
         (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")),
    )
    logs_conn.commit()

    staleness, is_stale = _compute_staleness(
        {"id": "task-2"}, logs_conn, now.replace(tzinfo=None)
    )
    assert staleness is not None
    assert is_stale


def test_compute_staleness_no_logs(temp_dbs):
    """Task with no logs returns None staleness."""
    agent_conn, logs_conn = temp_dbs
    staleness, is_stale = _compute_staleness(
        {"id": "no-logs"}, logs_conn, None
    )
    assert staleness is None
    assert not is_stale


# ── API token auth ───────────────────────────────────────────────────────────


def test_post_submit_returns_401_without_token(monkeypatch):
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")
    bare = app.test_client()
    resp = bare.post("/submit", data={"url": "https://example.com/x"})
    assert resp.status_code == 401
    data = json.loads(resp.data)
    assert data.get("error") == "Unauthorized"


def test_post_submit_returns_503_when_token_unset(client, monkeypatch):
    monkeypatch.delenv("KB_API_TOKEN", raising=False)
    bare = app.test_client()
    bare.environ_base["HTTP_AUTHORIZATION"] = "Bearer anything"
    resp = bare.post("/submit", data={"url": "https://example.com/x"})
    assert resp.status_code == 503
    data = json.loads(resp.data)
    assert "KB_API_TOKEN not configured" in data.get("error", "")


def test_post_submit_ok_with_bearer_token(client, monkeypatch, tmp_path):
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    resp = client.post("/submit", data={"url": "https://example.com/authed"})
    assert resp.status_code == 201


def test_post_submit_ok_with_x_api_key_header(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_API_TOKEN", "test-api-token")
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.setattr("agent.webhook.config.vault_path", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_X_API_KEY"] = "test-api-token"
        resp = c.post("/submit", data={"url": "https://example.com/x-api-key"})
    assert resp.status_code == 201


def test_dashboard_accessible_without_token(monkeypatch, tmp_path):
    # GET /dashboard is gated by Cloudflare Access (email), not the API token —
    # a browser cannot send a bearer header, so the token must not apply here.
    monkeypatch.setattr("agent.webhook.config.kb_agent_dir", tmp_path)
    monkeypatch.delenv("KB_API_TOKEN", raising=False)
    bare = app.test_client()
    resp = bare.get("/dashboard")
    assert resp.status_code == 200
