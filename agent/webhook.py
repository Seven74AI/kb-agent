"""Flask webhook — POST /submit and GET /dashboard.

Bearer token (`KB_API_TOKEN` / config `api_token`) required on mutating routes
and GET /dashboard. Flask runs sync in main thread; the asyncio consumer runs
in a background thread.
"""

import os
import secrets
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from agent.config import REPO_ROOT, Config, load_config
from agent.core import (
    compute_staleness,
    format_elapsed,
    list_tasks,
    plan_submission,
    resume_tasks,
    submit_plan,
)
from agent.core.resume import ResumeReasonError, normalize_reason_pattern
from agent.db import AGENT_SCHEMA, LOGS_SCHEMA, get_last_batch_run, list_ingest_queue, remove_ingest
from scripts.batch_ingest import refresh_parent_media_manifest
from agent.notify import sanitize_visible_message

app = Flask(__name__)
config: Config = load_config()

API_TOKEN_ENV = "KB_API_TOKEN"
_TOKEN_NOT_CONFIGURED_MSG = "KB_API_TOKEN not configured"
_PROTECTED_ENDPOINTS = frozenset({
    "submit",
    "resume",
    "upload_cookies",
    "remove_ingest_item",
    "dashboard",
})


def _config_yaml_path() -> Path:
    return config.kb_agent_dir / "config.yaml" if (config.kb_agent_dir / "config.yaml").exists() else REPO_ROOT / "config.yaml"


def resolve_api_token() -> str | None:
    """Shared API token: ``KB_API_TOKEN`` env overrides config.yaml ``api_token``."""
    env_val = os.environ.get(API_TOKEN_ENV)
    if env_val is not None:
        stripped = str(env_val).strip()
        return stripped or None
    path = _config_yaml_path()
    try:
        exists = path.exists()
    except (OSError, PermissionError):
        exists = False
    if not exists:
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    yaml_token = raw.get("api_token", "")
    if yaml_token is None:
        return None
    stripped = str(yaml_token).strip()
    return stripped or None


def extract_request_token(req) -> str | None:
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token or None
    api_key = req.headers.get("X-API-Key")
    if api_key:
        stripped = str(api_key).strip()
        return stripped or None
    return None


def api_token_configured() -> bool:
    return resolve_api_token() is not None


@app.before_request
def _require_api_token():
    endpoint = request.endpoint
    if endpoint not in _PROTECTED_ENDPOINTS:
        return None
    configured = resolve_api_token()
    if not configured:
        return jsonify({"error": _TOKEN_NOT_CONFIGURED_MSG}), 503
    provided = extract_request_token(request)
    if not provided or not secrets.compare_digest(provided, configured):
        return jsonify({"error": "Unauthorized"}), 401
    return None


# ── DB Helpers ────────────────────────────────────────────────────────────────

def _get_agent_conn() -> sqlite3.Connection:
    """Get a connection to agent.db (with WAL, row factory)."""
    db_path = _agent_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    # Ensure schema exists
    conn.executescript(AGENT_SCHEMA)
    return conn


def _get_logs_conn() -> sqlite3.Connection:
    """Get a connection to logs.db (with WAL, row factory)."""
    db_path = _logs_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(LOGS_SCHEMA)
    return conn


def _agent_db_path() -> Path:
    return config.kb_agent_dir / "data" / "agent.db"


def _logs_db_path() -> Path:
    return config.kb_agent_dir / "data" / "logs.db"


# Re-export for tests that import private helpers from webhook.
_format_elapsed = format_elapsed
_compute_staleness = compute_staleness


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Redirect to dashboard."""
    return redirect(url_for("dashboard"))


@app.route("/submit", methods=["POST"])
def submit():
    """Accept a URL for processing.

    Returns:
        201: Task created, queued for processing.
        200: Task skipped (already in vault or paused due to health check).
        400: Invalid request (no URL).
    """
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400

    plan = plan_submission(url, config)
    if plan["status"] in ("error", "skipped"):
        result = plan
    else:
        conn = _get_agent_conn()
        try:
            result = submit_plan(conn, plan)
        finally:
            conn.close()

    body = {k: v for k, v in result.items() if k not in ("http_status", "status")}
    body["status"] = result["status"]
    return jsonify(body), result["http_status"]


# ── Resume Endpoint ───────────────────────────────────────────────────────────

@app.route("/resume", methods=["POST"])
def resume():
    """Resume paused tasks matching a reason pattern.

    Query params:
        reason: Required. SQL LIKE pattern for the pause reason (e.g., 'cookies', 'firecrawl').

    Returns:
        JSON with count of resumed tasks, or 400 when reason is missing or empty.
    """
    reason_param = request.args.get("reason")
    if reason_param is None or not str(reason_param).strip():
        return jsonify({
            "error": "reason required (refusing to resume all paused tasks)",
        }), 400
    try:
        reason_pattern = normalize_reason_pattern(reason_param)
    except ResumeReasonError as exc:
        return jsonify({"error": str(exc)}), 400

    conn = _get_agent_conn()
    count = resume_tasks(conn, reason_param)
    conn.close()

    return jsonify({
        "status": "ok",
        "resumed": count,
        "reason_pattern": reason_pattern,
    })


# ── Cookie Upload Endpoint ────────────────────────────────────────────────────

ALLOWED_COOKIE_FILES = {
    "yt_cookies.txt": "YouTube / Instagram Reels (yt-dlp netscape format)",
    "ig_cookies.txt": "Instagram Posts (yt-dlp netscape format)",
    "threads_cookies.txt": "Threads (yt-dlp netscape format)",
    "substack_cookies.txt": "Substack (native video + paywalled posts)",
    "rumble_cookies.txt": "Rumble (yt-dlp netscape format)",
}


@app.route("/upload-cookies", methods=["POST"])
def upload_cookies():
    """Accept a cookie file upload and save it to the cookies directory.

    Expects multipart form with:
        cookie_type: one of 'yt_cookies.txt', 'ig_cookies.txt', 'threads_cookies.txt'
        cookie_file: the file upload

    Sets file permissions to 600 (read/write for owner only).
    """
    cookie_type = request.form.get("cookie_type", "").strip()
    if cookie_type not in ALLOWED_COOKIE_FILES:
        return jsonify({
            "error": f"Invalid cookie_type. Must be one of: {', '.join(ALLOWED_COOKIE_FILES.keys())}",
        }), 400

    uploaded_file = request.files.get("cookie_file")
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "cookie_file is required"}), 400

    # Ensure cookies directory exists
    cookie_dir = config.cookie_dir
    cookie_dir.mkdir(parents=True, exist_ok=True)

    # Save the file
    dest_path = cookie_dir / cookie_type
    uploaded_file.save(str(dest_path))

    # Set permissions: read/write for owner only (chmod 600)
    os.chmod(dest_path, stat.S_IRUSR | stat.S_IWUSR)

    # Verify
    file_size = dest_path.stat().st_size
    file_perms = oct(dest_path.stat().st_mode)[-3:]

    return jsonify({
        "status": "ok",
        "file": cookie_type,
        "size_bytes": file_size,
        "permissions": file_perms,
    }), 201


# ── Dashboard ─────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KB Agent — Dashboard{% if failure_count %} ⚠ {{ failure_count }}{% endif %}</title>
    <style>
        :root {
            --bg: #fafafa; --text: #222; --border: #ddd;
            --header-bg: #f0f0f0;
            --pending: #888; --in_progress: #0066cc; --done: #008000;
            --paused: #cc6600; --failed: #cc0000; --running: #0066cc;
        }
        body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; padding: 1rem; background: var(--bg); color: var(--text); }
        h1 { margin-bottom: 0.25rem; }
        .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
        .status-bar { display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
        .status-badge { padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem; }
        .pending { background: #eee; color: #555; }
        .in_progress { background: #cce5ff; color: #004085; }
        .done { background: #d4edda; color: #155724; }
        .paused { background: #fff3cd; color: #856404; }
        .failed { background: #f8d7da; color: #721c24; }
        .queued { background: #eee; color: #555; }
        .downloading { background: #cce5ff; color: #004085; }
        .staged { background: #d6d8f0; color: #3a3a6e; }
        .transcribing { background: #d6d8f0; color: #3a3a6e; }
        .dispatching { background: #cce5ff; color: #004085; }
        .dispatched { background: #d4edda; color: #155724; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }
        th, td { padding: 0.5rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
        th { background: var(--header-bg); font-weight: 600; position: sticky; top: 0; }
        tr:hover { background: #f5f5ff; }
        .url-cell { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .url-link { text-decoration: none; margin-left: 4px; }
        .step-list { font-size: 0.8rem; }
        .step-item { display: inline-block; margin: 1px 2px; padding: 1px 5px; border-radius: 3px; font-size: 0.75rem; }
        .step-pending { background: #f0f0f0; color: #888; }
        .step-running { background: #cce5ff; color: #004085; }
        .step-done { background: #d4edda; color: #155724; }
        .step-failed { background: #f8d7da; color: #721c24; }
        .step-paused { background: #fff3cd; color: #856404; }
        .log-line { font-size: 0.75rem; color: #666; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .staleness { font-size: 0.75rem; }
        .stale { color: #cc6600; font-weight: 600; }
        .fresh { color: #008000; }
        .actions { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center; }
        .actions form { display: inline; margin: 0; }
        .btn { padding: 0.35rem 0.8rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-size: 0.85rem; }
        .btn:hover { background: #f0f0f0; }
        .btn-resume { border-color: #cc6600; color: #cc6600; }
        .btn-resume:hover { background: #fff3cd; }
        .upload-section { border: 1px dashed #ccc; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }
        .upload-section h3 { margin-top: 0; }
        .refresh { color: #999; font-size: 0.85rem; margin-top: 1rem; }
        .empty-state { color: #888; font-style: italic; padding: 2rem 0; text-align: center; }
        .step-detail-row { background: #fafaff; }
        .step-detail-row td { font-size: 0.8rem; padding: 0.3rem 0.6rem 0.3rem 2rem; }
        .failure-banner { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <h1>KB Agent Dashboard</h1>
    <p class="subtitle">Cloudflare Access · Auto-refresh: 60s · {{ now }}</p>

    {% if failure_count %}
    <div class="failure-banner">⚠ {{ failure_count }} failure(s) — failed/paused tasks + errored ingest items below</div>
    {% endif %}

    <!-- Status counts -->
    <div class="status-bar">
        {% for status, label in [('pending','Pending'),('in_progress','In Progress'),
                                  ('done','Done'),('paused','Paused'),('failed','Failed')] %}
        <div class="status-badge {{ status }}" title="{{ status_help.get(status, '') }}">{{ label }}: {{ counts.get(status, 0) }}</div>
        {% endfor %}
        <div class="status-badge" style="background:#e8e8ff;color:#333;">Total: {{ counts.get('total', 0) }}</div>
    </div>

    <!-- Last batch run -->
    <p class="subtitle" style="margin-bottom:0.5rem;">Last batch run:
    {% if last_run %}
        <span class="status-badge {{ last_run.status }}" title="{{ status_help.get(last_run.status, '') }}">{{ last_run.status }}</span>
        {{ last_run.finished_at or last_run.started_at }}
        · {{ last_run.failed }} failed · {{ last_run.dispatched }} dispatched
        {% if last_run.summary %} · {{ last_run.summary }}{% endif %}
    {% else %}never{% endif %}
    </p>

    <!-- Actions: Resume buttons + Cookie upload -->
    <div class="actions">
        <span style="font-weight:600;">Quick Actions:</span>
        {% for reason, label in pause_reasons %}
        <form method="POST" action="/resume?reason={{ reason }}">
            <button type="submit" class="btn btn-resume">Resume: {{ label }}</button>
        </form>
        {% endfor %}
    </div>

    <!-- Cookie Upload -->
    <div class="upload-section">
        <h3>Upload Cookies</h3>
        <form method="POST" action="/upload-cookies" enctype="multipart/form-data">
            <select name="cookie_type" style="padding:0.3rem;margin-right:0.5rem;">
                {% for fname, desc in cookie_types.items() %}
                <option value="{{ fname }}">{{ fname }} — {{ desc }}</option>
                {% endfor %}
            </select>
            <input type="file" name="cookie_file" accept=".txt" style="margin-right:0.5rem;">
            <button type="submit" class="btn">Upload</button>
        </form>
    </div>

    <!-- Ingest Queue (pre-dispatch: batch collector) -->
    <h2>Ingest Queue</h2>
    {% if ingest %}
    <table>
        <thead>
            <tr>
                <th>Type</th>
                <th>URL</th>
                <th>Status</th>
                <th>Error</th>
                <th>Created</th>
                <th></th>
            </tr>
        </thead>
        <tbody>
            {% for q in ingest %}
            <tr>
                <td>{{ q.content_type }}</td>
                <td><div class="url-cell" title="{{ q.url }}">{{ q.url[:50] }}{% if q.url|length > 50 %}…{% endif %}</div>{% if q.url %}<a class="url-link" href="{{ q.url }}" target="_blank" rel="noopener" title="Open URL">🔗</a>{% endif %}</td>
                <td><span class="status-badge {{ q.status }}" title="{{ status_help.get(q.status, '') }}">{{ q.status }}</span></td>
                <td style="font-size:0.75rem;max-width:220px;overflow:hidden;text-overflow:ellipsis;" title="{{ q.error or '' }}">{{ q.error or '—' }}</td>
                <td style="font-size:0.8rem;">{{ q.created_at }}</td>
                <td><form method="POST" action="/remove-ingest" onsubmit="return confirm('Remove this ingest item?');"><input type="hidden" name="ingest_id" value="{{ q.id }}"><button type="submit" class="btn" style="font-size:0.75rem;padding:0.15rem 0.4rem;">Remove</button></form></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state"><p>No ingest queue items.</p></div>
    {% endif %}

    <!-- Tasks Table -->
    <h2>Tasks</h2>
    {% if tasks %}
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Type</th>
                <th>URL</th>
                <th>Status</th>
                <th>Steps</th>
                <th>Elapsed</th>
                <th>Staleness</th>
                <th>Last Log</th>
                <th>Error</th>
            </tr>
        </thead>
        <tbody>
            {% for task in tasks %}
            <tr>
                <td style="font-family:monospace;font-size:0.8rem;">{{ task.id[:8] }}</td>
                <td>{{ task.content_type }}</td>
                <td><div class="url-cell" title="{{ task.source_url }}">{{ task.source_url[:50] }}{% if task.source_url|length > 50 %}…{% endif %}</div>{% if task.source_url %}<a class="url-link" href="{{ task.source_url }}" target="_blank" rel="noopener" title="Open URL">🔗</a>{% endif %}</td>
                <td><span class="status-badge {{ task.status }}" title="{{ status_help.get(task.status, '') }}">{{ task.status }}</span></td>
                <td>
                    <div class="step-list">
                        {% for step in task.steps %}
                        <span class="step-item step-{{ step.status }}" title="{{ step.name }}: {{ status_help.get(step.status, '') }}">{{ step.name }}</span>
                        {% endfor %}
                    </div>
                </td>
                <td style="font-size:0.8rem;">{{ task.elapsed }}</td>
                <td>
                    {% if task.staleness %}
                    <span class="staleness {{ 'stale' if task.is_stale else 'fresh' }}">{{ task.staleness }}</span>
                    {% else %}
                    <span class="staleness">—</span>
                    {% endif %}
                </td>
                <td><div class="log-line" title="{{ task.last_log or '' }}">{{ task.last_log or '—' }}</div></td>
                <td style="font-size:0.75rem;max-width:150px;overflow:hidden;text-overflow:ellipsis;">{{ task.error or '' }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state">
        <p>No tasks yet.</p>
        <p>Submit a URL: <code>curl -X POST http://localhost:5000/submit -d 'url=https://example.com'</code></p>
    </div>
    {% endif %}

    <p class="refresh">Auto-refresh: <select id="polling-interval" onchange="setPolling(this.value)">
        <option value="5000">5s</option>
        <option value="30000">30s</option>
        <option value="60000" selected>1min</option>
    </select> · {{ now }}</p>
    <script>
    const POLL_KEY = 'kb-agent-polling-ms';
    const saved = localStorage.getItem(POLL_KEY) || '60000';
    document.getElementById('polling-interval').value = saved;
    let timer = null;

    function setPolling(val) {
        localStorage.setItem(POLL_KEY, val);
        clearTimeout(timer);
        scheduleRefresh(parseInt(val));
    }

    function scheduleRefresh(ms) {
        timer = setTimeout(() => location.reload(), ms);
    }

    // Start with saved preference
    scheduleRefresh(parseInt(saved));

    // Stop refresh when user interacts with forms to avoid data loss
    document.querySelectorAll('form').forEach(f => {
        f.addEventListener('submit', () => { clearTimeout(timer); });
    });
    </script>
</body>
</html>"""

# Known pause reason categories for the resume buttons
PAUSE_REASON_LABELS: list[tuple[str, str]] = [
    ("cookies", "Cookie Issues"),
    ("firecrawl", "Firecrawl Issues"),
    ("megadl", "Megadl Issues"),
]

# Status -> one-line explanation, shown as a hover tooltip on every status badge.
STATUS_HELP: dict[str, str] = {
    # task statuses
    "pending": "Queued — waiting for the consumer to claim it.",
    "in_progress": "Consumer is running this task's steps now.",
    "done": "Completed — note written to the vault.",
    "paused": "Blocked — awaiting manual resume (cookie/firecrawl/megadl fix).",
    "failed": "Failed — see the error column for the cause.",
    # ingest-queue statuses (pre-dispatch batch collector)
    "queued": "Awaiting the nightly batch collector (download + transcribe).",
    "downloading": "Batch collector is downloading the video now.",
    "staged": "Downloaded — staged for batch transcription.",
    "transcribing": "Batch transcription (Modal) is running.",
    "dispatching": "Transcribed — creating the synthesize task.",
    "dispatched": "Handed to the consumer as a synthesize task.",
    # batch-run statuses
    "running": "Currently running.",
    "ok": "Batch finished cleanly.",
    "crashed": "Batch process crashed before finishing.",
}


@app.route("/remove-ingest", methods=["POST"])
def remove_ingest_item():
    """Delete an ingest-queue row (escape hatch for permanently-broken links)."""
    ingest_id = (request.form.get("ingest_id") or "").strip()
    if not ingest_id:
        return jsonify({"error": "ingest_id required"}), 400
    conn = _get_agent_conn()
    parent_slug = remove_ingest(conn, ingest_id, temp_dir=config.temp_dir)
    if parent_slug is None:
        conn.close()
        return jsonify({"error": "not found"}), 404
    if parent_slug:
        refresh_parent_media_manifest(conn, config, parent_slug)
    conn.close()
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    """Show task queue status with step detail, elapsed time, and logs."""
    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)

    db_path = _agent_db_path()
    tasks: list[dict] = []
    ingest: list[dict] = []
    last_run: dict | None = None
    counts = {"pending": 0, "in_progress": 0, "done": 0, "paused": 0, "failed": 0, "total": 0}

    if db_path.exists():
        conn = _get_agent_conn()
        logs_conn = _get_logs_conn()
        state = list_tasks(conn, logs_conn=logs_conn, limit=100, now=now_naive)
        tasks = state["tasks"]
        counts = state["counts"]
        ingest = list_ingest_queue(conn, limit=100)
        last_run = get_last_batch_run(conn)
        conn.close()
        logs_conn.close()

    for q in ingest:
        if q.get("error"):
            q["error"] = sanitize_visible_message(q["error"])
    if last_run and last_run.get("summary"):
        last_run = {**last_run, "summary": sanitize_visible_message(last_run["summary"])}
    for task in tasks:
        if task.get("error"):
            task["error"] = sanitize_visible_message(task["error"])

    # Failure-first ordering + a single failure count for the banner/tab title.
    bad = {"failed", "paused"}
    tasks.sort(key=lambda t: 0 if t["status"] in bad else 1)
    ingest.sort(key=lambda q: 0 if (q.get("error") or q["status"] in bad) else 1)
    failure_count = (
        counts.get("failed", 0)
        + counts.get("paused", 0)
        + sum(1 for q in ingest if q.get("error") or q["status"] in bad)
    )

    return render_template_string(
        DASHBOARD_HTML,
        tasks=tasks,
        counts=counts,
        ingest=ingest,
        last_run=last_run,
        failure_count=failure_count,
        now=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        pause_reasons=PAUSE_REASON_LABELS,
        cookie_types=ALLOWED_COOKIE_FILES,
        status_help=STATUS_HELP,
    )


if __name__ == "__main__":
    app.run(host=config.webhook_host, port=config.webhook_port, debug=False)
