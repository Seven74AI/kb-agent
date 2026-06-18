"""Flask webhook — POST /submit and GET /dashboard.

Tailscale-only, no auth. Flask runs sync in main thread;
the asyncio consumer runs in a background thread.
"""

import os
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from agent.config import Config, load_config
from agent.db import (
    AGENT_SCHEMA, LOGS_SCHEMA,
    create_task, init_db, resume_tasks_by_reason,
    get_task_steps, get_all_tasks, get_last_log_line,
)
from agent.detection import dedup_check, detect_content_subtype, detect_platform, preflight_health

app = Flask(__name__)
config: Config = load_config()

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


# ── Pipeline step definitions per content type ────────────────────────────────

PIPELINE_STEPS: dict[str, list[str]] = {
    "web": [
        "firecrawl_extract", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "youtube": [
        "cookie_validate", "yt_dlp_download", "ffmpeg_extract_audio",
        "diarize", "transcribe", "synthesize_note",
        "quality_gate", "minio_upload", "git_push",
    ],
    "instagram_reel": [
        "cookie_validate", "yt_dlp_download", "ffmpeg_extract_audio",
        "diarize", "transcribe", "synthesize_note",
        "quality_gate", "minio_upload", "git_push",
    ],
    "instagram_post": [
        "cookie_validate", "extract_images", "synthesize_note",
        "quality_gate", "minio_upload", "git_push",
    ],
    "threads": [
        "cookie_validate", "extract_content", "synthesize_note",
        "quality_gate", "minio_upload", "git_push",
    ],
    "substack": [
        "firecrawl_extract", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "book": [
        "extract_text", "synthesize_note", "quality_gate",
        "minio_upload", "git_push",
    ],
    "mega": [
        "megadl_download", "classify_content", "delegate",
    ],
}


def slugify(url: str) -> str:
    """Derive a filesystem-safe slug from a URL."""
    import re
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[:100]


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

    # 1. Detect platform
    platform = detect_platform(url)

    # 2. Dedup check
    if dedup_check(url, config.vault_path):
        return jsonify({
            "status": "skipped",
            "reason": "Already in vault",
            "platform": platform,
        }), 200

    # 3. Pre-flight health check
    ok, reason = preflight_health(platform)
    if not ok:
        # Create paused task
        conn = _get_agent_conn()
        slug = slugify(url)
        steps = PIPELINE_STEPS.get(platform, PIPELINE_STEPS["web"])
        task_id = create_task(conn, url, platform, slug, steps, status="paused")
        # Set error reason on the paused task
        conn.execute(
            "UPDATE tasks SET error = ? WHERE id = ?", (reason, task_id)
        )
        conn.commit()
        conn.close()
        return jsonify({
            "status": "paused",
            "reason": reason,
            "platform": platform,
            "task_id": task_id,
        }), 200

    # 4. Detect content subtype
    subtype = detect_content_subtype(url, platform)

    # 5. Get pipeline steps for this content type
    steps = PIPELINE_STEPS.get(platform, PIPELINE_STEPS["web"])

    # 6. Create task in DB
    conn = _get_agent_conn()
    slug = slugify(url)
    task_id = create_task(conn, url, platform, slug, steps)

    return jsonify({
        "status": "pending",
        "task_id": task_id,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
    }), 201


# ── Resume Endpoint ───────────────────────────────────────────────────────────

@app.route("/resume", methods=["POST"])
def resume():
    """Resume paused tasks matching a reason pattern.

    Query params:
        reason: SQL LIKE pattern for the pause reason (e.g., 'cookies', 'firecrawl').
                Default: '%' (resume all paused).

    Returns:
        JSON with count of resumed tasks.
    """
    reason_param = request.args.get("reason", "%").strip()
    if not reason_param:
        reason_param = "%"

    # Convert simple keywords to LIKE patterns
    if reason_param != "%" and "%" not in reason_param:
        reason_param = f"%{reason_param}%"

    conn = _get_agent_conn()
    count = resume_tasks_by_reason(conn, reason_param)
    conn.close()

    return jsonify({
        "status": "ok",
        "resumed": count,
        "reason_pattern": reason_param,
    })


# ── Cookie Upload Endpoint ────────────────────────────────────────────────────

ALLOWED_COOKIE_FILES = {
    "yt_cookies.txt": "YouTube / Instagram Reels (yt-dlp netscape format)",
    "ig_cookies.txt": "Instagram Posts (yt-dlp netscape format)",
    "threads_cookies.txt": "Threads (yt-dlp netscape format)",
}


@app.route("/upload-cookies", methods=["POST"])
def upload_cookies():
    """Accept a cookie file upload and save it to the cookies directory.

    Expects multipart form with:
        cookie_type: one of 'yt_cookies.txt', 'ig_cookies.txt', 'threads_cookies.txt'
        cookie_file: the file upload

    Sets file permissions to 444 (read-only for all).
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

    # Set permissions: read-only for owner, group, others (chmod 444)
    os.chmod(dest_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

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
    <title>KB Agent — Dashboard</title>
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
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }
        th, td { padding: 0.5rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
        th { background: var(--header-bg); font-weight: 600; position: sticky; top: 0; }
        tr:hover { background: #f5f5ff; }
        .url-cell { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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
    </style>
</head>
<body>
    <h1>KB Agent Dashboard</h1>
    <p class="subtitle">Tailscale-only · Auto-refresh: 60s · {{ now }}</p>

    <!-- Status counts -->
    <div class="status-bar">
        {% for status, label in [('pending','Pending'),('in_progress','In Progress'),
                                  ('done','Done'),('paused','Paused'),('failed','Failed')] %}
        <div class="status-badge {{ status }}">{{ label }}: {{ counts.get(status, 0) }}</div>
        {% endfor %}
        <div class="status-badge" style="background:#e8e8ff;color:#333;">Total: {{ counts.get('total', 0) }}</div>
    </div>

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
                <td><div class="url-cell" title="{{ task.source_url }}">{{ task.source_url[:50] }}{% if task.source_url|length > 50 %}…{% endif %}</div></td>
                <td><span class="status-badge {{ task.status }}">{{ task.status }}</span></td>
                <td>
                    <div class="step-list">
                        {% for step in task.steps %}
                        <span class="step-item step-{{ step.status }}">{{ step.name }}</span>
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

    <p class="refresh">Auto-refresh: 60s · {{ now }}</p>
    <script>setTimeout(() => location.reload(), 60000);</script>
</body>
</html>"""

# Known pause reason categories for the resume buttons
PAUSE_REASON_LABELS: list[tuple[str, str]] = [
    ("cookies", "Cookie Issues"),
    ("firecrawl", "Firecrawl Issues"),
    ("megadl", "Megadl Issues"),
]


def _format_elapsed(created_at: str, now: datetime) -> str:
    """Format elapsed time from ISO timestamp to human-readable."""
    try:
        if "T" in created_at:
            created_dt = datetime.fromisoformat(created_at)
        else:
            created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        delta = now - created_dt

        if delta.total_seconds() < 0:
            return "just now"

        total_secs = int(delta.total_seconds())
        if total_secs < 60:
            return f"{total_secs}s"
        elif total_secs < 3600:
            return f"{total_secs // 60}m"
        elif total_secs < 86400:
            return f"{total_secs // 3600}h {int((total_secs % 3600) / 60)}m"
        else:
            return f"{total_secs // 86400}d {int((total_secs % 86400) / 3600)}h"
    except (ValueError, TypeError):
        return "?"


def _compute_staleness(task: dict, logs_conn: sqlite3.Connection, now: datetime) -> tuple[str | None, bool]:
    """Compute staleness string from the most recent log timestamp.

    Returns (staleness_str, is_stale) where is_stale=True if >2h since last update.
    """
    if logs_conn is None:
        return None, False

    row = logs_conn.execute(
        "SELECT logged_at FROM step_logs WHERE task_id = ? "
        "ORDER BY logged_at DESC LIMIT 1",
        (task["id"],),
    ).fetchone()

    if not row:
        return None, False

    try:
        logged_at = row[0]
        if "T" in logged_at:
            log_dt = datetime.fromisoformat(logged_at)
        else:
            log_dt = datetime.strptime(logged_at, "%Y-%m-%d %H:%M:%S")
        delta = now - log_dt
        total_secs = int(delta.total_seconds())

        if total_secs < 0:
            return "just now", False
        elif total_secs < 60:
            return f"{total_secs}s ago", False
        elif total_secs < 3600:
            return f"{total_secs // 60}m ago", False
        elif total_secs < 7200:
            return f"{total_secs // 3600}h ago", False
        elif total_secs < 86400:
            return f"{total_secs // 3600}h ago", True
        else:
            return f"{total_secs // 86400}d ago", True
    except (ValueError, TypeError):
        return None, False


@app.route("/dashboard")
def dashboard():
    """Show task queue status with step detail, elapsed time, and logs."""
    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)

    db_path = _agent_db_path()
    tasks = []
    counts = {"pending": 0, "in_progress": 0, "done": 0, "paused": 0, "failed": 0, "total": 0}

    if db_path.exists():
        conn = _get_agent_conn()
        logs_conn = _get_logs_conn()

        raw_tasks = get_all_tasks(conn, limit=100)
        counts["total"] = len(raw_tasks)

        for t in raw_tasks:
            status = t["status"]
            counts[status] = counts.get(status, 0) + 1

            # Get steps for this task
            steps = get_task_steps(conn, t["id"])

            # Get last log line
            last_log = get_last_log_line(logs_conn, t["id"])

            # Compute elapsed
            elapsed = _format_elapsed(t["created_at"], now_naive)

            # Compute staleness
            staleness_str, is_stale = _compute_staleness(t, logs_conn, now_naive)

            tasks.append({
                "id": t["id"],
                "source_url": t["source_url"],
                "content_type": t["content_type"],
                "status": status,
                "error": t["error"],
                "slug": t.get("slug"),
                "retries": t.get("retries", 0),
                "created_at": t["created_at"],
                "updated_at": t.get("updated_at"),
                "steps": steps,
                "elapsed": elapsed,
                "staleness": staleness_str,
                "is_stale": is_stale,
                "last_log": last_log,
            })

        conn.close()
        logs_conn.close()

    return render_template_string(
        DASHBOARD_HTML,
        tasks=tasks,
        counts=counts,
        now=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        pause_reasons=PAUSE_REASON_LABELS,
        cookie_types=ALLOWED_COOKIE_FILES,
    )


if __name__ == "__main__":
    app.run(host=config.webhook_host, port=config.webhook_port, debug=False)
