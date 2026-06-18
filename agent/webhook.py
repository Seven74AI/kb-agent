"""Flask webhook — POST /submit and GET /dashboard.

Tailscale-only, no auth. Flask runs sync in main thread;
the asyncio consumer runs in a background thread.
"""

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from agent.config import Config, load_config
from agent.detection import dedup_check, detect_content_subtype, detect_platform, preflight_health
from agent.db import create_task, init_db, AGENT_SCHEMA


app = Flask(__name__)
config: Config = load_config()

# Pipeline step definitions per content type
PIPELINE_STEPS: dict[str, list[str]] = {
    "web": [
        "firecrawl_extract", "synthesize_note", "quality_gate",
        "minio_upload", "git_push",
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
        "minio_upload", "git_push",
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
    # Take the last meaningful path segment or domain+path
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[?#].*", "", slug)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug[:100]  # Truncate to avoid path length issues


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
        return jsonify({
            "status": "paused",
            "reason": reason,
            "platform": platform,
        }), 200

    # 4. Detect content subtype
    subtype = detect_content_subtype(url, platform)

    # 5. Get pipeline steps for this content type
    steps = PIPELINE_STEPS.get(platform, PIPELINE_STEPS["web"])

    # 6. Create task in DB
    db_path = config.kb_agent_dir / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = slugify(url)
    task_id = create_task(conn, url, platform, slug, steps)

    return jsonify({
        "status": "pending",
        "task_id": task_id,
        "platform": platform,
        "content_type": subtype,
        "slug": slug,
    }), 201


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KB Agent — Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 1rem; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
        th, td { padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; }
        .pending { color: #666; }  .in_progress { color: #0066cc; }
        .done { color: #008000; }  .paused { color: #cc6600; }
        .failed { color: #cc0000; }
        .refresh { color: #999; font-size: 0.85rem; margin-top: 1rem; }
    </style>
</head>
<body>
    <h1>KB Agent Dashboard</h1>
    <p>Auto-refreshes every 60s. Last update: <span id="time">{{ now }}</span></p>

    <h2>Tasks</h2>
    <table>
        <thead><tr><th>ID</th><th>URL</th><th>Type</th><th>Status</th><th>Error</th><th>Created</th></tr></thead>
        <tbody>
            {% for task in tasks %}
            <tr>
                <td>{{ task.id[:8] }}...</td>
                <td style="max-width:300px;overflow:hidden">{{ task.source_url[:60] }}</td>
                <td>{{ task.content_type }}</td>
                <td class="{{ task.status }}">{{ task.status }}</td>
                <td>{{ task.error or '' }}</td>
                <td>{{ task.created_at }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% if not tasks %}
    <p>No tasks yet. Submit a URL via <code>POST /submit</code>.</p>
    {% endif %}

    <p class="refresh">Auto-refresh: 60s</p>
    <script>setTimeout(() => location.reload(), 60000);</script>
</body>
</html>"""


@app.route("/dashboard")
def dashboard():
    """Show task queue status and step detail."""
    import sqlite3
    db_path = config.kb_agent_dir / "agent.db"
    if not db_path.exists():
        return render_template_string(DASHBOARD_HTML, tasks=[], now=datetime.now().isoformat())

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, source_url, content_type, status, error, created_at "
        "FROM tasks ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    tasks = [dict(r) for r in rows]
    conn.close()

    return render_template_string(DASHBOARD_HTML, tasks=tasks, now=datetime.now().isoformat())


if __name__ == "__main__":
    app.run(host=config.webhook_host, port=config.webhook_port, debug=False)
