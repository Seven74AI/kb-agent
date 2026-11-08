"""Tests for agent.core — pure domain layer (no Flask)."""

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import Config
from agent.core.slugify import SLUG_MAX_LEN, batch_standalone_video_storage_slug, slug_path
from agent.pipelines.video_common import resolve_artifact_stem
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
from agent.core.bulk import PlaylistExpandError, expand_youtube_playlist
from agent.core.resume import ResumeReasonError
from agent.core.submit import plan_submission, submit_ingest, submit_plan
from agent.detection import ContentSubtypeDetection
from agent.db import (
    AGENT_SCHEMA,
    LOGS_SCHEMA,
    archive_llm_call,
    create_task,
    init_db,
    pause_task,
)
from agent.detection import DedupCheckError

METADATA_URL = "http://169.254.169.254/latest/meta-data/"
PRIVATE_HOST_URL = "http://127.0.0.1/playlist?list=PLfake"


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


def test_batch_standalone_video_storage_slug_substack_uses_ingest_id():
    shared = "author.substack.com_p_post"
    assert batch_standalone_video_storage_slug(
        ingest_id="ing-a",
        slug=shared,
        content_type="substack_video",
        parent_slug="",
    ) == "ing-a"
    assert batch_standalone_video_storage_slug(
        ingest_id="ing-b",
        slug=shared,
        content_type="substack_video",
        parent_slug=None,
    ) == "ing-b"
    assert batch_standalone_video_storage_slug(
        ingest_id="ing-emb",
        slug=shared,
        content_type="substack_video",
        parent_slug="parent_slug",
    ) == shared
    assert batch_standalone_video_storage_slug(
        ingest_id="ing-yt",
        slug="youtube.com_watch_v_abc",
        content_type="youtube",
        parent_slug="",
    ) == "youtube.com_watch_v_abc"


def test_resolve_artifact_stem_batch_uses_ingest_id_else_filesystem_slug():
    tid = "ingest-batch-uuid"
    task_batch = {"id": tid, "slug": "foo/bar", "_batch_ingest": True}
    assert resolve_artifact_stem(task_batch) == tid
    task = {"id": tid, "slug": "foo/bar?"}
    assert resolve_artifact_stem(task) == "foo_bar_"


def test_slug_path_sanitizes_and_caps_at_slug_max_len():
    assert slug_path("test?query=1") == "test_query_1"
    long_raw = "a" * 200 + "?x=1"
    assert len(slug_path(long_raw)) == SLUG_MAX_LEN
    expected = re.sub(r'[<>:"/\\|?*=&]', "_", long_raw)[:SLUG_MAX_LEN]
    assert slug_path(long_raw) == expected


def test_slugify_basic():
    assert slugify("https://example.com/foo/bar") == "example.com_foo_bar"


def test_slugify_strips_netloc_userinfo():
    """INVARIANT: credential userinfo never enters slug or derived durable keys."""
    cred_url = "https://user:sekret@example.com/path"
    slug = slugify(cred_url)
    assert "user" not in slug
    assert "sekret" not in slug
    assert slug == slugify("https://example.com/path")


def test_slugify_credential_free_url_unchanged():
    plain = "https://example.com/foo/bar"
    assert slugify(plain) == "example.com_foo_bar"


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


def test_get_pipeline_names_unknown_returns_none():
    from agent.pipelines import get_pipeline_names

    assert get_pipeline_names("nonexistent_typo_xyz", "also_missing") is None


def test_bare_substack_pipeline_matches_substack_text_embedded_media():
    """Bare content_type=substack (no subtype detect) must still run embedded media enqueue."""
    from agent.pipelines import get_pipeline_names

    substack = get_pipeline_names("substack", "")
    substack_text = get_pipeline_names("substack_text", "")
    assert substack is not None
    assert "substack_embedded_media" in substack
    assert substack == substack_text


def test_slugify_instagram_comment_preserves_comment_id():
    url = "https://www.instagram.com/reel/ABC123/?comment_id=999888777"
    a = slugify(url, content_type="instagram_comment")
    b = slugify(
        "https://www.instagram.com/reel/ABC123/?comment_id=111222333",
        content_type="instagram_comment",
    )
    assert a.endswith("_c_999888777")
    assert b.endswith("_c_111222333")
    assert a != b


REEL_BASE = "https://www.instagram.com/reel/ABC123/"


def test_submit_ingest_instagram_comment_slugs_differ(core_config, agent_conn):
    """INVARIANT: slug derives from the same content_type used for routing."""
    url_a = f"{REEL_BASE}?comment_id=999888777"
    url_b = f"{REEL_BASE}?comment_id=111222333"
    out_a = submit_ingest(url_a, core_config, agent_conn)
    out_b = submit_ingest(url_b, core_config, agent_conn)
    assert out_a["content_type"] == "instagram_comment"
    assert out_b["content_type"] == "instagram_comment"
    assert out_a["slug"] != out_b["slug"]
    assert out_a["slug"].endswith("_c_999888777")
    assert out_b["slug"].endswith("_c_111222333")


def test_submit_ingest_instagram_reel_slug_unchanged(core_config, agent_conn):
    out = submit_ingest(REEL_BASE, core_config, agent_conn)
    assert out["content_type"] == "instagram_reel"
    assert "_c_" not in out["slug"]
    assert out["slug"] == slugify(REEL_BASE, content_type="instagram_reel")


def test_plan_submission_and_submit_ingest_agree_on_slug(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    url = f"{REEL_BASE}?comment_id=42"
    plan = plan_submission(url, core_config)
    assert plan["status"] == "pending"
    ingest = submit_ingest(url, core_config, agent_conn)
    assert ingest["slug"] == plan["slug"]
    assert ingest["content_type"] == plan["content_type"]


def test_submit_ingest_returns_actual_status_for_exhausted_failed_duplicate(
    core_config, agent_conn,
):
    """INVARIANT: embed-success accounting must not treat dead URL duplicates as queued."""
    from agent.db import INGEST_MAX_MEDIA_ATTEMPTS, queue_ingest

    url = "https://substack-post-media.s3.amazonaws.com/public/images/dead_dup.webp"
    parent = "parent_dead_dup"
    queue_ingest(
        agent_conn, url, "substack_image", "img_dead",
        parent_slug=parent, media_kind="image",
    )
    agent_conn.execute(
        "UPDATE ingest_queue SET status='failed', error='gave up', attempts=? WHERE url=?",
        (INGEST_MAX_MEDIA_ATTEMPTS, url),
    )
    agent_conn.commit()

    out = submit_ingest(
        url, core_config, agent_conn, content_type="substack_image",
        parent_slug=parent, media_kind="image",
    )
    assert out["status"] == "failed"
    assert out["duplicate"] is True
    assert out["created_new"] is False
    assert out["ingest_id"]


def test_submit_ingest_revived_failed_duplicate_reports_queued(core_config, agent_conn):
    from agent.db import queue_ingest
    from agent.pipelines.web import _submit_embed_succeeded

    url = "https://substack-post-media.s3.amazonaws.com/public/images/revive_dup.webp"
    parent = "parent_revive_dup"
    queue_ingest(
        agent_conn, url, "substack_image", "img_revive",
        parent_slug=parent, media_kind="image",
    )
    agent_conn.execute(
        "UPDATE ingest_queue SET status='failed', error='ocr failed', attempts=1 WHERE url=?",
        (url,),
    )
    agent_conn.commit()

    out = submit_ingest(
        url, core_config, agent_conn, content_type="substack_image",
        parent_slug=parent, media_kind="image",
    )
    assert out["status"] == "queued"
    assert out["duplicate"] is True
    assert out["created_new"] is False
    assert _submit_embed_succeeded(out) is True


def test_plan_submission_normalizes_url_before_dedup(core_config, monkeypatch):
    """Dedup greps the display-safe source_url persisted in vault frontmatter."""
    from agent.redact import url_for_display

    tracked = "https://www.instagram.com/reel/AbCd/?igsh=noise&utm_campaign=x"
    note = core_config.vault_path / "Knowledge base" / "articles" / "existing.md"
    note.write_text(
        f"---\nsource_url: {url_for_display(tracked)}\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    result = plan_submission(tracked, core_config)
    assert result["status"] == "skipped"


def test_plan_submission_rejects_javascript_url(core_config):
    result = plan_submission("javascript:alert(1)", core_config)
    assert result["status"] == "error"
    assert result["http_status"] == 400
    assert "http or https" in result["error"].lower()


def test_plan_submission_unknown_content_type_not_enqueued(core_config, monkeypatch):
    monkeypatch.setattr(
        "agent.core.submit.detect_content_subtype_with_signals",
        lambda url, platform: ContentSubtypeDetection(subtype="unknown"),
    )
    result = plan_submission("https://mega.nz/folder/x", core_config)
    assert result["status"] == "error"
    assert result["http_status"] == 400
    assert "unknown" in result["error"].lower()


def test_plan_submission_unregistered_pipeline_error(core_config, monkeypatch):
    monkeypatch.setattr(
        "agent.core.submit.detect_content_subtype_with_signals",
        lambda url, platform: ContentSubtypeDetection(subtype="nonexistent_typo_xyz"),
    )
    monkeypatch.setattr("agent.core.submit.detect_platform", lambda url: "not_a_pipeline")
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    result = plan_submission("https://example.com/x", core_config)
    assert result["status"] == "error"
    assert result["http_status"] == 400


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


def test_detect_type_normalizes_tracking_params_like_submit():
    noisy = (
        "https://www.instagram.com/reel/ShortCode1/"
        "?igsh=track&igsi=track&comment_id=42"
    )
    clean = "https://www.instagram.com/reel/ShortCode1/?comment_id=42"
    assert detect_type(noisy) == detect_type(clean)
    assert detect_type(clean)["content_type"] == "instagram_comment"


def test_detect_type_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="http or https"):
        detect_type("javascript:alert(1)")


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


@pytest.mark.parametrize(
    "surface,invoke",
    [
        (
            "plan_submission",
            lambda cfg, conn, mp: plan_submission("https://example.com/x", cfg),
        ),
        (
            "submit_url",
            lambda cfg, conn, mp: submit_url("https://example.com/x", cfg, conn),
        ),
    ],
)
def test_submission_dedup_error_scrubs_paths(
    core_config, agent_conn, monkeypatch, surface, invoke
):
    """HTTP/MCP clients must not see vault paths from dedup grep stderr."""

    def boom(url, vault_path):
        raise DedupCheckError(
            "Vault dedup grep failed (exit 2): grep: /root/Documents/Obsidian Vault/x: denied"
        )

    monkeypatch.setattr("agent.core.submit.dedup_check", boom)
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    result = invoke(core_config, agent_conn, monkeypatch)
    assert result["status"] == "error"
    assert "/root/" not in result["error"]
    assert "dedup check failed" in result["error"]


def test_submit_url_dedup_grep_failure_does_not_enqueue(core_config, agent_conn, monkeypatch):
    """Vault grep errors surface as structured errors without creating tasks."""
    import subprocess

    def fail_grep(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="permission denied")

    monkeypatch.setattr("agent.detection.subprocess.run", fail_grep)
    before = agent_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    result = submit_url("https://example.com/new", core_config, agent_conn)
    after = agent_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert after == before
    assert result["status"] == "error"
    assert result["http_status"] == 500
    assert "dedup check failed" in result["error"]


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


@patch("agent.core.bulk.subprocess.run")
def test_expand_youtube_playlist_rejects_metadata_url(mock_run):
    with pytest.raises(ValueError, match="not allowed"):
        expand_youtube_playlist(METADATA_URL)
    mock_run.assert_not_called()


@patch("agent.core.bulk.subprocess.run")
def test_expand_youtube_playlist_rejects_private_host_url(mock_run):
    with pytest.raises(ValueError, match="not allowed"):
        expand_youtube_playlist(PRIVATE_HOST_URL)
    mock_run.assert_not_called()


@patch("agent.core.bulk.subprocess.run")
def test_expand_youtube_playlist_raises_on_nonzero_exit(mock_run):
    import subprocess

    mock_run.return_value = subprocess.CompletedProcess(
        [], 1, stdout="", stderr="Sign in to confirm you're not a bot"
    )
    with pytest.raises(PlaylistExpandError, match="exit 1"):
        expand_youtube_playlist("https://youtube.com/playlist?list=PLtest")


def test_ingest_playlist_surfaces_expand_failure(monkeypatch, core_config, agent_conn):
    def fail(_url):
        raise PlaylistExpandError("playlist expansion timed out")

    monkeypatch.setattr("agent.core.bulk.expand_youtube_playlist", fail)
    result = ingest_playlist("https://youtube.com/playlist?list=PLx", core_config, agent_conn)
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert "no videos found" not in result["error"]


def test_ingest_playlist_sanitizes_expand_error_detail(monkeypatch, core_config, agent_conn):
    leak = (
        "playlist expansion failed (exit 1): ERROR: cookie at /root/kb-agent/cookies/youtube.txt "
        "api_key=deadbeef"
    )

    def fail(_url):
        raise PlaylistExpandError(leak)

    monkeypatch.setattr("agent.core.bulk.expand_youtube_playlist", fail)
    result = ingest_playlist("https://youtube.com/playlist?list=PLx", core_config, agent_conn)
    assert result["status"] == "error"
    assert "deadbeef" not in result["error"]
    assert "/root/kb-agent" not in result["error"]
    assert "[credential redacted]" in result["error"]
    assert "[path]" in result["error"]


def test_submit_url_health_pause(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr(
        "agent.core.submit.preflight_health",
        lambda p: (False, "cookies invalid"),
    )
    result = submit_url("https://youtube.com/watch?v=abc", core_config, agent_conn)
    assert result["status"] == "paused"
    assert result["http_status"] == 200
    assert result["task_id"]


def test_submit_plan_pause_reason_scrubbed_in_task_row(core_config, agent_conn, monkeypatch):
    monkeypatch.setattr(
        "agent.core.submit.preflight_health",
        lambda p: (False, "missing cookies at /root/kb-agent/cookies/youtube.txt"),
    )
    plan = plan_submission("https://youtube.com/watch?v=abc", core_config)
    assert plan["status"] == "paused"
    assert "/root/kb-agent/cookies/youtube.txt" in (plan["reason"] or "")
    result = submit_plan(agent_conn, plan)
    row = agent_conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (result["task_id"],)
    ).fetchone()
    assert "/root/kb-agent/cookies/youtube.txt" in (row["error"] or "")
    assert "/root/" not in (result["reason"] or "")


def test_archive_llm_call_redacts_paths_and_secrets_at_insert(logs_conn, monkeypatch):
    monkeypatch.delenv("KB_LLM_ARCHIVE_RAW", raising=False)
    archive_llm_call(
        logs_conn,
        "task-1",
        "synthesis",
        "read /root/secret/vault/note.md\napi_key=supersecret\n"
        "SOURCE URL: https://cdn.example.com/v.mp4?token=archive-secret",
        "Bearer eyJhbG.token\n/home/user/out.txt",
        "test-model",
        tokens_in=1,
        tokens_out=2,
    )
    row = logs_conn.execute(
        "SELECT prompt, response FROM llm_archives WHERE task_id = ?", ("task-1",)
    ).fetchone()
    assert "/root/" not in row["prompt"]
    assert "supersecret" not in row["prompt"]
    assert "[credential redacted]" in row["prompt"]
    assert "token=archive-secret" not in row["prompt"]
    assert "https://cdn.example.com/v.mp4" in row["prompt"]
    assert "eyJhbG" not in row["response"]
    assert "[redacted]" in row["response"]
    assert "/home/" not in row["response"]


def test_archive_llm_call_store_raw_opt_in(logs_conn, monkeypatch):
    monkeypatch.setenv("KB_LLM_ARCHIVE_RAW", "1")
    raw_prompt = "token=abc123 at /root/kb-agent/data"
    archive_llm_call(
        logs_conn,
        "task-raw",
        "synthesis",
        raw_prompt,
        "ok",
        "test-model",
    )
    row = logs_conn.execute(
        "SELECT prompt FROM llm_archives WHERE task_id = ?", ("task-raw",)
    ).fetchone()
    assert row["prompt"] == raw_prompt


def test_list_tasks_counts_completed_toward_done(core_config, agent_conn, logs_conn):
    create_task(
        agent_conn,
        "https://a.com",
        "web",
        "done_slug",
        ["extract"],
        status="done",
    )
    create_task(
        agent_conn,
        "https://b.com",
        "web",
        "completed_slug",
        ["extract"],
        status="completed",
    )
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    assert listed["counts"]["done"] == 2
    assert listed["counts"].get("completed", 0) == 0


def test_list_tasks_and_get_task(core_config, agent_conn, logs_conn):
    task_id = create_task(agent_conn, "https://a.com", "web", "a_slug", ["extract"])
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    assert listed["counts"]["total"] == 1
    assert len(listed["tasks"]) == 1

    one = get_task(agent_conn, task_id, logs_conn=logs_conn)
    assert one is not None
    assert one["id"] == task_id
    assert one["steps"][0]["name"] == "extract"


def test_list_tasks_sanitizes_last_log(core_config, agent_conn, logs_conn):
    task_id = create_task(agent_conn, "https://a.com", "web", "a_slug", ["extract"])
    secret = "/root/kb-agent/private/run.log"
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content) VALUES (?, ?, ?, ?)",
        (task_id, "extract", "stderr", f"failed reading {secret}: {'e' * 400}"),
    )
    logs_conn.commit()
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    last_log = listed["tasks"][0]["last_log"]
    assert secret not in last_log
    assert len(last_log) < 500


def test_list_tasks_sanitizes_error_and_step_output(core_config, agent_conn, logs_conn):
    secret = "/root/kb-agent/secret/pipeline.py"
    task_id = create_task(agent_conn, "https://a.com", "web", "a_slug", ["extract"])
    agent_conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (f'broke at "{secret}"', task_id),
    )
    agent_conn.execute(
        "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'extract'",
        (f'stderr from {secret}', task_id),
    )
    agent_conn.commit()
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    task = listed["tasks"][0]
    assert secret not in (task["error"] or "")
    assert secret not in (task["steps"][0]["output"] or "")
    one = get_task(agent_conn, task_id, logs_conn=logs_conn)
    assert secret not in (one["error"] or "")
    assert secret not in (one["steps"][0]["output"] or "")


def test_list_tasks_redacts_tokenized_urls_on_stale_rows(core_config, agent_conn, logs_conn):
    """Read-path defense: legacy rows with query-bearing URLs are redacted on display."""
    tokenized = "https://cdn.example.com/video.mp4?token=SECRET&expires=999"
    task_id = create_task(agent_conn, "https://a.com", "web", "stale_slug", ["extract"])
    agent_conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (f"fetch failed: {tokenized}", task_id),
    )
    agent_conn.execute(
        "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'extract'",
        (f'{{"description": "see {tokenized}"}}', task_id),
    )
    logs_conn.execute(
        "INSERT INTO step_logs (task_id, step_name, stream, content) VALUES (?, ?, ?, ?)",
        (task_id, "extract", "stderr", f"stderr: {tokenized}"),
    )
    agent_conn.commit()
    logs_conn.commit()
    listed = list_tasks(agent_conn, logs_conn=logs_conn, limit=10)
    task = listed["tasks"][0]
    assert "token=SECRET" not in (task["error"] or "")
    assert "token=SECRET" not in (task["steps"][0]["output"] or "")
    assert "token=SECRET" not in (task["last_log"] or "")
    assert "?" not in (task["error"] or "")


def test_get_task_missing(agent_conn):
    assert get_task(agent_conn, "nonexistent-id") is None


def test_normalize_reason_pattern():
    with pytest.raises(ResumeReasonError):
        normalize_reason_pattern("")
    with pytest.raises(ResumeReasonError):
        normalize_reason_pattern("%")
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
