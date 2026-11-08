"""Tests for ingest queue dedup/tag updates and schema migration."""

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.core.resume import retry_task
from agent.core.submit import dispatch_synthesize
from agent.config import Config
from agent.db import (
    AGENT_SCHEMA,
    INGEST_MAX_MEDIA_ATTEMPTS,
    _ACTIVE_SYNTHESIZE_SLUG_INDEX,
    _migrate,
    claim_task,
    count_pending_media,
    create_task,
    find_active_task,
    has_active_synthesize_task,
    has_completed_parent_synthesis,
    has_completed_synthesize_task,
    has_synthesize_task_signal,
    get_next_step,
    get_task_steps,
    init_db,
    is_deferred_media_pause,
    task_in_deferred_media_cycle,
    queue_ingest,
    recover_failed_media,
    recover_stale_ingest,
    resume_tasks_by_reason,
    serialize_tags,
    complete_task,
    fail_task,
    pause_task,
    supersede_stale_deferral_synthesize_tasks,
    update_step,
)


def _stamp_parent_raw_media(conn, temp_dir: Path, parent_slug: str) -> None:
    """Write append markers for every sibling row (dispatch gate requires on-disk bodies)."""
    from agent.db import media_append_marker

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id FROM ingest_queue WHERE parent_slug = ?",
        (parent_slug,),
    ).fetchall()
    parts = ["# Parent article\n\n"]
    for row in rows:
        ingest_id = row["id"]
        parts.append(
            f"\n\n## Media\n\nembedded\n\n{media_append_marker(ingest_id)}\n"
        )
    raw = temp_dir / f"{parent_slug}_raw.md"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("".join(parts), encoding="utf-8")
    conn.execute(
        "UPDATE ingest_queue SET media_appended = 1 WHERE parent_slug = ?",
        (parent_slug,),
    )
    conn.commit()


def test_queue_ingest_duplicate_updates_tags(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    url = "https://example.com/a"
    first_id, created = queue_ingest(conn, url, "web", "slug_a", tags=["alpha"])
    assert created is True

    second_id, created = queue_ingest(conn, url, "web", "slug_a", tags=["beta", "gamma"])
    assert created is False
    assert second_id == first_id

    row = conn.execute("SELECT tags FROM ingest_queue WHERE url = ?", (url,)).fetchone()
    assert row["tags"] == serialize_tags(["beta", "gamma"])


def test_queue_ingest_duplicate_empty_tags_unchanged(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://example.com/b"
    queue_ingest(conn, url, "web", "slug_b", tags=["keep"])
    queue_ingest(conn, url, "web", "slug_b", tags=None)
    row = conn.execute("SELECT tags FROM ingest_queue WHERE url = ?", (url,)).fetchone()
    assert row["tags"] == serialize_tags(["keep"])


def test_migrate_adds_missing_ingest_queue_columns(tmp_path):
    db_path = tmp_path / "data" / "agent.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE ingest_queue (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            content_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued'
        )"""
    )
    conn.commit()

    _migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ingest_queue)").fetchall()}
    for required in ("slug", "tags", "error", "created_at", "dispatched_at", "parent_slug", "media_kind", "attempts"):
        assert required in cols, f"missing column {required}"

    conn.execute(
        "INSERT INTO ingest_queue (id, url, content_type, slug) VALUES (?, ?, ?, ?)",
        ("x", "https://example.com/x", "web", "x_slug"),
    )
    conn.commit()


def test_queue_ingest_parent_linkage(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    queue_ingest(
        conn, "https://substack-post-media.s3.amazonaws.com/public/images/a_800x600.png",
        "substack_image", "a_800x600", parent_slug="article_slug", media_kind="image",
    )
    row = conn.execute(
        "SELECT parent_slug, media_kind FROM ingest_queue WHERE parent_slug = ?",
        ("article_slug",),
    ).fetchone()
    assert row["parent_slug"] == "article_slug"
    assert row["media_kind"] == "image"


def test_count_pending_media(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    queue_ingest(conn, "https://example.com/i1", "substack_image", "i1", parent_slug="p", media_kind="image")
    queue_ingest(conn, "https://example.com/i2", "substack_image", "i2", parent_slug="p", media_kind="image")
    queue_ingest(conn, "https://example.com/v1", "youtube", "v1", parent_slug="p", media_kind="video")
    queue_ingest(conn, "https://example.com/other", "web", "other")  # no parent

    assert count_pending_media(conn, "p") == 3

    # mark two done → 1 remaining
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE url IN ('https://example.com/i1', 'https://example.com/v1')")
    conn.commit()
    assert count_pending_media(conn, "p") == 1

    # a failed sibling still blocks synthesis (must be retried to success)
    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url='https://example.com/i2'")
    conn.commit()
    assert count_pending_media(conn, "p") == 1


def test_recover_failed_media_requeues_up_to_max_attempts(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    queue_ingest(conn, "https://example.com/i1", "substack_image", "i1", parent_slug="p", media_kind="image")
    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url='https://example.com/i1'")
    conn.commit()

    # retry 1
    assert recover_failed_media(conn, max_attempts=3) == 1
    row = conn.execute("SELECT status, attempts FROM ingest_queue WHERE url='https://example.com/i1'").fetchone()
    assert row["status"] == "queued"
    assert row["attempts"] == 1

    # fail again, retry 2, fail again, retry 3
    for _ in range(2):
        conn.execute("UPDATE ingest_queue SET status='failed' WHERE url='https://example.com/i1'")
        conn.commit()
        assert recover_failed_media(conn, max_attempts=3) == 1
    row = conn.execute("SELECT attempts FROM ingest_queue WHERE url='https://example.com/i1'").fetchone()
    assert row["attempts"] == 3

    # 4th failure stays failed (attempts exhausted) — no more requeue
    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url='https://example.com/i1'")
    conn.commit()
    assert recover_failed_media(conn, max_attempts=3) == 0
    row = conn.execute("SELECT status FROM ingest_queue WHERE url='https://example.com/i1'").fetchone()
    assert row["status"] == "failed"


def test_queue_ingest_same_url_different_parent_creates_two_rows(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://substack-post-media.s3.amazonaws.com/public/images/shared.png"

    id_a, created_a = queue_ingest(
        conn, url, "substack_image", "img_a", parent_slug="article_a", media_kind="image",
    )
    id_b, created_b = queue_ingest(
        conn, url, "substack_image", "img_b", parent_slug="article_b", media_kind="image",
    )
    assert created_a and created_b
    assert id_a != id_b
    assert conn.execute("SELECT COUNT(*) FROM ingest_queue WHERE url = ?", (url,)).fetchone()[0] == 2


@pytest.mark.parametrize(
    "pause_reason",
    [
        "awaiting media: 2 images enqueued",
        "awaiting substack cookies — paywalled/login HTML without article body",
        "deferred to batch synthesis — parent must not inline synthesize",
        "batch synthesis dispatched or completed — parent must not re-enter synthesis",
        "media gate: ingest rows 1 != manifest total 2",
    ],
)
def test_resume_deferred_media_denies_bulk_resume(tmp_path, pause_reason):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.commit()
    assert is_deferred_media_pause(pause_reason)
    assert resume_tasks_by_reason(conn, f"%{pause_reason[:12]}%") == 0
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "paused"


def test_get_next_step_blocks_synthesize_on_substack_paywall_pause(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = (
        "awaiting substack cookies — paywalled/login HTML without article body"
    )
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    defer_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, defer_id),
    )
    conn.commit()

    assert is_deferred_media_pause(pause_reason)
    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "substack_embedded_media"
    assert nxt["status"] == "paused"


def test_get_next_step_blocks_on_failed_embed_before_pending_synthesize(tmp_path):
    """Failed media-gate steps are barriers, not only paused ones (deferral retry preserve_embed)."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="in_progress",
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'failed', output = 'embed boom' WHERE id = ?",
        (embed_id,),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'pending' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "substack_embedded_media"
    assert nxt["status"] == "failed"


def test_get_next_step_blocks_pending_synthesize_when_embedded_media_paused(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("awaiting media: 2 images enqueued", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    defer_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = 'defer' WHERE id = ?",
        (defer_id,),
    )
    conn.commit()

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "substack_embedded_media"
    assert nxt["status"] == "paused"


@pytest.mark.parametrize(
    "pause_reason",
    [
        "deferred to batch synthesis — parent must not inline synthesize",
        "batch synthesis dispatched or completed — parent must not re-enter synthesis",
    ],
)
def test_get_next_step_does_not_skip_paused_synthesize_deferral(tmp_path, pause_reason):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="pending",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' "
        "WHERE task_id = ? AND name IN ('substack_cleanup', 'substack_embedded_media')",
        (task_id,),
    )
    synth_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, synth_id),
    )
    conn.commit()

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "synthesize_note"
    assert nxt["status"] == "paused"


@pytest.mark.parametrize(
    "pause_reason",
    [
        "deferred synthesis paused — sibling media still pending",
        "deferred synthesis paused — parent media not complete at execute time",
        "deferred synthesis dispatch failed: unknown content_type",
        "unsupported embed: instagram",
        "unsupported embed: tiktok",
    ],
)
def test_is_deferred_media_pause_execute_gate_synthesis(pause_reason):
    assert is_deferred_media_pause(pause_reason)


def test_task_in_deferred_media_cycle_unsupported_embed(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "unsupported embed: instagram"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.commit()
    assert task_in_deferred_media_cycle(conn, task_id, error=pause_reason)


def test_task_in_deferred_media_cycle_substack_cleanup_marker_only(tmp_path):
    """Durable deferral on ``substack_cleanup`` without row/error/embed output still in-cycle."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "cleanup_marker_only"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/marker-only",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute("UPDATE tasks SET error = NULL WHERE id = ?", (task_id,))
    conn.execute(
        """
        UPDATE task_steps SET status = 'completed', output = 'enqueued prior'
        WHERE task_id = ? AND name = 'substack_cleanup'
        """,
        (task_id,),
    )
    conn.execute(
        """
        UPDATE task_steps SET output = ?
        WHERE task_id = ? AND name = 'substack_cleanup'
        """,
        (f"cleaned\nmedia_deferred:{slug}", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = 'clean' WHERE id = ?",
        (embed_id,),
    )
    conn.commit()

    assert task_in_deferred_media_cycle(conn, task_id, slug=slug, error=None)


def test_retry_task_preserves_embed_when_only_cleanup_deferral_marker(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "retry_cleanup_marker_parent"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/retry-marker",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "prior enqueue complete"
    conn.execute("UPDATE tasks SET error = NULL WHERE id = ?", (task_id,))
    conn.execute(
        """
        UPDATE task_steps SET status = 'completed', output = ?
        WHERE task_id = ? AND name = 'substack_cleanup'
        """,
        (f"cleaned\nmedia_deferred:{slug}", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, embed_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'running' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()

    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (embed_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["synthesize_note"] == "pending"


def test_resume_unsupported_embed_denies_bulk_resume(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "unsupported embed: instagram"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.commit()
    assert resume_tasks_by_reason(conn, "%unsupported%") == 0
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "paused"


def test_is_deferred_media_pause_media_gate_mismatch():
    reason = "media gate: ingest rows 1 != manifest total 2"
    assert is_deferred_media_pause(reason)


def test_get_next_step_blocks_later_pending_on_non_deferred_paused_step(tmp_path):
    """Paused steps are barriers for every pause reason, not only deferred media."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "firecrawl rate limit — retry after cooldown"
    task_id = create_task(
        conn,
        "https://example.com/article",
        "web",
        "web_slug",
        ["fetch", "clean", "synthesize_note"],
        status="pending",
    )
    fetch_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'fetch'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, fetch_id),
    )
    conn.commit()
    assert not is_deferred_media_pause(pause_reason)

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "fetch"
    assert nxt["status"] == "paused"


def test_resume_tasks_by_reason_resets_paused_steps_before_later_pending(tmp_path):
    """Bulk resume must not leave task pending while an earlier step stays paused."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "firecrawl rate limit — retry after cooldown"
    task_id = create_task(
        conn,
        "https://example.com/article",
        "web",
        "web_slug",
        ["fetch", "clean", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    fetch_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'fetch'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, fetch_id),
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, "%firecrawl%") == 1
    task_status = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert task_status[0] == "pending"
    assert task_status[1] is None
    fetch_status = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?", (fetch_id,),
    ).fetchone()
    assert fetch_status[0] == "pending"
    assert fetch_status[1] is None

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "fetch"
    assert nxt["status"] == "pending"
    clean = conn.execute(
        "SELECT status FROM task_steps WHERE task_id = ? AND name = 'clean'",
        (task_id,),
    ).fetchone()[0]
    assert clean == "pending"


def test_get_next_step_blocks_on_media_gate_pause(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "media gate: ingest rows 1 != manifest total 2"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note", "quality_gate"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    defer_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, defer_id),
    )
    conn.commit()

    assert is_deferred_media_pause(pause_reason)
    assert resume_tasks_by_reason(conn, "%media gate%") == 0
    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "substack_embedded_media"
    assert nxt["status"] == "paused"


def test_bootstrap_parent_media_deferral_siblings_no_spurious_media_changed(tmp_path):
    from agent.db import (
        bootstrap_parent_media_deferral_from_siblings,
        get_parent_media_deferral,
    )
    import uuid

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "aligned_parent"
    url = "https://blog.substack.com/p/aligned"
    create_task(
        conn,
        url,
        "substack_text",
        parent,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    for i in range(2):
        conn.execute(
            """
            INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                f"https://example.com/{i}.png",
                "substack_image",
                f"img{i}",
                "queued",
                parent,
                "image",
            ),
        )
    manifest = tmp_path / f"{parent}_media.json"
    manifest.write_text(
        json.dumps({"total": 2, "url": url, "content_type": "substack_text"}),
        encoding="utf-8",
    )
    conn.commit()

    bootstrap_parent_media_deferral_from_siblings(conn, parent, tmp_path)
    deferral = get_parent_media_deferral(conn, parent)
    assert deferral is not None
    assert deferral["expected_total"] == 2
    assert deferral["media_changed"] == 0


def test_bootstrap_parent_media_deferral_high_water_enqueues_shortfall(tmp_path):
    from agent.db import (
        bootstrap_parent_media_deferral_from_siblings,
        get_parent_media_deferral,
        parent_deferral_synthesize_execute_blocked,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "hw_parent"
    url = "https://blog.substack.com/p/high-water"
    task_id = create_task(
        conn,
        url,
        "substack_text",
        parent,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        """
        UPDATE task_steps SET status = 'paused', output = ?
        WHERE task_id = ? AND name = 'substack_embedded_media'
        """,
        ("awaiting media: 2 images + 0 videos enqueued", task_id),
    )
    import uuid

    conn.execute(
        """
        INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            "https://example.com/one.png",
            "substack_image",
            "img1",
            "queued",
            parent,
            "image",
        ),
    )
    conn.commit()

    bootstrap_parent_media_deferral_from_siblings(conn, parent, tmp_path)
    deferral = get_parent_media_deferral(conn, parent)
    assert deferral is not None
    assert deferral["expected_total"] == 2
    assert deferral["media_changed"] == 1

    conn.execute(
        "UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()
    raw = tmp_path / f"{parent}_raw.md"
    raw.write_text("# parent\n", encoding="utf-8")
    reason = parent_deferral_synthesize_execute_blocked(conn, parent, raw_path=raw)
    assert reason == "deferred synthesis paused — parent media not complete at execute time"


def test_queue_ingest_bumps_parent_deferral_expected_without_manifest(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "queue_growth_parent"
    url = "https://blog.substack.com/p/queue-growth"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn,
        "https://example.com/i1.png",
        "substack_image",
        "i1",
        parent_slug=parent,
        media_kind="image",
    )
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 1

    queue_ingest(
        conn,
        "https://example.com/i2.png",
        "substack_image",
        "i2",
        parent_slug=parent,
        media_kind="image",
    )
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 2


def test_parent_deferral_synthesize_blocked_without_deferral_row_pending_siblings(
    tmp_path,
):
    from agent.db import parent_deferral_synthesize_execute_blocked

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "defer_missing_parent"
    queue_ingest(
        conn,
        "https://example.com/img.png",
        "substack_image",
        "img1",
        parent_slug=parent,
        media_kind="image",
        status="queued",
    )
    conn.commit()
    raw = tmp_path / f"{parent}_raw.md"
    raw.write_text("# parent\n", encoding="utf-8")
    reason = parent_deferral_synthesize_execute_blocked(
        conn, parent, raw_path=raw,
    )
    assert reason == "deferred synthesis paused — sibling media still pending"


def test_parent_deferral_synthesize_not_blocked_standalone_no_deferral_row(tmp_path):
    from agent.db import parent_deferral_synthesize_execute_blocked

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "standalone_article"
    conn.commit()
    raw = tmp_path / f"{slug}_raw.md"
    raw.write_text("# standalone\n", encoding="utf-8")
    assert parent_deferral_synthesize_execute_blocked(conn, slug, raw_path=raw) is None


def test_parent_deferral_synthesize_blocked_dispatched_siblings_no_deferral_row(
    tmp_path,
):
    """Lost deferral row + all-dispatched-but-unverified siblings must fail closed."""
    from agent.db import parent_deferral_synthesize_execute_blocked

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "lost_deferral_parent"
    url = "https://blog.substack.com/p/lost"
    create_task(conn, url, "substack_text", parent, ["substack_cleanup"], status="paused")
    queue_ingest(
        conn,
        "https://example.com/img.png",
        "substack_image",
        "img1",
        parent_slug=parent,
        media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=0 WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()
    raw = tmp_path / f"{parent}_raw.md"
    raw.write_text("# parent without markers\n", encoding="utf-8")
    reason = parent_deferral_synthesize_execute_blocked(conn, parent, raw_path=raw)
    assert reason is not None
    assert "deferred synthesis paused" in reason


def test_find_active_task_supersedes_terminal_synthesize_missing_deferral_with_siblings(
    tmp_path,
):
    from agent.db import get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/orphan-dup"
    slug = "orphan_dup_parent"
    create_task(conn, url, "substack_text", slug, ["substack_cleanup"], status="paused")
    import uuid

    conn.execute(
        """
        INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            "https://example.com/i.png",
            "substack_image",
            "i1",
            "dispatched",
            slug,
            "image",
        ),
    )
    done_id = create_task(
        conn,
        url,
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.commit()
    assert get_parent_media_deferral(conn, slug) is None
    assert find_active_task(conn, slug, "substack_text_synthesize") is None
    row = conn.execute("SELECT status FROM tasks WHERE id=?", (done_id,)).fetchone()
    assert row[0] == "failed"

    config = _batch_test_config(tmp_path)
    result = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert result.get("duplicate") is not True
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE slug=? AND content_type GLOB '*_synthesize'",
        (slug,),
    ).fetchone()[0] == 2


def test_parent_deferral_synthesize_execute_blocked_bootstraps_deferral_from_manifest(
    tmp_path,
):
    from agent.db import (
        bootstrap_parent_media_deferral_from_manifest,
        get_parent_media_deferral,
        parent_deferral_synthesize_execute_blocked,
        queue_ingest,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "manifest_parent"
    queue_ingest(
        conn,
        "https://example.com/vid.mp4",
        "youtube",
        "vid1",
        parent_slug=parent,
        media_kind="video",
        status="queued",
    )
    manifest = tmp_path / f"{parent}_media.json"
    manifest.write_text(
        json.dumps(
            {
                "total": 1,
                "url": "https://blog.substack.com/p/post",
                "content_type": "substack_text",
            }
        ),
        encoding="utf-8",
    )
    conn.commit()

    bootstrap_parent_media_deferral_from_manifest(conn, parent, tmp_path)
    deferral = get_parent_media_deferral(conn, parent)
    assert deferral is not None
    assert deferral["expected_total"] == 1

    raw = tmp_path / f"{parent}_raw.md"
    raw.write_text("# parent\n", encoding="utf-8")
    reason = parent_deferral_synthesize_execute_blocked(conn, parent, raw_path=raw)
    assert reason == "deferred synthesis paused — sibling media still pending"


@pytest.mark.parametrize(
    "content_type,pause_reason",
    [
        (
            "substack_text_synthesize",
            "deferred synthesis paused — sibling media still pending",
        ),
        (
            "substack_text_synthesize",
            "deferred synthesis paused — parent media not complete at execute time",
        ),
        (
            "substack_image_synthesize",
            "deferred synthesis paused — sibling media still pending",
        ),
        (
            "substack_image_synthesize",
            "deferred synthesis paused — parent media not complete at execute time",
        ),
    ],
)
def test_get_next_step_does_not_skip_paused_synthesize_execute_gate(
    tmp_path, content_type, pause_reason,
):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        content_type,
        "parent_slug",
        ["synthesize_note", "quality_gate", "minio_upload"],
        status="pending",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    synth_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (pause_reason, synth_id),
    )
    conn.commit()

    nxt = get_next_step(conn, task_id)
    assert nxt is not None
    assert nxt["name"] == "synthesize_note"
    assert nxt["status"] == "paused"


def test_resume_awaiting_media_does_not_skip_embedded_step(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    defer_step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("awaiting media: 2 images enqueued", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = 'defer' WHERE id = ?",
        (defer_step_id,),
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, "%awaiting media%") == 0
    task_row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert task_row[0] == "paused"
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["substack_embedded_media"] == "paused"
    assert steps["synthesize_note"] == "pending"


def test_retry_awaiting_media_preserves_embed_step(tmp_path):
    """Enqueue shortfall pause (no parent_media_deferral row) still preserves embed on retry."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/post",
        "substack_text",
        "parent_slug",
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    defer_step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "defer"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("awaiting media: 1 images enqueued", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'running', output = 'mid' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, defer_step_id),
    )
    conn.commit()

    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (defer_step_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["substack_cleanup"] == "running"
    assert steps["synthesize_note"] == "pending"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "pending"


def test_retry_media_gate_preserves_embed_without_deferral_row(tmp_path):
    """``media gate`` in ``tasks.error`` with no deferral row preserves embed on retry."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "media_gate_no_row"
    pause_reason = "media gate: ingest rows 1 != manifest total 2"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/gate",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "manifest partial"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, embed_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()
    assert task_in_deferred_media_cycle(conn, task_id, slug=slug, error=pause_reason)

    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (embed_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["synthesize_note"] == "pending"


def test_retry_terminal_success_embed_rewinds_embed_step(tmp_path):
    """Terminal-success embed must reset on retry so HTML re-parse can re-enqueue media."""
    from agent.db import ensure_parent_media_deferral, task_in_deferred_media_cycle

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "terminal_embed_retry"
    url = "https://blog.substack.com/p/terminal-embed"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    terminal_output = "media ready — deferred to batch"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("batch synthesis dispatched or completed — parent must not re-enter synthesis", task_id),
    )
    conn.execute(
        """
        UPDATE task_steps SET status = 'done', output = ?
        WHERE task_id = ? AND name IN ('substack_cleanup', 'substack_embedded_media')
        """,
        (f"cleaned\nmedia_deferred:{slug}", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET output = ? WHERE id = ?",
        (terminal_output, embed_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()

    assert task_in_deferred_media_cycle(conn, task_id, slug=slug, error=None)
    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (embed_id,),
    ).fetchone()
    assert embed[0] == "pending"
    assert embed[1] is None


def test_retry_failed_deferral_cycle_preserves_embed_step(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "failed_deferral_parent"
    url = "https://blog.substack.com/p/failed-deferral"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="failed",
    )
    defer_step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "awaiting media: 1 images enqueued"
    pause_reason = "awaiting media: 1 images enqueued"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, defer_step_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'failed' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()

    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (defer_step_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["synthesize_note"] == "pending"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "pending"


def test_retry_in_deferral_cycle_without_embed_step_resets_failed_steps(tmp_path):
    """Open deferral on slug must not block step reset when task lacks embed step (slug collision)."""
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "deferral_slug_collision"
    url = "https://blog.substack.com/p/collision"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    task_id = create_task(
        conn,
        "https://example.com/unrelated",
        "web",
        slug,
        ["fetch", "clean", "synthesize_note"],
        status="failed",
    )
    pause_reason = "awaiting media: 1 images enqueued"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    failed_step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'clean'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE task_steps SET status = 'failed', output = 'err' WHERE id = ?",
        (failed_step_id,),
    )
    conn.commit()
    assert task_in_deferred_media_cycle(conn, task_id, slug=slug, error=pause_reason)

    assert retry_task(conn, task_id) is True
    row = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (failed_step_id,),
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] is None
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "pending"


def test_retry_failed_without_deferral_cycle_resets_steps(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn,
        "https://example.com/article",
        "web",
        "web_fail_retry",
        ["fetch", "clean", "synthesize_note"],
        status="failed",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'clean'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("transient 503", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'failed', output = 'err' WHERE id = ?",
        (embed_id,),
    )
    conn.commit()

    assert retry_task(conn, task_id) is True
    row = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (embed_id,),
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] is None


def test_dispatch_failure_pause_appended_and_blocks_resume_retry(tmp_path):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "dispatch_fail_parent"
    url = "https://blog.substack.com/p/dispatch-fail"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    prior_pause = "awaiting media: 1 images enqueued"
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    defer_step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "awaiting media: 1 images + 0 videos"
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (prior_pause, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, defer_step_id),
    )
    conn.commit()

    batch_ingest._record_parent_deferral_dispatch_failure(
        conn, slug, "unknown content_type: substack_text_synthesize",
    )
    conn.commit()

    row = conn.execute("SELECT error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    error = row[0] or ""
    assert prior_pause in error
    assert "deferred synthesis dispatch failed" in error
    assert is_deferred_media_pause(error)
    assert resume_tasks_by_reason(conn, "%deferred synthesis%") == 0

    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (defer_step_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "pending"


def test_dispatch_failure_strips_query_tokens_in_tasks_error(tmp_path):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "dispatch_fail_secret_parent"
    url = "https://blog.substack.com/p/dispatch-fail-secret"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.commit()

    secret_url = "https://cdn.example.com/v?token=SECRET"
    batch_ingest._record_parent_deferral_dispatch_failure(
        conn,
        slug,
        f"upstream refused {secret_url}",
    )
    conn.commit()

    row = conn.execute("SELECT error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    error = row[0] or ""
    assert "deferred synthesis dispatch failed" in error
    assert "token=SECRET" not in error
    assert "SECRET" not in error
    assert "upstream refused" in error


def test_queue_ingest_duplicate_revives_failed_sibling(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://example.com/embed.png"
    queue_ingest(
        conn, url, "substack_image", "img1",
        parent_slug="article", media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='failed', error='ocr failed', attempts=1 WHERE url=?",
        (url,),
    )
    conn.commit()

    _, created = queue_ingest(
        conn, url, "substack_image", "img1",
        parent_slug="article", media_kind="image",
    )
    assert created is False
    row = conn.execute(
        "SELECT status, error, attempts FROM ingest_queue WHERE url=?", (url,),
    ).fetchone()
    assert row["status"] == "queued"
    assert row["error"] is None
    assert row["attempts"] == 2


def test_queue_ingest_duplicate_keeps_failed_when_attempts_exhausted(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://example.com/embed2.png"
    queue_ingest(
        conn, url, "substack_image", "img2",
        parent_slug="article", media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='failed', error='gave up', attempts=3 WHERE url=?",
        (url,),
    )
    conn.commit()

    queue_ingest(conn, url, "substack_image", "img2", parent_slug="article", media_kind="image")
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE url=?", (url,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "gave up"


def test_queue_ingest_revive_and_recover_single_attempt_accounting(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://example.com/revive.png"
    queue_ingest(
        conn, url, "substack_image", "img_rev",
        parent_slug="article", media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='failed', error='ocr failed', attempts=1 WHERE url=?",
        (url,),
    )
    conn.commit()

    queue_ingest(
        conn, url, "substack_image", "img_rev",
        parent_slug="article", media_kind="image",
    )
    assert recover_failed_media(conn, max_attempts=3) == 0
    row = conn.execute("SELECT status, attempts FROM ingest_queue WHERE url=?", (url,)).fetchone()
    assert row["status"] == "queued"
    assert row["attempts"] == 2

    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url=?", (url,))
    conn.commit()
    assert recover_failed_media(conn, max_attempts=3) == 1
    row = conn.execute("SELECT status, attempts FROM ingest_queue WHERE url=?", (url,)).fetchone()
    assert row["status"] == "queued"
    assert row["attempts"] == 3

    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url=?", (url,))
    conn.commit()
    assert recover_failed_media(conn, max_attempts=3) == 0
    row = conn.execute("SELECT status, attempts FROM ingest_queue WHERE url=?", (url,)).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 3


def test_queue_ingest_duplicate_revive_reaches_max_attempts(tmp_path):
    """Permanent failures exhaust INGEST_MAX_MEDIA_ATTEMPTS via batch recover only."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://example.com/stuck.png"
    max_attempts = 3
    queue_ingest(
        conn, url, "substack_image", "stuck",
        parent_slug="article", media_kind="image",
        max_attempts=max_attempts,
    )
    conn.execute(
        "UPDATE ingest_queue SET status='failed', error='ocr failed', attempts=0 WHERE url=?",
        (url,),
    )
    conn.commit()
    for expected in (1, 2, 3):
        assert recover_failed_media(conn, max_attempts=max_attempts) == 1
        row = conn.execute(
            "SELECT status, attempts FROM ingest_queue WHERE url=?", (url,),
        ).fetchone()
        assert row["status"] == "queued"
        assert row["attempts"] == expected
        conn.execute("UPDATE ingest_queue SET status='failed' WHERE url=?", (url,))
        conn.commit()
    assert recover_failed_media(conn, max_attempts=max_attempts) == 0
    row = conn.execute(
        "SELECT status, attempts FROM ingest_queue WHERE url=?", (url,),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 3
    queue_ingest(
        conn, url, "substack_image", "stuck",
        parent_slug="article", media_kind="image",
        max_attempts=max_attempts,
    )
    row = conn.execute(
        "SELECT status, attempts FROM ingest_queue WHERE url=?", (url,),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 3


def test_has_active_synthesize_task_excludes_failed(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "parent_slug"
    create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="failed",
    )
    assert has_active_synthesize_task(conn, slug) is False
    create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="pending",
    )
    assert has_active_synthesize_task(conn, slug) is True


def test_has_synthesize_task_signal_includes_completed(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "signal_slug"
    create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    assert has_completed_synthesize_task(conn, slug) is True
    assert has_synthesize_task_signal(conn, slug) is True
    assert has_active_synthesize_task(conn, slug) is False


def test_prior_generation_done_synthesize_not_signal_for_new_deferral(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "gen_signal_parent"
    url = "https://blog.substack.com/p/gen-signal"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    create_task(
        conn,
        url,
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') "
        "WHERE slug = ? AND content_type GLOB '*_synthesize'",
        (anchor, slug),
    )
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.commit()
    assert get_parent_media_deferral(conn, slug)["media_changed"] == 1
    assert has_synthesize_task_signal(conn, slug) is False
    assert has_completed_synthesize_task(conn, slug) is False


def test_has_active_synthesize_false_when_only_done_row(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "done_only_parent"
    url = "https://blog.substack.com/p/done-only"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    create_task(
        conn,
        url,
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.commit()
    assert has_active_synthesize_task(conn, slug) is False


def test_has_completed_parent_synthesis_rejects_stale_synth_with_vault(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    slug = "stale_vault_synth"
    vault = tmp_path / "vault"
    url = "https://blog.substack.com/p/stale-synth"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    (note_dir / f"{slug}.md").write_text("# old note\n", encoding="utf-8")
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/s.png", "substack_image", "s",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    media_tmp = tmp_path / "media_tmp"
    _stamp_parent_raw_media(conn, media_tmp, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now', '-2 days') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False


def test_retry_task_allowed_when_only_stale_terminal_synthesize(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "retry_stale_syn"
    url = "https://author.substack.com/p/retry-stale"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    parent_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("awaiting media: 1 images enqueued", parent_id),
    )
    create_task(
        conn,
        url,
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') "
        "WHERE slug = ? AND content_type GLOB '*_synthesize'",
        (anchor, slug),
    )
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.commit()

    assert has_active_synthesize_task(conn, slug) is False
    assert retry_task(conn, parent_id) is True


def test_retry_task_skips_when_synthesize_sibling_active(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "deferred_parent"
    parent_id = create_task(
        conn,
        "https://author.substack.com/p/post",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("awaiting media: 1 images enqueued", parent_id),
    )
    create_task(
        conn,
        "https://author.substack.com/p/post",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="in_progress",
    )
    conn.commit()

    assert retry_task(conn, parent_id) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (parent_id,)).fetchone()[0] == "paused"


def test_resume_tasks_by_reason_skips_when_synthesize_sibling_active(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "bulk_synth_guard_parent"
    pause_reason = "firecrawl rate limit — retry after cooldown"
    parent_id = create_task(
        conn,
        "https://author.substack.com/p/post",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, parent_id),
    )
    create_task(
        conn,
        "https://author.substack.com/p/post",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="in_progress",
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, "%firecrawl%") == 0
    assert (
        conn.execute("SELECT status FROM tasks WHERE id = ?", (parent_id,)).fetchone()[0]
        == "paused"
    )


def test_resume_tasks_by_reason_excludes_open_deferral_without_error_marker(tmp_path):
    """Bulk resume must not bypass deferral when only cleanup marker / embed output signal it."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "bulk_deferral_marker_parent"
    pause_reason = "firecrawl rate limit — retry after cooldown"
    task_id = create_task(
        conn,
        "https://blog.substack.com/p/bulk-defer",
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    embed_output = "awaiting media: 1 images + 0 videos"
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (pause_reason, task_id))
    conn.execute(
        """
        UPDATE task_steps SET status = 'completed', output = ?
        WHERE task_id = ? AND name = 'substack_cleanup'
        """,
        (f"cleaned\nmedia_deferred:{slug}", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE id = ?",
        (embed_output, embed_id),
    )
    conn.commit()

    assert task_in_deferred_media_cycle(conn, task_id, slug=slug, error=pause_reason)
    assert resume_tasks_by_reason(conn, "%firecrawl%") == 0
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE id = ?",
        (embed_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] == embed_output


def test_resume_tasks_by_reason_resets_running_steps_like_retry(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    pause_reason = "firecrawl rate limit — retry after cooldown"
    task_id = create_task(
        conn,
        "https://example.com/article",
        "web",
        "web_bulk_running",
        ["fetch", "clean", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        (pause_reason, task_id),
    )
    conn.execute(
        """
        UPDATE task_steps SET status = 'paused', output = ?
        WHERE task_id = ? AND name = 'fetch'
        """,
        (pause_reason, task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'running' WHERE task_id = ? AND name = 'synthesize_note'",
        (task_id,),
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, "%firecrawl%") == 1
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["fetch"] == "pending"
    assert steps["synthesize_note"] == "pending"


def test_find_active_task_synthesize_slug_wide(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "shared_parent"
    synth_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="in_progress",
    )
    assert find_active_task(conn, slug, "substack_image_synthesize") == synth_id
    assert find_active_task(conn, slug, "substack_text") is None


def test_find_active_task_prefers_oldest_row_when_duplicates_exist(tmp_path):
    """Invariant: LIMIT 1 without ORDER BY is nondeterministic — oldest wins."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "dup_parent"
    older_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="pending",
    )
    # Legacy DBs could hold duplicate in-flight owners; simulate for ORDER BY semantics.
    conn.execute(f"DROP INDEX IF EXISTS {_ACTIVE_SYNTHESIZE_SLUG_INDEX}")
    newer_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_image_synthesize",
        slug,
        ["synthesize_note"],
        status="pending",
    )
    conn.execute(
        "UPDATE tasks SET created_at = datetime('now', '-2 hours') WHERE id = ?",
        (older_id,),
    )
    conn.execute(
        "UPDATE tasks SET created_at = datetime('now', '-1 hour') WHERE id = ?",
        (newer_id,),
    )
    conn.commit()
    assert find_active_task(conn, slug, "substack_text_synthesize") == older_id


def test_recover_stale_ingest_increments_attempts(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    ingest_id, _ = queue_ingest(conn, "https://example.com/v.mp4", "youtube", "vid1")
    conn.execute(
        "UPDATE ingest_queue SET status='downloading', attempts=0, "
        "updated_at=datetime('now', '-200 minutes') WHERE id=?",
        (ingest_id,),
    )
    conn.commit()

    assert recover_stale_ingest(conn, max_age_minutes=120) == 1
    row = conn.execute(
        "SELECT status, attempts, error FROM ingest_queue WHERE id=?",
        (ingest_id,),
    ).fetchone()
    assert row["status"] == "queued"
    assert row["attempts"] == 1
    assert "recovered from stale" in row["error"]


def test_recover_stale_ingest_fails_when_attempts_exhausted(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    ingest_id, _ = queue_ingest(conn, "https://example.com/v2.mp4", "youtube", "vid2")
    conn.execute(
        "UPDATE ingest_queue SET status='transcribing', "
        f"attempts={INGEST_MAX_MEDIA_ATTEMPTS - 1}, "
        "updated_at=datetime('now', '-200 minutes') WHERE id=?",
        (ingest_id,),
    )
    conn.commit()

    assert recover_stale_ingest(conn, max_age_minutes=120) == 0
    row = conn.execute(
        "SELECT status, attempts, error FROM ingest_queue WHERE id=?",
        (ingest_id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == INGEST_MAX_MEDIA_ATTEMPTS
    assert "exceeded max attempts" in row["error"]


def test_dispatch_synthesize_idempotent_when_task_pending(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/x"
    slug = "parent_slug"
    create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="pending",
    )
    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    first = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    second = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert first["duplicate"] is True
    assert second["duplicate"] is True
    assert first["task_id"] == second["task_id"]
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE slug=?", (slug,)).fetchone()[0] == 1


def test_reconcile_deferred_parent_syntheses_orphan(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "orphan_parent"
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        '{"url": "https://blog.substack.com/p/x", "content_type": "substack_text", "total": 1}',
        encoding="utf-8",
    )
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    calls: list[str] = []

    def fake_dispatch(url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t1"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()
    batch_ingest._reconcile_deferred_parent_syntheses(conn, config)
    assert calls == [parent]


def test_embedded_video_download_failure_sets_failed(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(
        conn, "https://www.youtube.com/watch?v=abcdEFghijk", "youtube", "yt1",
        parent_slug="p1", media_kind="video",
    )
    row = conn.execute("SELECT * FROM ingest_queue").fetchone()
    q = dict(row)

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (None, "download failed"))

    batch_ingest._process_embedded_videos([q], config, conn, 1)
    status = conn.execute("SELECT status FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()["status"]
    assert status == "failed"


def test_has_completed_parent_synthesis_parent_task_or_synthesize(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "synth_owner"
    vault = tmp_path / "vault"
    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is False

    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.commit()
    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is True

    conn.execute("DELETE FROM tasks WHERE id = ?", (parent_id,))
    conn.commit()
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    (note_dir / f"{slug}.md").write_text("# Note\n", encoding="utf-8")
    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is False

    syn_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="completed",
    )
    conn.commit()
    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is True
    conn.execute("DELETE FROM tasks WHERE id = ?", (syn_id,))


def test_has_completed_parent_synthesis_vault_requires_deferral_reconciled(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    slug = "stale_vault_parent"
    vault = tmp_path / "vault"
    url = "https://blog.substack.com/p/stale"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    (note_dir / f"{slug}.md").write_text("# stale\n", encoding="utf-8")

    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is False

    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    queue_ingest(
        conn, "https://example.com/a.png", "substack_image", "a",
        parent_slug=slug, media_kind="image",
    )
    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is False

    queue_ingest(
        conn, "https://example.com/b.png", "substack_image", "b",
        parent_slug=slug, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (slug,))
    conn.commit()
    assert get_parent_media_deferral(conn, slug)["expected_total"] == 2
    media_tmp = tmp_path / "media_tmp"
    _stamp_parent_raw_media(conn, media_tmp, slug)
    note_path = note_dir / f"{slug}.md"
    note_path.write_text("# stale after media\n", encoding="utf-8")
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False

    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is True


def test_queue_ingest_rejects_javascript_scheme(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    with pytest.raises(ValueError, match="http or https"):
        queue_ingest(conn, "javascript:alert(1)", "web", "jev")
    assert conn.execute("SELECT COUNT(*) FROM ingest_queue").fetchone()[0] == 0


def test_queue_ingest_revive_rejects_non_http_legacy_row(tmp_path):
    """Failed revive must not bypass http(s) scheme guard on pre-existing rows."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    bad = "javascript:alert(1)"
    conn.execute(
        "INSERT INTO ingest_queue (id, url, content_type, slug, status, attempts) "
        "VALUES ('legacy-id', ?, 'web', 'jev', 'failed', 0)",
        (bad,),
    )
    conn.commit()
    with pytest.raises(ValueError, match="http or https"):
        queue_ingest(conn, bad, "web", "jev")
    row = conn.execute(
        "SELECT status FROM ingest_queue WHERE id='legacy-id'",
    ).fetchone()
    assert row[0] == "failed"


def test_has_completed_parent_synthesis_vault_shortcut_needs_terminal_or_fresh_note(tmp_path):
    import os
    import time

    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    slug = "vault_shortcut_gate"
    vault = tmp_path / "vault"
    url = "https://blog.substack.com/p/shortcut"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    (note_dir / f"{slug}.md").write_text("# pre-existing\n", encoding="utf-8")
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/x.png", "substack_image", "x",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    conn.commit()
    media_tmp = tmp_path / "media_tmp"
    _stamp_parent_raw_media(conn, media_tmp, slug)
    stale = time.time() - 3600
    os.utime(note_dir / f"{slug}.md", (stale, stale))
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False

    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is True


def test_has_completed_parent_synthesis_vault_without_raw_unverified(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "vault_only_parent"
    vault = tmp_path / "vault"
    url = "https://blog.substack.com/p/vault-only"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    (note_dir / f"{slug}.md").write_text("# note\n", encoding="utf-8")
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 0)
    conn.commit()

    assert has_completed_parent_synthesis(conn, slug, vault_path=vault) is False


def test_reingest_new_media_redispatches_after_prior_cycle(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "reingest_parent"
    url = "https://blog.substack.com/p/reingest"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    parent_id = create_task(
        conn, url, "substack_text", parent,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("deferred synthesis dispatched; parent ingestion superseded", parent_id),
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", parent,
        ["synthesize_note"], status="completed",
    )
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    _stamp_parent_raw_media(conn, config.temp_dir, parent)
    note = config.vault_path / "Knowledge base" / "articles" / f"{parent}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# cycle 1\n", encoding="utf-8")

    assert has_completed_parent_synthesis(
        conn, parent, vault_path=config.vault_path, temp_dir=config.temp_dir,
    ) is True

    deferral_anchor = get_parent_media_deferral(conn, parent)["updated_at"]
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (deferral_anchor, syn_id),
    )
    conn.commit()

    queue_ingest(
        conn, "https://example.com/i2", "substack_image", "i2",
        parent_slug=parent, media_kind="image",
    )
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched' WHERE id IN "
        "(SELECT id FROM ingest_queue WHERE parent_slug=?)",
        (parent,),
    )
    conn.commit()
    _stamp_parent_raw_media(conn, config.temp_dir, parent)
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 2
    assert get_parent_media_deferral(conn, parent)["media_changed"] == 1

    assert has_completed_parent_synthesis(
        conn, parent, vault_path=config.vault_path, temp_dir=config.temp_dir,
    ) is False

    calls: list[str] = []

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t-reingest", "duplicate": False}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]
    old_syn = conn.execute(
        "SELECT status, error FROM tasks WHERE slug=? AND content_type GLOB '*_synthesize'",
        (parent,),
    ).fetchone()
    assert old_syn["status"] == "failed"
    assert "superseded by deferral media cycle" in (old_syn["error"] or "")


def test_fresh_deferral_generation_ignores_superseded_parent(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_parent"
    url = "https://blog.substack.com/p/sup"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    tid = create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("deferred synthesis dispatched; parent ingestion superseded", tid),
    )
    conn.commit()
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.commit()

    assert has_completed_parent_synthesis(conn, slug) is False


def test_supersede_deferred_parent_task_canonical_reason(tmp_path):
    """Batch and web paths share one supersede error string."""
    from agent.db import (
        SUPERSEDE_DEFERRED_PARENT_REASON,
        supersede_deferred_parent_task,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_canonical"
    url = "https://blog.substack.com/p/canonical"
    parent_id = create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="paused",
    )
    conn.commit()

    assert supersede_deferred_parent_task(conn, slug) == 1
    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (parent_id,),
    ).fetchone()
    assert row["status"] == "completed"
    assert row["error"] == SUPERSEDE_DEFERRED_PARENT_REASON


def test_supersede_deferred_parent_task_cas_stays_terminal(tmp_path):
    """Stale complete_task cannot revive a batch-superseded parent row."""
    from agent.db import complete_task, supersede_deferred_parent_task

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_cas"
    url = "https://blog.substack.com/p/cas"
    parent_id = create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.commit()

    assert supersede_deferred_parent_task(conn, slug) == 1
    assert complete_task(conn, parent_id) is False
    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (parent_id,),
    ).fetchone()
    assert row["status"] == "completed"
    assert "superseded" in (row["error"] or "").lower()


def test_supersede_deferred_parent_task_idempotent(tmp_path):
    from agent.db import supersede_deferred_parent_task

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_idem"
    url = "https://blog.substack.com/p/idem"
    create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="paused",
    )
    conn.commit()

    assert supersede_deferred_parent_task(conn, slug) == 1
    assert supersede_deferred_parent_task(conn, slug) == 0


def test_has_completed_parent_synthesis_monotonic_pending_with_fresh_synth_deferral(
    tmp_path,
):
    """Fresh terminal *_synthesize must not count while a sibling is still pending."""
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mono_deferral"
    url = "https://blog.substack.com/p/mono-deferral"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/pending.png", "substack_image", "pend",
        parent_slug=slug, media_kind="image",
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False


def test_has_completed_parent_synthesis_monotonic_pending_with_fresh_synth_embedded(
    tmp_path,
):
    """Deferral-less embedded context: pending siblings block terminal synthesis."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mono_embedded"
    url = "https://blog.substack.com/p/mono-embedded"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    manifest = media_tmp / f"{slug}_media.json"
    manifest.write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    queue_ingest(
        conn, "https://example.com/pending.png", "substack_image", "pend",
        parent_slug=slug, media_kind="image",
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(conn, slug, temp_dir=media_tmp) is False


def test_media_changed_consumed_after_reingest_growth_and_media_landed(tmp_path):
    """Growth sets media_changed; satisfied siblings + landed raw must clear it."""
    from agent.db import (
        ensure_parent_media_deferral,
        find_active_task,
        get_parent_media_deferral,
        parent_media_complete,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "consumed_growth"
    url = "https://blog.substack.com/p/consumed-growth"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1.png", "substack_image", "i1",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    queue_ingest(
        conn, "https://example.com/i2.png", "substack_image", "i2",
        parent_slug=slug, media_kind="image",
    )
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    raw = media_tmp / f"{slug}_raw.md"
    assert parent_media_complete(conn, slug, raw_path=raw) is True
    deferral_anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        (deferral_anchor, syn_id),
    )
    conn.commit()

    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is True
    assert get_parent_media_deferral(conn, slug)["media_changed"] == 0
    assert find_active_task(conn, slug, "substack_text_synthesize") == syn_id


def test_media_changed_blocks_completion_while_sibling_shortfall(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "shortfall_block"
    url = "https://blog.substack.com/p/shortfall"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1.png", "substack_image", "i1",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)

    deferral = get_parent_media_deferral(conn, slug)
    assert deferral["expected_total"] == 2
    assert int(deferral["media_changed"]) == 1
    assert has_completed_parent_synthesis(
        conn, slug, temp_dir=media_tmp,
    ) is False


def test_shortfall_marker_does_not_clear_media_changed_without_raw(tmp_path):
    """INVARIANT: enqueue shortfall is set-only; raw verification clears media_changed."""
    from agent.db import (
        ensure_parent_media_deferral,
        get_parent_media_deferral,
        mark_parent_media_changed,
        parent_media_complete,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "shortfall_no_clear"
    url = "https://blog.substack.com/p/shortfall-no-clear"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    mark_parent_media_changed(conn, slug)
    queue_ingest(
        conn, "https://example.com/a.png", "substack_image", "a",
        parent_slug=slug, media_kind="image",
    )
    queue_ingest(
        conn, "https://example.com/b.png", "substack_image", "b",
        parent_slug=slug, media_kind="image",
    )
    deferral = get_parent_media_deferral(conn, slug)
    assert deferral["expected_total"] == 2
    assert int(deferral["media_changed"]) == 1
    assert parent_media_complete(conn, slug) is False


def test_requeue_task_for_retry_cas_refuses_superseded_before_update(tmp_path):
    from agent.db import requeue_task_for_retry_cas

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "web", "cas_slug", ["only"], status="failed",
    )
    conn.execute(
        "UPDATE tasks SET error = 'superseded: stale owner' WHERE id = ?", (task_id,),
    )
    conn.commit()
    with conn:
        assert requeue_task_for_retry_cas(conn, task_id) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "failed"


def test_retry_task_cas_skips_step_reset_when_superseded_race(tmp_path):
    """CAS update must fail when error becomes superseded after SELECT."""
    import agent.core.resume as resume_mod
    from agent.db import get_task_steps, requeue_task_for_retry_cas

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "web", "race_slug", ["s0", "s1"], status="failed",
    )
    conn.execute(
        "UPDATE task_steps SET status = 'failed' WHERE task_id = ?", (task_id,),
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", ("transient fail", task_id))
    conn.commit()

    def requeue_after_supersede(connection, tid, **kwargs):
        connection.execute(
            "UPDATE tasks SET error = 'superseded: between select and update' WHERE id = ?",
            (tid,),
        )
        return requeue_task_for_retry_cas(connection, tid, **kwargs)

    with patch.object(resume_mod, "requeue_task_for_retry_cas", side_effect=requeue_after_supersede):
        assert retry_task(conn, task_id) is False

    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "failed"
    steps = get_task_steps(conn, task_id)
    assert all(s["status"] == "failed" for s in steps)


def test_resume_tasks_by_reason_cas_skips_superseded_race(tmp_path):
    """Bulk resume CAS must not resurrect superseded paused rows."""
    from agent.db import get_task_steps

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    reason = "cooldown — retry later"
    task_id = create_task(
        conn, "https://example.com", "web", "resume_race", ["only"], status="paused",
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (reason, task_id))
    conn.execute(
        "UPDATE task_steps SET status = 'paused' WHERE task_id = ?", (task_id,),
    )
    conn.commit()

    rows = conn.execute(
        "SELECT id, slug, error FROM tasks WHERE status = 'paused' AND error LIKE ?",
        (f"%{reason[:8]}%",),
    ).fetchall()
    assert len(rows) == 1

    conn.execute(
        "UPDATE tasks SET error = 'superseded: bulk resume race' WHERE id = ?", (task_id,),
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, f"%{reason[:8]}%") == 0
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "paused"
    assert all(s["status"] == "paused" for s in get_task_steps(conn, task_id))


def test_has_completed_parent_synthesis_monotonic_complete_media_and_fresh_terminal(
    tmp_path,
):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mono_complete"
    url = "https://blog.substack.com/p/mono-complete"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/ok.png", "substack_image", "ok",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is True


def test_has_completed_parent_synthesis_monotonic_stale_terminal_with_pending_sibling(
    tmp_path,
):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mono_stale_pending"
    url = "https://blog.substack.com/p/mono-stale"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/pending.png", "substack_image", "pend",
        parent_slug=slug, media_kind="image",
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now', '-2 days') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False


def test_has_completed_parent_synthesis_standalone_terminal(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mono_standalone"
    url = "https://blog.substack.com/p/standalone"
    create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="completed",
    )
    conn.commit()
    assert has_completed_parent_synthesis(conn, slug) is True


def test_has_completed_parent_synthesis_manifest_no_deferral_stale_terminal(
    tmp_path,
):
    """Manifest-only embedded context: stale unscoped *_synthesize must not complete."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "manifest_stale_syn"
    url = "https://blog.substack.com/p/manifest-stale"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    (media_tmp / f"{slug}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now', '-2 days') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(conn, slug, temp_dir=media_tmp) is False


def test_has_completed_parent_synthesis_manifest_reconciled_fresh_terminal(
    tmp_path,
):
    """Manifest bootstraps deferral; verified raw + fresh terminal counts as complete."""
    from agent.db import bootstrap_parent_media_deferral_from_manifest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "manifest_reconciled"
    url = "https://blog.substack.com/p/manifest-recon"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    media_tmp.mkdir(parents=True)
    (media_tmp / f"{slug}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    queue_ingest(
        conn, "https://example.com/ok.png", "substack_image", "ok",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    bootstrap_parent_media_deferral_from_manifest(conn, slug, media_tmp)
    conn.commit()
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is True


def test_has_completed_parent_synthesis_genuine_complete_idempotent(tmp_path):
    """A fully complete parent stays complete across repeated guard checks."""
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "idempotent_complete"
    url = "https://blog.substack.com/p/idempotent"
    vault = tmp_path / "vault"
    media_tmp = tmp_path / "media_tmp"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/ok.png", "substack_image", "ok",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    _stamp_parent_raw_media(conn, media_tmp, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime('now') WHERE id = ?",
        (syn_id,),
    )
    conn.commit()

    kwargs = {"vault_path": vault, "temp_dir": media_tmp}
    assert has_completed_parent_synthesis(conn, slug, **kwargs) is True
    assert has_completed_parent_synthesis(conn, slug, **kwargs) is True


@patch("agent.pipelines.web._fetch_substack_html")
def test_step_substack_embedded_media_requires_raw_markers_when_pending_zero(
    mock_fetch, tmp_path,
):
    from agent.db import AGENT_SCHEMA, ensure_parent_media_deferral, init_db, queue_ingest, update_ingest_status
    from agent.exceptions import ErrorType, StepError
    from agent.pipelines.web import step_substack_embedded_media

    kb = tmp_path / "kb"
    (kb / "data").mkdir(parents=True)
    media_tmp = tmp_path / "tmp"
    media_tmp.mkdir(parents=True)
    config = Config(
        kb_agent_dir=kb,
        vault_path=tmp_path / "vault",
        temp_dir=media_tmp,
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    slug = "raw_gate_slug"
    post = "https://author.substack.com/p/raw-gate"
    vid = "a7315fa8-8027-4432-a7e3-652479bc5275"
    task = {
        "id": "task-raw-gate",
        "slug": slug,
        "source_url": post,
        "content_type": "substack_text",
        "tags": [],
    }
    conn = init_db(kb / "data" / "agent.db", AGENT_SCHEMA)
    ensure_parent_media_deferral(conn, slug, post, "substack_text", 1)
    ingest_id, _ = queue_ingest(
        conn, f"{post}#{vid}", "substack_video", slug,
        parent_slug=slug, media_kind="video",
    )
    update_ingest_status(conn, ingest_id, "dispatched")
    conn.close()
    mock_fetch.return_value = (
        f'<html><body><script>"videoUpload":{{"id":"{vid}"}}</script></body></html>'
    )
    with pytest.raises(StepError) as exc:
        step_substack_embedded_media(task, config)
    assert exc.value.error_type == ErrorType.PAUSE
    assert "media gate" in str(exc.value).lower()


@patch("agent.pipelines.web._fetch_substack_html")
def test_step_substack_embedded_media_ensures_deferral_on_enqueue(
    mock_fetch, tmp_path,
):
    from agent.db import AGENT_SCHEMA, get_parent_media_deferral, init_db
    from agent.pipelines.web import step_substack_embedded_media

    kb = tmp_path / "kb"
    (kb / "data").mkdir(parents=True)
    media_tmp = tmp_path / "tmp"
    media_tmp.mkdir(parents=True)
    config = Config(
        kb_agent_dir=kb,
        vault_path=tmp_path / "vault",
        temp_dir=media_tmp,
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    slug = "embed_deferral"
    post = "https://author.substack.com/p/embed"
    vid = "a7315fa8-8027-4432-a7e3-652479bc5275"
    task = {
        "id": "task-embed-deferral",
        "slug": slug,
        "source_url": post,
        "content_type": "substack_text",
        "tags": [],
    }
    mock_fetch.return_value = (
        f'<html><body><script>"videoUpload":{{"id":"{vid}"}}</script></body></html>'
    )

    from agent.exceptions import StepError

    with pytest.raises(StepError, match="awaiting media"):
        step_substack_embedded_media(task, config)

    conn = init_db(kb / "data" / "agent.db", AGENT_SCHEMA)
    deferral = get_parent_media_deferral(conn, slug)
    conn.close()
    assert deferral is not None
    assert deferral["expected_total"] == 1
    assert deferral["source_url"] == post


def test_ensure_parent_media_deferral_grows_expected_total(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "grow_parent"
    url = "https://blog.substack.com/p/grow"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 1

    ensure_parent_media_deferral(conn, parent, url, "substack_text", 3)
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 3

    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 3


def test_maybe_dispatch_skips_when_parent_synthesis_completed(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "standalone_done_parent"
    url = "https://blog.substack.com/p/standalone-done"
    create_task(conn, url, "substack_text", parent, ["synthesize_note"], status="completed")
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    dispatch = MagicMock()
    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    dispatch.assert_not_called()


def test_maybe_dispatch_does_not_skip_when_siblings_after_inline_complete(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "inline_done_parent"
    url = "https://blog.substack.com/p/inline-done"
    parent_id = create_task(conn, url, "substack_text", parent, ["synthesize_note"], status="completed")
    conn.execute(
        "UPDATE tasks SET updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (parent_id,),
    )
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now') WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    assert has_completed_parent_synthesis(
        conn, parent, vault_path=config.vault_path, temp_dir=config.temp_dir,
    ) is False

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        return {"status": "pending", "task_id": "t-batch", "duplicate": False}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True


def test_has_completed_parent_synthesis_deferral_less_ignores_superseded_parent(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_deferral_less"
    url = "https://blog.substack.com/p/sup-less"
    parent_id = create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("deferred synthesis dispatched; parent ingestion superseded", parent_id),
    )
    queue_ingest(
        conn, "https://example.com/i.png", "substack_image", "i",
        parent_slug=slug, media_kind="image",
    )
    conn.commit()

    assert has_completed_parent_synthesis(conn, slug) is False


def test_has_completed_parent_synthesis_deferral_mtime_requires_fresh_terminal(tmp_path):
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "mtime_only_parent"
    vault = tmp_path / "vault"
    url = "https://blog.substack.com/p/mtime"
    note_dir = vault / "Knowledge base" / "articles"
    note_dir.mkdir(parents=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/x.png", "substack_image", "x",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now'), updated_at=datetime('now') WHERE parent_slug=?",
        (slug,),
    )
    conn.commit()
    media_tmp = tmp_path / "media_tmp"
    _stamp_parent_raw_media(conn, media_tmp, slug)
    note_path = note_dir / f"{slug}.md"
    note_path.write_text("# fresh note\n", encoding="utf-8")
    assert has_completed_parent_synthesis(
        conn, slug, vault_path=vault, temp_dir=media_tmp,
    ) is False


def test_maybe_dispatch_unlocks_when_expected_total_grows(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "embed_growth_parent"
    url = "https://blog.substack.com/p/growth"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    # Gate growth is owned by deferral writers (e.g. web embed step), not batch manifest url.
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"content_type": "substack_text", "total": 2}),
        encoding="utf-8",
    )
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    calls: list[str] = []

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t-growth"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    assert calls == []

    queue_ingest(
        conn, "https://example.com/i2", "substack_image", "i2",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()
    _stamp_parent_raw_media(conn, config.temp_dir, parent)
    assert get_parent_media_deferral(conn, parent)["expected_total"] == 2
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]


def test_maybe_dispatch_parent_synthesize_manifest_count_and_idempotent(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "article_slug"
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    from agent.config import Config

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(
        conn, parent, "https://blog.substack.com/p/x", "substack_text", 2,
    )
    (config.temp_dir / f"{parent}_media.json").write_text(
        '{"content_type": "substack_text", "total": 2}',
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_dispatch(url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t1"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    assert calls == []

    (config.temp_dir / f"{parent}_media.json").write_text(
        '{"content_type": "substack_text", "total": 1}',
        encoding="utf-8",
    )
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False

    queue_ingest(
        conn, "https://example.com/i2", "substack_image", "i2",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()
    _stamp_parent_raw_media(conn, config.temp_dir, parent)
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    assert len(calls) == 1


def test_maybe_dispatch_fails_when_dispatched_without_raw_markers(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "dispatched_only_parent"
    url = "https://blog.substack.com/p/markers"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )

    dispatch = MagicMock()
    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    dispatch.assert_not_called()


def test_maybe_dispatch_fails_closed_without_deferral_record(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "manifest_missing_parent"
    url = "https://blog.substack.com/p/reconcile"
    create_task(conn, url, "substack_text", parent, ["substack_cleanup"], status="paused")
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    calls: list[str] = []

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t-reconcile"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    assert calls == []


def test_maybe_dispatch_unlocks_when_deferral_and_manifest_match(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "deferral_match_parent"
    url = "https://blog.substack.com/p/ok"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t-ok"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]


def test_remove_ingest_reconciles_expected_total_and_unlocks_dispatch(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral, remove_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "removed_sibling_parent"
    url = "https://blog.substack.com/p/two"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    id_a, _ = queue_ingest(
        conn, "https://example.com/a", "substack_image", "a",
        parent_slug=parent, media_kind="image",
    )
    id_b, _ = queue_ingest(
        conn, "https://example.com/b", "substack_image", "b",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE id=?", (id_a,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 2}),
        encoding="utf-8",
    )
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    remove_ingest(conn, id_b, temp_dir=config.temp_dir)
    deferral = get_parent_media_deferral(conn, parent)
    assert deferral["expected_total"] == 1
    assert deferral.get("media_changed") == 1

    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE id=?", (id_a,))
    conn.commit()
    _stamp_parent_raw_media(conn, config.temp_dir, parent)

    batch_ingest.refresh_parent_media_manifest(conn, config, parent)
    manifest = json.loads((config.temp_dir / f"{parent}_media.json").read_text(encoding="utf-8"))
    assert manifest["total"] == 1
    assert manifest.get("media_changed") is True

    calls: list[str] = []

    def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
        calls.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "t-after-remove"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]


def test_remove_ingest_expected_total_monotonic_down_respects_high_water(tmp_path):
    """Lowering stops at embedded/manifest high-water (never above committed media count)."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral, remove_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "remove_hw_cap"
    url = "https://blog.substack.com/p/hw-cap"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 3)
    ids = []
    for label in ("a", "b", "c"):
        ingest_id, _ = queue_ingest(
            conn,
            f"https://example.com/{label}",
            "substack_image",
            label,
            parent_slug=parent,
            media_kind="image",
        )
        ids.append(ingest_id)
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    (temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack_text", "total": 2}),
        encoding="utf-8",
    )

    remove_ingest(conn, ids[2], temp_dir=temp_dir)
    deferral = get_parent_media_deferral(conn, parent)
    assert deferral["expected_total"] == 2


def test_maybe_dispatch_fails_closed_when_manifest_total_null(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "null_total_parent"
    queue_ingest(
        conn, "https://example.com/i1", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE parent_slug=?", (parent,))
    conn.commit()

    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    (config.temp_dir / f"{parent}_media.json").write_text(
        '{"url": "https://blog.substack.com/p/x", "content_type": "substack_text"}',
        encoding="utf-8",
    )

    dispatch = MagicMock()
    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()

    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    dispatch.assert_not_called()


def test_update_ingest_status_stores_sanitized_error(tmp_path):
    from agent.db import update_ingest_status

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    ingest_id, _ = queue_ingest(conn, "https://example.com/v.mp4", "youtube", "vid_scrub")
    secret_path = "/root/kb-agent/secret/cookies.txt"
    update_ingest_status(
        conn,
        ingest_id,
        "failed",
        error=f'download failed: File "{secret_path}", line 9, in run\nstderr: {"x" * 300}',
    )
    row = conn.execute(
        "SELECT error FROM ingest_queue WHERE id=?", (ingest_id,),
    ).fetchone()
    assert secret_path in row["error"]
    assert len(row["error"]) > 300


def test_update_ingest_status_redacts_urls_at_write(tmp_path):
    from agent.db import finish_batch_run, record_batch_run, update_ingest_status

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    ingest_id, _ = queue_ingest(conn, "https://example.com/v.mp4", "youtube", "vid_url_scrub")
    leaky = "GET https://cdn.example.com/seg.m3u8?sig=deadbeef failed"
    update_ingest_status(conn, ingest_id, "failed", error=leaky)
    row = conn.execute(
        "SELECT error FROM ingest_queue WHERE id=?", (ingest_id,),
    ).fetchone()
    assert "sig=deadbeef" not in (row["error"] or "")
    assert "?" not in (row["error"] or "")

    run_id = record_batch_run(conn)
    finish_batch_run(
        conn,
        run_id,
        "failed",
        1,
        1,
        0,
        f"vid: {leaky}",
    )
    batch = conn.execute(
        "SELECT summary FROM batch_runs WHERE id=?", (run_id,),
    ).fetchone()
    assert "sig=deadbeef" not in (batch["summary"] or "")


def test_claim_task_fifo_by_created_at(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    id_old = create_task(conn, "https://example.com/old", "web", "old", ["step"])
    id_new = create_task(conn, "https://example.com/new", "web", "new", ["step"])
    conn.execute(
        "UPDATE tasks SET created_at = ? WHERE id = ?",
        ("2000-01-01 00:00:00", id_old),
    )
    conn.execute(
        "UPDATE tasks SET created_at = ? WHERE id = ?",
        ("2000-01-02 00:00:00", id_new),
    )
    conn.commit()

    claimed = claim_task(conn)
    assert claimed is not None
    assert claimed["id"] == id_old


def test_update_step_running_sets_started_at_not_finished_at(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    step = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ?", (task_id,)
    ).fetchone()
    step_id = step["id"]

    update_step(conn, step_id, "running")
    row = conn.execute(
        "SELECT status, started_at, finished_at FROM task_steps WHERE id = ?",
        (step_id,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["started_at"] is not None
    assert row["finished_at"] is None

    update_step(conn, step_id, "done", "ok")
    row = conn.execute(
        "SELECT status, finished_at FROM task_steps WHERE id = ?",
        (step_id,),
    ).fetchone()
    assert row["status"] == "done"
    assert row["finished_at"] is not None


def test_update_step_failed_sets_finished_at(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ?", (task_id,)
    ).fetchone()["id"]

    update_step(conn, step_id, "running")
    update_step(conn, step_id, "failed", "boom")
    row = conn.execute(
        "SELECT status, finished_at FROM task_steps WHERE id = ?",
        (step_id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["finished_at"] is not None


def test_update_step_persists_scrubbed_output_at_rest(tmp_path):
    """Invariant: task_steps.output uses boundary4 (URLs/credentials), not display path scrub."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ?", (task_id,)
    ).fetchone()["id"]

    raw = "wrote /root/kb-agent/data/out.md via yt-dlp: cookiefile /root/secret/cookies.txt"
    update_step(conn, step_id, "done", raw)
    row = conn.execute(
        "SELECT output FROM task_steps WHERE id = ?", (step_id,)
    ).fetchone()
    assert "/root/kb-agent/data/out.md" in (row["output"] or "")
    assert "[path]" not in (row["output"] or "")
    assert "cookiefile" in (row["output"] or "").lower()


def test_fail_task_persists_scrubbed_error_at_rest(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    leaky = (
        "Traceback (most recent call last):\n"
        '  File "/root/kb-agent/agent/pipelines/video.py", line 9\n'
        "RuntimeError: disk at /root/kb-agent/tmp/foo\n"
    )
    fail_task(conn, task_id, leaky)
    row = conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert "Traceback" in (row["error"] or "")
    assert "/root/kb-agent/tmp/foo" in (row["error"] or "")
    assert "RuntimeError" in (row["error"] or "")


def test_fail_task_redacts_credential_shapes_at_write(tmp_path):
    """Invariant: tasks.error must not store raw api_key/Bearer values at write time."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    fail_task(
        conn,
        task_id,
        "subprocess stderr: api_key=abc123 refused; header Bearer xyz",
    )
    row = conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    err = row["error"] or ""
    assert "abc123" not in err
    assert "xyz" not in err or "Bearer [redacted]" in err
    assert "[credential redacted]" in err
    assert "Bearer [redacted]" in err


def test_persist_visible_error_redacts_cookie_session_pairs():
    """Invariant: cookie-style name=value secrets are scrubbed before durable storage."""
    from agent.db import _persist_visible_error

    out = _persist_visible_error("err sessionid=supersecret token")
    assert "supersecret" not in out
    assert "sessionid=[REDACTED]" in out


def test_fail_task_redacts_cookie_in_steperror_stderr_slice(tmp_path):
    """stderr[:200] embedded in StepError messages must not leak cookies at persist."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "instagram", "slug", ["only"])
    stderr_snippet = "ERROR: auth failed sessionid=leaked-cookie-value9; retry"
    fail_task(
        conn,
        task_id,
        f"yt-dlp reel download failed: {stderr_snippet[:200]}",
    )
    row = conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    err = row["error"] or ""
    assert "leaked-cookie-value9" not in err
    assert "sessionid=[REDACTED]" in err


def test_pause_task_persists_scrubbed_reason_at_rest(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    pause_task(conn, task_id, "cookies missing at /root/kb-agent/cookies/yt_cookies.txt")
    row = conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert "/root/kb-agent/cookies/yt_cookies.txt" in (row["error"] or "")
    assert "[path]" not in (row["error"] or "")


_TOKENIZED_CDN = "https://cdn.example.com/video.mp4?token=SECRET&expires=999"


def test_fail_task_redacts_http_url_query_at_write(tmp_path):
    """Invariant: tasks.error strips URL query params at write time."""
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    fail_task(conn, task_id, f"StepError: fetch failed {_TOKENIZED_CDN}")
    row = conn.execute("SELECT error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    err = row["error"] or ""
    assert "token=SECRET" not in err
    assert "?" not in err


def test_pause_task_redacts_http_url_query_at_write(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    pause_task(conn, task_id, f"cookies invalid for {_TOKENIZED_CDN}")
    row = conn.execute("SELECT error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    err = row["error"] or ""
    assert "token=SECRET" not in err
    assert "?" not in err


def test_update_step_redacts_url_bearing_metadata_output_at_write(tmp_path):
    """Invariant: task_steps.output redacts embedded http URLs at write time."""
    import json

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "youtube", "slug", ["extract_metadata"])
    step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ?", (task_id,)
    ).fetchone()["id"]
    payload = json.dumps(
        {
            "title": "Episode",
            "description": f"Chapters at {_TOKENIZED_CDN}",
        }
    )
    update_step(conn, step_id, "done", payload)
    row = conn.execute("SELECT output FROM task_steps WHERE id = ?", (step_id,)).fetchone()
    stored = row["output"] or ""
    assert "token=SECRET" not in stored
    assert "?" not in stored


def test_update_step_paused_sets_finished_at(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row

    task_id = create_task(conn, "https://example.com", "web", "slug", ["only"])
    step_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ?", (task_id,)
    ).fetchone()["id"]

    update_step(conn, step_id, "running")
    update_step(conn, step_id, "paused", "waiting")
    row = conn.execute(
        "SELECT status, finished_at FROM task_steps WHERE id = ?",
        (step_id,),
    ).fetchone()
    assert row["status"] == "paused"
    assert row["finished_at"] is not None


def _batch_test_config(tmp_path) -> Config:
    config = Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_parent_media_complete_fail_closed_without_raw_path(tmp_path):
    from agent.db import ensure_parent_media_deferral, parent_media_complete, queue_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    parent = "db_only_no_raw"
    ensure_parent_media_deferral(conn, parent, "https://blog.substack.com/p/x", "substack_text", 1)
    ingest_id, _ = queue_ingest(
        conn, "https://example.com/i.png", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=1 WHERE id=?",
        (ingest_id,),
    )
    conn.commit()
    assert parent_media_complete(conn, parent, raw_path=None) is False
    assert parent_media_complete(conn, parent, raw_path=None, allow_db_only=True) is True


def test_parent_media_complete_rejects_marker_only_block(tmp_path):
    from agent.db import ensure_parent_media_deferral, media_append_marker, parent_media_complete, queue_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "marker_only_parent"
    ensure_parent_media_deferral(conn, parent, "https://blog.substack.com/p/m", "substack_text", 1)
    ingest_id, _ = queue_ingest(
        conn, "https://example.com/i.png", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=1 WHERE id=?",
        (ingest_id,),
    )
    conn.commit()
    raw = tmp_path / f"{parent}_raw.md"
    raw.write_text(f"# article\n\n{media_append_marker(ingest_id)}\n", encoding="utf-8")
    assert parent_media_complete(conn, parent, raw_path=raw) is False


def test_queue_ingest_resets_dispatched_siblings_when_media_changed(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "reset_siblings"
    url = "https://blog.substack.com/p/reset"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1.png", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=1 WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()

    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    assert get_parent_media_deferral(conn, parent)["media_changed"] == 1
    queue_ingest(
        conn, "https://example.com/i2.png", "substack_image", "i2",
        parent_slug=parent, media_kind="image",
    )
    rows = conn.execute(
        "SELECT status, media_appended, dispatched_at FROM ingest_queue WHERE parent_slug=?",
        (parent,),
    ).fetchall()
    assert all(r["status"] == "queued" for r in rows)
    assert all(r["media_appended"] == 0 for r in rows)
    assert all(r["dispatched_at"] is None for r in rows)


def test_deferral_growth_requeue_clears_dispatched_at(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "growth_requeue_clear"
    url = "https://blog.substack.com/p/growth-requeue"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/i1.png", "substack_image", "i1",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at=datetime('now') WHERE parent_slug=?",
        (parent,),
    )
    conn.commit()

    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    row = conn.execute(
        "SELECT status, dispatched_at FROM ingest_queue WHERE parent_slug=?",
        (parent,),
    ).fetchone()
    assert row["status"] == "queued"
    assert row["dispatched_at"] is None
    assert get_parent_media_deferral(conn, parent)["media_changed"] == 1


def test_deferral_media_landing_anchor_ignores_non_dispatched_rows(tmp_path):
    """INVARIANT: landing anchor only considers dispatched (or failed+dispatched_at) siblings."""
    from agent.db import _deferral_media_landing_anchor, ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "landing_anchor_filter"
    url = "https://blog.substack.com/p/anchor-filter"
    ensure_parent_media_deferral(conn, parent, url, "substack_text", 2)
    deferral = get_parent_media_deferral(conn, parent)
    conn.execute(
        "UPDATE parent_media_deferral SET updated_at = '2024-01-15 10:00:00' WHERE parent_slug = ?",
        (parent,),
    )
    deferral = get_parent_media_deferral(conn, parent)

    queue_ingest(
        conn, "https://example.com/stale.png", "substack_image", "stale",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='queued', "
        "dispatched_at='2024-06-01 12:00:00' WHERE slug='stale'",
    )
    queue_ingest(
        conn, "https://example.com/live.png", "substack_image", "live",
        parent_slug=parent, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', "
        "dispatched_at='2024-01-15 11:00:00' WHERE slug='live'",
    )
    conn.commit()

    anchor = _deferral_media_landing_anchor(conn, parent, deferral)
    assert anchor == "2024-01-15 11:00:00"


def test_terminal_after_anchor_treats_same_second_as_fresh(tmp_path):
    """INVARIANT: terminal-after-anchor probes use deferral_cycle_cmp (>=), not strict >."""
    from agent.db import (
        _parent_terminal_after_anchor,
        _synthesize_terminal_after_anchor,
        ensure_parent_media_deferral,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "anchor_same_second"
    url = "https://blog.substack.com/p/anchor-same"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = "2024-03-01 12:00:00"

    parent_id = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="completed",
    )
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = NULL, updated_at = ? WHERE id IN (?, ?)",
        (anchor, parent_id, syn_id),
    )
    conn.commit()

    assert _parent_terminal_after_anchor(conn, slug, anchor) is True
    assert _synthesize_terminal_after_anchor(conn, slug, anchor) is True


def test_dispatch_synthesize_redispatches_stale_done_under_media_changed(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/x"
    slug = "stale_done_parent"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    done_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="done",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (anchor, done_id),
    )
    conn.commit()
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.commit()

    config = _batch_test_config(tmp_path)
    assert find_active_task(conn, slug, "substack_text_synthesize") is None
    result = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert result.get("duplicate") is not True
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE slug=?", (slug,)).fetchone()[0] == 2
    old = conn.execute("SELECT status FROM tasks WHERE id=?", (done_id,)).fetchone()[0]
    assert old == "failed"


def test_deferral_growth_supersedes_pending_synthesize(tmp_path):
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "pending_super_parent"
    url = "https://blog.substack.com/p/pending-super"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    pending_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="pending",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (anchor, pending_id),
    )
    conn.commit()

    ensure_parent_media_deferral(conn, slug, url, "substack_text", 2)
    conn.commit()

    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (pending_id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert "superseded by deferral media cycle" in (row["error"] or "")
    assert find_active_task(conn, slug, "substack_text_synthesize") is None


def test_retry_task_refuses_superseded_failed_synthesize(tmp_path):
    """INVARIANT: retry_task must not resurrect terminal superseded *_synthesize rows."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/no-retry-syn"
    slug = "no_retry_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    deferral = get_parent_media_deferral(conn, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (deferral["updated_at"], syn_id),
    )
    conn.commit()
    supersede_stale_deferral_synthesize_tasks(conn, slug, deferral)
    conn.commit()

    before = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (syn_id,),
    ).fetchone()
    assert before["status"] == "failed"
    assert "superseded" in (before["error"] or "").lower()

    assert retry_task(conn, syn_id) is False
    after = conn.execute("SELECT status, error FROM tasks WHERE id = ?", (syn_id,)).fetchone()
    assert after["status"] == "failed"
    assert after["error"] == before["error"]


def test_complete_task_leaves_superseded_failed_synthesize_terminal(tmp_path):
    """Superseded *_synthesize rows must not be revived to done by stale consumers."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/zombie-syn"
    slug = "zombie_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    deferral = get_parent_media_deferral(conn, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (deferral["updated_at"], syn_id),
    )
    conn.commit()
    supersede_stale_deferral_synthesize_tasks(conn, slug, deferral)
    conn.commit()

    assert complete_task(conn, syn_id) is False
    row = conn.execute("SELECT status, error FROM tasks WHERE id = ?", (syn_id,)).fetchone()
    assert row["status"] == "failed"
    assert "superseded" in (row["error"] or "").lower()
    assert find_active_task(conn, slug, "substack_text_synthesize") is None
    assert has_completed_synthesize_task(conn, slug) is False


def test_has_completed_synthesize_task_ignores_superseded_terminal_done_row(tmp_path):
    """Terminal freshness queries must filter superseded errors on *_synthesize rows."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/syn-sup-done"
    slug = "syn_sup_done"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    deferral = get_parent_media_deferral(conn, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="done",
    )
    conn.execute(
        "UPDATE tasks SET error = ?, updated_at = datetime('now') WHERE id = ?",
        ("superseded by deferral media cycle", syn_id),
    )
    conn.commit()

    assert has_completed_synthesize_task(conn, slug) is False
    assert find_active_task(conn, slug, "substack_text_synthesize") is None


def test_find_active_task_terminal_synthesize_keeps_media_changed_without_raw_on_disk(
    tmp_path,
):
    """media_changed must not clear from DB-only parent_media_complete (missing raw bundle)."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://blog.substack.com/p/raw-missing"
    slug = "raw_missing_active"
    media_tmp = tmp_path / "tmp"
    media_tmp.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    ingest_id, _ = queue_ingest(
        conn, "https://example.com/m.png", "substack_image", "m1",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=1 WHERE id=?",
        (ingest_id,),
    )
    conn.commit()
    deferral = get_parent_media_deferral(conn, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="done",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        (deferral["updated_at"], syn_id),
    )
    conn.execute(
        "UPDATE parent_media_deferral SET media_changed = 1 WHERE parent_slug = ?",
        (slug,),
    )
    conn.commit()

    assert find_active_task(
        conn, slug, "substack_text_synthesize", temp_dir=media_tmp,
    ) is None
    assert int(get_parent_media_deferral(conn, slug)["media_changed"]) == 1


def test_find_active_task_terminal_synthesize_clears_media_changed_when_raw_landed(
    tmp_path,
):
    """Disk markers present: find_active_task may consume media_changed like has_completed_parent_synthesis."""
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    url = "https://blog.substack.com/p/raw-landed"
    slug = "raw_landed_active"
    media_tmp = tmp_path / "tmp"
    media_tmp.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    ingest_id, _ = queue_ingest(
        conn, "https://example.com/m.png", "substack_image", "m1",
        parent_slug=slug, media_kind="image",
    )
    conn.execute(
        "UPDATE ingest_queue SET status='dispatched', media_appended=1 WHERE id=?",
        (ingest_id,),
    )
    conn.commit()
    _stamp_parent_raw_media(conn, media_tmp, slug)
    deferral = get_parent_media_deferral(conn, slug)
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="done",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        (deferral["updated_at"], syn_id),
    )
    conn.execute(
        "UPDATE parent_media_deferral SET media_changed = 1 WHERE parent_slug = ?",
        (slug,),
    )
    conn.commit()

    assert find_active_task(
        conn, slug, "substack_text_synthesize", temp_dir=media_tmp,
    ) == syn_id
    assert int(get_parent_media_deferral(conn, slug)["media_changed"]) == 0


def test_find_active_task_supersedes_stale_pending_synthesize_after_media_changed(tmp_path):
    from agent.db import (
        ensure_parent_media_deferral,
        get_parent_media_deferral,
        mark_parent_media_changed,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/stale-pending"
    slug = "stale_pending_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor = get_parent_media_deferral(conn, slug)["updated_at"]
    pending_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="pending",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (anchor, pending_id),
    )
    conn.commit()

    mark_parent_media_changed(conn, slug)
    conn.commit()

    assert find_active_task(conn, slug, "substack_text_synthesize") is None
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (pending_id,)).fetchone()
    assert row[0] == "failed"

    config = _batch_test_config(tmp_path)
    result = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert result.get("duplicate") is not True


def test_find_active_task_accepts_synthesize_when_updated_at_equals_deferral(tmp_path):
    """Same-second SQLite timestamps: equal updated_at is generation-fresh (not superseded)."""
    from agent.db import (
        ensure_parent_media_deferral,
        get_parent_media_deferral,
        mark_parent_media_changed,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/same-second"
    slug = "same_second_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    mark_parent_media_changed(conn, slug)
    conn.commit()

    deferral = get_parent_media_deferral(conn, slug)
    assert int(deferral["media_changed"]) == 1
    anchor = deferral["updated_at"]

    pending_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="pending",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        (anchor, pending_id),
    )
    conn.commit()

    assert find_active_task(conn, slug, "substack_text_synthesize") == pending_id
    assert has_active_synthesize_task(conn, slug) is True
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (pending_id,)).fetchone()
    assert row[0] == "pending"


def test_find_active_task_supersedes_synthesize_strictly_older_than_deferral(tmp_path):
    from agent.db import (
        ensure_parent_media_deferral,
        get_parent_media_deferral,
        mark_parent_media_changed,
    )

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/older-than-deferral"
    slug = "older_than_deferral_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    anchor_before = get_parent_media_deferral(conn, slug)["updated_at"]
    pending_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="pending",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = datetime(?, '-1 day') WHERE id = ?",
        (anchor_before, pending_id),
    )
    conn.commit()

    mark_parent_media_changed(conn, slug)
    conn.commit()

    assert find_active_task(conn, slug, "substack_text_synthesize") is None
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (pending_id,)).fetchone()
    assert row[0] == "failed"


def test_dispatch_synthesize_idempotent_when_task_done(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/x"
    slug = "parent_slug"
    done_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="done",
    )
    config = _batch_test_config(tmp_path)
    result = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert result["duplicate"] is True
    assert result["task_id"] == done_id
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE slug=?", (slug,)).fetchone()[0] == 1


def test_dispatch_synthesize_redispatches_after_failed_task(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/x"
    slug = "parent_slug"
    create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="failed",
    )
    config = _batch_test_config(tmp_path)
    result = dispatch_synthesize(
        url, config, conn, content_type="substack_text_synthesize", slug=slug,
    )
    assert result.get("duplicate") is not True
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE slug=?", (slug,)).fetchone()[0] == 2


def test_process_videos_download_failure_sets_failed(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=standAlone", "youtube", "yt_stand")
    row = conn.execute("SELECT * FROM ingest_queue").fetchone()
    q = dict(row)
    config = _batch_test_config(tmp_path)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (None, "download failed"))

    failed, dispatched, _ = batch_ingest._process_videos([q], config, conn, 1)
    assert failed == 1
    assert dispatched == 0
    assert conn.execute("SELECT status FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()["status"] == "failed"


def test_process_videos_batch_transcribe_failure_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=tx1", "youtube", "yt_tx")
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_tx.mp4"
    fake_video.write_bytes(b"x" * 200)

    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

    failed, dispatched, causes = batch_ingest._process_videos([q], config, conn, 2)
    assert failed == 1
    assert dispatched == 0
    assert conn.execute("SELECT status FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()["status"] == "failed"
    assert any("batch_transcribe" in c for c in causes)


def test_process_videos_missing_merged_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=tx2", "youtube", "yt_merge")
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_merge.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

    failed, _, _ = batch_ingest._process_videos([q], config, conn, 3)
    assert failed == 1
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()
    assert row["status"] == "failed"
    assert "merged transcript" in row["error"]


def test_process_videos_upload_failure_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=tx3", "youtube", "yt_up")
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_up.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{q['id']}.json").write_text('{"segments": [{"text": "hello"}]}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
    def _upload_fail(*_a, **_k):
        raise RuntimeError("minio down")

    monkeypatch.setattr(batch_ingest, "upload_file_to_minio", _upload_fail)
    monkeypatch.setattr(
        batch_ingest, "dispatch_synthesize",
        lambda *a, **k: {"status": "pending", "task_id": "never"},
    )

    failed, dispatched, _ = batch_ingest._process_videos([q], config, conn, 4)
    assert failed == 1
    assert dispatched == 0
    assert conn.execute("SELECT status FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()["status"] == "failed"


def test_process_videos_dispatch_failure_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=tx4", "youtube", "yt_dispatch")
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_dispatch.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{q['id']}.json").write_text('{"segments": [{"text": "hello"}]}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(
        batch_ingest, "upload_file_to_minio", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        batch_ingest,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "error", "error": "unknown content_type: bad"},
    )

    failed, dispatched, causes = batch_ingest._process_videos([q], config, conn, 5)
    assert failed == 1
    assert dispatched == 0
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()
    assert row["status"] == "failed"
    assert "unknown content_type" in row["error"]
    assert any("unknown content_type" in c for c in causes)


def test_process_videos_dispatch_success_marks_dispatched(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://www.youtube.com/watch?v=tx5", "youtube", "yt_ok")
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_ok.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{q['id']}.json").write_text('{"segments": [{"text": "hello"}]}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(batch_ingest, "upload_file_to_minio", lambda *_a, **_k: None)
    monkeypatch.setattr(
        batch_ingest,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "pending", "task_id": "synth-1", "duplicate": True},
    )

    failed, dispatched, _ = batch_ingest._process_videos([q], config, conn, 6)
    assert failed == 0
    assert dispatched == 1
    assert conn.execute("SELECT status FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()["status"] == "dispatched"


def test_embedded_images_batch_ocr_failure_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(
        conn, "https://example.com/chart.png", "substack_image", "img_ocr",
        parent_slug="parent_a", media_kind="image",
    )
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "curl":
            dest = Path(cmd[cmd.index("-o") + 1])
            dest.write_bytes(b"\x00" * 200)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "batch_ocr.py" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 1, "", "ocr modal failed")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

    batch_ingest._process_embedded_images([q], config, conn, 5)
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "no OCR output"


def test_embedded_videos_transcribe_failure_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(
        conn, "https://www.youtube.com/watch?v=embedTx", "youtube", "yt_emb",
        parent_slug="parent_v", media_kind="video",
    )
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_emb.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 2, "", "transcribe failed")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

    batch_ingest._process_embedded_videos([q], config, conn, 6)
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()
    assert row["status"] == "failed"
    assert "batch_transcribe" in row["error"]


def test_embedded_videos_empty_transcript_sets_failed(tmp_path, monkeypatch):
    import subprocess

    import scripts.batch_ingest as batch_ingest

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(
        conn, "https://www.youtube.com/watch?v=emptyTx", "youtube", "yt_empty",
        parent_slug="parent_v2", media_kind="video",
    )
    q = dict(conn.execute("SELECT * FROM ingest_queue").fetchone())
    config = _batch_test_config(tmp_path)
    fake_video = config.temp_dir / "yt_empty.mp4"
    fake_video.write_bytes(b"x" * 200)
    monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

    def fake_run(cmd, **kwargs):
        if "batch_transcribe.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{q['id']}.json").write_text('{"segments": [{"text": "  "}]}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

    batch_ingest._process_embedded_videos([q], config, conn, 7)
    row = conn.execute("SELECT status, error FROM ingest_queue WHERE id=?", (q["id"],)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "empty transcript"


def test_lookup_parent_article_metadata_tie_breaks_on_id(tmp_path):
    from agent.db import _lookup_parent_article_metadata

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    slug = "dup_slug_tiebreak"
    older_id = create_task(
        conn, "https://example.com/older", "web", slug, ["download"],
    )
    newer_id = create_task(
        conn, "https://example.com/newer", "web", slug, ["download"],
    )
    ts = "2024-06-01 12:00:00"
    conn.execute(
        "UPDATE tasks SET created_at = ? WHERE id IN (?, ?)",
        (ts, older_id, newer_id),
    )
    conn.execute("UPDATE tasks SET id = ? WHERE id = ?", ("id-aaa", older_id))
    conn.execute("UPDATE tasks SET id = ? WHERE id = ?", ("id-bbb", newer_id))
    conn.commit()

    source_url, content_type = _lookup_parent_article_metadata(conn, slug)
    assert source_url == "https://example.com/older"
    assert content_type == "web"


def test_bootstrap_parent_media_deferral_manifest_tie_breaks_on_id(tmp_path):
    from agent.db import (
        bootstrap_parent_media_deferral_from_manifest,
        get_parent_media_deferral,
    )

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    slug = "manifest_tiebreak_parent"
    older_id = create_task(
        conn, "https://example.com/first", "substack_text", slug, ["synthesize"],
    )
    newer_id = create_task(
        conn, "https://example.com/second", "substack_text", slug, ["synthesize"],
    )
    ts = "2024-06-01 12:00:00"
    conn.execute(
        "UPDATE tasks SET created_at = ? WHERE id IN (?, ?)",
        (ts, older_id, newer_id),
    )
    conn.execute("UPDATE tasks SET id = ? WHERE id = ?", ("man-aaa", older_id))
    conn.execute("UPDATE tasks SET id = ? WHERE id = ?", ("man-bbb", newer_id))
    conn.commit()

    manifest = tmp_path / f"{slug}_media.json"
    manifest.write_text(json.dumps({"total": 2}), encoding="utf-8")
    bootstrap_parent_media_deferral_from_manifest(conn, slug, tmp_path)
    row = get_parent_media_deferral(conn, slug)
    assert row is not None
    assert row["source_url"] == "https://example.com/first"
    assert row["content_type"] == "substack_text"


def test_dispatch_synthesize_concurrent_single_active_owner(tmp_path):
    """Two concurrent dispatch_synthesize calls yield exactly one in-flight owner."""
    db_path = tmp_path / "agent.db"
    init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/concurrent"
    slug = "concurrent_parent"
    config = _batch_test_config(tmp_path)
    barrier = threading.Barrier(2, timeout=10)
    results: list[dict] = []
    lock = threading.Lock()

    def worker():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            barrier.wait()
            with lock:
                result = dispatch_synthesize(
                    url,
                    config,
                    conn,
                    content_type="substack_text_synthesize",
                    slug=slug,
                )
            results.append(result)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert len(results) == 2
    task_ids = {r["task_id"] for r in results}
    assert len(task_ids) == 1
    dup_flags = [r.get("duplicate") for r in results]
    assert dup_flags.count(True) == 1
    assert dup_flags.count(None) + dup_flags.count(False) == 1

    verify = sqlite3.connect(str(db_path))
    in_flight = verify.execute(
        "SELECT COUNT(*) FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        "AND status IN ('pending', 'in_progress', 'paused')",
        (slug,),
    ).fetchone()[0]
    assert in_flight == 1
    verify.close()


def test_create_delegate_task_concurrent_single_active_child(tmp_path):
    """Concurrent delegate child resolution yields one in-flight row per (slug, content_type)."""
    from agent.db import create_delegate_task

    db_path = tmp_path / "agent.db"
    init_db(db_path, AGENT_SCHEMA)
    slug = "batch_parent_alpha"
    pipeline = "book_epub"
    file_path = str(tmp_path / "alpha.epub")
    Path(file_path).write_text("epub-bytes", encoding="utf-8")
    steps = ["extract_text", "synthesize_note"]
    barrier = threading.Barrier(2, timeout=10)
    results: list[str] = []
    errors: list[BaseException] = []

    def worker():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            barrier.wait()
            task_id = create_delegate_task(conn, file_path, pipeline, slug, steps)
            results.append(task_id)
        except BaseException as exc:
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert not errors
    assert len(results) == 2
    assert len(set(results)) == 1

    verify = sqlite3.connect(str(db_path))
    in_flight = verify.execute(
        "SELECT COUNT(*) FROM tasks WHERE slug = ? AND content_type = ? "
        "AND status IN ('pending', 'in_progress', 'paused')",
        (slug, pipeline),
    ).fetchone()[0]
    assert in_flight == 1
    verify.close()


def test_has_stale_deferral_synthesize_terminal_ignores_superseded_rows(tmp_path):
    """INVARIANT: stale synthesize probe shares _TERMINAL_SUPERSEDE_FILTER with sibling probes."""
    from agent.db import (
        _has_stale_deferral_synthesize_terminal,
        ensure_parent_media_deferral,
        get_parent_media_deferral,
    )
    from agent.redact import url_for_display

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = url_for_display("https://blog.substack.com/p/stale-sup")
    slug = "stale_sup_syn"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    deferral = get_parent_media_deferral(conn, slug)
    anchor = deferral["updated_at"]
    conn.execute(
        "UPDATE parent_media_deferral SET updated_at = '2024-06-01 00:00:00' WHERE parent_slug = ?",
        (slug,),
    )
    anchor = "2024-06-02 00:00:00"

    syn_sup = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = ?, updated_at = '2024-06-01 12:00:00' WHERE id = ?",
        ("superseded by deferral media cycle", syn_sup),
    )
    conn.commit()

    assert _has_stale_deferral_synthesize_terminal(conn, slug, anchor) is False

    syn_valid = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = NULL, updated_at = '2024-06-01 11:00:00' WHERE id = ?",
        (syn_valid,),
    )
    conn.commit()

    assert _has_stale_deferral_synthesize_terminal(conn, slug, anchor) is True


def test_terminal_probes_return_valid_when_superseded_row_is_older(tmp_path):
    """INVARIANT: terminal probes filter superseded rows before LIMIT 1 (older superseded first)."""
    from agent.db import (
        _non_superseded_parent_terminal,
        _parent_terminal_after_anchor,
        _synthesize_terminal_after_anchor,
        ensure_parent_media_deferral,
        get_parent_media_deferral,
    )
    from agent.redact import url_for_display

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = url_for_display("https://blog.substack.com/p/term-order")
    slug = "term_sup_first"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    conn.execute(
        "UPDATE parent_media_deferral SET updated_at = '2024-01-01 00:00:00' WHERE parent_slug = ?",
        (slug,),
    )
    deferral = get_parent_media_deferral(conn, slug)
    anchor = deferral["updated_at"]

    bad_parent = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="completed",
    )
    good_parent = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="completed",
    )
    bad_syn = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    good_syn = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = ?, created_at = '2024-01-01 00:00:01', "
        "updated_at = '2024-01-02 00:00:01' WHERE id IN (?, ?)",
        ("superseded by deferral media cycle", bad_parent, bad_syn),
    )
    conn.execute(
        "UPDATE tasks SET error = NULL, created_at = '2024-01-01 00:00:02', "
        "updated_at = '2024-01-02 00:00:02' WHERE id IN (?, ?)",
        (good_parent, good_syn),
    )
    conn.commit()

    assert has_completed_synthesize_task(conn, slug) is True
    assert _non_superseded_parent_terminal(conn, slug) is True
    assert _parent_terminal_after_anchor(conn, slug, anchor) is True
    assert _synthesize_terminal_after_anchor(conn, slug, anchor) is True


def test_terminal_probes_return_valid_when_superseded_row_is_newer(tmp_path):
    """INVARIANT: superseded row must not win when a valid terminal exists (newer superseded)."""
    from agent.db import _non_superseded_parent_terminal
    from agent.redact import url_for_display

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = url_for_display("https://blog.substack.com/p/term-newer-bad")
    slug = "term_sup_last"
    good_parent = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="completed",
    )
    bad_parent = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="completed",
    )
    good_syn = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    bad_syn = create_task(
        conn, url, "substack_text_synthesize", slug, ["synthesize_note"], status="completed",
    )
    conn.execute(
        "UPDATE tasks SET error = NULL, created_at = '2024-01-01 00:00:01', "
        "updated_at = '2024-01-02 00:00:01' WHERE id IN (?, ?)",
        (good_parent, good_syn),
    )
    conn.execute(
        "UPDATE tasks SET error = ?, created_at = '2024-01-01 00:00:02', "
        "updated_at = '2024-01-02 00:00:02' WHERE id IN (?, ?)",
        ("superseded by deferral media cycle", bad_parent, bad_syn),
    )
    conn.commit()

    assert has_completed_synthesize_task(conn, slug) is True
    assert _non_superseded_parent_terminal(conn, slug) is True


def test_bootstrap_parent_media_deferral_manifest_preserves_media_changed(tmp_path):
    from agent.db import (
        bootstrap_parent_media_deferral_from_manifest,
        get_parent_media_deferral,
    )

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    slug = "manifest_media_changed"
    manifest = tmp_path / f"{slug}_media.json"
    manifest.write_text(
        json.dumps(
            {
                "total": 2,
                "url": "https://blog.substack.com/p/post",
                "content_type": "substack_text",
                "media_changed": True,
            }
        ),
        encoding="utf-8",
    )
    bootstrap_parent_media_deferral_from_manifest(conn, slug, tmp_path)
    row = get_parent_media_deferral(conn, slug)
    assert row is not None
    assert int(row["media_changed"]) == 1


def test_bootstrap_parent_media_deferral_normalizes_manifest_url(tmp_path):
    from agent.db import (
        bootstrap_parent_media_deferral_from_manifest,
        get_parent_media_deferral,
    )
    from agent.redact import url_for_display

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    slug = "manifest_url_norm"
    raw_url = "https://user:sekret@blog.substack.com/p/post?token=SECRET#frag"
    manifest = tmp_path / f"{slug}_media.json"
    manifest.write_text(
        json.dumps(
            {
                "total": 1,
                "url": raw_url,
                "content_type": "substack_text",
            }
        ),
        encoding="utf-8",
    )
    bootstrap_parent_media_deferral_from_manifest(conn, slug, tmp_path)
    row = get_parent_media_deferral(conn, slug)
    assert row is not None
    assert row["source_url"] == url_for_display(raw_url)
    assert "sekret" not in row["source_url"]
    assert "?" not in row["source_url"]


def test_resume_tasks_by_reason_skips_active_deferral_resumes_stale_row(tmp_path):
    """Bulk resume: active cycle (pending media) blocked; stale deferral row may resume."""
    from agent.db import ensure_parent_media_deferral

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    url = "https://blog.substack.com/p/resume-stale"
    active_slug = "bulk_resume_active"
    stale_slug = "bulk_resume_stale"
    pause_reason = "firecrawl rate limit — retry after cooldown"

    ensure_parent_media_deferral(conn, active_slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/pending.png", "substack_image", "pend",
        parent_slug=active_slug, media_kind="image",
    )
    ensure_parent_media_deferral(conn, stale_slug, url, "substack_text", 1)
    queue_ingest(
        conn, "https://example.com/done.png", "substack_image", "done",
        parent_slug=stale_slug, media_kind="image", status="dispatched",
    )

    active_id = create_task(
        conn, url, "substack_text", active_slug,
        ["substack_cleanup", "synthesize_note"], status="paused",
    )
    stale_id = create_task(
        conn, url, "substack_text", stale_slug,
        ["substack_cleanup", "synthesize_note"], status="paused",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id IN (?, ?)",
        (pause_reason, active_id, stale_id),
    )
    conn.commit()

    assert resume_tasks_by_reason(conn, "%firecrawl%") == 1
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (active_id,),
    ).fetchone()[0] == "paused"
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (stale_id,),
    ).fetchone()[0] == "pending"
