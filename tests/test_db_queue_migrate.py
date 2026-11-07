"""Tests for ingest queue dedup/tag updates and schema migration."""

import sqlite3
from pathlib import Path

import pytest

from agent.core.resume import retry_task
from agent.db import (
    AGENT_SCHEMA,
    _migrate,
    claim_task,
    count_pending_media,
    create_task,
    get_task_steps,
    init_db,
    queue_ingest,
    recover_failed_media,
    resume_tasks_by_reason,
    serialize_tags,
    update_step,
)


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


def test_retry_awaiting_media_unblocks_embedded_step_not_synthesize(tmp_path):
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
        ("awaiting media: 1 images enqueued", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'substack_cleanup'",
        (task_id,),
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = 'defer' WHERE id = ?",
        (defer_step_id,),
    )
    conn.commit()

    assert retry_task(conn, task_id) is True
    steps = {s["name"]: s["status"] for s in get_task_steps(conn, task_id)}
    assert steps["substack_embedded_media"] == "pending"
    assert steps["synthesize_note"] == "pending"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "pending"


def test_maybe_dispatch_parent_synthesize_manifest_count_and_idempotent(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest

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
    (config.temp_dir / f"{parent}_media.json").write_text(
        '{"url": "https://blog.substack.com/p/x", "content_type": "substack_text", "total": 2}',
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
        '{"url": "https://blog.substack.com/p/x", "content_type": "substack_text", "total": 1}',
        encoding="utf-8",
    )
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert calls == [parent]
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
    assert len(calls) == 1


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
