"""Tests for ingest queue dedup/tag updates and schema migration."""

import sqlite3
from pathlib import Path

import pytest

from agent.db import (
    AGENT_SCHEMA,
    _migrate,
    count_pending_media,
    init_db,
    queue_ingest,
    serialize_tags,
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
    for required in ("slug", "tags", "error", "created_at", "dispatched_at", "parent_slug", "media_kind"):
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

    # a failed sibling also counts as "done" (don't block forever)
    conn.execute("UPDATE ingest_queue SET status='failed' WHERE url='https://example.com/i2'")
    conn.commit()
    assert count_pending_media(conn, "p") == 0
