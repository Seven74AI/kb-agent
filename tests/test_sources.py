"""Tests for watched-source registry and last-seen persistence."""

from __future__ import annotations

import sqlite3

import pytest

from agent.core.sources import (
    lookup_all_sources,
    register_source,
    update_source_state,
)
from agent.db import AGENT_SCHEMA, init_db


@pytest.fixture
def agent_conn(tmp_path):
    db_path = tmp_path / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_register_source_persists_url_and_cadence(agent_conn, tmp_path):
    source_id = register_source(
        agent_conn,
        "https://example.com/feed",
        86400,
    )
    assert source_id

    agent_conn.close()
    db_path = tmp_path / "data" / "agent.db"
    reopened = init_db(db_path, AGENT_SCHEMA)
    reopened.row_factory = sqlite3.Row

    sources = lookup_all_sources(reopened)
    assert len(sources) == 1
    row = sources[0]
    assert row["source_id"] == source_id
    assert row["url"] == "https://example.com/feed"
    assert row["cadence"] == 86400
    assert row["last_seen"] is None
    assert row["last_content_hash"] is None
    reopened.close()


def test_update_source_state_returns_exact_stored_values(agent_conn):
    source_id = register_source(agent_conn, "https://news.example.com/rss", 3600)
    last_seen = "2026-11-04T12:00:00+00:00"
    content_hash = "sha256:abc123"

    updated = update_source_state(agent_conn, source_id, last_seen, content_hash)
    assert updated == {
        "source_id": source_id,
        "url": "https://news.example.com/rss",
        "cadence": 3600,
        "last_seen": last_seen,
        "last_content_hash": content_hash,
    }

    listed = lookup_all_sources(agent_conn)
    assert listed[0] == updated


def test_register_source_rejects_invalid_url(agent_conn):
    with pytest.raises(ValueError, match="url"):
        register_source(agent_conn, "not-a-url", 60)
    with pytest.raises(ValueError, match="url"):
        register_source(agent_conn, "", 60)
    with pytest.raises(ValueError, match="url"):
        register_source(agent_conn, "ftp://example.com/feed", 60)
    with pytest.raises(ValueError, match="url"):
        register_source(agent_conn, "https:///no-host", 60)
    assert lookup_all_sources(agent_conn) == []


def test_register_source_rejects_duplicate_url(agent_conn):
    register_source(agent_conn, "https://example.com/feed", 3600)
    with pytest.raises(ValueError, match="already registered"):
        register_source(agent_conn, "https://example.com/feed", 86400)
    # One row only — the duplicate never hit the table.
    assert len(lookup_all_sources(agent_conn)) == 1


def test_register_source_rejects_invalid_cadence(agent_conn):
    with pytest.raises(ValueError, match="cadence"):
        register_source(agent_conn, "https://example.com/a", 0)
    with pytest.raises(ValueError, match="cadence"):
        register_source(agent_conn, "https://example.com/a", -1)
    with pytest.raises(ValueError, match="cadence"):
        register_source(agent_conn, "https://example.com/a", True)
    with pytest.raises(ValueError, match="cadence"):
        register_source(agent_conn, "https://example.com/a", 60.0)
    assert lookup_all_sources(agent_conn) == []


def test_update_source_state_unknown_id(agent_conn):
    assert update_source_state(agent_conn, "missing-id", "2026-01-01T00:00:00Z", "hash") is None
