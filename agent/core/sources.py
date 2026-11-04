"""Watched-source registry — URL + cadence with durable last-seen state."""

from __future__ import annotations

import sqlite3
import uuid
from urllib.parse import urlparse


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("url required")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("url must be an http or https URL with a host")
    return url


def _validate_cadence(cadence: int) -> int:
    if not isinstance(cadence, int) or isinstance(cadence, bool):
        raise ValueError("cadence must be a positive integer (seconds)")
    if cadence <= 0:
        raise ValueError("cadence must be a positive integer (seconds)")
    return cadence


def _source_from_row(row: sqlite3.Row) -> dict:
    return {
        "source_id": row["id"],
        "url": row["url"],
        "cadence": row["cadence"],
        "last_seen": row["last_seen"],
        "last_content_hash": row["last_content_hash"],
    }


def register_source(agent_conn: sqlite3.Connection, url: str, cadence: int) -> str:
    """Register a watched source. Returns the new source_id.

    Raises ValueError if the URL is already registered (idempotency is the
    caller's concern — a duplicate is a domain error, not a silent no-op).
    """
    url = _validate_url(url)
    cadence = _validate_cadence(cadence)
    existing = agent_conn.execute(
        "SELECT id FROM sources WHERE url = ?", (url,)
    ).fetchone()
    if existing is not None:
        raise ValueError(f"source already registered: {url}")
    source_id = str(uuid.uuid4())
    with agent_conn:
        agent_conn.execute(
            """
            INSERT INTO sources (id, url, cadence, last_seen, last_content_hash)
            VALUES (?, ?, ?, NULL, NULL)
            """,
            (source_id, url, cadence),
        )
    return source_id


def lookup_all_sources(agent_conn: sqlite3.Connection) -> list[dict]:
    """Return all registered sources with their persisted state."""
    rows = agent_conn.execute(
        "SELECT id, url, cadence, last_seen, last_content_hash FROM sources ORDER BY created_at"
    ).fetchall()
    return [_source_from_row(row) for row in rows]


def update_source_state(
    agent_conn: sqlite3.Connection,
    source_id: str,
    last_seen: str,
    last_content_hash: str,
) -> dict | None:
    """Update last-seen timestamp and content hash for a source."""
    with agent_conn:
        agent_conn.execute(
            """
            UPDATE sources
            SET last_seen = ?, last_content_hash = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (last_seen, last_content_hash, source_id),
        )
    row = agent_conn.execute(
        "SELECT id, url, cadence, last_seen, last_content_hash FROM sources WHERE id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        return None
    return _source_from_row(row)
