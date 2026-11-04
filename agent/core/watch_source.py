"""Recurring source watch — poll pages, diff items, auto-submit new URLs."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agent.config import Config
from agent.core.submit import submit_url

# Appended to AGENT_SCHEMA in agent/db.py (also exported for tests that build schema).
WATCH_SOURCES_SCHEMA = """
CREATE TABLE IF NOT EXISTS watch_sources (
    id TEXT PRIMARY KEY,
    page_url TEXT NOT NULL UNIQUE,
    item_pattern TEXT NOT NULL,
    item_url_template TEXT NOT NULL,
    cadence_seconds INTEGER NOT NULL DEFAULT 86400,
    last_seen_json TEXT NOT NULL DEFAULT '[]',
    last_polled_at TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

SubmitFn = Callable[[str, Config, sqlite3.Connection], dict]
FetchHtml = Callable[[str], str]


def extract_items(html: str, item_pattern: str) -> frozenset[str]:
    """Return unique capture-group values from *html* matching *item_pattern*."""
    return frozenset(re.findall(item_pattern, html))


def diff_items(current: frozenset[str], last_seen: frozenset[str]) -> list[str]:
    """Sorted item ids present in *current* but not in *last_seen*."""
    return sorted(current - last_seen)


def items_to_urls(items: list[str], item_url_template: str) -> list[str]:
    """Expand item ids to submission URLs via ``{item}`` template placeholder."""
    return [item_url_template.format(item=item) for item in sorted(items)]


def parse_last_seen(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(data, list):
        return frozenset()
    return frozenset(str(x) for x in data)


def serialize_last_seen(items: frozenset[str]) -> str:
    return json.dumps(sorted(items))


def merge_last_seen_after_poll(
    last_seen: frozenset[str],
    current: frozenset[str],
    newly_submitted: frozenset[str],
) -> frozenset[str]:
    """Union prior last-seen, everything on the page now, and items we attempted."""
    return last_seen | current | newly_submitted


def is_poll_due(
    last_polled_at: str | None,
    cadence_seconds: int,
    now: datetime,
) -> bool:
    if last_polled_at is None:
        return True
    try:
        # SQLite datetime('now') format
        last = datetime.strptime(last_polled_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elapsed = (now - last).total_seconds()
    return elapsed >= cadence_seconds


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def register_watch_source(
    conn: sqlite3.Connection,
    page_url: str,
    *,
    item_pattern: str,
    item_url_template: str,
    cadence_seconds: int = 86400,
    fetch_html: FetchHtml | None = None,
    baseline: bool = True,
) -> dict[str, Any]:
    """Register a page to watch. Optionally baseline *last_seen* from a live fetch."""
    page_url = page_url.strip()
    if not page_url:
        raise ValueError("page_url required")
    if cadence_seconds < 1:
        raise ValueError("cadence_seconds must be >= 1")
    if "{item}" not in item_url_template:
        raise ValueError("item_url_template must contain {item}")

    existing = conn.execute(
        "SELECT id FROM watch_sources WHERE page_url = ?",
        (page_url,),
    ).fetchone()
    if existing:
        raise ValueError(f"watch source already registered: {page_url}")

    last_seen: frozenset[str] = frozenset()
    if baseline and fetch_html is not None:
        html = fetch_html(page_url)
        last_seen = extract_items(html, item_pattern)

    source_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            """INSERT INTO watch_sources
               (id, page_url, item_pattern, item_url_template, cadence_seconds, last_seen_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                page_url,
                item_pattern,
                item_url_template,
                cadence_seconds,
                serialize_last_seen(last_seen),
            ),
        )
    row = conn.execute(
        "SELECT * FROM watch_sources WHERE id = ?",
        (source_id,),
    ).fetchone()
    return _row_to_dict(row)


def get_watch_source(conn: sqlite3.Connection, page_url: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM watch_sources WHERE page_url = ?",
        (page_url.strip(),),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_watch_sources(conn: sqlite3.Connection, *, enabled_only: bool = True) -> list[dict[str, Any]]:
    if enabled_only:
        rows = conn.execute(
            "SELECT * FROM watch_sources WHERE enabled = 1 ORDER BY created_at"
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM watch_sources ORDER BY created_at").fetchall()
    return [_row_to_dict(r) for r in rows]


def poll_watch_source(
    conn: sqlite3.Connection,
    page_url: str,
    config: Config,
    *,
    fetch_html: FetchHtml,
    now: datetime | None = None,
    submit_fn: SubmitFn | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Poll one registered source; submit URLs for newly discovered items."""
    row = get_watch_source(conn, page_url)
    if row is None:
        raise ValueError(f"watch source not registered: {page_url}")
    if not row["enabled"]:
        return {"page_url": page_url, "skipped": True, "reason": "disabled"}

    now = now or datetime.now(timezone.utc)
    if not force and not is_poll_due(row["last_polled_at"], row["cadence_seconds"], now):
        return {"page_url": page_url, "skipped": True, "reason": "not_due"}

    submit = submit_fn or submit_url
    html = fetch_html(page_url)
    current = extract_items(html, row["item_pattern"])
    last_seen = parse_last_seen(row["last_seen_json"])
    new_items = diff_items(current, last_seen)
    urls = items_to_urls(new_items, row["item_url_template"])

    submissions: list[dict[str, Any]] = []
    for url in urls:
        submissions.append(submit(url, config, conn))

    merged = merge_last_seen_after_poll(last_seen, current, frozenset(new_items))
    polled_at = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            """UPDATE watch_sources
               SET last_seen_json = ?, last_polled_at = ?
               WHERE page_url = ?""",
            (serialize_last_seen(merged), polled_at, page_url),
        )

    return {
        "page_url": page_url,
        "new_items": new_items,
        "submissions": submissions,
        "last_seen_count": len(merged),
    }


def poll_all_due_sources(
    conn: sqlite3.Connection,
    config: Config,
    *,
    fetch_html: FetchHtml,
    now: datetime | None = None,
    submit_fn: SubmitFn | None = None,
) -> list[dict[str, Any]]:
    """Poll every enabled source whose cadence has elapsed."""
    now = now or datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for source in list_watch_sources(conn, enabled_only=True):
        outcome = poll_watch_source(
            conn,
            source["page_url"],
            config,
            fetch_html=fetch_html,
            now=now,
            submit_fn=submit_fn,
        )
        if not outcome.get("skipped"):
            results.append(outcome)
    return results


def default_fetch_html(url: str) -> str:
    """Fetch page HTML (stdlib). Used by scheduler and CLI."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "kb-agent-watch/1.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read().decode("utf-8", "replace")


async def run_watch_scheduler(
    config: Config,
    agent_conn: sqlite3.Connection,
    *,
    poll_interval_seconds: float = 60.0,
    fetch_html: FetchHtml | None = None,
    submit_fn: SubmitFn | None = None,
    stopping: Callable[[], bool] | None = None,
) -> None:
    """Background loop: poll due watch sources every *poll_interval_seconds*."""
    import asyncio

    fetch = fetch_html or default_fetch_html
    is_stopping = stopping or (lambda: False)

    while not is_stopping():
        try:
            poll_all_due_sources(
                agent_conn,
                config,
                fetch_html=fetch,
                submit_fn=submit_fn,
            )
        except Exception as exc:
            print(f"[watch_source] poll error: {exc}", flush=True)
        await asyncio.sleep(poll_interval_seconds)
