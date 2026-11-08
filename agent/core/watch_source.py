"""Recurring source watch — poll pages, diff items, auto-submit new URLs."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from agent.config import Config
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.notify import sanitize_visible_message
from agent.redact import url_for_display

SubmitFn = Callable[[str, Config, sqlite3.Connection], dict]
FetchHtml = Callable[[str], str]


def extract_items(html: str, item_pattern: str) -> frozenset[str]:
    """Return unique captured values from *html* matching *item_pattern*.

    Uses the first capture group when the pattern has one (the common ``{item}``
    case), else the full match. ``re.findall`` would return tuples for patterns
    with multiple capture groups, which breaks ``serialize_last_seen``
    (``json.dumps`` rejects tuples) and ``items_to_urls``.
    """
    return frozenset(
        m.group(1) if m.lastindex else m.group(0)
        for m in re.finditer(item_pattern, html)
    )


def extract_items_available(html: str, item_pattern: str, now: datetime | None = None) -> frozenset[str]:
    """Extract items, skipping cards whose ``data-open`` is in the future.

    Pages that pre-list not-yet-live content (the terragrandir TDAH summit) mark
    each conference card with ``data-open=<ISO timestamp>``. Cards opening in the
    future ("À venir") must not be treated as available items — otherwise a watch
    baselines speakers for days that haven't happened yet. Cards without
    ``data-open`` are always kept (not time-gated).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    def _drop_future(m: re.Match) -> str:
        try:
            opens = datetime.fromisoformat(m.group(1))
        except ValueError:
            return m.group(0)  # unparseable timestamp — keep the card
        return "" if opens > now else m.group(0)

    html = re.sub(
        r'<li\b[^>]*data-open="([^"]+)"[^>]*>.*?</li>',
        _drop_future,
        html,
        flags=re.DOTALL,
    )
    return extract_items(html, item_pattern)


def diff_items(current: frozenset[str], last_seen: frozenset[str]) -> list[str]:
    """Sorted item ids present in *current* but not in *last_seen*."""
    return sorted(current - last_seen)


def items_to_urls(items: list[str], item_url_template: str) -> list[str]:
    """Expand item ids to submission URLs via ``{item}`` template placeholder."""
    return [
        item_url_template.replace("{item}", item) for item in sorted(items)
    ]


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
    newly_seen: frozenset[str],
) -> frozenset[str]:
    """Union prior last-seen with items successfully handled this poll.

    Items whose submission failed are left out of ``newly_seen`` so the next
    poll retries them instead of dropping them permanently.
    """
    return last_seen | newly_seen


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


def _row_for_display(row: dict[str, Any]) -> dict[str, Any]:
    """Outward-facing watch source row: redact URL fields; DB keeps full values."""
    out = dict(row)
    if "page_url" in out:
        out["page_url"] = url_for_display(out["page_url"])
    if "item_url_template" in out:
        out["item_url_template"] = url_for_display(out["item_url_template"])
    return out


def _display_page_url(page_url: str) -> str:
    return url_for_display(page_url)


def _validate_page_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("page_url required")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("page_url must be an http or https URL with a host")
    return url


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
    page_url = _validate_page_url(page_url)
    if cadence_seconds < 1:
        raise ValueError("cadence_seconds must be >= 1")
    if "{item}" not in item_url_template:
        raise ValueError("item_url_template must contain {item}")
    try:
        re.compile(item_pattern)
    except re.error as exc:
        raise ValueError(f"item_pattern is not a valid regex: {exc}") from exc
    if baseline and fetch_html is None:
        raise ValueError(
            "baseline=True requires fetch_html to seed last_seen_json "
            "(pass a fetcher, or baseline=False)"
        )

    existing = conn.execute(
        "SELECT id FROM watch_sources WHERE page_url = ?",
        (page_url,),
    ).fetchone()
    if existing:
        raise ValueError(f"watch source already registered: {page_url}")

    last_seen: frozenset[str] = frozenset()
    if baseline and fetch_html is not None:
        html = fetch_html(page_url)
        last_seen = extract_items_available(html, item_pattern)

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
    return _row_for_display(_row_to_dict(row))


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
        return {
            "page_url": _display_page_url(page_url),
            "skipped": True,
            "reason": "disabled",
        }

    now = now or datetime.now(timezone.utc)
    if not force and not is_poll_due(row["last_polled_at"], row["cadence_seconds"], now):
        return {
            "page_url": _display_page_url(page_url),
            "skipped": True,
            "reason": "not_due",
        }

    submit = submit_fn or submit_routed
    html = fetch_html(page_url)
    current = extract_items_available(html, row["item_pattern"], now=now)
    last_seen = parse_last_seen(row["last_seen_json"])
    new_items = diff_items(current, last_seen)
    urls = items_to_urls(new_items, row["item_url_template"])

    submissions: list[dict[str, Any]] = []
    newly_seen: set[str] = set()
    for item, url in zip(new_items, urls):
        try:
            result = submit(url, config, conn)
        except Exception as exc:  # isolate per-item failures so one bad item doesn't abort the poll
            result = {
                "status": "error",
                "error": sanitize_visible_message(f"{type(exc).__name__}: {exc}"),
            }
        submissions.append(result)
        # Only mark an item as seen when it was actually enqueued, paused, or
        # already in the vault. Failed submissions stay unseen so the next poll
        # retries them instead of dropping them permanently.
        if result.get("status") != "error":
            newly_seen.add(item)

    merged = merge_last_seen_after_poll(last_seen, frozenset(newly_seen))
    failed = any(s.get("status") == "error" for s in submissions)
    polled_at = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        if failed:
            # Hold last_polled_at so the next scheduler tick (60s) retries the
            # failed items instead of waiting a full cadence. Successfully
            # handled items are still marked seen so they aren't re-submitted.
            conn.execute(
                "UPDATE watch_sources SET last_seen_json = ? WHERE page_url = ?",
                (serialize_last_seen(merged), page_url),
            )
        else:
            conn.execute(
                """UPDATE watch_sources
                   SET last_seen_json = ?, last_polled_at = ?
                   WHERE page_url = ?""",
                (serialize_last_seen(merged), polled_at, page_url),
            )

    return {
        "page_url": _display_page_url(page_url),
        "new_items": new_items,
        "submissions": submissions,
        "last_seen_count": len(merged),
        "retry_pending": failed,
    }


def poll_all_due_sources(
    conn: sqlite3.Connection,
    config: Config,
    *,
    fetch_html: FetchHtml,
    now: datetime | None = None,
    submit_fn: SubmitFn | None = None,
) -> list[dict[str, Any]]:
    """Poll every enabled source whose cadence has elapsed.

    A failure on one source is captured as an error entry and does not abort
    polling of the remaining sources.
    """
    now = now or datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for source in list_watch_sources(conn, enabled_only=True):
        try:
            outcome = poll_watch_source(
                conn,
                source["page_url"],
                config,
                fetch_html=fetch_html,
                now=now,
                submit_fn=submit_fn,
            )
        except Exception as exc:  # isolate per-source failures
            results.append(
                {
                    "page_url": _display_page_url(source["page_url"]),
                    "error": sanitize_visible_message(f"{type(exc).__name__}: {exc}"),
                }
            )
            continue
        if not outcome.get("skipped"):
            results.append(outcome)
    return results


def default_fetch_html(url: str) -> str:
    """Fetch page HTML via curl (--max-redirs 0). Used by scheduler and CLI."""
    _assert_fetch_url_allowed(url)
    cmd = [
        "curl", "-s", "-L", "--max-redirs", "0", "--max-time", "40",
        "-H", "User-Agent: kb-agent-watch/1.0",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"curl failed ({result.returncode})")
    return result.stdout


VIDEO_CONTENT_TYPES = {
    "youtube",
    "bunny_stream",
    "rumble",
    "substack_video",
    "instagram_reel",
    "threads_video",
    "mega_video",
}


def submit_routed(url: str, config: Config, conn: sqlite3.Connection, *, tags=None) -> dict:
    """Submit a watch-discovered URL: video → deferred ingest, else immediate.

    Video types go through ``submit_ingest`` so the nightly batch collector
    downloads + batch-transcribes them (one Modal per type) instead of paying a
    Modal cold-start per video. Non-video types submit immediately.
    """
    from agent.core.submit import submit_ingest, submit_url
    from agent.detection import detect_content_subtype, detect_platform, normalize_url

    url = normalize_url(url)
    platform = detect_platform(url)
    subtype = detect_content_subtype(url, platform)
    if subtype in VIDEO_CONTENT_TYPES:
        return submit_ingest(url, config, conn, content_type=subtype, tags=tags)
    return submit_url(url, config, conn, tags=tags)


def _poll_due_sources_in_thread(
    config: Config,
    fetch_html: FetchHtml,
    submit_fn: SubmitFn | None,
) -> list[dict[str, Any]]:
    """Poll all due sources on a dedicated connection (runs off the event loop)."""
    from agent.db import AGENT_SCHEMA, init_db

    db_path = config.kb_agent_dir / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    try:
        return poll_all_due_sources(conn, config, fetch_html=fetch_html, submit_fn=submit_fn)
    finally:
        conn.close()


async def run_watch_scheduler(
    config: Config,
    *,
    poll_interval_seconds: float = 60.0,
    fetch_html: FetchHtml | None = None,
    submit_fn: SubmitFn | None = None,
    stopping: Callable[[], bool] | None = None,
) -> None:
    """Background loop: poll due watch sources every *poll_interval_seconds*.

    Each tick does blocking HTTP fetch + SQLite writes, so it is offloaded to a
    worker thread (``asyncio.to_thread``) to avoid stalling the consumer's event
    loop and delaying task claiming.
    """
    import asyncio

    fetch = fetch_html or default_fetch_html
    is_stopping = stopping or (lambda: False)

    while not is_stopping():
        try:
            await asyncio.to_thread(_poll_due_sources_in_thread, config, fetch, submit_fn)
        except Exception as exc:
            print(f"[watch_source] poll error: {exc}", flush=True)
        await asyncio.sleep(poll_interval_seconds)
