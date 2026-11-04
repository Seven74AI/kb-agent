"""Tests for watch_source — polling, diff, and last-seen dedup."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent.config import Config
from agent.core.watch_source import (
    diff_items,
    extract_items,
    is_poll_due,
    items_to_urls,
    merge_last_seen_after_poll,
    parse_last_seen,
    poll_all_due_sources,
    poll_watch_source,
    register_watch_source,
    serialize_last_seen,
)
from agent.db import AGENT_SCHEMA, init_db


TDAH_PAGE = "https://www.terragrandir.com/conferences-tdah-adulte-2026/"
TDAH_PATTERN = r"/tdah-adulte-(j\d+-[a-z0-9-]+)/"
TDAH_TEMPLATE = "https://www.terragrandir.com/tdah-adulte-{item}/"

SAMPLE_HTML = """
<html><body>
  <a href="/tdah-adulte-j1-intro/">Intro</a>
  <a href="/tdah-adulte-j2-keynote/">Keynote</a>
</body></html>
"""

SAMPLE_HTML_NEW = """
<html><body>
  <a href="/tdah-adulte-j1-intro/">Intro</a>
  <a href="/tdah-adulte-j2-keynote/">Keynote</a>
  <a href="/tdah-adulte-j3-panel/">Panel</a>
</body></html>
"""


@pytest.fixture
def agent_conn(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def core_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


def test_extract_items_from_html():
    items = extract_items(SAMPLE_HTML, TDAH_PATTERN)
    assert items == frozenset({"j1-intro", "j2-keynote"})


def test_extract_items_multi_group_uses_first_capture():
    pattern = r"/tdah-adulte-(j\d+)-([a-z-]+)/"
    items = extract_items(SAMPLE_HTML, pattern)
    assert items == frozenset({"j1", "j2"})


def test_extract_items_no_group_uses_full_match():
    pattern = r"tdah-adulte-j\d+-[a-z0-9-]+"
    items = extract_items(SAMPLE_HTML, pattern)
    assert items == frozenset({"tdah-adulte-j1-intro", "tdah-adulte-j2-keynote"})


def test_diff_items_returns_only_new():
    last = frozenset({"j1-intro", "j2-keynote"})
    current = frozenset({"j1-intro", "j2-keynote", "j3-panel"})
    assert diff_items(current, last) == ["j3-panel"]


def test_diff_items_empty_last_sees_all_current():
    current = frozenset({"j1-intro", "j2-keynote"})
    assert diff_items(current, frozenset()) == ["j1-intro", "j2-keynote"]


def test_items_to_urls_sorted():
    urls = items_to_urls(["j2-b", "j1-a"], TDAH_TEMPLATE)
    assert urls == [
        "https://www.terragrandir.com/tdah-adulte-j1-a/",
        "https://www.terragrandir.com/tdah-adulte-j2-b/",
    ]


def test_last_seen_json_roundtrip():
    raw = serialize_last_seen(frozenset({"a", "b"}))
    assert parse_last_seen(raw) == frozenset({"a", "b"})


def test_merge_last_seen_after_poll():
    last = frozenset({"j1-intro"})
    newly_seen = frozenset({"j2-keynote", "j3-panel"})
    merged = merge_last_seen_after_poll(last, newly_seen)
    assert merged == frozenset({"j1-intro", "j2-keynote", "j3-panel"})


def test_is_poll_due_never_polled():
    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert is_poll_due(None, 3600, now) is True


def test_is_poll_due_respects_cadence():
    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    last = (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    assert is_poll_due(last, 3600, now) is False
    assert is_poll_due(last, 900, now) is True


def test_register_watch_source_baseline(agent_conn):
    row = register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        cadence_seconds=86400,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    assert row["page_url"] == TDAH_PAGE
    assert row["cadence_seconds"] == 86400
    assert parse_last_seen(row["last_seen_json"]) == frozenset({"j1-intro", "j2-keynote"})


def test_register_duplicate_raises(agent_conn):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_watch_source(
            agent_conn,
            TDAH_PAGE,
            item_pattern=TDAH_PATTERN,
            item_url_template=TDAH_TEMPLATE,
            fetch_html=lambda url: SAMPLE_HTML,
        )


def test_poll_watch_source_submits_new_only(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    submitted: list[str] = []

    def track_submit(url, config, conn):
        submitted.append(url)
        return {"status": "pending", "http_status": 201, "task_id": "t1", "platform": "bunny_stream"}

    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=track_submit,
        now=datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result["new_items"] == ["j3-panel"]
    assert len(submitted) == 1
    assert submitted[0] == "https://www.terragrandir.com/tdah-adulte-j3-panel/"

    row = agent_conn.execute(
        "SELECT last_seen_json FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    seen = parse_last_seen(row["last_seen_json"])
    assert "j3-panel" in seen


def test_poll_watch_source_no_duplicate_resubmit(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    submit_count = {"n": 0}

    def submit(url, config, conn):
        submit_count["n"] += 1
        return {"status": "pending", "http_status": 201, "task_id": "t1", "platform": "web"}

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    poll_watch_source(
        agent_conn, TDAH_PAGE, core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW, submit_fn=submit, now=now,
    )
    # Force a second poll past cadence so the "not_due" gate cannot be what
    # suppresses the resubmit — dedup via last_seen_json must be what does.
    later = now + timedelta(days=1)
    poll_watch_source(
        agent_conn, TDAH_PAGE, core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW, submit_fn=submit, now=later, force=True,
    )
    assert submit_count["n"] == 1


@pytest.mark.parametrize("status", ["skipped", "paused"])
def test_poll_watch_source_non_error_status_marks_seen(
    core_config, agent_conn, monkeypatch, status
):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def non_error_submit(url, config, conn):
        return {"status": status, "reason": "handled"}

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=non_error_submit,
        now=now,
    )
    assert result["new_items"] == ["j3-panel"]
    row = agent_conn.execute(
        "SELECT last_seen_json FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    # Non-error outcome (skip/pause) marks the item seen — no retry next poll.
    assert "j3-panel" in parse_last_seen(row["last_seen_json"])

    later = now + timedelta(days=1)
    result2 = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=non_error_submit,
        now=later,
        force=True,
    )
    assert result2["new_items"] == []


def test_poll_watch_source_disabled_skips(core_config, agent_conn):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    agent_conn.execute(
        "UPDATE watch_sources SET enabled = 0 WHERE page_url = ?",
        (TDAH_PAGE,),
    )
    agent_conn.commit()

    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        now=datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result == {"page_url": TDAH_PAGE, "skipped": True, "reason": "disabled"}


def test_poll_all_due_sources_skips_not_due(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        cadence_seconds=3600,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    agent_conn.execute(
        "UPDATE watch_sources SET last_polled_at = ? WHERE page_url = ?",
        (now.strftime("%Y-%m-%d %H:%M:%S"), TDAH_PAGE),
    )
    agent_conn.commit()

    results = poll_all_due_sources(
        agent_conn,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        now=now,
    )
    assert results == []


def test_register_watch_source_invalid_regex(agent_conn):
    with pytest.raises(ValueError, match="not a valid regex"):
        register_watch_source(
            agent_conn,
            TDAH_PAGE,
            item_pattern="(unclosed",
            item_url_template=TDAH_TEMPLATE,
            fetch_html=lambda url: SAMPLE_HTML,
        )


def test_register_watch_source_baseline_requires_fetcher(agent_conn):
    with pytest.raises(ValueError, match="baseline=True requires fetch_html"):
        register_watch_source(
            agent_conn,
            TDAH_PAGE,
            item_pattern=TDAH_PATTERN,
            item_url_template=TDAH_TEMPLATE,
            baseline=True,
            fetch_html=None,
        )


def test_poll_watch_source_failed_submit_not_marked_seen(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def failing_submit(url, config, conn):
        return {"status": "error", "http_status": 500, "error": "boom"}

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=failing_submit,
        now=now,
    )
    # Item not marked seen → next poll retries it.
    assert result["new_items"] == ["j3-panel"]
    row = agent_conn.execute(
        "SELECT last_seen_json FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    assert parse_last_seen(row["last_seen_json"]) == frozenset({"j1-intro", "j2-keynote"})

    # Second poll still sees j3-panel as new (retried), once cadence elapses.
    later = now + timedelta(days=1)
    result2 = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=failing_submit,
        now=later,
    )
    assert result2["new_items"] == ["j3-panel"]


def test_poll_watch_source_failed_submit_holds_last_polled(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    old = "2026-06-01 00:00:00"
    agent_conn.execute(
        "UPDATE watch_sources SET last_polled_at = ? WHERE page_url = ?",
        (old, TDAH_PAGE),
    )
    agent_conn.commit()

    def failing_submit(url, config, conn):
        return {"status": "error", "http_status": 500, "error": "boom"}

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=failing_submit,
        now=now,
    )
    # Failed submission → last_polled_at held so the next tick retries fast.
    assert result["retry_pending"] is True
    row = agent_conn.execute(
        "SELECT last_polled_at FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    assert row["last_polled_at"] == old


def test_poll_watch_source_success_advances_last_polled(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def ok_submit(url, config, conn):
        return {"status": "pending", "http_status": 201, "task_id": "t1", "platform": "web"}

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=ok_submit,
        now=now,
    )
    assert result["retry_pending"] is False
    row = agent_conn.execute(
        "SELECT last_polled_at FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    assert row["last_polled_at"] == now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def test_poll_watch_source_submit_exception_isolated(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def exploding_submit(url, config, conn):
        raise RuntimeError("db locked")

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    result = poll_watch_source(
        agent_conn,
        TDAH_PAGE,
        core_config,
        fetch_html=lambda url: SAMPLE_HTML_NEW,
        submit_fn=exploding_submit,
        now=now,
    )
    assert result["new_items"] == ["j3-panel"]
    assert result["submissions"][0]["status"] == "error"
    row = agent_conn.execute(
        "SELECT last_seen_json FROM watch_sources WHERE page_url = ?",
        (TDAH_PAGE,),
    ).fetchone()
    assert parse_last_seen(row["last_seen_json"]) == frozenset({"j1-intro", "j2-keynote"})


def test_poll_all_due_sources_isolates_source_errors(core_config, agent_conn, monkeypatch):
    register_watch_source(
        agent_conn,
        TDAH_PAGE,
        item_pattern=TDAH_PATTERN,
        item_url_template=TDAH_TEMPLATE,
        fetch_html=lambda url: SAMPLE_HTML,
    )
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    def fetch_boom(url):
        raise RuntimeError("network down")

    now = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
    results = poll_all_due_sources(
        agent_conn,
        core_config,
        fetch_html=fetch_boom,
        now=now,
        submit_fn=lambda url, config, conn: {"status": "pending", "http_status": 201},
    )
    assert len(results) == 1
    assert results[0]["page_url"] == TDAH_PAGE
    assert "network down" in results[0]["error"]
