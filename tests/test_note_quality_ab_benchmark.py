"""Note-quality A/B benchmark harness (kb-agent vs Hermes)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.benchmark.note_quality import (
    build_review_html,
    compare_url_pair,
    load_url_set,
    resolve_note_in_dir,
    write_comparison_report,
)
from agent.config import Config
from agent.core.submit import plan_submission, submit_url
from agent.db import AGENT_SCHEMA, init_db

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "note_quality_ab"
FIXTURE_URLS = FIXTURE_ROOT / "urls.jsonl"
KB_DIR = FIXTURE_ROOT / "kb_agent"
HERMES_DIR = FIXTURE_ROOT / "hermes"


@pytest.fixture
def benchmark_config(tmp_path):
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        article_min_quotes=2,
        article_min_size=100,
        article_min_fact_checks=2,
    )


def test_load_url_set_reads_jsonl():
    rows = load_url_set(FIXTURE_URLS)
    assert len(rows) == 2
    assert rows[0]["url"].startswith("https://")


def test_resolve_note_in_dir_by_slug():
    path = resolve_note_in_dir(
        "https://example.com/benchmark-article-a",
        KB_DIR,
        content_type="web",
    )
    assert path is not None
    assert path.name == "example.com_benchmark-article-a.md"


def test_compare_url_pair_reports_quality_metrics(benchmark_config):
    kb_path = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", KB_DIR, content_type="web"
    )
    hermes_path = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", HERMES_DIR, content_type="web"
    )
    result = compare_url_pair(
        "https://example.com/benchmark-article-a",
        kb_path,
        hermes_path,
        content_type="web",
        config=benchmark_config,
    )
    assert result["kb_agent"]["metrics"]["char_count"] > 0
    assert result["hermes"]["metrics"]["char_count"] > 0
    assert result["kb_agent"]["passed"] is True
    assert result["hermes"]["passed"] is False
    assert any("Too few quotes" in f for f in result["hermes"]["failures"])


def test_write_comparison_report_json_and_html(benchmark_config, tmp_path):
    kb_a = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", KB_DIR, content_type="web"
    )
    hermes_a = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", HERMES_DIR, content_type="web"
    )
    entries = [
        compare_url_pair(
            "https://example.com/benchmark-article-a",
            kb_a,
            hermes_a,
            content_type="web",
            config=benchmark_config,
        )
    ]
    out_dir = tmp_path / "report"
    paths = write_comparison_report(entries, out_dir)
    assert paths["json"].is_file()
    assert paths["html"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["url_count"] == 1
    html = paths["html"].read_text(encoding="utf-8")
    assert "kb-agent" in html.lower() or "KB Agent" in html
    assert "Hermes" in html
    assert "example.com/benchmark-article-a" in html


def test_build_review_html_side_by_side_content():
    html = build_review_html(
        [
            {
                "url": "https://example.com/x",
                "content_type": "web",
                "kb_agent": {
                    "note_path": "kb.md",
                    "passed": True,
                    "failures": [],
                    "metrics": {"char_count": 10},
                    "content": "# KB note\n\nBody",
                },
                "hermes": {
                    "note_path": "hermes.md",
                    "passed": False,
                    "failures": ["Too short"],
                    "metrics": {"char_count": 5},
                    "content": "# Hermes note",
                },
            }
        ]
    )
    assert "KB note" in html
    assert "Hermes note" in html
    assert "Too short" in html


def test_plan_submission_ignore_dedup(benchmark_config):
    note = benchmark_config.vault_path / "Knowledge base" / "articles" / "dup.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("source_url: https://example.com/dup\n", encoding="utf-8")

    skipped = plan_submission("https://example.com/dup", benchmark_config)
    assert skipped["status"] == "skipped"

    pending = plan_submission(
        "https://example.com/dup", benchmark_config, ignore_dedup=True
    )
    assert pending["status"] in ("pending", "paused")
    assert pending.get("slug")


def test_submit_url_ignore_dedup(benchmark_config, monkeypatch):
    note = benchmark_config.vault_path / "Knowledge base" / "articles" / "dup2.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("source_url: https://example.com/dup2\n", encoding="utf-8")
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))

    db_path = benchmark_config.kb_agent_dir / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    try:
        result = submit_url(
            "https://example.com/dup2",
            benchmark_config,
            conn,
            ignore_dedup=True,
        )
        assert result["status"] == "pending"
    finally:
        conn.close()
