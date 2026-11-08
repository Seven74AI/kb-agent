"""Note-quality A/B benchmark harness (kb-agent vs Hermes)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from agent.benchmark.note_quality import (
    build_review_html,
    compare_from_vaults,
    compare_url_pair,
    find_note_in_vault,
    find_notes_in_vault,
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
RELAXED_CONFIG = FIXTURE_ROOT / "benchmark_config.yaml"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "note_quality_ab_benchmark.py"


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
    )
    assert path is not None
    assert path.name == "example.com_benchmark-article-a.md"


def test_compare_url_pair_reports_quality_metrics(benchmark_config):
    kb_path = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", KB_DIR
    )
    hermes_path = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", HERMES_DIR
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
    assert result["url"] == "https://example.com/benchmark-article-a"


def test_write_comparison_report_json_and_html(benchmark_config, tmp_path):
    kb_a = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", KB_DIR
    )
    hermes_a = resolve_note_in_dir(
        "https://example.com/benchmark-article-a", HERMES_DIR
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


def test_write_comparison_report_json_note_path_display_safe(benchmark_config, tmp_path):
    """comparison.json must not persist absolute host paths (same policy as review bodies)."""
    host_like = tmp_path / "root" / "kb-agent" / "vault" / "Knowledge base" / "articles"
    host_like.mkdir(parents=True)
    kb_note = host_like / "benchmark-article-a.md"
    kb_note.write_text(
        "---\nsource_url: https://example.com/benchmark-article-a\n---\n"
        "## Summary\nKB body with enough text.\n"
        "## Key Concepts\n### One\n"
        "> quote one\n> quote two\n"
        "## Critical Analysis\n**Fact check 1:** ok\n**Fact check 2:** ok\n",
        encoding="utf-8",
    )
    hermes_note = host_like / "hermes-a.md"
    hermes_note.write_text("# short\n", encoding="utf-8")

    entries = [
        compare_url_pair(
            "https://example.com/benchmark-article-a",
            kb_note,
            hermes_note,
            content_type="web",
            config=benchmark_config,
        )
    ]
    assert Path(entries[0]["kb_agent"]["note_path"]).is_absolute()

    out_dir = tmp_path / "report"
    paths = write_comparison_report(entries, out_dir)
    raw_json = paths["json"].read_text(encoding="utf-8")
    assert "/root/kb-agent" not in raw_json
    assert str(host_like) not in raw_json

    entry = json.loads(raw_json)["entries"][0]
    for side in ("kb_agent", "hermes"):
        note_path = entry[side]["note_path"]
        assert note_path is not None
        assert not Path(note_path).is_absolute()
        assert note_path in {"benchmark-article-a.md", "hermes-a.md"}


def test_build_review_html_redacts_tracking_query_on_url():
    tracked = "https://example.com/article?utm_source=track&sig=deadbeef"
    html = build_review_html(
        [
            {
                "url": tracked,
                "content_type": "web",
                "kb_agent": {
                    "note_path": "kb.md",
                    "passed": True,
                    "failures": [],
                    "metrics": {},
                    "content": "---\nsource_url: " + tracked + "\n---\n",
                },
                "hermes": {
                    "note_path": "hermes.md",
                    "passed": True,
                    "failures": [],
                    "metrics": {},
                    "content": "(missing)",
                },
            }
        ]
    )
    assert "utm_source" not in html
    assert "sig=deadbeef" not in html
    assert "https://example.com/article" in html


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


# ── Vault resolution path ───────────────────────────────────────────────────────


def _write_vault_note(vault: Path, url: str, name: str) -> Path:
    knowledge = vault / "Knowledge base" / "articles"
    knowledge.mkdir(parents=True, exist_ok=True)
    note = knowledge / name
    note.write_text(f"source_url: {url}\n\n## Summary\nBody text.\n", encoding="utf-8")
    return note


def test_find_note_in_vault_unique(tmp_path):
    vault = tmp_path / "vault"
    url = "https://example.com/vault-article"
    expected = _write_vault_note(vault, url, "vault-article.md")

    assert find_notes_in_vault(url, vault) == [expected]
    assert find_note_in_vault(url, vault) == expected


def test_find_note_in_vault_matches_display_safe_submit_url(tmp_path):
    """Lookup uses url_for_display so query/fragment on submit URL still resolves."""
    from agent.redact import url_for_display

    vault = tmp_path / "vault"
    stored = "https://mega.nz/file/abc123"
    submit = "https://mega.nz/file/abc123#key-material"
    expected = _write_vault_note(vault, stored, "mega.md")
    assert find_notes_in_vault(submit, vault) == [expected]
    assert find_note_in_vault(submit, vault) == expected
    assert url_for_display(submit) == stored


def test_find_note_in_vault_missing(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Knowledge base" / "articles").mkdir(parents=True, exist_ok=True)

    assert find_notes_in_vault("https://example.com/absent", vault) == []
    assert find_note_in_vault("https://example.com/absent", vault) is None


def test_find_note_in_vault_ambiguous_returns_none(tmp_path):
    vault = tmp_path / "vault"
    url = "https://example.com/ambiguous"
    _write_vault_note(vault, url, "copy-a.md")
    _write_vault_note(vault, url, "copy-b.md")

    matches = find_notes_in_vault(url, vault)
    assert len(matches) == 2
    # Ambiguous matches must not be silently resolved to one file.
    assert find_note_in_vault(url, vault) is None


def test_compare_from_vaults_resolves_both_sides(benchmark_config, tmp_path):
    kb_vault = tmp_path / "kb-vault"
    hermes_vault = tmp_path / "hermes-vault"
    url = "https://example.com/vault-pair"
    _write_vault_note(kb_vault, url, "kb.md")
    _write_vault_note(hermes_vault, url, "hermes.md")

    entries = compare_from_vaults(
        [{"url": url, "content_type": "web"}],
        kb_vault=kb_vault,
        hermes_vault=hermes_vault,
        config=benchmark_config,
    )
    assert len(entries) == 1
    assert entries[0]["kb_agent"]["note_path"] is not None
    assert entries[0]["hermes"]["note_path"] is not None


# ── CLI compare summary semantics + validation guards ───────────────────────────


def _run_compare(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "compare",
         "--urls", str(FIXTURE_URLS),
         "--kb-agent-dir", str(KB_DIR),
         "--hermes-dir", str(HERMES_DIR),
         "--config", str(RELAXED_CONFIG),
         *extra],
        capture_output=True,
        text=True,
    )


def test_cli_compare_summary_semantics(tmp_path):
    out_dir = tmp_path / "report"
    result = _run_compare("--output", str(out_dir))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)["summary"]
    # With relaxed thresholds: kb-agent article-a passes, article-b fails the
    # structural "unbalanced" gate (both sides), and both Hermes notes fail.
    assert summary == {
        "url_count": 2,
        "kb_agent_pass": 1,
        "hermes_pass": 0,
        "both_pass": 0,
    }


def test_cli_compare_requires_both_vaults(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "compare",
         "--urls", str(FIXTURE_URLS),
         "--config", str(RELAXED_CONFIG),
         "--kb-vault", str(tmp_path / "kb"),
         "--output", str(tmp_path / "report")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BOTH --kb-vault and --hermes-vault" in result.stderr


def test_cli_compare_rejects_same_vault(tmp_path):
    shared = tmp_path / "shared-vault"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "compare",
         "--urls", str(FIXTURE_URLS),
         "--config", str(RELAXED_CONFIG),
         "--kb-vault", str(shared),
         "--hermes-vault", str(shared),
         "--output", str(tmp_path / "report")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "same path" in result.stderr
