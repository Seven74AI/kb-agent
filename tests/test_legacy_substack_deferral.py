"""Regression: legacy content_type=substack must use deferral + synthesize-only dispatch."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.pipelines.web import step_synthesize_web


@pytest.fixture
def config(tmp_path):
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )


def test_get_pipeline_names_unknown_synthesize_returns_none_not_platform_fallback():
    from agent.pipelines import get_pipeline_names

    steps = get_pipeline_names("substack_synthesize", "substack")
    assert steps is None

    assert get_pipeline_names("nonexistent_xyz_synthesize", "substack") is None


def test_get_pipeline_names_known_synthesize_still_resolves():
    from agent.pipelines import get_pipeline_names

    steps = get_pipeline_names("substack_text_synthesize", "substack")
    assert steps is not None
    assert steps == ["synthesize_note", "quality_gate", "storage_upload", "see_also", "git_push"]
    assert "firecrawl_extract" not in steps
    assert "substack_embedded_media" not in steps


def test_ensure_parent_media_deferral_normalizes_legacy_substack(tmp_path):
    from agent.db import AGENT_SCHEMA, ensure_parent_media_deferral, get_parent_media_deferral, init_db

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "blog.substack.com_p_legacy"
    ensure_parent_media_deferral(
        conn, slug, "https://blog.substack.com/p/legacy", "substack", 2,
    )
    row = get_parent_media_deferral(conn, slug)
    assert row["content_type"] == "substack_text"
    ensure_parent_media_deferral(
        conn, slug, "https://blog.substack.com/p/legacy", "substack", 3,
    )
    row = get_parent_media_deferral(conn, slug)
    assert row["content_type"] == "substack_text"
    assert int(row["expected_total"]) == 3


def test_legacy_substack_parent_pauses_on_deferral_and_queued_sibling(config, tmp_path):
    from agent.db import (
        AGENT_SCHEMA,
        ensure_parent_media_deferral,
        init_db,
        queue_ingest,
    )

    config.kb_agent_dir = tmp_path / "kb"
    (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    slug = "blog.substack.com_p_old"
    post = "https://blog.substack.com/p/old"
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    ensure_parent_media_deferral(conn, slug, post, "substack", 1)
    queue_ingest(
        conn, f"{post}#img", "substack_image", "img1",
        parent_slug=slug, media_kind="image",
    )
    conn.commit()
    conn.close()
    (config.temp_dir / f"{slug}_raw.md").write_text("# Post\n\nBody.\n", encoding="utf-8")

    task = {
        "id": "legacy-substack-parent",
        "source_url": post,
        "content_type": "substack",
        "slug": slug,
    }
    with pytest.raises(StepError) as exc:
        step_synthesize_web(task, config)
    assert exc.value.error_type == ErrorType.PAUSE
    assert "deferred to batch synthesis" in str(exc.value).lower()


def test_dispatch_synthesize_substack_synthesize_errors_fail_closed(tmp_path):
    from agent.config import Config
    from agent.core.submit import dispatch_synthesize
    from agent.db import AGENT_SCHEMA, init_db

    cfg = Config(
        kb_agent_dir=tmp_path / "kb",
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
    )
    (cfg.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
    conn = init_db(cfg.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    rr = dispatch_synthesize(
        "https://blog.substack.com/p/x",
        cfg,
        conn,
        content_type="substack_synthesize",
        slug="blog.substack.com_p_x",
    )
    assert rr["status"] == "error"
    assert "unknown content_type: substack_synthesize" in (rr.get("error") or "")


def test_maybe_dispatch_legacy_deferral_uses_synthesize_only_pipeline(
    tmp_path, monkeypatch, config,
):
    import scripts.batch_ingest as batch_ingest
    from agent.db import (
        AGENT_SCHEMA,
        ensure_parent_media_deferral,
        init_db,
        media_append_marker,
        queue_ingest,
        update_ingest_status,
    )

    config.kb_agent_dir = tmp_path / "kb"
    (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    parent = "blog.substack.com_p_dispatch"
    url = "https://blog.substack.com/p/dispatch"
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    ensure_parent_media_deferral(conn, parent, url, "substack", 1)
    ingest_id, _ = queue_ingest(
        conn, f"{url}#vid", "substack_video", "vid",
        parent_slug=parent, media_kind="video",
    )
    update_ingest_status(conn, ingest_id, "dispatched")
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"url": url, "content_type": "substack", "total": 1}),
        encoding="utf-8",
    )
    marker = media_append_marker(ingest_id)
    (config.temp_dir / f"{parent}_raw.md").write_text(
        f"# Post\n\nBody.\n\n## Media\n\nembedded\n\n{marker}\n",
        encoding="utf-8",
    )
    conn.execute(
        "UPDATE ingest_queue SET media_appended = 1 WHERE parent_slug = ?",
        (parent,),
    )
    conn.commit()

    captured = {}

    def fake_dispatch(_url, _cfg, _conn, *, content_type, slug, tags=None):
        captured["content_type"] = content_type
        return {"status": "pending", "task_id": "syn-1"}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert captured["content_type"] == "substack_text_synthesize"
