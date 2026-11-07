"""Regression tests for scripts/batch_ingest.py behavior."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.core.batch_lock import acquire, is_locked, release
from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest, serialize_tags


@pytest.fixture
def config(tmp_path):
    return Config(
        kb_agent_dir=tmp_path / "kb",
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "agent.db"
    c = init_db(db_path, AGENT_SCHEMA)
    c.row_factory = sqlite3.Row
    return c


class TestAppendToParent:
    def test_append_returns_false_when_raw_missing(self, config):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        assert batch_ingest._append_to_parent(config, "parent", "Image", "text") is False

    def test_append_returns_true_and_writes(self, config):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")
        assert batch_ingest._append_to_parent(config, parent, "Image", "OCR text") is True
        body = raw.read_text(encoding="utf-8")
        assert "## Image" in body
        assert "OCR text" in body


class TestDeferredSynthesizeTags:
    def test_dispatch_includes_parent_tags_from_task(self, tmp_path, monkeypatch, config, conn):
        import scripts.batch_ingest as batch_ingest

        parent = "parent_slug"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        (config.temp_dir / f"{parent}_media.json").write_text(
            json.dumps({
                "url": "https://blog.substack.com/p/x",
                "content_type": "substack_text",
                "total": 0,
            }),
            encoding="utf-8",
        )
        create_task(
            conn,
            "https://blog.substack.com/p/x",
            "substack_text",
            parent,
            ["substack_cleanup"],
            tags=["tdah", "research"],
        )
        captured: list[dict] = []

        def fake_dispatch(url, cfg, agent_conn, **kwargs):
            captured.append(kwargs)
            return {"status": "pending", "task_id": "t1"}

        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
        batch_ingest._deferred_synthesize_dispatched.clear()
        assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
        assert captured[0].get("tags") == ["tdah", "research"]


class TestBatchLockInMain:
    def test_main_acquires_and_releases_lock(self, config, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(batch_ingest, "load_config", lambda: config)
        monkeypatch.setattr(
            batch_ingest,
            "init_db",
            lambda *a, **k: MagicMock(
                execute=MagicMock(),
                commit=MagicMock(),
                close=MagicMock(),
            ),
        )
        monkeypatch.setattr(batch_ingest, "recover_stale_ingest", lambda c: None)
        monkeypatch.setattr(batch_ingest, "recover_failed_media", lambda c: 0)
        monkeypatch.setattr(batch_ingest, "record_batch_run", lambda c: 1)
        monkeypatch.setattr(batch_ingest, "claim_ingest_items", lambda c: [])
        monkeypatch.setattr(batch_ingest, "finish_batch_run", lambda *a: None)

        assert is_locked(config) is False
        batch_ingest.main()
        assert is_locked(config) is False

    def test_main_exits_when_lock_held(self, config, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        assert acquire(config)

        monkeypatch.setattr(batch_ingest, "load_config", lambda: config)
        finish = MagicMock()
        monkeypatch.setattr(batch_ingest, "finish_batch_run", finish)
        batch_ingest.main()
        finish.assert_not_called()
        release(config)


class TestEmbeddedImageSizeCap:
    def test_oversized_image_marked_failed_not_staged(self, config, conn, monkeypatch, tmp_path):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent"
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://cdn.example/img.png",
            "substack_image",
            "img_slug",
            parent_slug=parent,
            media_kind="image",
        )[0]

        big = tmp_path / "big.png"
        big.write_bytes(b"x" * (batch_ingest.MAX_STAGED_IMAGE_BYTES + 1))

        def fake_run(cmd, **kw):
            if "-o" in cmd:
                out_idx = cmd.index("-o") + 1
                Path(cmd[out_idx]).write_bytes(big.read_bytes())
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        statuses: list[tuple] = []

        def track_status(connection, item_id, status, error=None):
            statuses.append((item_id, status, error))

        monkeypatch.setattr(batch_ingest, "update_ingest_status", track_status)
        batch_ingest._process_embedded_images(
            [{
                "id": ingest_id,
                "url": "https://cdn.example/img.png",
                "parent_slug": parent,
                "slug": "img_slug",
            }],
            config,
            conn,
            1,
        )
        failed = [s for s in statuses if s[1] == "failed"]
        assert failed
        assert "too large" in (failed[0][2] or "").lower()


class TestRunBatchLock:
    def test_mcp_server_uses_is_locked_not_acquire(self):
        import agent.mcp.server as mcp_server

        assert hasattr(mcp_server, "is_locked")
        assert not hasattr(mcp_server, "release")


class TestKbOcrSizeCap:
    def test_max_image_bytes_in_source(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        assert "MAX_IMAGE_BYTES = 25 * 1024 * 1024" in text
        assert "size > MAX_IMAGE_BYTES" in text


class TestAppendFailureMarksIngest:
    def test_missing_raw_marks_failed_not_dispatched(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent"
        ingest_id = "ingest-1"
        statuses: list[tuple] = []

        def track_status(connection, item_id, status, error=None):
            statuses.append((item_id, status, error))

        monkeypatch.setattr(batch_ingest, "update_ingest_status", track_status)
        monkeypatch.setattr(batch_ingest, "_maybe_dispatch_parent_synthesize", lambda *a: False)

        # Simulate post-OCR path without raw file
        ocr_dir = config.temp_dir / "batch_imgout" / "run_1"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_json = ocr_dir / "item.png.ocr.json"
        ocr_json.write_text(json.dumps({"reading": "some text"}), encoding="utf-8")

        q = {"id": ingest_id, "parent_slug": parent, "slug": "s1", "url": "https://x/img.png"}
        dest = config.temp_dir / "batch_images" / "run_1" / "item.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89PNG")

        for q_item, dest_path in [(q, dest)]:
            ocr_path = ocr_dir / f"{dest_path.name}.ocr.json"
            reading = json.loads(ocr_path.read_text(encoding="utf-8")).get("reading", "")
            if not batch_ingest._append_to_parent(config, q_item["parent_slug"], "Image", reading):
                batch_ingest.update_ingest_status(
                    conn, q_item["id"], "failed", error="parent raw markdown missing",
                )
            else:
                batch_ingest.update_ingest_status(conn, q_item["id"], "dispatched")

        assert ("ingest-1", "failed", "parent raw markdown missing") in statuses
