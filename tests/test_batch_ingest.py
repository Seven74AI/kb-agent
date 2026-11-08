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
    def test_append_without_conn_refuses(self, config):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="conn and ingest_id"):
            batch_ingest._append_to_parent(config, "parent", "Image", "text")

    def test_append_returns_false_when_raw_missing(self, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        assert batch_ingest._append_to_parent(
            config, parent, "Image", "text", conn=conn, ingest_id=ingest_id,
        ) is False

    def test_append_returns_true_and_writes(self, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import media_append_marker, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")
        assert batch_ingest._append_to_parent(
            config, parent, "Image", "OCR text", conn=conn, ingest_id=ingest_id,
        ) is True
        body = raw.read_text(encoding="utf-8")
        assert "## Image" in body
        assert "OCR text" in body
        assert media_append_marker(ingest_id) in body

    def test_append_idempotent_on_reprocess(self, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")
        batch_ingest._append_to_parent(
            config, parent, "Image", "OCR text", conn=conn, ingest_id=ingest_id,
        )
        batch_ingest._append_to_parent(
            config, parent, "Image", "OCR text", conn=conn, ingest_id=ingest_id,
        )
        body = raw.read_text(encoding="utf-8")
        assert body.count("## Image") == 1
        assert body.count("OCR text") == 1

    def test_append_marker_prevents_double_write_after_crash(self, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import media_append_marker, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        marker = media_append_marker(ingest_id)
        raw.write_text(
            f"# Article\n\nBody.\n\n## Image\n\nOCR text\n\n{marker}\n",
            encoding="utf-8",
        )
        batch_ingest._append_to_parent(
            config, parent, "Image", "OCR text again", conn=conn, ingest_id=ingest_id,
        )
        body = raw.read_text(encoding="utf-8")
        assert body.count("## Image") == 1
        assert "OCR text again" not in body
        row = conn.execute(
            "SELECT media_appended FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["media_appended"] == 1

    def test_append_parent_media_atomic_writes_disk_before_media_appended(self, config, conn, monkeypatch):
        from agent.db import append_parent_media_atomic, media_append_marker, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")
        order: list[str] = []

        real_replace = os.replace

        def tracking_replace(src, dst):
            order.append("replace")
            assert conn.in_transaction is False
            row = conn.execute(
                "SELECT media_appended FROM ingest_queue WHERE id=?",
                (ingest_id,),
            ).fetchone()
            assert row["media_appended"] == 0
            assert src.read_text(encoding="utf-8") != raw.read_text(encoding="utf-8")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking_replace)

        assert append_parent_media_atomic(conn, ingest_id, raw, "Image", "OCR text") is True
        assert order == ["replace"]
        body = raw.read_text(encoding="utf-8")
        assert "OCR text" in body
        assert media_append_marker(ingest_id) in body
        row = conn.execute(
            "SELECT media_appended FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["media_appended"] == 1
        assert not list(config.temp_dir.glob("*.tmp"))

    def test_append_parent_media_atomic_repairs_full_block_when_flag_without_body(
        self, config, conn,
    ):
        from agent.db import append_parent_media_atomic, media_append_marker, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")
        conn.execute(
            "UPDATE ingest_queue SET media_appended = 1 WHERE id = ?",
            (ingest_id,),
        )
        conn.commit()

        assert append_parent_media_atomic(conn, ingest_id, raw, "Image", "OCR text") is True
        body = raw.read_text(encoding="utf-8")
        assert "## Image" in body
        assert "OCR text" in body
        assert media_append_marker(ingest_id) in body

    def test_append_parent_media_atomic_repairs_marker_when_flag_without_marker(self, config, conn):
        from agent.db import append_parent_media_atomic, media_append_marker, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text(
            "# Article\n\nBody.\n\n## Image\n\nOCR text\n",
            encoding="utf-8",
        )
        conn.execute(
            "UPDATE ingest_queue SET media_appended = 1 WHERE id = ?",
            (ingest_id,),
        )
        conn.commit()

        assert append_parent_media_atomic(conn, ingest_id, raw, "Image", "OCR text") is True
        body = raw.read_text(encoding="utf-8")
        assert body.count("## Image") == 1
        assert body.count("OCR text") == 1
        assert media_append_marker(ingest_id) in body

    def test_append_parent_media_atomic_redacts_urls_in_transcript_body(self, config, conn):
        from agent.db import append_parent_media_atomic, queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "article_slug"
        tracking = "https://cdn.example.com/frame.jpg?token=SECRET&expires=999"
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text("# Article\n\nBody.\n", encoding="utf-8")

        body = f"OCR line with source {tracking}"
        assert append_parent_media_atomic(conn, ingest_id, raw, "Image", body) is True
        written = raw.read_text(encoding="utf-8")
        assert "token=SECRET" not in written
        assert "?" not in written.split("## Image", 1)[-1]
        assert "https://cdn.example.com/frame.jpg" in written


class TestDeferredSynthesizeTags:
    def test_parent_tags_ignores_synthesize_task_row(self, conn):
        import scripts.batch_ingest as batch_ingest

        parent = "parent_with_synth_dup"
        create_task(
            conn,
            "https://blog.substack.com/p/x",
            "substack_text",
            parent,
            ["substack_cleanup"],
            tags=["article-tag"],
        )
        create_task(
            conn,
            "https://blog.substack.com/p/x",
            "substack_text_synthesize",
            parent,
            ["synthesize_note"],
            tags='["synth-only"]',
        )
        assert batch_ingest._parent_tags(conn, parent) == ["article-tag"]

    def test_dispatch_includes_parent_tags_from_task(self, tmp_path, monkeypatch, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import ensure_parent_media_deferral

        parent = "parent_slug"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        post_url = "https://blog.substack.com/p/x"
        ensure_parent_media_deferral(conn, parent, post_url, "substack_text", 0)
        (config.temp_dir / f"{parent}_media.json").write_text(
            json.dumps({"content_type": "substack_text", "total": 0}),
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

    def test_dispatch_supersedes_paused_parent_task(self, tmp_path, monkeypatch, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import ensure_parent_media_deferral

        parent = "parent_slug"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        post_url = "https://blog.substack.com/p/x"
        ensure_parent_media_deferral(conn, parent, post_url, "substack_text", 0)
        (config.temp_dir / f"{parent}_media.json").write_text(
            json.dumps({"content_type": "substack_text", "total": 0}),
            encoding="utf-8",
        )
        parent_id = create_task(
            conn,
            "https://blog.substack.com/p/x",
            "substack_text",
            parent,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
            status="paused",
        )
        conn.execute(
            "UPDATE tasks SET error = ? WHERE id = ?",
            ("awaiting media: 1 images enqueued", parent_id),
        )
        conn.commit()

        def fake_dispatch(url, cfg, agent_conn, **kwargs):
            return {"status": "pending", "task_id": "synth-task-1", "duplicate": False}

        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
        batch_ingest._deferred_synthesize_dispatched.clear()
        assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True

        row = conn.execute(
            "SELECT status, error FROM tasks WHERE id = ?",
            (parent_id,),
        ).fetchone()
        assert row["status"] == "completed"
        assert "parent ingestion superseded" in row["error"]
        assert "synth-task-1" in row["error"]

    def test_dispatch_error_leaves_parent_unsuperseded(self, tmp_path, monkeypatch, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import ensure_parent_media_deferral, media_append_marker, queue_ingest

        parent = "parent_dispatch_err"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        url = "https://blog.substack.com/p/dispatch-err"
        ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
        (config.temp_dir / f"{parent}_media.json").write_text(
            json.dumps({"content_type": "substack_text", "total": 1}),
            encoding="utf-8",
        )
        parent_id = create_task(
            conn,
            url,
            "substack_text",
            parent,
            ["substack_cleanup"],
            status="paused",
        )
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/i1.png",
            "substack_image",
            "i1",
            parent_slug=parent,
            media_kind="image",
        )
        conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE id=?", (ingest_id,))
        conn.commit()
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text(
            f"# Parent\n\n## Media\n\n{media_append_marker(ingest_id)}\n",
            encoding="utf-8",
        )
        conn.execute(
            "UPDATE ingest_queue SET media_appended = 1 WHERE parent_slug = ?",
            (parent,),
        )
        conn.commit()

        def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
            return {
                "status": "error",
                "error": "unknown content_type: substack_text_synthesize",
            }

        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
        batch_ingest._deferred_synthesize_dispatched.clear()
        assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
        assert parent not in batch_ingest._deferred_synthesize_dispatched

        row = conn.execute(
            "SELECT status, error FROM tasks WHERE id = ?",
            (parent_id,),
        ).fetchone()
        assert row["status"] == "paused"
        assert "parent ingestion superseded" not in (row["error"] or "")
        assert "deferred synthesis dispatch failed" in (row["error"] or "")
        assert "unknown content_type: substack_text_synthesize" in (row["error"] or "")

    def test_dispatch_duplicate_still_supersedes_parent(self, tmp_path, monkeypatch, config, conn):
        import scripts.batch_ingest as batch_ingest
        from agent.db import ensure_parent_media_deferral, media_append_marker, queue_ingest

        parent = "parent_dispatch_dup"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        url = "https://blog.substack.com/p/dispatch-dup"
        ensure_parent_media_deferral(conn, parent, url, "substack_text", 1)
        (config.temp_dir / f"{parent}_media.json").write_text(
            json.dumps({"content_type": "substack_text", "total": 1}),
            encoding="utf-8",
        )
        parent_id = create_task(
            conn,
            url,
            "substack_text",
            parent,
            ["substack_cleanup"],
            status="paused",
        )
        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/i1.png",
            "substack_image",
            "i1",
            parent_slug=parent,
            media_kind="image",
        )
        conn.execute("UPDATE ingest_queue SET status='dispatched' WHERE id=?", (ingest_id,))
        conn.commit()
        raw = config.temp_dir / f"{parent}_raw.md"
        raw.write_text(
            f"# Parent\n\n## Media\n\n{media_append_marker(ingest_id)}\n",
            encoding="utf-8",
        )
        conn.execute(
            "UPDATE ingest_queue SET media_appended = 1 WHERE parent_slug = ?",
            (parent,),
        )
        conn.commit()

        def fake_dispatch(dispatch_url, cfg, agent_conn, **kwargs):
            return {
                "status": "pending",
                "task_id": "existing-synth",
                "duplicate": True,
            }

        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
        batch_ingest._deferred_synthesize_dispatched.clear()
        assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
        assert parent in batch_ingest._deferred_synthesize_dispatched

        row = conn.execute(
            "SELECT status, error FROM tasks WHERE id = ?",
            (parent_id,),
        ).fetchone()
        assert row["status"] == "completed"
        assert "parent ingestion superseded" in row["error"]


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


class TestEmbeddedImageFetchGuard:
    def test_prevalidated_image_curl_disables_redirects(self, config, conn, monkeypatch, tmp_path):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_redirect"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img_redirect",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q = {
            "id": ingest_id,
            "url": "https://example.com/img.png",
            "parent_slug": parent,
            "slug": "img_redirect",
        }
        redirect_target = "http://127.0.0.1/redirected"
        with pytest.raises(ValueError, match="not allowed"):
            batch_ingest._assert_fetch_url_allowed(redirect_target)

        curl_calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            curl_calls.append(list(cmd))
            if cmd[0] == "curl" and "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x00" * 200)
                return MagicMock(returncode=0)
            return MagicMock(returncode=1)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q], config, conn, 12)
        assert curl_calls
        curl_cmd = curl_calls[0]
        assert "--max-redirs" in curl_cmd
        assert curl_cmd[curl_cmd.index("--max-redirs") + 1] == "0"

    def test_blocked_image_url_marks_failed(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        ingest_id = queue_ingest(
            conn,
            "http://169.254.169.254/latest/meta-data/",
            "substack_image",
            "img_blocked",
            parent_slug="parent_blocked",
            media_kind="image",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        curl_calls: list = []
        monkeypatch.setattr(
            batch_ingest.subprocess,
            "run",
            lambda cmd, **kw: (curl_calls.append(cmd) or MagicMock(returncode=0)),
        )
        batch_ingest._process_embedded_images([q], config, conn, 99)
        assert not curl_calls
        row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "blocked fetch url" in row["error"].lower()


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
            "https://example.com/img.png",
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
                "url": "https://example.com/img.png",
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

    def test_run_batch_log_path_is_relative_to_kb_agent_dir(self, config):
        from agent.mcp.server import agent_relative_path

        log_path = config.kb_agent_dir / "data" / "batch_manual.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        visible = agent_relative_path(log_path, config.kb_agent_dir)
        assert not Path(visible).is_absolute()
        assert visible == "data/batch_manual.log"
        assert str(config.kb_agent_dir) not in visible


class TestKbOcrSizeCap:
    def test_max_image_bytes_in_source(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        assert "MAX_IMAGE_BYTES = 25 * 1024 * 1024" in text
        assert "size > MAX_IMAGE_BYTES" in text
        assert "MAX_IMAGE_PIXELS" in text
        assert "MAX_NEW_OCR_TOKENS" in text
        assert "w * h > MAX_IMAGE_PIXELS" in text

    def test_read_images_returns_structured_errors_per_item(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        start = text.index("def read_images")
        body = text[start : text.index("@app.local_entrypoint", start)]
        assert "for item in images:" in body
        assert "except Exception" in body
        assert '"error":safe' in body.replace(" ", "") or '"error": safe' in body

    def test_main_writes_error_ocr_json(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        assert 'r.get("error")' in text
        assert "ocr_error_json" in text

    def test_main_redacts_reading_urls_before_write(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        assert "redact_http_urls_in_text" in text
        assert 'r["reading"]' in text or "r['reading']" in text

    def test_batch_ingest_rejects_truncated_ocr_flag(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_trunc"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://example.com/huge.png",
            "substack_image",
            "img_trunc",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())

        def fake_run(cmd, **kw):
            if cmd[0] == "curl":
                dest = Path(cmd[cmd.index("-o") + 1])
                dest.write_bytes(b"\x89PNG" + b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    if staged.name.startswith(ingest_id):
                        (out_dir / f"{staged.name}.ocr.json").write_text(
                            json.dumps({"reading": "partial", "truncated": True}),
                            encoding="utf-8",
                        )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q], config, conn, 99)
        row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?", (ingest_id,)
        ).fetchone()
        assert row["status"] == "failed"
        assert "truncated" in (row["error"] or "").lower()


class TestBatchOcrPartialFailure:
    def test_nonzero_exit_still_dispatches_images_with_output(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_partial"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        id_ok = queue_ingest(
            conn,
            "https://example.com/ok.png",
            "substack_image",
            "img_ok",
            parent_slug=parent,
            media_kind="image",
        )[0]
        id_bad = queue_ingest(
            conn,
            "https://example.com/bad.png",
            "substack_image",
            "img_bad",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q_ok = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_ok,)).fetchone())
        q_bad = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_bad,)).fetchone())

        def fake_run(cmd, **kw):
            if cmd[0] == "curl":
                dest = Path(cmd[cmd.index("-o") + 1])
                dest.write_bytes(b"\x89PNG" + b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    if staged.name.startswith(id_ok):
                        (out_dir / f"{staged.name}.ocr.json").write_text(
                            json.dumps({"reading": "chart text"}),
                            encoding="utf-8",
                        )
                return MagicMock(returncode=1, stdout="", stderr="partial modal failure")
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest, "OCR_BATCH_MAX_FAILURE_RATE", 0.5)
        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q_ok, q_bad], config, conn, 42)

        ok_row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (id_ok,),
        ).fetchone()
        bad_row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (id_bad,),
        ).fetchone()
        assert ok_row["status"] == "dispatched"
        assert bad_row["status"] == "failed"
        assert bad_row["error"] == "no OCR output"

    def test_ocr_error_json_marks_failed_with_message(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_ocr_err"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://example.com/bad.png",
            "substack_image",
            "img_err",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())

        def fake_run(cmd, **kw):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG" + b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    (out_dir / f"{staged.name}.ocr.json").write_text(
                        json.dumps({"error": "CUDA OOM"}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q], config, conn, 43)
        row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "CUDA OOM" in row["error"]
        assert row["error"] != "no OCR output"

    def test_high_ocr_failure_rate_fails_entire_batch(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_batch_fail"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        id_ok = queue_ingest(
            conn,
            "https://example.com/ok.png",
            "substack_image",
            "img_ok2",
            parent_slug=parent,
            media_kind="image",
        )[0]
        id_bad = queue_ingest(
            conn,
            "https://example.com/bad.png",
            "substack_image",
            "img_bad2",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q_ok = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_ok,)).fetchone())
        q_bad = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_bad,)).fetchone())

        def fake_run(cmd, **kw):
            if cmd[0] == "curl":
                dest = Path(cmd[cmd.index("-o") + 1])
                dest.write_bytes(b"\x89PNG" + b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    if staged.name.startswith(id_ok):
                        (out_dir / f"{staged.name}.ocr.json").write_text(
                            json.dumps({"reading": "good chart"}),
                            encoding="utf-8",
                        )
                    else:
                        (out_dir / f"{staged.name}.ocr.json").write_text(
                            json.dumps({"error": "decode failed"}),
                            encoding="utf-8",
                        )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q_ok, q_bad], config, conn, 44)

        ok_row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (id_ok,),
        ).fetchone()
        bad_row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (id_bad,),
        ).fetchone()
        assert ok_row["status"] == "failed"
        assert "failure rate exceeded" in ok_row["error"]
        assert bad_row["status"] == "failed"
        assert "decode failed" in bad_row["error"]


class TestAppendFailureMarksIngest:
    def test_missing_raw_marks_failed_not_dispatched(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent"
        from agent.db import queue_ingest

        ingest_id, _ = queue_ingest(
            conn,
            "https://example.com/img.png",
            "substack_image",
            "img1",
            parent_slug=parent,
            media_kind="image",
        )
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
            if not batch_ingest._append_to_parent(
                config,
                q_item["parent_slug"],
                "Image",
                reading,
                conn=conn,
                ingest_id=q_item["id"],
            ):
                batch_ingest.update_ingest_status(
                    conn, q_item["id"], "failed", error="parent raw markdown missing",
                )
            else:
                batch_ingest.update_ingest_status(conn, q_item["id"], "dispatched")

        assert (ingest_id, "failed", "parent raw markdown missing") in statuses


class TestResolveMediaKind:
    def test_infers_image_from_content_type_when_media_kind_null(self):
        import scripts.batch_ingest as batch_ingest

        q = {
            "content_type": "substack_image",
            "media_kind": None,
            "parent_slug": "p",
            "slug": "img",
        }
        kind, err = batch_ingest._resolve_media_kind(q)
        assert kind == "image"
        assert err is None

    def test_typo_media_kind_inferred_from_content_type(self):
        import scripts.batch_ingest as batch_ingest

        kind, err = batch_ingest._resolve_media_kind({
            "content_type": "youtube",
            "media_kind": "Video",
            "slug": "v",
        })
        assert kind == "video"
        assert err is None

    def test_unresolvable_media_kind_returns_error(self):
        import scripts.batch_ingest as batch_ingest

        kind, err = batch_ingest._resolve_media_kind({
            "content_type": "web",
            "media_kind": "audio",
            "slug": "x",
        })
        assert kind is None
        assert "unsupported media_kind" in err


class TestMediaKindRoutingInMain:
    def test_orphan_embedded_row_marked_failed_not_stuck_downloading(
        self, config, conn, monkeypatch, tmp_path,
    ):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)

        ingest_id = queue_ingest(
            conn,
            "https://example.com/file.bin",
            "web",
            "orphan_media",
            parent_slug="parent_x",
            media_kind="audio",
        )[0]
        conn.execute(
            "UPDATE ingest_queue SET status='downloading' WHERE id=?",
            (ingest_id,),
        )
        conn.commit()
        claimed = [dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())]

        monkeypatch.setattr(batch_ingest, "load_config", lambda: config)
        monkeypatch.setattr(batch_ingest, "acquire", lambda _c: True)
        monkeypatch.setattr(batch_ingest, "release", lambda _c: None)
        monkeypatch.setattr(
            batch_ingest,
            "init_db",
            lambda *a, **k: conn,
        )
        monkeypatch.setattr(batch_ingest, "recover_stale_ingest", lambda c: None)
        monkeypatch.setattr(batch_ingest, "recover_failed_media", lambda c: 0)
        monkeypatch.setattr(batch_ingest, "record_batch_run", lambda c: 1)
        monkeypatch.setattr(batch_ingest, "claim_ingest_items", lambda c: claimed)
        monkeypatch.setattr(batch_ingest, "_reconcile_deferred_parent_syntheses", lambda *a: None)
        monkeypatch.setattr(batch_ingest, "finish_batch_run", lambda *a: None)
        from agent.db import update_ingest_status as real_update

        updates: list[tuple] = []

        def track_update(connection, item_id, status, error=None):
            updates.append((item_id, status, error))
            real_update(connection, item_id, status, error=error)

        monkeypatch.setattr(batch_ingest, "update_ingest_status", track_update)

        batch_ingest.main()
        failed = [u for u in updates if u[0] == ingest_id and u[1] == "failed"]
        assert failed
        assert "unsupported media_kind" in (failed[0][2] or "")
        assert all(u[1] != "downloading" for u in updates if u[0] == ingest_id)


class TestVideoStagingUsesQueueId:
    """Batch video staging/transcripts must use ingest_queue.id, not slug."""

    @staticmethod
    def _shared_substack_post_slug():
        from agent.core.slugify import slugify

        post = "https://author.substack.com/p/my-post"
        assert slugify(f"{post}#vid-a") == slugify(f"{post}#vid-b")
        return slugify(post)

    def test_two_native_videos_same_slug_distinct_staged_files(
        self, config, conn, monkeypatch, tmp_path,
    ):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = self._shared_substack_post_slug()
        post = "https://author.substack.com/p/my-post"
        parent = "parent_multi_vid"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        id_a = queue_ingest(
            conn,
            f"{post}#native-a",
            "substack_video",
            slug,
            parent_slug=parent,
            media_kind="video",
        )[0]
        id_b = queue_ingest(
            conn,
            f"{post}#native-b",
            "substack_video",
            slug,
            parent_slug=parent,
            media_kind="video",
        )[0]
        q_a = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_a,)).fetchone())
        q_b = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_b,)).fetchone())
        assert q_a["slug"] == q_b["slug"]

        v_a = tmp_path / "a.mp4"
        v_b = tmp_path / "b.mp4"
        v_a.write_bytes(b"a" * 200)
        v_b.write_bytes(b"b" * 200)

        def fake_download(q, _cfg):
            return (v_a if q["id"] == id_a else v_b, None)

        monkeypatch.setattr(batch_ingest, "_download_video", fake_download)
        staged_paths: list[Path] = []

        def fake_run(cmd, **kw):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                bdir = Path(cmd[2])
                out_dir.mkdir(parents=True, exist_ok=True)
                for staged in bdir.iterdir():
                    staged_paths.append(staged)
                    (out_dir / f"{staged.stem}.json").write_text(
                        json.dumps({"segments": [{"text": f"tx-{staged.stem}"}]}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_videos([q_a, q_b], config, conn, 77)

        assert len(staged_paths) == 2
        names = {p.name for p in staged_paths}
        assert f"{id_a}.mp4" in names
        assert f"{id_b}.mp4" in names
        assert id_a != id_b
        assert all(conn.execute(
            "SELECT status FROM ingest_queue WHERE id=?",
            (i,),
        ).fetchone()["status"] == "dispatched" for i in (id_a, id_b))

    def test_two_standalone_native_videos_same_slug_distinct_minio_and_dispatch_keys(
        self, config, conn, monkeypatch, tmp_path,
    ):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = self._shared_substack_post_slug()
        post = "https://author.substack.com/p/my-post"
        id_a = queue_ingest(
            conn,
            f"{post}#native-a",
            "substack_video",
            slug,
            media_kind="video",
        )[0]
        id_b = queue_ingest(
            conn,
            f"{post}#native-b",
            "substack_video",
            slug,
            media_kind="video",
        )[0]
        q_a = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_a,)).fetchone())
        q_b = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_b,)).fetchone())
        assert q_a["slug"] == q_b["slug"]
        assert not q_a.get("parent_slug")

        v_a = tmp_path / "a.mp4"
        v_b = tmp_path / "b.mp4"
        v_a.write_bytes(b"a" * 200)
        v_b.write_bytes(b"b" * 200)

        def fake_download(q, _cfg):
            return (v_a if q["id"] == id_a else v_b, None)

        monkeypatch.setattr(batch_ingest, "_download_video", fake_download)

        upload_keys: list[str] = []

        def track_upload(_src, dest_key):
            upload_keys.append(dest_key)

        monkeypatch.setattr(batch_ingest, "upload_file_to_minio", track_upload)

        dispatch_slugs: list[str] = []

        def fake_dispatch(*_a, **kwargs):
            dispatch_slugs.append(kwargs.get("slug"))
            return {"status": "pending", "task_id": f"task-{kwargs.get('slug')}"}

        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)

        def fake_run(cmd, **kw):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                for ingest_id in (id_a, id_b):
                    (out_dir / f"{ingest_id}.json").write_text(
                        json.dumps({"segments": [{"text": f"tx-{ingest_id}"}]}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        failed, dispatched, _ = batch_ingest._process_videos([q_a, q_b], config, conn, 79)
        assert failed == 0
        assert dispatched == 2
        assert len(upload_keys) == 4
        joined = " ".join(upload_keys)
        assert f"videos/{id_a}.mp4" in joined
        assert f"videos/{id_b}.mp4" in joined
        assert f"transcripts/{id_a}.json" in joined
        assert f"transcripts/{id_b}.json" in joined
        assert f"videos/{slug}." not in joined
        assert f"transcripts/{slug}.json" not in joined
        assert dispatch_slugs == [id_a, id_b]

    def test_same_youtube_embed_two_parents_distinct_staged_files(
        self, config, conn, monkeypatch, tmp_path,
    ):
        import scripts.batch_ingest as batch_ingest
        from agent.core.slugify import slugify
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        yt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        slug = slugify(yt, content_type="youtube")
        for parent in ("parent_one", "parent_two"):
            (config.temp_dir / f"{parent}_raw.md").write_text(f"# {parent}\n", encoding="utf-8")
        id_one = queue_ingest(
            conn, yt, "youtube", slug, parent_slug="parent_one", media_kind="video",
        )[0]
        id_two = queue_ingest(
            conn, yt, "youtube", slug, parent_slug="parent_two", media_kind="video",
        )[0]
        q_one = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_one,)).fetchone())
        q_two = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (id_two,)).fetchone())
        assert q_one["slug"] == q_two["slug"]

        v_one = tmp_path / "one.mp4"
        v_two = tmp_path / "two.mp4"
        v_one.write_bytes(b"1" * 200)
        v_two.write_bytes(b"2" * 200)

        def fake_download(q, _cfg):
            return (v_one if q["id"] == id_one else v_two, None)

        monkeypatch.setattr(batch_ingest, "_download_video", fake_download)
        staged_paths: list[Path] = []

        def fake_run(cmd, **kw):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                bdir = Path(cmd[2])
                out_dir.mkdir(parents=True, exist_ok=True)
                for staged in bdir.iterdir():
                    staged_paths.append(staged)
                    (out_dir / f"{staged.stem}.json").write_text(
                        json.dumps({"segments": [{"text": f"words-{staged.stem}"}]}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_videos([q_one, q_two], config, conn, 88)

        assert len(staged_paths) == 2
        assert {p.name for p in staged_paths} == {f"{id_one}.mp4", f"{id_two}.mp4"}
        raw_one = (config.temp_dir / "parent_one_raw.md").read_text(encoding="utf-8")
        raw_two = (config.temp_dir / "parent_two_raw.md").read_text(encoding="utf-8")
        assert f"words-{id_one}" in raw_one
        assert f"words-{id_two}" in raw_two


class TestDownloadVideoResolvesByQueueId:
    """Temp-dir lookup must use ingest id so shared slugs do not cross-pick siblings."""

    def test_picks_id_keyed_mp4_not_slug_sibling(self, config, monkeypatch, tmp_path):
        import scripts.batch_ingest as batch_ingest

        class _Step:
            def __init__(self, name):
                self.name = name

            def fn(self, _task, _cfg):
                return None

        monkeypatch.setattr(
            "agent.pipelines.get_pipeline",
            lambda ct: [_Step(n) for n in batch_ingest.VIDEO_DOWNLOAD_STEPS[ct]],
        )

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        shared_slug = "author.substack.com_p_my-post"
        id_want = "ingest-want-this-one"
        id_other = "ingest-other-sibling"
        (config.temp_dir / f"{id_want}.mp4").write_bytes(b"w" * 200)
        (config.temp_dir / f"{id_other}.mp4").write_bytes(b"o" * 200)
        (config.temp_dir / f"{shared_slug}.mp4").write_bytes(b"s" * 200)

        q = {
            "id": id_want,
            "slug": shared_slug,
            "url": f"https://author.substack.com/p/my-post#{id_want}",
            "content_type": "substack_video",
        }
        path, err = batch_ingest._download_video(q, config)
        assert err is None
        assert path == config.temp_dir / f"{id_want}.mp4"
        assert path.read_bytes()[0:1] == b"w"


class TestDownloadVideoUnsupportedContentType:
    def test_unknown_content_type_returns_error_not_keyerror(self, config, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        q = {
            "id": "i1",
            "slug": "s1",
            "url": "https://example.com/v",
            "content_type": "not_a_video_pipeline",
        }
        path, err = batch_ingest._download_video(q, config)
        assert path is None
        assert err == "unsupported embedded content_type: not_a_video_pipeline"


class TestBatchPipelineIdKeyedDownloadArtifacts:
    """Embedded YouTube/Rumble downloads must write temp artifacts keyed by ingest id."""

    def test_embedded_youtube_produces_id_keyed_video_and_metadata_pairing(
        self, config, monkeypatch,
    ):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        ingest_id = "ingest-yt-embedded-001"
        shared_slug = "youtube.com_watch_v_dQw4w9WgXcQ"
        q = {
            "id": ingest_id,
            "slug": shared_slug,
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "content_type": "youtube",
        }

        def fake_run(cmd, **_kw):
            joined = " ".join(str(c) for c in cmd)
            if "%(title)s" in joined:
                return MagicMock(returncode=0, stdout="T||U||60||1||20240101")
            if "%(chapters)s" in joined:
                return MagicMock(returncode=0, stdout="[]")
            if "%(description)s" in joined:
                return MagicMock(returncode=0, stdout="desc")
            if "-o" in cmd:
                out = Path(cmd[cmd.index("-o") + 1])
                out.write_bytes(b"\x00" * 2000)
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr("agent.pipelines.youtube._validate_cookies", lambda _cfg: None)
        monkeypatch.setattr("agent.pipelines.youtube.subprocess.run", fake_run)
        monkeypatch.setattr("agent.pipelines.youtube.ensure_h264", lambda p: p)

        path, err = batch_ingest._download_video(q, config)
        assert err is None
        assert path == config.temp_dir / f"{ingest_id}.mp4"
        meta = config.temp_dir / f"{ingest_id}_metadata.json"
        assert meta.is_file()
        assert json.loads(meta.read_text(encoding="utf-8"))["title"] == "T"
        assert not (config.temp_dir / f"{shared_slug}.mp4").exists()
        assert not (config.temp_dir / f"{shared_slug}_metadata.json").exists()

    def test_embedded_rumble_same_url_different_parents_no_collision(
        self, config, monkeypatch,
    ):
        import scripts.batch_ingest as batch_ingest
        from agent.core.slugify import slugify

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        rumble_url = "https://rumble.com/v2vk68c-example"
        shared_slug = slugify(rumble_url, content_type="rumble")
        cdn_url = "https://hugh.cdn.rumble.cloud/video/same-asset.mp4"

        monkeypatch.setattr(
            "agent.pipelines.rumble.resolve_rumble",
            lambda _url: ("Title", [{"url": cdn_url, "height": 480}]),
        )

        def fake_run(cmd, **_kw):
            if cmd[0] == "curl" and "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x00" * 2000)
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr("agent.pipelines.rumble.subprocess.run", fake_run)
        monkeypatch.setattr("agent.pipelines.rumble.ensure_h264", lambda p: p)

        id_a = "ingest-rumble-parent-a"
        id_b = "ingest-rumble-parent-b"
        results: list[Path] = []
        for ingest_id, parent in ((id_a, "parent_a"), (id_b, "parent_b")):
            q = {
                "id": ingest_id,
                "slug": shared_slug,
                "url": rumble_url,
                "content_type": "rumble",
            }
            path, err = batch_ingest._download_video(q, config)
            assert err is None
            assert path == config.temp_dir / f"{ingest_id}.mp4"
            results.append(path)

        assert results[0] != results[1]
        meta_a = json.loads((config.temp_dir / f"{id_a}_metadata.json").read_text())
        meta_b = json.loads((config.temp_dir / f"{id_b}_metadata.json").read_text())
        assert meta_a["source_url"] == rumble_url
        assert meta_b["source_url"] == rumble_url
        assert not (config.temp_dir / f"{shared_slug}.mp4").exists()

    def test_embedded_instagram_reel_produces_id_keyed_video(self, config, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.cookie_dir.mkdir(parents=True, exist_ok=True)
        (config.cookie_dir / "ig_cookies.txt").write_text(
            "instagram.com\tTRUE\t/\tFALSE\t0\tsessionid\ttest\n",
            encoding="utf-8",
        )
        ingest_id = "ingest-ig-reel-001"
        shared_slug = "instagram.com_reel_ABC123"
        q = {
            "id": ingest_id,
            "slug": shared_slug,
            "url": "https://www.instagram.com/reel/ABC123/",
            "content_type": "instagram_reel",
        }

        def fake_run(cmd, **_kw):
            if "-o" in cmd or "--output" in cmd:
                key = "-o" if "-o" in cmd else "--output"
                out = Path(cmd[cmd.index(key) + 1])
                target = out.with_suffix(".mp4") if out.suffix else Path(str(out) + ".mp4")
                target.write_bytes(b"\x00" * 2000)
                return MagicMock(returncode=0, stderr="", stdout="")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr("agent.pipelines.instagram.subprocess.run", fake_run)
        monkeypatch.setattr("agent.pipelines.instagram.ensure_h264", lambda p: p)

        path, err = batch_ingest._download_video(q, config)
        assert err is None
        assert path == config.temp_dir / f"{ingest_id}.mp4"
        slug_reel = config.temp_dir / f"{shared_slug}_reel.mp4"
        assert not slug_reel.exists()

    def test_embedded_threads_video_same_url_different_parents_no_collision(
        self, config, monkeypatch,
    ):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.cookie_dir.mkdir(parents=True, exist_ok=True)
        (config.cookie_dir / "threads_cookies.txt").write_text("# ok\n", encoding="utf-8")

        threads_url = "https://www.threads.net/@user/post/XYZ"
        shared_slug = "threads.net_user_post_XYZ"
        video_html = (
            '{"video_versions":[{"url":"https://scontent.cdninstagram.com/v/threads-v.mp4",'
            '"height":720}],"caption":{"text":"clip"},"permalink":"https://www.threads.com/'
            '\\u0040user/post/XYZ","is_reply":false}<div class="_ab6c">x</div>'
        )

        monkeypatch.setattr(
            "agent.pipelines.threads._curl_threads",
            lambda *_a, **_kw: video_html,
        )

        def fake_run(cmd, **_kw):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x00" * 2000)
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr("agent.pipelines.threads.subprocess.run", fake_run)
        monkeypatch.setattr("agent.pipelines.threads.ensure_h264", lambda p: p)

        id_a = "ingest-threads-parent-a"
        id_b = "ingest-threads-parent-b"
        for ingest_id in (id_a, id_b):
            q = {
                "id": ingest_id,
                "slug": shared_slug,
                "url": threads_url,
                "content_type": "threads_video",
            }
            path, err = batch_ingest._download_video(q, config)
            assert err is None
            assert path == config.temp_dir / f"{ingest_id}.mp4"
            assert (config.temp_dir / f"{ingest_id}_threads_raw.json").is_file()

        assert not (config.temp_dir / f"{shared_slug}_threads_video.mp4").exists()
        assert (config.temp_dir / f"{id_a}.mp4").exists()
        assert (config.temp_dir / f"{id_b}.mp4").exists()

    def test_copy_threads_raw_for_synthesize_slug_keyed_sidecar(self, config):
        import scripts.batch_ingest as batch_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        ingest_id = "ingest-threads-copy-raw"
        storage_slug = "threads.net_user_post_COPY"
        q = {"id": ingest_id, "slug": storage_slug}
        batch_payload = {
            "author": "copyauthor",
            "captions": ["batch detect caption"],
            "parent_context": "parent ctx",
        }
        (config.temp_dir / f"{ingest_id}_threads_raw.json").write_text(
            json.dumps(batch_payload),
            encoding="utf-8",
        )
        batch_ingest.copy_threads_raw_for_synthesize(config, q, storage_slug)
        slug_sidecar = config.temp_dir / f"{storage_slug}_threads_raw.json"
        assert slug_sidecar.is_file()
        loaded = json.loads(slug_sidecar.read_text(encoding="utf-8"))
        assert loaded["author"] == "copyauthor"
        assert loaded["parent_context"] == "parent ctx"


class TestChildProcessLogScrubbing:
    """Child stdout/stderr tails in batch logs must be scrubbed at write time."""

    def test_process_videos_scrubs_transcribe_child_stderr(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        ingest_id = queue_ingest(
            conn, "https://www.youtube.com/watch?v=scrub1", "youtube", "yt_scrub_log",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        fake_video = config.temp_dir / "yt_scrub_log.mp4"
        fake_video.write_bytes(b"x" * 200)
        monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

        sensitive = (
            "modal run ok prefix "
            "yt-dlp: cookiefile /root/kb-agent/cookies/yt.txt "
            "api_key=supersecret99 "
            "suffix"
        )

        def fake_run(cmd, **kwargs):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{ingest_id}.json").write_text(
                    json.dumps({"segments": [{"text": "hello"}]}),
                    encoding="utf-8",
                )
                return MagicMock(returncode=0, stdout="", stderr=sensitive)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest, "upload_file_to_minio", lambda *_a, **_k: None)
        monkeypatch.setattr(
            batch_ingest,
            "dispatch_synthesize",
            lambda *_a, **_k: {"status": "pending", "task_id": "t1"},
        )
        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)

        batch_ingest._process_videos([q], config, conn, 501)
        logged = capsys.readouterr().out
        assert "/root/kb-agent" not in logged
        assert "supersecret99" not in logged
        assert "[path]" in logged
        assert "[credential redacted]" in logged
        assert "modal run ok prefix" in logged

    def test_process_embedded_images_scrubs_ocr_child_stderr(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_ocr_scrub"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://example.com/chart.png",
            "substack_image",
            "img_scrub_log",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        sensitive = (
            "ocr status: reading complete "
            "Netscape cookie at /home/ubuntu/secret/cookies.txt "
            "token=abcdef "
            "done"
        )

        def fake_run(cmd, **kw):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    (out_dir / f"{staged.name}.ocr.json").write_text(
                        json.dumps({"reading": "chart labels"}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0, stdout=sensitive, stderr="")
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q], config, conn, 502)
        logged = capsys.readouterr().out
        assert "/home/ubuntu" not in logged
        assert "abcdef" not in logged
        assert "[path]" in logged
        assert "[credential redacted]" in logged
        assert "ocr status: reading complete" in logged

    def test_process_videos_scrubs_download_failure_line(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        ingest_id = queue_ingest(
            conn, "https://www.youtube.com/watch?v=dlfail", "youtube", "yt_dl_scrub",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        leaky = "download_video failed: cookiefile /root/kb-agent/cookies/yt.txt password=sekret"
        monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (None, leaky))

        batch_ingest._process_videos([q], config, conn, 503)
        logged = capsys.readouterr().out
        assert "/root/kb-agent" not in logged
        assert "sekret" not in logged
        assert "[path]" in logged
        assert "[credential redacted]" in logged
        assert "download_video failed" in logged

    def test_process_videos_scrubs_upload_and_dispatch_lines(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
        ingest_id = queue_ingest(
            conn, "https://www.youtube.com/watch?v=upfail", "youtube", "yt_up_scrub",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        fake_video = config.temp_dir / "yt_up_scrub.mp4"
        fake_video.write_bytes(b"x" * 200)
        monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

        class _UploadExc(Exception):
            pass

        upload_calls = {"n": 0}

        def flaky_upload(*_a, **_k):
            upload_calls["n"] += 1
            if upload_calls["n"] == 1:
                raise _UploadExc("minio: /root/.minio/creds token=upload_leak")

        monkeypatch.setattr(batch_ingest, "upload_file_to_minio", flaky_upload)

        ingest_dispatch = queue_ingest(
            conn, "https://www.youtube.com/watch?v=disfail", "youtube", "yt_dispatch_scrub",
        )[0]
        q_dispatch = dict(
            conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_dispatch,)).fetchone(),
        )
        fake_video_2 = config.temp_dir / "yt_dispatch_scrub.mp4"
        fake_video_2.write_bytes(b"y" * 200)

        def fake_run(cmd, **kwargs):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                for item in (ingest_id, ingest_dispatch):
                    (out_dir / f"{item}.json").write_text(
                        json.dumps({"segments": [{"text": "hi"}]}),
                        encoding="utf-8",
                    )
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        monkeypatch.setattr(
            batch_ingest,
            "dispatch_synthesize",
            lambda *_a, **_k: {
                "status": "error",
                "error": "dispatch denied at /root/kb-agent/data token=badkey",
            },
        )

        batch_ingest._process_videos([q, q_dispatch], config, conn, 504)
        logged = capsys.readouterr().out
        assert "upload_leak" not in logged
        assert "badkey" not in logged
        assert "/root" not in logged
        assert "[credential redacted]" in logged
        assert "upload failed" in logged
        assert "dispatch failed" in logged

    def test_main_scrubs_media_kind_and_submit_errors(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        queue_ingest(
            conn,
            "https://example.com/x.png",
            "substack",
            "bad_kind",
            parent_slug="parent_bad",
            media_kind="bogus",
        )
        queue_ingest(conn, "https://example.com/page", "web", "nonvid_fail")

        monkeypatch.setattr(batch_ingest, "load_config", lambda: config)
        monkeypatch.setattr(batch_ingest, "acquire", lambda _c: True)
        monkeypatch.setattr(batch_ingest, "release", lambda _c: None)
        monkeypatch.setattr(batch_ingest, "recover_stale_ingest", lambda _c: None)
        monkeypatch.setattr(batch_ingest, "recover_failed_media", lambda _c: 0)
        monkeypatch.setattr(batch_ingest, "record_batch_run", lambda _c: 9)
        monkeypatch.setattr(
            batch_ingest,
            "submit_standalone_non_video_for_batch",
            lambda *_a, **_k: {
                "status": "error",
                "error": "firecrawl refused /root/.hermes/profiles api_key=xyz",
            },
        )
        monkeypatch.setattr(batch_ingest, "finish_batch_run", lambda *a: None)
        monkeypatch.setattr(batch_ingest, "_reconcile_deferred_parent_syntheses", lambda *a: None)

        def init_with_test_db(*a, **k):
            conn.row_factory = sqlite3.Row
            return conn

        monkeypatch.setattr(batch_ingest, "init_db", init_with_test_db)

        def claim_all(_c):
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM ingest_queue WHERE status='queued'",
                ).fetchall()
            ]

        monkeypatch.setattr(batch_ingest, "claim_ingest_items", claim_all)

        batch_ingest.main()
        logged = capsys.readouterr().out
        assert "xyz" not in logged
        assert "/root/.hermes" not in logged
        assert "[credential redacted]" in logged
        assert "[failed]" in logged

    def test_main_scrubs_requeue_stdout(self, config, conn, monkeypatch, capsys):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        queue_ingest(conn, "https://example.com/page", "web", "nonvid_pause")

        monkeypatch.setattr(batch_ingest, "load_config", lambda: config)
        monkeypatch.setattr(batch_ingest, "acquire", lambda _c: True)
        monkeypatch.setattr(batch_ingest, "release", lambda _c: None)
        monkeypatch.setattr(batch_ingest, "recover_stale_ingest", lambda _c: None)
        monkeypatch.setattr(batch_ingest, "recover_failed_media", lambda _c: 0)
        monkeypatch.setattr(batch_ingest, "record_batch_run", lambda _c: 10)
        monkeypatch.setattr(
            batch_ingest,
            "submit_standalone_non_video_for_batch",
            lambda *_a, **_k: {
                "status": "paused",
                "reason": "missing cookies at /root/kb-agent/cookies/web.txt token=leaked",
            },
        )
        monkeypatch.setattr(batch_ingest, "finish_batch_run", lambda *a: None)
        monkeypatch.setattr(batch_ingest, "_reconcile_deferred_parent_syntheses", lambda *a: None)

        def init_with_test_db(*a, **k):
            conn.row_factory = sqlite3.Row
            return conn

        monkeypatch.setattr(batch_ingest, "init_db", init_with_test_db)

        def claim_all(_c):
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM ingest_queue WHERE status='queued'",
                ).fetchall()
            ]

        monkeypatch.setattr(batch_ingest, "claim_ingest_items", claim_all)

        batch_ingest.main()
        logged = capsys.readouterr().out
        assert "leaked" not in logged
        assert "/root/kb-agent" not in logged
        assert "[requeue]" in logged
        assert "[credential redacted]" in logged
        assert "[path]" in logged


class TestBatchIngestLogScrubExemptions:
    """Scripts outside batch_ingest/transcribe/ocr with raw stderr tails are exempt.

    Persistent batch cron stdout is captured in batch_manual.log; other scripts either
    scrub at rest (rotate_logs.py) or write only to ephemeral operator stderr
    (tdah_daily_check.py, watch_sources_poll.py, diarize.py usage hints).
    """

    def test_exempt_operator_scripts_exist_on_disk(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        for name in (
            "rotate_logs.py",
            "tdah_daily_check.py",
            "watch_sources_poll.py",
        ):
            assert (scripts_dir / name).is_file(), name


class TestCorruptJsonHandling:
    def test_corrupt_ocr_json_marks_failed(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_ocr"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://example.com/chart.png",
            "substack_image",
            "img_bad_ocr",
            parent_slug=parent,
            media_kind="image",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())

        def fake_run(cmd, **kw):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x00" * 200)
                return MagicMock(returncode=0)
            if "batch_ocr.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                stage_dir = Path(cmd[2])
                for staged in stage_dir.iterdir():
                    (out_dir / f"{staged.name}.ocr.json").write_text("{not-json", encoding="utf-8")
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_images([q], config, conn, 9)
        row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "invalid OCR JSON" in row["error"]

    def test_corrupt_transcript_json_marks_failed(self, config, conn, monkeypatch):
        import subprocess

        import scripts.batch_ingest as batch_ingest
        from agent.db import queue_ingest

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        parent = "parent_tx"
        (config.temp_dir / f"{parent}_raw.md").write_text("# Post\n", encoding="utf-8")
        ingest_id = queue_ingest(
            conn,
            "https://www.youtube.com/watch?v=badjson",
            "youtube",
            "vid_bad_json",
            parent_slug=parent,
            media_kind="video",
        )[0]
        q = dict(conn.execute("SELECT * FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone())
        fake_video = config.temp_dir / "vid_bad_json.mp4"
        fake_video.write_bytes(b"x" * 200)
        monkeypatch.setattr(batch_ingest, "_download_video", lambda _q, _cfg: (fake_video, None))

        def fake_run(cmd, **kwargs):
            if "batch_transcribe.py" in cmd[1]:
                out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{ingest_id}.json").write_text("<<bad>>", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.run(cmd, **kwargs)

        monkeypatch.setattr(batch_ingest.subprocess, "run", fake_run)
        batch_ingest._process_embedded_videos([q], config, conn, 10)
        row = conn.execute(
            "SELECT status, error FROM ingest_queue WHERE id=?",
            (ingest_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "invalid transcript JSON" in row["error"]

    def test_corrupt_manifest_skips_dispatch_without_crash(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        parent = "parent_manifest"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        (config.temp_dir / f"{parent}_media.json").write_text("{broken", encoding="utf-8")
        dispatch = MagicMock()
        monkeypatch.setattr(batch_ingest, "dispatch_synthesize", dispatch)
        batch_ingest._deferred_synthesize_dispatched.clear()
        assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False
        dispatch.assert_not_called()


class TestStandaloneNonVideoDispatch:
    """batch_ingest standalone non-video dispatch + requeue invariants."""

    def test_skipped_submit_requeues_ingest(self, conn):
        import scripts.batch_ingest as batch_ingest

        ingest_id, _ = queue_ingest(conn, "https://example.com/page", "web", "web_slug")
        conn.execute("UPDATE ingest_queue SET status='downloading' WHERE id=?", (ingest_id,))
        conn.commit()
        outcome = batch_ingest.apply_standalone_non_video_result(
            conn, ingest_id, {"status": "skipped", "reason": "Already in vault"},
        )
        assert outcome is None
        row = conn.execute("SELECT status FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone()
        assert row["status"] == "queued"

    def test_paused_submit_requeues_ingest(self, conn):
        import scripts.batch_ingest as batch_ingest

        ingest_id, _ = queue_ingest(conn, "https://example.com/page2", "web", "web_slug2")
        conn.execute("UPDATE ingest_queue SET status='downloading' WHERE id=?", (ingest_id,))
        conn.commit()
        outcome = batch_ingest.apply_standalone_non_video_result(
            conn, ingest_id, {"status": "paused", "reason": "cookies"},
        )
        assert outcome is None
        row = conn.execute("SELECT status FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone()
        assert row["status"] == "queued"

    def test_pending_with_task_id_dispatches(self, conn):
        import scripts.batch_ingest as batch_ingest

        ingest_id, _ = queue_ingest(conn, "https://example.com/page3", "web", "web_slug3")
        conn.execute("UPDATE ingest_queue SET status='downloading' WHERE id=?", (ingest_id,))
        conn.commit()
        outcome = batch_ingest.apply_standalone_non_video_result(
            conn, ingest_id, {"status": "pending", "task_id": "new-task"},
        )
        assert outcome == "dispatched"
        row = conn.execute("SELECT status FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone()
        assert row["status"] == "dispatched"

    def test_batch_preflight_pause_skips_paused_task(self, config, conn, monkeypatch):
        import scripts.batch_ingest as batch_ingest

        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        monkeypatch.setattr(
            batch_ingest,
            "plan_submission",
            lambda url, cfg, ignore_dedup=False: {
                "status": "paused",
                "reason": "cookie jar empty",
                "platform": "youtube",
            },
        )
        r = batch_ingest.submit_standalone_non_video_for_batch(
            conn, "https://youtube.com/watch?v=abc", config,
        )
        assert r["status"] == "paused"
        assert "task_id" not in r
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
