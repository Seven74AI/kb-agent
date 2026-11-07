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

    def test_dispatch_supersedes_paused_parent_task(self, tmp_path, monkeypatch, config, conn):
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


class TestKbOcrSizeCap:
    def test_max_image_bytes_in_source(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        assert "MAX_IMAGE_BYTES = 25 * 1024 * 1024" in text
        assert "size > MAX_IMAGE_BYTES" in text

    def test_read_images_skips_bad_images_per_item(self):
        text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
        start = text.index("def read_images")
        body = text[start : text.index("@app.local_entrypoint", start)]
        assert "for item in images:" in body
        assert "except Exception" in body
        assert "skipped:" in body


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
