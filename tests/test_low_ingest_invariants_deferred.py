"""Regression tests for deferred-media / low ingest invariants (round)."""

import asyncio
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.consumer import Consumer
from agent.db import AGENT_SCHEMA, claim_ingest_items, init_db, queue_ingest
from agent.detection import _detect_threads_subtype
from agent.exceptions import ErrorType, StepError
from agent.pipelines.see_also import _see_also_link_count, step_see_also
from agent.pipelines.instagram import step_ig_reel_merge
from agent.pipelines.threads import step_threads_merge
from agent.pipelines.video_common import _artifact_stem, _slug_path, resolve_artifact_stem
from agent.pipelines.youtube import (
    step_merge_and_identify,
    step_minio_upload,
)
from agent.quality import _resynthesize
from agent.consumer import recover_orphan_tasks

TOKENIZED = "https://cdn.example.com/article?token=secret&expires=999"


@pytest.fixture
def agent_db(tmp_path):
    return init_db(tmp_path / "agent.db", AGENT_SCHEMA)


@pytest.fixture
def consumer(config, agent_db, tmp_path):
    from agent.db import LOGS_SCHEMA

    logs = init_db(tmp_path / "logs.db", LOGS_SCHEMA)
    c = Consumer(config)
    c.agent_conn = agent_db
    c.logs_conn = logs
    c._video_semaphore = asyncio.BoundedSemaphore(1)
    return c


@pytest.fixture
def config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    temp = tmp_path / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=vault,
        temp_dir=temp,
    )


def test_see_also_counts_links_only_in_see_also_section():
    body = (
        "[[one]] [[two]] [[three]]\n\n"
        "## See Also\n"
        "- [[only-one]]\n"
    )
    assert _see_also_link_count(body, "## See Also") == 1
    assert _see_also_link_count(body, "## Voir aussi") == 0


def test_step_see_also_resolves_slug_path_for_sanitized_vault_note(config):
    """IG/Threads notes live at _slug_path(slug).md — raw slug paths must not no-op."""
    raw_slug = "instagram.com/reel/ABC123"
    safe_slug = _slug_path(raw_slug)
    task_dict = {"slug": raw_slug, "content_type": "instagram_reel", "id": "ig-see-also-1"}
    folder = config.vault_path / "Knowledge base" / "instagram"
    folder.mkdir(parents=True, exist_ok=True)
    note_path = folder / f"{safe_slug}.md"
    note_path.write_text("# Reel\n\nBody without See Also.\n", encoding="utf-8")
    other = folder / "peer-reel.md"
    other.write_text("# Peer\n", encoding="utf-8")

    with patch("agent.pipelines.see_also._search_vault", return_value=[other]), patch(
        "agent.pipelines.see_also.get_llm_client"
    ) as mock_llm:
        mock_llm.return_value.call.return_value = MagicMock(
            content="## See Also\n- [[peer-reel]] — related\n"
        )
        result = step_see_also(task_dict, config)

    assert "No note to add" not in result
    updated = note_path.read_text(encoding="utf-8")
    assert "[[peer-reel]]" in updated
    assert "Added 1 See Also links" in result


def test_step_see_also_appends_section_when_header_missing(config):
    """Notes without a See Also stub must gain the section; count is section-only."""
    task_dict = {"slug": "no-see-also-header", "content_type": "web", "id": "task-no-header"}
    folder = config.vault_path / "Knowledge base" / "articles"
    note_path = folder / f"{task_dict['slug']}.md"
    note_path.write_text(
        "# Book summary\n\nBody only — no See Also section.\n",
        encoding="utf-8",
    )
    other = folder / "peer.md"
    other.write_text("# Peer\nOverlap.\n", encoding="utf-8")

    with patch("agent.pipelines.see_also._search_vault", return_value=[other]), patch(
        "agent.pipelines.see_also.get_llm_client"
    ) as mock_llm:
        mock_llm.return_value.call.return_value = MagicMock(
            content="## See Also\n- [[peer]] — related\n- [[peer-two]] — also\n"
        )
        result = step_see_also(task_dict, config)

    updated = note_path.read_text(encoding="utf-8")
    assert updated.endswith("## See Also\n- [[peer]] — related\n- [[peer-two]] — also")
    assert "## See Also" in updated
    assert "Added 2 See Also links" in result
    assert _see_also_link_count(updated, "## See Also") == 2


def test_step_see_also_reported_count_ignores_body_wikilinks(config):
    """Status message must not count wikilinks outside the See Also section."""
    task_dict = {"slug": "body-wikilinks-only", "content_type": "web", "id": "task-body-wl"}
    folder = config.vault_path / "Knowledge base" / "articles"
    note_path = folder / f"{task_dict['slug']}.md"
    note_path.write_text(
        "Mentions [[alpha]] and [[beta]] and [[gamma]] in the body.\n",
        encoding="utf-8",
    )
    other = folder / "delta.md"
    other.write_text("# Delta\n", encoding="utf-8")

    with patch("agent.pipelines.see_also._search_vault", return_value=[other]), patch(
        "agent.pipelines.see_also.get_llm_client"
    ) as mock_llm:
        mock_llm.return_value.call.return_value = MagicMock(
            content="## See Also\n- [[delta]] — one link\n"
        )
        result = step_see_also(task_dict, config)

    assert "Added 1 See Also links" in result
    assert "Added 4" not in result


def test_step_see_also_redacts_tracking_urls_in_llm_section(config):
    """See Also LLM output must not persist query/fragment URLs into the vault note."""
    task_dict = {"slug": "see-also-url-redact", "content_type": "web", "id": "task-see-url"}
    folder = config.vault_path / "Knowledge base" / "articles"
    note_path = folder / f"{task_dict['slug']}.md"
    note_path.write_text(
        "---\nsource_url: https://example.com/base\n---\n\n# Note\n\nBody.\n",
        encoding="utf-8",
    )
    other = folder / "peer.md"
    other.write_text("# Peer\n", encoding="utf-8")

    with patch("agent.pipelines.see_also._search_vault", return_value=[other]), patch(
        "agent.pipelines.see_also.get_llm_client"
    ) as mock_llm:
        mock_llm.return_value.call.return_value = MagicMock(
            content=(
                f"## See Also\n"
                f"- [[peer]] — see {TOKENIZED} for context\n"
            )
        )
        step_see_also(task_dict, config)

    written = note_path.read_text(encoding="utf-8")
    assert "token=secret" not in written
    assert "https://cdn.example.com/article" in written


def test_resynthesize_scrubs_source_url_on_vault_write(tmp_path, config):
    """Re-synthesize must scrub frontmatter source_url before writing the vault note."""
    slug = "resynth_url_scrub"
    note_path = tmp_path / f"{slug}.md"
    note_path.write_text(
        f"---\nsource_url: {TOKENIZED}\n---\n\n# Old\n",
        encoding="utf-8",
    )
    (Path(config.temp_dir)).mkdir(parents=True, exist_ok=True)
    (Path(config.temp_dir) / f"{slug}_raw.md").write_text("source text", encoding="utf-8")

    llm = MagicMock()
    llm.call.return_value = MagicMock(
        content=(
            f"---\nsource_url: {TOKENIZED}\n---\n\n"
            f"# Rewritten\n\nAlso links {TOKENIZED}.\n"
        )
    )

    assert _resynthesize(note_path, config, llm, ["too short"]) is True
    written = note_path.read_text(encoding="utf-8")
    assert "token=secret" not in written
    assert "https://cdn.example.com/article" in written


def test_step_see_also_refreshes_when_body_has_many_wikilinks(config):
    task_dict = {"slug": "see-also-body-links", "content_type": "web", "id": "task-see-also"}
    folder = config.vault_path / "Knowledge base" / "articles"
    note_path = folder / f"{task_dict['slug']}.md"
    note_path.write_text(
        "[[a]] [[b]] [[c]] [[d]]\n\n## See Also\n_No related notes found._\n",
        encoding="utf-8",
    )
    other = folder / "other-note.md"
    other.write_text("# Other\nTopic overlap.\n", encoding="utf-8")

    with patch("agent.pipelines.see_also._search_vault", return_value=[other]), patch(
        "agent.pipelines.see_also.get_llm_client"
    ) as mock_llm:
        mock_llm.return_value.call.return_value = MagicMock(
            content="## See Also\n- [[other-note]] — related\n"
        )
        result = step_see_also(task_dict, config)

    assert "already populated" not in result
    updated = note_path.read_text(encoding="utf-8")
    assert "[[other-note]]" in updated


def test_detect_threads_subtype_returns_text_on_fetch_failure(monkeypatch, caplog):
    def boom(cmd, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    caplog.set_level(logging.WARNING)
    subtype, fetch_failed = _detect_threads_subtype("https://threads.net/@u/post/1")
    assert subtype == "threads_text"
    assert fetch_failed is True
    assert any("Threads subtype fetch failed" in r.message for r in caplog.records)


def test_resynthesize_logs_on_llm_failure(tmp_path, config, caplog):
    caplog.set_level(logging.WARNING)
    slug = "resynth_slug"
    note_path = tmp_path / f"{slug}.md"
    note_path.write_text("# Note\n", encoding="utf-8")
    (Path(config.temp_dir)).mkdir(parents=True, exist_ok=True)
    (Path(config.temp_dir) / f"{slug}_raw.md").write_text("source", encoding="utf-8")

    llm = MagicMock()
    llm.call.side_effect = RuntimeError("api down")

    assert _resynthesize(note_path, config, llm, ["too short"]) is False
    assert any("Re-synthesize LLM call failed" in r.message for r in caplog.records)


def test_resynthesize_logs_when_raw_source_missing(tmp_path, config, caplog):
    caplog.set_level(logging.WARNING)
    slug = "resynth_no_raw"
    note_path = tmp_path / f"{slug}.md"
    note_path.write_text("# Note\n", encoding="utf-8")
    (Path(config.temp_dir)).mkdir(parents=True, exist_ok=True)

    llm = MagicMock()

    assert _resynthesize(note_path, config, llm, ["too short"]) is False
    llm.call.assert_not_called()
    assert any("raw source missing" in r.message for r in caplog.records)


def test_artifact_stem_batch_uses_ingest_id_and_resolve_slug_otherwise():
    tid = "ingest-batch-uuid"
    task_batch = {"id": tid, "slug": "foo/bar", "_batch_ingest": True}
    assert resolve_artifact_stem(task_batch) == tid
    assert _artifact_stem(task_batch) == tid
    task = {"id": tid, "slug": "foo/bar?"}
    assert resolve_artifact_stem(task) == "foo_bar_"
    assert _artifact_stem(task) == "foo_bar_"


def test_merge_and_identify_uses_artifact_stem_paths(config, tmp_path):
    task_id = "youtube-ingest-item-1"
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "id": task_id,
        "slug": "yt-slug",
        "_batch_ingest": True,
    }
    (config.temp_dir / f"{task_id}_diarization.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "speaker": "S1"}]}),
        encoding="utf-8",
    )
    (config.temp_dir / f"{task_id}_transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}),
        encoding="utf-8",
    )
    step_merge_and_identify(task, config)
    assert (config.temp_dir / f"{task_id}_merged.json").is_file()


def test_ig_reel_merge_uses_artifact_stem_paths(config):
    task_id = "ig-batch-ingest-uuid"
    task = {
        "id": task_id,
        "slug": "instagram.com/reel/ABC",
        "_batch_ingest": True,
    }
    (config.temp_dir / f"{task_id}_diarization.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "speaker": "S1"}]}),
        encoding="utf-8",
    )
    (config.temp_dir / f"{task_id}_transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "speech long enough to avoid music-only detection threshold",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    step_ig_reel_merge(task, config)
    assert (config.temp_dir / f"{task_id}_merged.json").is_file()


def test_threads_merge_uses_artifact_stem_paths(config):
    task_id = "threads-batch-ingest-uuid"
    task = {
        "id": task_id,
        "slug": "threads.net/@u/post/1",
        "_batch_ingest": True,
    }
    (config.temp_dir / f"{task_id}_diarization.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "speaker": "S1"}]}),
        encoding="utf-8",
    )
    (config.temp_dir / f"{task_id}_transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}),
        encoding="utf-8",
    )
    step_threads_merge(task, config)
    assert (config.temp_dir / f"{task_id}_merged.json").is_file()


@patch("agent.archival.upload_file_to_minio", side_effect=lambda src, dest, **kw: dest)
@patch("agent.pipelines.video_common.subprocess.run")
def test_mega_minio_upload_resolves_webm_video(mock_run, _mock_upload, config):
    """Mega download normalizes to {stem}.webm; MinIO upload must not require .mp4 only."""
    slug = "mega_video_test"
    task = {
        "id": "test-mega-video-1",
        "slug": slug,
        "content_type": "mega_video",
    }
    (config.temp_dir / f"{slug}.webm").write_bytes(b"fake-webm")
    (config.temp_dir / f"{slug}_merged.json").write_text("{}", encoding="utf-8")
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "---\nsource_files:\n  video: http://placeholder\n---\n# Note\n",
        encoding="utf-8",
    )
    def fake_ffmpeg(cmd, **kwargs):
        if cmd and str(cmd[0]) == "ffmpeg":
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-mp3")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_ffmpeg
    result = step_minio_upload(task, config)
    assert "videos/" in result
    assert f"videos/{slug}.webm" in result
    assert f"videos/{slug}.mp3" in result


def test_kb_ocr_truncated_payload_includes_flag():
    text = (Path(__file__).resolve().parents[1] / "modal" / "kb_ocr.py").read_text(
        encoding="utf-8"
    )
    assert '"truncated": True' in text


def test_claim_ingest_items_claims_in_created_at_order(tmp_path):
    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    queue_ingest(conn, "https://a.example", "substack_image", "a", media_kind="image")
    queue_ingest(conn, "https://b.example", "substack_image", "b", media_kind="image")
    expected_order = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM ingest_queue WHERE status='queued' "
            "ORDER BY created_at ASC, id ASC"
        ).fetchall()
    ]
    claimed = claim_ingest_items(conn)
    assert [c["id"] for c in claimed] == expected_order


def test_refresh_parent_media_manifest_syncs_media_changed_from_db(tmp_path):
    import scripts.batch_ingest as batch_ingest
    from agent.db import ensure_parent_media_deferral, get_parent_media_deferral

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "manifest_parent"
    config = Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "temp",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(
        conn, parent, "https://example.com/p", "substack_text", expected_total=1,
    )
    with conn:
        conn.execute(
            "UPDATE parent_media_deferral SET media_changed = 0 WHERE parent_slug = ?",
            (parent,),
        )
    batch_ingest.refresh_parent_media_manifest(conn, config, parent)
    manifest = json.loads((config.temp_dir / f"{parent}_media.json").read_text(encoding="utf-8"))
    assert manifest.get("media_changed") is False
    assert int(get_parent_media_deferral(conn, parent)["media_changed"]) == 0


def test_maybe_dispatch_parent_synthesize_without_manifest_url(tmp_path, monkeypatch):
    """Deferral gate must come from DB/bootstrap, not manifest ``url`` (redacted sidecars)."""
    import scripts.batch_ingest as batch_ingest
    from agent.db import create_task, ensure_parent_media_deferral, media_append_marker, queue_ingest

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "url_redacted_parent"
    post = "https://author.substack.com/p/post"
    config = Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "temp",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(conn, parent, post, "substack_text", 1)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    create_task(
        conn,
        post,
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

    called: list[str] = []

    def fake_dispatch(_url, _cfg, _agent_conn, **kwargs):
        called.append(kwargs.get("slug") or "")
        return {"status": "pending", "task_id": "synth-1", "duplicate": False}

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is True
    assert called == [parent]
    assert "url" not in json.loads(
        (config.temp_dir / f"{parent}_media.json").read_text(encoding="utf-8"),
    )


def test_parent_deferral_dispatch_error_records_scrubbed_pause(tmp_path, monkeypatch):
    import scripts.batch_ingest as batch_ingest
    from agent.db import create_task, ensure_parent_media_deferral, media_append_marker, queue_ingest

    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    conn.row_factory = sqlite3.Row
    parent = "parent_dispatch_scrub"
    secret_url = "https://blog.substack.com/p/x?api_key=LEAK"
    config = Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "temp",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(conn, parent, secret_url, "substack_text", 1)
    (config.temp_dir / f"{parent}_media.json").write_text(
        json.dumps({"content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )
    parent_id = create_task(
        conn,
        secret_url,
        "substack_text",
        parent,
        ["substack_embedded_media"],
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

    def fake_dispatch(_url, _cfg, _agent_conn, **kwargs):
        return {
            "status": "error",
            "error": "dispatch denied api_key=LEAK",
        }

    monkeypatch.setattr(batch_ingest, "dispatch_synthesize", fake_dispatch)
    batch_ingest._deferred_synthesize_dispatched.clear()
    assert batch_ingest._maybe_dispatch_parent_synthesize(conn, config, parent) is False

    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?",
        (parent_id,),
    ).fetchone()
    assert row["status"] == "paused"
    assert "deferred synthesis dispatch failed" in (row["error"] or "")
    assert "LEAK" not in (row["error"] or "")
    assert "api_key" not in (row["error"] or "")


def test_zero_embeds_count_pending_media_raises_step_error_not_unbound(config, tmp_path):
    """Deferral gate must raise StepError when pending lookup fails (no UnboundLocalError)."""
    from agent.pipelines.web import step_substack_embedded_media

    config.kb_agent_dir = tmp_path / "kb"
    (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    slug = "deferral-db-fail"
    task_dict = {
        "id": "task-db-fail",
        "slug": slug,
        "source_url": "https://author.substack.com/p/post",
        "content_type": "substack_text",
    }
    (config.temp_dir / f"{slug}_media.json").write_text(
        json.dumps({"content_type": "substack_text", "total": 1}),
        encoding="utf-8",
    )

    pending_err = StepError(
        "awaiting media — re-fetch found no embeds but siblings pending",
        error_type=ErrorType.PAUSE,
        step_name="substack_embedded_media",
    )

    with patch("agent.pipelines.web._fetch_substack_html", return_value="<html><body>empty</body></html>"), patch(
        "agent.db.count_pending_media", side_effect=pending_err,
    ):
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
    assert exc.value is pending_err
    assert not isinstance(exc.value, UnboundLocalError)


def test_consumer_skips_orphan_recovery_when_run_already_did(monkeypatch, consumer):
    import agent.consumer as consumer_mod

    monkeypatch.setenv("KB_AGENT_ORPHAN_RECOVERY_DONE", "1")
    called = []

    def fake_recover(conn, **kwargs):
        called.append(1)

    monkeypatch.setattr(consumer_mod, "recover_orphan_tasks", fake_recover)
    monkeypatch.setattr("agent.pipelines.import_all_pipelines", lambda: None)
    monkeypatch.setattr("agent.pipelines.PIPELINES", {})

    asyncio.run(consumer.startup())
    assert called == []


def test_recover_orphan_tasks_shared_helper(agent_db):
    conn = agent_db
    task_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status) "
            "VALUES (?, ?, ?, ?, 'in_progress')",
            (task_id, "https://example.com", "web", "shared_orphan"),
        )
        conn.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status) "
            "VALUES (?, 0, 's0', 'done'), (?, 1, 's1', 'pending')",
            (task_id, task_id),
        )
    recover_orphan_tasks(conn, log_prefix="[test]")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row[0] == "pending"


def test_recover_orphan_tasks_skips_parent_with_active_synthesize(agent_db):
    from agent.db import create_task

    conn = agent_db
    slug = "orphan_synth_guard"
    url = "https://example.com/parent"
    parent_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="in_progress",
    )
    with conn:
        conn.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status, output) "
            "VALUES (?, 0, 'substack_cleanup', 'done', NULL), "
            "(?, 1, 'substack_embedded_media', 'running', 'pid=99999999')",
            (parent_id, parent_id),
        )
    create_task(
        conn,
        url,
        "substack_text_synthesize",
        slug,
        ["synthesize_note"],
        status="in_progress",
    )
    recover_orphan_tasks(conn, log_prefix="[test]")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (parent_id,)).fetchone()
    assert row[0] == "in_progress"
