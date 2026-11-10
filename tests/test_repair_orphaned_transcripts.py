"""Unit tests for scripts/repair_orphaned_transcripts.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "repair_orphaned_transcripts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("repair_orphaned_transcripts", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(_REPO))
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mc_bare_json_lines(filenames: list[str]) -> str:
    """``mc ls --json`` lines: ``key`` is the bare object name under the listed prefix."""
    return "\n".join(json.dumps({"type": "file", "key": name}) for name in filenames) + "\n"


def _note(
    *,
    slug: str,
    subdir: str = "",
    source_url: str,
    video_key: str,
    transcript_key: str | None = None,
    with_full_transcript: bool = False,
) -> str:
    sf = f"  video: {video_key}\n"
    if transcript_key:
        sf += f"  transcript: {transcript_key}\n"
    body = "## Summary\n\nShort.\n"
    if with_full_transcript:
        body += "\n## Full Transcript\n\nHello.\n"
    return f"""---
source_url: {source_url}
source_files:
{sf}---

{body}
"""


@pytest.fixture
def repair_mod():
    return _load_module()


@pytest.fixture
def repair_config(tmp_path):
    from agent.config import Config

    data = tmp_path / "data"
    data.mkdir()
    vault = tmp_path / "vault"
    kb = vault / "Knowledge base"
    kb.mkdir(parents=True)
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=vault,
        temp_dir=tmp_path / "tmp",
        storage_bucket="knowledge-base",
        storage_alias="tigris",
    )


@pytest.fixture
def agent_conn(repair_config):
    from agent.db import AGENT_SCHEMA, init_db

    return init_db(repair_config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)


def test_storage_slug_matches_resolve_slug(repair_mod):
    from agent.core.slugify import batch_standalone_video_storage_slug, slugify
    from agent.pipelines.video_common import _resolve_slug

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    _, base, storage_slug = repair_mod.resolve_video_synthesize_plan(url)
    task_slug = slugify(url, content_type=base)
    assert storage_slug == _resolve_slug({"id": str(uuid.uuid4()), "slug": task_slug})

    ingest_id = "ingest-queue-uuid-1234"
    substack_url = "https://author.substack.com/p/example-post"
    _, base_ss, slug_ss = repair_mod.resolve_video_synthesize_plan(
        substack_url, ingest_id=ingest_id
    )
    assert base_ss == "substack_video"
    assert slug_ss == batch_standalone_video_storage_slug(
        ingest_id=ingest_id,
        slug=slugify(substack_url, content_type="substack_video"),
        content_type="substack_video",
        parent_slug=None,
    )


def test_resolve_video_synthesize_plan_mega_video_returns_none(repair_mod):
    assert repair_mod.resolve_video_synthesize_plan("https://mega.nz/file/abc#key") is None


def test_discover_redeploy_and_migrate_from_reels(
    repair_mod, repair_config, monkeypatch
):
    slug = "gandhi-construction-coloniale-simon-legalliot"
    reels_bare = f"{slug}_transcript.json"
    kb = repair_config.vault_path / "Knowledge base"
    note_path = kb / f"{slug}.md"
    note_path.write_text(
        _note(
            slug=slug,
            source_url="https://www.instagram.com/reel/abc123/",
            video_key=f"reels/{slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            prefix = cmd[4]
            if prefix.endswith("/reels/"):
                return MagicMock(returncode=0, stdout=_mc_bare_json_lines([reels_bare]), stderr="")
            if prefix.endswith("/transcripts/"):
                return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected mc", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == slug
    assert plan.redeploy[0].content_type == "instagram_reel_synthesize"
    assert len(plan.migrate) == 1
    assert plan.migrate[0].src == f"reels/{reels_bare}"
    assert plan.migrate[0].dst == f"transcripts/{slug}.json"
    assert plan.missing_transcript == []


def test_discover_prefers_video_key_slug_over_url_storage_slug(
    repair_mod, repair_config, monkeypatch
):
    video_slug = "my-reel-clean-slug"
    url_slug = "www.instagram.com_reel_xyz_"
    kb = repair_config.vault_path / "Knowledge base"
    note_path = kb / "orphan-display-name.md"
    note_path.write_text(
        _note(
            slug="orphan-display-name",
            source_url="https://www.instagram.com/reel/xyz/",
            video_key=f"reels/{video_slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            prefix = cmd[4]
            if prefix.endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines(
                        [f"{url_slug}.json", f"{video_slug}.json"]
                    ),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == video_slug


def test_discover_substack_finds_transcript_at_url_slug_when_video_uses_ingest_id(
    repair_mod, repair_config, monkeypatch
):
    from agent.core.slugify import slugify
    from agent.pipelines.video_common import _resolve_slug

    ingest_id = "ing-queue-only-uuid"
    substack_url = "https://author.substack.com/p/example-post"
    url_slug = _resolve_slug(
        {
            "id": str(uuid.uuid4()),
            "slug": slugify(substack_url, content_type="substack_video"),
        }
    )
    assert url_slug != ingest_id

    kb = repair_config.vault_path / "Knowledge base" / "articles"
    kb.mkdir(parents=True)
    note_path = kb / f"{url_slug}.md"
    note_path.write_text(
        _note(
            slug=url_slug,
            source_url=substack_url,
            video_key=f"videos/{ingest_id}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            if cmd[4].endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines([f"{url_slug}.json"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == url_slug
    assert plan.redeploy[0].content_type == "substack_video_synthesize"


def test_discover_substack_embedded_transcript_by_ingest_id(
    repair_mod, repair_config, monkeypatch
):
    ingest_id = "ing-emb-uuid"
    parent_slug = "author.substack.com_p_parent"
    kb = repair_config.vault_path / "Knowledge base" / "articles"
    kb.mkdir(parents=True)
    note_path = kb / f"{parent_slug}.md"
    note_path.write_text(
        _note(
            slug=parent_slug,
            source_url="https://author.substack.com/p/parent#video-slot",
            video_key=f"videos/{ingest_id}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            if cmd[4].endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines([f"{ingest_id}.json"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == ingest_id
    assert plan.redeploy[0].content_type == "substack_video_synthesize"


def test_discover_redeploy_when_transcript_in_canonical_prefix(
    repair_mod, repair_config, monkeypatch
):
    display_slug = "my-reel-note"
    storage_slug = "www.instagram.com_reel_xyz_"
    kb = repair_config.vault_path / "Knowledge base" / "instagram"
    kb.mkdir(parents=True)
    (kb / f"{display_slug}.md").write_text(
        _note(
            slug=display_slug,
            source_url="https://www.instagram.com/reel/xyz/",
            video_key=f"videos/{storage_slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            if cmd[4].endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines([f"{storage_slug}.json"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == storage_slug
    assert plan.migrate == []
    assert plan.missing_transcript == []


def test_discover_missing_transcript_report_only(repair_mod, repair_config, monkeypatch):
    slug = "no-transcript-anywhere"
    kb = repair_config.vault_path / "Knowledge base" / "videos"
    kb.mkdir(parents=True)
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    (kb / f"{slug}.md").write_text(
        _note(
            slug=slug,
            source_url=url,
            video_key=f"videos/{slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            if cmd[4].endswith("/videos/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines([f"{slug}.mp4"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert plan.redeploy == []
    assert plan.migrate == []
    assert len(plan.missing_transcript) == 1
    assert plan.missing_transcript[0].source_url == url


def test_apply_dry_run_skips_dispatch_and_mc_cp(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.core.slugify import slugify
    from agent.detection import normalize_url

    slug = "dry-run-slug"
    url = "https://rumble.com/v12345-title.html"
    url_norm = normalize_url(url)
    storage_slug = slugify(url_norm, content_type="rumble")

    kb = repair_config.vault_path / "Knowledge base"
    (kb / f"{slug}.md").write_text(
        _note(
            slug=slug,
            source_url=url,
            video_key=f"videos/{storage_slug}.mp4",
        ),
        encoding="utf-8",
    )

    dispatch_calls = []
    mc_calls = []

    def fake_run(cmd, **kwargs):
        mc_calls.append(list(cmd))
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            if cmd[4].endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_bare_json_lines([f"{storage_slug}.json"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["mc", "cp"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "t"}

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.discover(repair_config)
    assert len(plan.redeploy) == 1

    repair_mod.apply(plan, repair_config, agent_conn, dry_run=True)
    assert dispatch_calls == []
    assert not any(c[:2] == ["mc", "cp"] for c in mc_calls)

    dispatch_calls.clear()
    mc_calls.clear()
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][1]["content_type"] == "rumble_synthesize"
    assert dispatch_calls[0][1]["slug"] == storage_slug

    agent_conn.close()


def test_discover_skips_note_with_youtube_transcript_heading(
    repair_mod, repair_config, monkeypatch
):
    kb = repair_config.vault_path / "Knowledge base" / "videos"
    kb.mkdir(parents=True)
    slug = "already-has-transcript"
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    (kb / f"{slug}.md").write_text(
        _note(
            slug=slug,
            source_url=url,
            video_key=f"videos/{slug}.mp4",
        ).replace("## Summary", "## Transcript\n\nDone.\n\n## Summary"),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stderr="unexpected", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)
    assert plan.redeploy == []
    assert plan.missing_transcript == []


def test_apply_active_synthesize_guard_does_not_fail_existing_tasks(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task

    slug = "orphan_dup_parent"
    url = "https://blog.substack.com/p/orphan-dup"
    parent_id = create_task(
        agent_conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup"],
        status="paused",
    )
    with agent_conn:
        agent_conn.execute(
            """
            INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                "https://example.com/i.png",
                "substack_image",
                "i1",
                "dispatched",
                slug,
                "image",
            ),
        )
    synth_id = create_task(
        agent_conn,
        url,
        "substack_video_synthesize",
        slug,
        ["synthesize_note"],
        status="pending",
    )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "new"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="substack_video_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)

    row = agent_conn.execute("SELECT status FROM tasks WHERE id=?", (synth_id,)).fetchone()
    assert row[0] == "pending"
    assert dispatch_calls == []
    assert parent_id  # parent row untouched; guard must not supersede synthesize tasks

    agent_conn.close()


def test_apply_dispatches_when_synthesize_task_is_paused(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task
    from agent.pipelines import get_pipeline_names

    slug = "orphan_paused_synth"
    url = "https://www.youtube.com/watch?v=pausedorphan1"
    steps = get_pipeline_names("youtube_synthesize")
    synth_id = create_task(
        agent_conn,
        url,
        "youtube_synthesize",
        slug,
        steps,
        status="paused",
    )
    agent_conn.execute(
        "UPDATE task_steps SET status = 'paused' WHERE task_id = ? AND name = 'fetch_transcript'",
        (synth_id,),
    )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": synth_id, "redispatched": True}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="youtube_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)

    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][1]["slug"] == slug

    agent_conn.close()


def test_apply_skips_dispatch_when_done_synth_parent_siblings_no_deferral(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task

    slug = "orphan_done_parent"
    url = "https://blog.substack.com/p/orphan-done"
    synth_id = create_task(
        agent_conn,
        url,
        "substack_video_synthesize",
        slug,
        ["synthesize_note"],
        status="done",
    )
    with agent_conn:
        agent_conn.execute(
            """
            INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                "https://example.com/i.png",
                "substack_image",
                "i1",
                "dispatched",
                slug,
                "image",
            ),
        )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "new"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="substack_video_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)

    row = agent_conn.execute("SELECT status FROM tasks WHERE id=?", (synth_id,)).fetchone()
    assert row[0] == "done"
    assert dispatch_calls == []
    assert plan.skipped_parent_reconcile == []

    agent_conn.close()


def test_apply_skips_dispatch_when_done_synth_parent_siblings_stale_deferral(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task, ensure_parent_media_deferral

    slug = "orphan_done_parent_stale_def"
    url = "https://blog.substack.com/p/orphan-done-stale"
    synth_id = create_task(
        agent_conn,
        url,
        "substack_video_synthesize",
        slug,
        ["synthesize_note"],
        status="done",
    )
    with agent_conn:
        agent_conn.execute(
            """
            INSERT INTO ingest_queue (id, url, content_type, slug, status, parent_slug, media_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                "https://example.com/i.png",
                "substack_image",
                "i1",
                "dispatched",
                slug,
                "image",
            ),
        )
        ensure_parent_media_deferral(
            agent_conn,
            slug,
            url,
            "substack_text",
            expected_total=1,
        )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "new"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="substack_video_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)

    row = agent_conn.execute("SELECT status FROM tasks WHERE id=?", (synth_id,)).fetchone()
    assert row[0] == "done"
    assert dispatch_calls == []
    assert plan.skipped_parent_reconcile == []

    agent_conn.close()


def test_apply_skips_dispatch_when_synthesize_task_done(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task

    slug = "orphan_done_no_parent"
    url = "https://www.youtube.com/watch?v=doneorphan1"
    create_task(
        agent_conn,
        url,
        "youtube_synthesize",
        slug,
        ["fetch_transcript", "synthesize_youtube"],
        status="done",
    )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "new"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="youtube_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)
    assert dispatch_calls == []

    agent_conn.close()


def test_apply_skips_dispatch_when_paused_on_non_fetch_step(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    from agent.db import create_task
    from agent.pipelines import get_pipeline_names

    slug = "orphan_paused_synth_step"
    url = "https://www.youtube.com/watch?v=pausedorphan2"
    steps = get_pipeline_names("youtube_synthesize")
    synth_id = create_task(
        agent_conn,
        url,
        "youtube_synthesize",
        slug,
        steps,
        status="paused",
    )
    agent_conn.execute(
        "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND name = 'fetch_transcript'",
        (synth_id,),
    )
    agent_conn.execute(
        "UPDATE task_steps SET status = 'paused' WHERE task_id = ? AND name = 'synthesize_youtube'",
        (synth_id,),
    )
    agent_conn.commit()

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "new"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=repair_config.vault_path / "Knowledge base" / "x.md",
                source_url=url,
                content_type="youtube_synthesize",
                storage_slug=slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)
    assert dispatch_calls == []

    agent_conn.close()


def test_apply_skips_when_canonical_target_note_already_has_transcript(
    repair_mod, repair_config, agent_conn, monkeypatch
):
    storage_slug = "gandhi-construction-coloniale-simon-legalliot"
    orphan_slug = "gandhi-stub-duplicate"
    url = "https://www.instagram.com/reel/duplicate123/"
    kb = repair_config.vault_path / "Knowledge base" / "instagram"
    kb.mkdir(parents=True)
    canonical = kb / f"{storage_slug}.md"
    canonical.write_text(
        _note(
            slug=storage_slug,
            source_url=url,
            video_key=f"reels/{storage_slug}.mp4",
            with_full_transcript=True,
        ),
        encoding="utf-8",
    )
    orphan = kb.parent / f"{orphan_slug}.md"
    orphan.write_text(
        _note(
            slug=orphan_slug,
            source_url=url,
            video_key=f"reels/{storage_slug}.mp4",
        ),
        encoding="utf-8",
    )

    dispatch_calls = []

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "t"}

    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.RepairPlan(
        redeploy=[
            repair_mod.RedeployItem(
                note_path=orphan,
                source_url=url,
                content_type="instagram_reel_synthesize",
                storage_slug=storage_slug,
            )
        ],
    )
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=False)
    assert dispatch_calls == []

    agent_conn.close()
