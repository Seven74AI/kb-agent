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


def _mc_json_lines(keys: list[str]) -> str:
    lines = []
    bucket = "knowledge-base"
    for key in keys:
        lines.append(json.dumps({"type": "file", "key": f"{bucket}/{key}"}))
    return "\n".join(lines) + "\n"


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
    folder = subdir.strip("/")
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
    from agent.pipelines.video_common import _resolve_slug

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    _, _, storage_slug = repair_mod.resolve_video_synthesize_plan(url)
    task_slug = repair_mod.slugify_for_storage(url, "youtube")
    assert storage_slug == _resolve_slug({"id": str(uuid.uuid4()), "slug": task_slug})


def test_discover_redeploy_and_migrate_from_reels(
    repair_mod, repair_config, monkeypatch
):
    slug = "gandhi-construction-coloniale-simon-legalliot"
    reels_key = f"reels/{slug}_transcript.json"
    kb = repair_config.vault_path / "Knowledge base"
    note_path = kb / f"{slug}.md"
    note_path.write_text(
        _note(
            slug=slug,
            source_url="https://www.instagram.com/reel/abc123/",
            video_key=f"videos/{slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            prefix = cmd[4]
            if prefix.endswith("/reels/"):
                return MagicMock(returncode=0, stdout=_mc_json_lines([reels_key]), stderr="")
            if prefix.endswith("/transcripts/"):
                return MagicMock(returncode=0, stdout="", stderr="")
            if prefix.endswith("/videos/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_json_lines([f"videos/{slug}.mp4"]),
                    stderr="",
                )
        return MagicMock(returncode=1, stderr="unexpected mc", stdout="")

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    plan = repair_mod.discover(repair_config)

    assert len(plan.redeploy) == 1
    assert plan.redeploy[0].storage_slug == slug  # matched via video object key stem
    assert plan.redeploy[0].content_type == "instagram_reel_synthesize"
    assert len(plan.migrate) == 1
    assert plan.migrate[0].storage_slug == slug
    assert plan.migrate[0].src.endswith(f"reels/{slug}_transcript.json")
    assert plan.migrate[0].dst.endswith(f"transcripts/{slug}.json")
    assert plan.missing_transcript == []


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
            video_key=f"videos/{display_slug}.mp4",
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            prefix = cmd[4]
            if prefix.endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_json_lines([f"transcripts/{storage_slug}.json"]),
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
                    stdout=_mc_json_lines([f"videos/{slug}.mp4"]),
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
    from agent.db import AGENT_SCHEMA, init_db

    slug = "dry-run-slug"
    kb = repair_config.vault_path / "Knowledge base"
    (kb / f"{slug}.md").write_text(
        _note(
            slug=slug,
            source_url="https://rumble.com/v12345-title.html",
            video_key=f"videos/{slug}.mp4",
        ),
        encoding="utf-8",
    )

    dispatch_calls = []
    mc_calls = []

    storage_slug = "rumble.com_v12345_title.html"

    def fake_run(cmd, **kwargs):
        mc_calls.append(list(cmd))
        if cmd[:4] == ["mc", "ls", "--recursive", "--json"]:
            prefix = cmd[4]
            if prefix.endswith("/transcripts/"):
                return MagicMock(
                    returncode=0,
                    stdout=_mc_json_lines([f"transcripts/{storage_slug}.json"]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    def fake_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return {"status": "pending", "task_id": "t"}

    monkeypatch.setattr(repair_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(repair_mod, "dispatch_synthesize", fake_dispatch)

    plan = repair_mod.discover(repair_config)
    repair_mod.apply(plan, repair_config, agent_conn, dry_run=True)

    assert dispatch_calls == []
    assert not any(c[:2] == ["mc", "cp"] for c in mc_calls)

    agent_conn.close()
    conn = init_db(repair_config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    conn.close()
    assert count == 0

