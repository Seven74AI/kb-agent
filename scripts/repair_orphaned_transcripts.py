#!/usr/bin/env python3
"""Discover and repair video notes missing synthesis when transcripts exist in object storage.

Cross-references MinIO (via ``mc``), the Obsidian vault, and the agent DB. By default
runs in dry-run mode: prints planned migrations and synthesize dispatches. Pass ``--apply``
to copy legacy ``reels/`` transcripts into ``transcripts/`` and enqueue synthesize tasks.

``--report`` prints notes that still need a full re-ingest (video archived, no transcript).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from agent.config import Config, load_config  # noqa: E402
from agent.core.slugify import batch_standalone_video_storage_slug, slugify, slug_path  # noqa: E402
from agent.core.submit import dispatch_synthesize  # noqa: E402
from agent.db import (  # noqa: E402
    AGENT_SCHEMA,
    _parent_sibling_media_count,
    init_db,
    normalize_tags,
)
from agent.detection import detect_content_subtype, detect_platform, normalize_url  # noqa: E402
from agent.pipelines import get_pipeline  # noqa: E402
from agent.pipelines.video_common import _resolve_slug  # noqa: E402

VIDEO_BASE_TYPES = frozenset(
    {
        "youtube",
        "rumble",
        "bunny_stream",
        "instagram_reel",
        "threads_video",
        "substack_video",
    }
)

_LOCAL_PLATFORM_VIDEO_BASE: dict[str, str] = {
    "youtube": "youtube",
    "rumble": "rumble",
    "bunny_stream": "bunny_stream",
    "instagram_reel": "instagram_reel",
    "threads": "threads_video",
    "substack": "substack_video",
}

_SYNTHESIZE_NOTE_SUBDIR: dict[str, str] = {
    "instagram_reel_synthesize": "instagram",
    "threads_video_synthesize": "threads",
    "substack_video_synthesize": "videos",
    "bunny_stream_synthesize": "",
    "youtube_synthesize": "videos",
    "rumble_synthesize": "videos",
}

_FULL_TRANSCRIPT_HEADING = re.compile(
    r"^##\s+(Full )?Transcript\s*$", re.MULTILINE
)
_SOURCE_URL = re.compile(r"^source_url:\s*(.+)\s*$", re.MULTILINE)
_TAGS_LINE = re.compile(r"^tags:\s*\[(.*)\]\s*$", re.MULTILINE)


@dataclass
class RedeployItem:
    note_path: Path
    source_url: str
    content_type: str
    storage_slug: str
    tags: list[str] | None = None


@dataclass
class MigrateItem:
    storage_slug: str
    src: str
    dst: str


@dataclass
class MissingTranscriptItem:
    note_path: Path
    source_url: str
    storage_slug: str | None
    video_key: str | None = None


@dataclass
class RepairPlan:
    redeploy: list[RedeployItem] = field(default_factory=list)
    migrate: list[MigrateItem] = field(default_factory=list)
    missing_transcript: list[MissingTranscriptItem] = field(default_factory=list)
    skipped_parent_reconcile: list[RedeployItem] = field(default_factory=list)


def slugify_for_storage(url: str, base_content_type: str) -> str:
    """Task slug before ``_resolve_slug`` (filesystem-safe path segment)."""
    return slugify(url, content_type=base_content_type)


def _local_video_base_from_platform(platform: str) -> str | None:
    """URL-pattern video base type without network I/O (excludes mega)."""
    return _LOCAL_PLATFORM_VIDEO_BASE.get(platform)


def resolve_video_synthesize_plan(
    url: str,
    *,
    ingest_id: str | None = None,
    parent_slug: str | None = None,
) -> tuple[str, str, str] | None:
    """Return (synthesize_content_type, base_content_type, storage_slug) for a video URL."""
    raw = (url or "").strip()
    if not raw:
        return None
    try:
        url = normalize_url(raw)
    except ValueError:
        return None

    platform = detect_platform(url)
    base = _local_video_base_from_platform(platform)
    if base is None:
        fetched = detect_content_subtype(url, platform)
        if fetched in VIDEO_BASE_TYPES:
            base = fetched
    if base not in VIDEO_BASE_TYPES:
        return None

    synthesize_ct = base if base.endswith("_synthesize") else f"{base}_synthesize"
    if get_pipeline(synthesize_ct) is None:
        return None

    task_slug = slugify_for_storage(url, base)
    if base == "substack_video" and ingest_id:
        storage_slug = batch_standalone_video_storage_slug(
            ingest_id=ingest_id,
            slug=task_slug,
            content_type=base,
            parent_slug=parent_slug,
        )
    else:
        storage_slug = _resolve_slug(
            {"id": "00000000-0000-0000-0000-000000000099", "slug": task_slug}
        )
    return synthesize_ct, base, storage_slug


def expected_synthesize_note_path(
    config: Config, *, storage_slug: str, content_type: str
) -> Path:
    """Vault path the synthesize pipeline will write for the given dispatch slug."""
    safe = slug_path(storage_slug)
    subdir = _SYNTHESIZE_NOTE_SUBDIR.get(content_type, "videos")
    kb = config.vault_path / "Knowledge base"
    if not subdir:
        return kb / f"{safe}.md"
    return kb / subdir / f"{safe}.md"


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    fm = text[3:end].lstrip("\n")
    body = text[end + 4 :].lstrip("\n")
    return fm, body


def _parse_source_files(fm: str) -> dict[str, str]:
    out: dict[str, str] = {}
    in_block = False
    for line in fm.splitlines():
        if line.strip() == "source_files:":
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r"^\s+(\w+):\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
            continue
        if line and not line[0].isspace():
            break
    return out


def _parse_tags(fm: str) -> list[str] | None:
    m = _TAGS_LINE.search(fm)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    parts = [p.strip().strip("'\"") for p in inner.split(",")]
    tags = [p for p in parts if p]
    return normalize_tags(tags) if tags else None


def _note_needs_synthesis(fm: str, body: str, source_files: dict[str, str]) -> bool:
    if "video" not in source_files:
        return False
    if "transcript" in source_files:
        return False
    if _FULL_TRANSCRIPT_HEADING.search(body):
        return False
    return True


def _storage_prefix(config: Config, folder: str) -> str:
    return f"{config.storage_alias}/{config.storage_bucket}/{folder}/"


def mc_ls_keys(config: Config, folder: str) -> list[str]:
    """List object keys under ``folder/`` relative to the bucket (e.g. ``videos/foo.mp4``)."""
    target = _storage_prefix(config, folder)
    result = subprocess.run(
        ["mc", "ls", "--recursive", "--json", target],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mc ls failed for {folder}: {(result.stderr or result.stdout or '').strip()[:300]}"
        )
    bucket = config.storage_bucket
    keys: list[str] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = entry.get("key") or entry.get("objectName") or ""
        if not raw:
            continue
        if raw.startswith(f"{bucket}/"):
            raw = raw[len(bucket) + 1 :]
        elif "/" in raw and not raw.startswith(folder):
            # Some mc builds emit alias/bucket/key — keep the tail after bucket.
            parts = raw.split("/")
            try:
                idx = parts.index(bucket)
                raw = "/".join(parts[idx + 1 :])
            except ValueError:
                pass
        if not raw.startswith(f"{folder}/"):
            raw = f"{folder}/{raw}"
        keys.append(raw)
    return keys


def _reels_slug_from_key(key: str) -> str:
    """Map a ``reels/`` object key to the storage slug ``step_fetch_transcript`` uses."""
    name = Path(key).name
    if not name.endswith(".json"):
        return Path(name).stem
    stem = Path(name).stem
    if stem.endswith("_transcript"):
        return stem[: -len("_transcript")]
    return stem


@dataclass
class _TranscriptIndex:
    canonical: set[str]
    reels_only: dict[str, str]

    def locate(self, storage_slug: str) -> tuple[str, str] | None:
        if storage_slug in self.canonical:
            return "transcripts", f"transcripts/{storage_slug}.json"
        if storage_slug in self.reels_only:
            return "reels", self.reels_only[storage_slug]
        return None


def _build_transcript_index(config: Config) -> _TranscriptIndex:
    canonical: set[str] = set()
    for key in mc_ls_keys(config, "transcripts"):
        if key.startswith("transcripts/") and key.endswith(".json"):
            canonical.add(Path(key).stem)

    reels_only: dict[str, str] = {}
    for key in mc_ls_keys(config, "reels"):
        if not (key.startswith("reels/") and key.endswith(".json")):
            continue
        slug = _reels_slug_from_key(key)
        if slug in canonical:
            continue
        reels_only[slug] = key

    return _TranscriptIndex(canonical=canonical, reels_only=reels_only)


def _slug_from_video_key(video_key: str) -> str | None:
    if not video_key:
        return None
    path = Path(video_key)
    if path.suffix:
        return path.stem
    return path.name or None


def _substack_url_storage_slug(url: str) -> str | None:
    """Storage slug from the normalized URL (live pipeline), not ingest queue id."""
    plan = resolve_video_synthesize_plan(url)
    if plan is None or plan[1] != "substack_video":
        return None
    task_slug = slugify_for_storage(url, "substack_video")
    return _resolve_slug(
        {"id": "00000000-0000-0000-0000-000000000099", "slug": task_slug}
    )


def _candidate_storage_slugs(
    *,
    note_path: Path,
    video_key: str | None,
    url_storage_slug: str | None,
    extra_slugs: list[str] | None = None,
) -> list[str]:
    """Slugs to try when locating a transcript (prefer uploaded object keys)."""
    seen: set[str] = set()
    out: list[str] = []

    def add(slug: str | None) -> None:
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)

    add(_slug_from_video_key(video_key or ""))
    add(url_storage_slug)
    for slug in extra_slugs or []:
        add(slug)
    add(note_path.stem)
    return out


@dataclass(frozen=True)
class _SynthesizeDispatchDecision:
    """Pure-read classification for whether ``apply`` should call ``dispatch_synthesize``."""

    dispatch: bool
    skip_message: str | None = None


def _synthesize_task_row(conn, slug: str, content_type: str):
    return conn.execute(
        """
        SELECT id, status FROM tasks
        WHERE slug = ? AND content_type = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (slug, content_type),
    ).fetchone()


def _paused_fetch_transcript_reset_eligible(conn, task_id: str) -> bool:
    """Mirror ``reset_paused_synthesize_fetch_transcript`` preconditions (read-only)."""
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row[0] != "paused":
        return False
    step = conn.execute(
        """
        SELECT status, step_order FROM task_steps
        WHERE task_id = ? AND name = 'fetch_transcript'
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if step is None or step[0] not in ("paused", "failed"):
        return False
    fetch_order = step[1]
    earlier = conn.execute(
        """
        SELECT 1 FROM task_steps
        WHERE task_id = ? AND step_order < ? AND status NOT IN ('done', 'skipped')
        LIMIT 1
        """,
        (task_id, fetch_order),
    ).fetchone()
    return earlier is None


def _primary_paused_step_name(conn, task_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT name FROM task_steps
        WHERE task_id = ? AND status = 'paused'
        ORDER BY step_order
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    return row[0] if row else None


def classify_synthesize_dispatch(
    conn, slug: str, content_type: str
) -> _SynthesizeDispatchDecision:
    """Classify existing synthesize task state before ``dispatch_synthesize`` (no side effects)."""
    row = _synthesize_task_row(conn, slug, content_type)
    if row is None:
        return _SynthesizeDispatchDecision(dispatch=True)
    task_id, status = row[0], row[1]
    if status in ("pending", "in_progress"):
        return _SynthesizeDispatchDecision(
            dispatch=False,
            skip_message=f"[skip] active synthesize already queued for slug={slug}",
        )
    if status == "paused":
        if _paused_fetch_transcript_reset_eligible(conn, task_id):
            return _SynthesizeDispatchDecision(dispatch=True)
        step_name = _primary_paused_step_name(conn, task_id) or "unknown"
        return _SynthesizeDispatchDecision(
            dispatch=False,
            skip_message=f"[skip] paused on {step_name}, manual slug={slug}",
        )
    if status in ("done", "completed"):
        return _SynthesizeDispatchDecision(
            dispatch=False,
            skip_message=(
                f"[skip] already done but note lacks transcript, manual slug={slug}"
            ),
        )
    return _SynthesizeDispatchDecision(dispatch=True)


def _target_note_already_synthesized(config: Config, item: RedeployItem) -> bool:
    """True when synthesize would write a different note that already has a transcript."""
    synth_path = expected_synthesize_note_path(
        config, storage_slug=item.storage_slug, content_type=item.content_type
    )
    if not synth_path.is_file():
        return False
    if synth_path.resolve() == item.note_path.resolve():
        return False
    text = synth_path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    source_files = _parse_source_files(fm)
    if "transcript" in source_files:
        return True
    if _FULL_TRANSCRIPT_HEADING.search(body):
        return True
    return False


def _needs_parent_reconcile_skip(conn, slug: str) -> bool:
    """Pure-read: parent slugs with ingest children use the deferral flow, not this repair."""
    return _parent_sibling_media_count(conn, slug) > 0


def discover(config: Config) -> RepairPlan:
    """Enumerate repair actions by cross-referencing storage and vault notes."""
    index = _build_transcript_index(config)
    kb_root = config.vault_path / "Knowledge base"
    if not kb_root.is_dir():
        raise RuntimeError(f"Knowledge base not found under vault: {kb_root}")

    plan = RepairPlan()
    seen_redeploy: set[tuple[str, str]] = set()

    for note_path in sorted(kb_root.rglob("*.md")):
        text = note_path.read_text(encoding="utf-8")
        fm, body = _split_frontmatter(text)
        source_files = _parse_source_files(fm)
        if not _note_needs_synthesis(fm, body, source_files):
            continue

        m = _SOURCE_URL.search(fm)
        source_url = (m.group(1).strip().strip('"').strip("'") if m else "") or ""
        tags = _parse_tags(fm)

        video_key = source_files.get("video")
        video_stem = _slug_from_video_key(video_key or "")
        plan_params = (
            resolve_video_synthesize_plan(source_url, ingest_id=video_stem)
            if source_url
            else None
        )
        url_storage_slug = plan_params[2] if plan_params else None

        extra_slugs: list[str] = []
        if plan_params and plan_params[1] == "substack_video" and source_url:
            url_slug = _substack_url_storage_slug(source_url)
            if url_slug:
                extra_slugs.append(url_slug)

        loc: tuple[str, str] | None = None
        storage_slug: str | None = None
        for candidate in _candidate_storage_slugs(
            note_path=note_path,
            video_key=video_key,
            url_storage_slug=url_storage_slug,
            extra_slugs=extra_slugs,
        ):
            loc = index.locate(candidate)
            if loc is not None:
                storage_slug = candidate
                break

        if loc is None or not storage_slug:
            plan.missing_transcript.append(
                MissingTranscriptItem(
                    note_path=note_path,
                    source_url=source_url,
                    storage_slug=url_storage_slug,
                    video_key=video_key,
                )
            )
            continue

        if not plan_params or not source_url:
            plan.missing_transcript.append(
                MissingTranscriptItem(
                    note_path=note_path,
                    source_url=source_url,
                    storage_slug=storage_slug,
                    video_key=video_key,
                )
            )
            continue

        synthesize_ct, _, _ = plan_params

        dedupe_key = (source_url, storage_slug)
        if dedupe_key in seen_redeploy:
            continue
        seen_redeploy.add(dedupe_key)

        plan.redeploy.append(
            RedeployItem(
                note_path=note_path,
                source_url=source_url,
                content_type=synthesize_ct,
                storage_slug=storage_slug,
                tags=tags,
            )
        )

        prefix, reels_rel = loc
        if prefix == "reels":
            dst = f"transcripts/{storage_slug}.json"
            if storage_slug not in index.canonical:
                plan.migrate.append(
                    MigrateItem(storage_slug=storage_slug, src=reels_rel, dst=dst)
                )

    return plan


def _mc_copy(config: Config, src_rel: str, dst_rel: str) -> None:
    bucket = config.storage_bucket
    alias = config.storage_alias
    src = f"{alias}/{bucket}/{src_rel}"
    dst = f"{alias}/{bucket}/{dst_rel}"
    result = subprocess.run(
        ["mc", "cp", src, dst],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mc cp {src_rel} -> {dst_rel} failed: {(result.stderr or result.stdout or '').strip()[:300]}"
        )


def apply(plan: RepairPlan, config: Config, conn, *, dry_run: bool = True) -> None:
    """Run migrations first, then synthesize dispatches."""
    for item in plan.migrate:
        line = f"migrate transcript {item.src} -> {item.dst}"
        if dry_run:
            print(f"[dry-run] {line}")
            continue
        _mc_copy(config, item.src, item.dst)
        print(f"[migrate] {line}")

    for item in plan.redeploy:
        action = (
            f"dispatch_synthesize url={item.source_url!r} "
            f"content_type={item.content_type} slug={item.storage_slug}"
        )
        if dry_run:
            print(f"[dry-run] {action}")
            continue

        if _target_note_already_synthesized(config, item):
            synth_path = expected_synthesize_note_path(
                config, storage_slug=item.storage_slug, content_type=item.content_type
            )
            print(
                f"[skip] target already synthesized slug={item.storage_slug} "
                f"target={synth_path} orphan={item.note_path}"
            )
            continue

        decision = classify_synthesize_dispatch(
            conn, item.storage_slug, item.content_type
        )
        if not decision.dispatch:
            print(decision.skip_message)
            continue

        if _needs_parent_reconcile_skip(conn, item.storage_slug):
            print(
                f"[skip] slug={item.storage_slug} is a parent with ingest siblings "
                "— needs manual reconciliation"
            )
            plan.skipped_parent_reconcile.append(item)
            continue

        result = dispatch_synthesize(
            item.source_url,
            config,
            conn,
            content_type=item.content_type,
            slug=item.storage_slug,
            tags=item.tags,
        )
        if result.get("status") == "error":
            raise RuntimeError(result.get("error") or "dispatch_synthesize failed")
        tid = result.get("task_id", "?")
        flag = "redispatched" if result.get("redispatched") else (
            "duplicate" if result.get("duplicate") else "created"
        )
        print(f"[dispatch] {item.storage_slug} -> task {tid} ({flag})")
        synth_path = expected_synthesize_note_path(
            config, storage_slug=item.storage_slug, content_type=item.content_type
        )
        if synth_path.resolve() != item.note_path.resolve():
            print(
                f"[note] orphan={item.note_path} "
                f"synthesize will write={synth_path}"
            )


def _print_missing(plan: RepairPlan) -> None:
    if not plan.missing_transcript:
        print("No missing-transcript candidates.")
        return
    print("Missing transcript in storage (re-ingest required):")
    for item in plan.missing_transcript:
        print(f"  - {item.source_url or '(no source_url)'}  note={item.note_path}")
        if item.storage_slug:
            print(f"      storage_slug={item.storage_slug}")
        if item.video_key:
            print(f"      video={item.video_key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy legacy transcripts and dispatch synthesize tasks (default: dry-run).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print missing-transcript report and exit (no dispatch).",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config()
        plan = discover(config)
        if args.report:
            _print_missing(plan)
            return 0

        dry_run = not args.apply
        if dry_run:
            print("Dry-run (pass --apply to mutate storage and dispatch).")
        print(
            f"Plan: redeploy={len(plan.redeploy)} migrate={len(plan.migrate)} "
            f"missing_transcript={len(plan.missing_transcript)}"
        )

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        try:
            apply(plan, config, conn, dry_run=dry_run)
        finally:
            conn.close()

        if plan.skipped_parent_reconcile:
            print(
                f"Skipped parent reconcile (manual): {len(plan.skipped_parent_reconcile)}"
            )
            for item in plan.skipped_parent_reconcile:
                print(f"  - slug={item.storage_slug} url={item.source_url!r}")

        if plan.missing_transcript:
            print()
            _print_missing(plan)
        return 0
    except Exception as exc:
        print(f"repair_orphaned_transcripts failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
