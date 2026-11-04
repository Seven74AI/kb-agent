"""Enqueue URLs for pipeline processing."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.detect import detect_platform
from core.pipelines import resolve_pipeline_steps
from core.types import SubmitOutcome, SubmitResult
from core.util import slugify

DedupCheck = Callable[[str, Path], bool]
PreflightHealth = Callable[[str], tuple[bool, str]]
DetectContentSubtype = Callable[[str, str], str]
CreateTask = Callable[
    [sqlite3.Connection, str, str, str, list[str], str],
    str,
]


@dataclass(frozen=True)
class SubmitDependencies:
    """Injectable side-effect boundaries for submit_url (keeps core testable)."""

    dedup_check: DedupCheck
    preflight_health: PreflightHealth
    detect_content_subtype: DetectContentSubtype
    create_task: CreateTask


def submit_url(
    conn: sqlite3.Connection,
    url: str,
    *,
    vault_path: Path,
    deps: SubmitDependencies,
) -> SubmitResult:
    """Enqueue a URL using the same decision order as POST /submit.

    1. Detect platform
    2. Vault dedup
    3. Pre-flight health (may create paused task)
    4. Detect content subtype + resolve pipeline steps
    5. Create pending task

    Args:
        conn: Open agent.db connection.
        url: Source URL (non-empty).
        vault_path: Obsidian vault root for dedup grep.
        deps: Side-effect functions (today wired from agent.detection + agent.db).

    Returns:
        SubmitResult describing skip, pause, or enqueue outcome.
    """
    url = url.strip()
    if not url:
        raise ValueError("url required")

    platform = detect_platform(url)

    if deps.dedup_check(url, vault_path):
        return SubmitResult(
            outcome=SubmitOutcome.SKIPPED,
            platform=platform,
            reason="Already in vault",
        )

    ok, reason = deps.preflight_health(platform)
    slug = slugify(url)
    subtype = deps.detect_content_subtype(url, platform)
    steps = resolve_pipeline_steps(subtype, platform)

    if not ok:
        task_id = deps.create_task(conn, url, platform, slug, steps, "paused")
        conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (reason, task_id))
        conn.commit()
        return SubmitResult(
            outcome=SubmitOutcome.PAUSED,
            platform=platform,
            task_id=task_id,
            content_type=subtype,
            slug=slug,
            reason=reason,
        )

    task_id = deps.create_task(conn, url, platform, slug, steps, "pending")
    return SubmitResult(
        outcome=SubmitOutcome.CREATED,
        platform=platform,
        task_id=task_id,
        content_type=subtype,
        slug=slug,
    )
