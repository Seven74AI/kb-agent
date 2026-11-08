"""Pure predicates and documented deferral lifecycle for deferred-media ingestion.

Leaf module: stdlib imports only. Connection-taking CAS helpers live in ``agent.db``.
"""

from __future__ import annotations

import enum
from typing import Any

# Substrings in task/step pause reasons that mean deferred media still blocks synthesis.
DEFERRED_MEDIA_PAUSE_MARKERS: tuple[str, ...] = (
    "awaiting media",
    "awaiting substack cookies",
    "deferred to batch synthesis",
    "batch synthesis dispatched or completed",
    "deferred synthesis paused",
    "deferred synthesis dispatch failed",
    "media gate",
    "media_deferred:",
    "unsupported embed",
)

_EMBED_STEP_TERMINAL_SUCCESS_HEADS = (
    "no embedded media",
    "media ready — deferred to batch",
)

SYNTHESIZE_IN_FLIGHT_STATUSES = "'pending', 'in_progress', 'paused'"

REQUEUE_SUPERSEDE_GUARD = (
    "AND (error IS NULL OR instr(LOWER(COALESCE(error, '')), 'superseded') = 0)"
)

SUPERSEDE_DEFERRED_PARENT_REASON = (
    "deferred synthesis dispatched; parent ingestion superseded"
)


class DeferralState(enum.Enum):
    """Logical deferral lifecycle for a parent ingestion row (documentation + classifier)."""

    IDLE = "idle"
    AWAITING_MEDIA = "awaiting_media"
    SYNTHESIZE_OWNED = "synthesize_owned"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DONE = "done"


# Single source of truth for legal deferral-state moves. ``agent.db`` CAS predicates
# must stay consistent with this table (see tests/test_deferral_fsm.py).
ALLOWED_TRANSITIONS: dict[DeferralState, tuple[DeferralState, ...]] = {
    DeferralState.IDLE: (
        DeferralState.AWAITING_MEDIA,
        DeferralState.SYNTHESIZE_OWNED,
        DeferralState.FAILED,
        DeferralState.DONE,
        DeferralState.SUPERSEDED,
    ),
    DeferralState.AWAITING_MEDIA: (
        DeferralState.SYNTHESIZE_OWNED,
        DeferralState.FAILED,
        DeferralState.SUPERSEDED,
        DeferralState.DONE,
        DeferralState.IDLE,
    ),
    DeferralState.SYNTHESIZE_OWNED: (
        DeferralState.DONE,
        DeferralState.SUPERSEDED,
        DeferralState.FAILED,
        DeferralState.AWAITING_MEDIA,
        DeferralState.IDLE,
    ),
    # FAILED recovers to IDLE via retry/requeue (requeue_task_for_retry_cas, retry_task).
    DeferralState.FAILED: (DeferralState.IDLE,),
    DeferralState.SUPERSEDED: (),
    DeferralState.DONE: (),
}


def is_deferred_media_pause(reason: str | None) -> bool:
    """True when a pause reason signals the parent must not pass ``synthesize_note``."""
    if not reason:
        return False
    low = reason.lower()
    return any(marker in low for marker in DEFERRED_MEDIA_PAUSE_MARKERS)


def embed_step_output_terminal_success(output: str | None) -> bool:
    """True when ``substack_embedded_media`` finished with a non-pause terminal message."""
    if not output:
        return False
    head = output.split("\n", 1)[0].strip()
    if head == _EMBED_STEP_TERMINAL_SUCCESS_HEADS[0]:
        return True
    return head.startswith(_EMBED_STEP_TERMINAL_SUCCESS_HEADS[1])


def is_synthesize_content_type(content_type: str) -> bool:
    """True when *content_type* is a deferred synthesize-only pipeline variant."""
    return (content_type or "").endswith("_synthesize")


def is_superseded_parent_error(error: str | None) -> bool:
    return bool(error and "superseded" in error.lower())


def deferral_cycle_cmp(deferral: dict[str, Any]) -> str:
    """SQL comparison operator for deferral-generation freshness.

    Invariant: a task created at or after ``deferral.updated_at`` is never stale.
    SQLite ``datetime('now')`` is second-resolution, so strict ``>`` would mark
    same-second rows stale when ``media_changed`` is set.
    """
    del deferral
    return ">="


def classify_deferral_state(
    *,
    status: str,
    error: str | None,
    in_deferred_cycle: bool,
    has_synthesize_owner: bool,
) -> DeferralState:
    """Name the current deferral lifecycle state from task signals (no database I/O)."""
    if is_superseded_parent_error(error):
        return DeferralState.SUPERSEDED
    normalized = (status or "").strip().lower()
    if normalized in ("done", "completed"):
        return DeferralState.DONE
    if normalized == "failed":
        return DeferralState.FAILED
    if has_synthesize_owner:
        return DeferralState.SYNTHESIZE_OWNED
    if in_deferred_cycle:
        return DeferralState.AWAITING_MEDIA
    return DeferralState.IDLE
