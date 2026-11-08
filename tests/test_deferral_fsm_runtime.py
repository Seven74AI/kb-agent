"""Runtime parity: parent deferral CAS paths in agent.db vs deferral_fsm ALLOWED_TRANSITIONS.

Uses the same in-memory sqlite fixtures as tests/test_db_queue_migrate.py and
tests/test_deferral_fsm.py (init_db + AGENT_SCHEMA + create_task).
"""

from __future__ import annotations

import sqlite3
import pytest

from agent.core.resume import retry_task
from agent.db import (
    AGENT_SCHEMA,
    _clear_parent_media_changed_if_satisfied,
    _mark_parent_media_enqueue_shortfall,
    _reconcile_parent_media_deferral_on_new_sibling,
    complete_task,
    create_task,
    ensure_parent_media_deferral,
    fail_task,
    get_parent_media_deferral,
    has_active_synthesize_task,
    init_db,
    pause_task,
    queue_ingest,
    requeue_task_for_retry_cas,
    reset_task_steps_for_retry,
    resume_tasks_by_reason,
    supersede_deferred_parent_task,
    supersede_stale_deferral_synthesize_tasks,
    task_in_deferred_media_cycle,
)
from agent.deferral_fsm import (
    ALLOWED_TRANSITIONS,
    DeferralState,
    classify_deferral_state,
    is_synthesize_content_type,
)

# Documented CAS ops exercised below (parent ingestion row unless noted).
RUNTIME_CAS_OPS = (
    "supersede_deferred_parent_task",
    "complete_task",
    "requeue_task_for_retry_cas",
    "fail_task",
    "pause_task",
    "reset_task_steps_for_retry",
    "retry_task",
    "resume_tasks_by_reason",
    "supersede_stale_deferral_synthesize_tasks",
    "_mark_parent_media_enqueue_shortfall",
    "_reconcile_parent_media_deferral_on_new_sibling",
    "_clear_parent_media_changed_if_satisfied",
)

def _open_conn(tmp_path) -> sqlite3.Connection:
    db_path = tmp_path / "agent.db"
    return init_db(db_path, AGENT_SCHEMA)


def _classify_parent_row(conn: sqlite3.Connection, task_id: str) -> DeferralState:
    row = conn.execute(
        "SELECT status, error, slug, content_type FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row is not None
    status, error, slug, content_type = row
    assert not is_synthesize_content_type(content_type or "")
    in_cycle = task_in_deferred_media_cycle(
        conn, task_id, slug=slug, error=error,
    )
    has_owner = bool(slug and has_active_synthesize_task(conn, slug))
    return classify_deferral_state(
        status=status,
        error=error,
        in_deferred_cycle=in_cycle,
        has_synthesize_owner=has_owner,
    )


def _assert_fsm_edge(
    before: DeferralState,
    after: DeferralState,
    *,
    op: str,
) -> None:
    if before == after:
        return
    if after in ALLOWED_TRANSITIONS[before]:
        return
    pytest.fail(
        f"{op}: illegal deferral transition {before.value} → {after.value} "
        f"(not in ALLOWED_TRANSITIONS[{before.name}])",
    )


def _parent_fixture(
    conn: sqlite3.Connection,
    slug: str,
    *,
    status: str = "paused",
    error: str | None = "awaiting media: 1 images + 0 videos",
    steps: list[str] | None = None,
    with_deferral: bool = True,
) -> str:
    url = f"https://blog.substack.com/p/{slug}"
    steps = steps or ["substack_cleanup", "substack_embedded_media", "synthesize_note"]
    task_id = create_task(conn, url, "substack_text", slug, steps, status=status)
    if error is not None:
        conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (error, task_id))
    if with_deferral:
        ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    if status == "paused" and "substack_embedded_media" in steps:
        conn.execute(
            """
            UPDATE task_steps SET status = 'paused', output = ?
            WHERE task_id = ? AND name = 'substack_embedded_media'
            """,
            ("awaiting media: 1 images + 0 videos", task_id),
        )
    conn.commit()
    return task_id


@pytest.mark.parametrize("status", ["paused", "pending", "in_progress"])
def test_supersede_deferred_parent_task_fsm(tmp_path, status):
    conn = _open_conn(tmp_path)
    slug = f"rt_sup_{status}"
    parent_id = _parent_fixture(conn, slug, status=status)
    before = _classify_parent_row(conn, parent_id)
    conn.commit()
    changed = supersede_deferred_parent_task(conn, slug)
    assert changed == 1
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="supersede_deferred_parent_task")


def test_complete_task_parent_done_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_complete"
    url = "https://blog.substack.com/p/rt-complete"
    parent_id = create_task(
        conn, url, "substack_text", slug, ["synthesize_note"], status="in_progress",
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert complete_task(conn, parent_id) is True
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="complete_task")


def test_requeue_task_for_retry_cas_paused_deferral_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_requeue_pause"
    parent_id = _parent_fixture(conn, slug, status="paused")
    before = _classify_parent_row(conn, parent_id)
    assert requeue_task_for_retry_cas(conn, parent_id, allowed_statuses=("paused",))
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="requeue_task_for_retry_cas")


def test_requeue_task_for_retry_cas_failed_parent_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_requeue_fail"
    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/fail",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="failed",
    )
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("transient network", parent_id),
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert requeue_task_for_retry_cas(conn, parent_id)
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="requeue_task_for_retry_cas")


def test_fail_task_parent_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_fail"
    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/rt-fail",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="in_progress",
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    fail_task(conn, parent_id, "pipeline error")
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="fail_task")


def test_pause_task_deferral_marker_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_pause"
    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/rt-pause",
        "substack_text",
        slug,
        ["substack_embedded_media"],
        status="in_progress",
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    pause_task(conn, parent_id, "awaiting media: 2 images + 0 videos")
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="pause_task")


def test_reset_task_steps_for_retry_preserves_embed_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_reset_steps"
    parent_id = _parent_fixture(conn, slug, status="paused")
    before = _classify_parent_row(conn, parent_id)
    conn.commit()
    requeue_task_for_retry_cas(conn, parent_id, allowed_statuses=("paused",))
    row = conn.execute(
        "SELECT slug, error FROM tasks WHERE id = ?", (parent_id,),
    ).fetchone()
    reset_task_steps_for_retry(
        conn, parent_id, slug=row[0], error=row[1],
    )
    conn.commit()
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="reset_task_steps_for_retry")


def test_retry_task_deferred_parent_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_retry_def"
    parent_id = _parent_fixture(conn, slug, status="paused")
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert retry_task(conn, parent_id) is True
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="retry_task")


def test_retry_task_failed_parent_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_retry_failed"
    parent_id = create_task(
        conn,
        "https://example.com",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="failed",
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert retry_task(conn, parent_id) is True
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="retry_task")


def test_resume_tasks_by_reason_non_deferral_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_resume"
    reason = "firecrawl rate limit — retry after cooldown"
    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/resume",
        "substack_text",
        slug,
        ["synthesize_note"],
        status="paused",
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (reason, parent_id))
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    n = resume_tasks_by_reason(conn, "%firecrawl%")
    assert n == 1
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="resume_tasks_by_reason")


def test_supersede_stale_deferral_synthesize_tasks_stale_child_only(tmp_path):
    """Fails prior-generation *_synthesize rows; parent classify unchanged when no active owner."""
    conn = _open_conn(tmp_path)
    slug = "rt_sup_stale"
    url = "https://blog.substack.com/p/stale"
    parent_id = _parent_fixture(conn, slug, status="paused", with_deferral=True)
    deferral = get_parent_media_deferral(conn, slug)
    assert deferral is not None
    syn_id = create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (syn_id,),
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert not has_active_synthesize_task(conn, slug)
    assert supersede_stale_deferral_synthesize_tasks(conn, slug, deferral) == 1
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="supersede_stale_deferral_synthesize_tasks")


def test_supersede_stale_deferral_synthesize_tasks_active_owner_unchanged(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_sup_active"
    url = "https://blog.substack.com/p/stale-active"
    parent_id = _parent_fixture(conn, slug, status="paused", with_deferral=True)
    deferral = get_parent_media_deferral(conn, slug)
    assert deferral is not None
    create_task(
        conn, url, "substack_text_synthesize", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE slug = ? AND content_type GLOB '*_synthesize'",
        (deferral["updated_at"], slug),
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    assert before is DeferralState.SYNTHESIZE_OWNED
    assert supersede_stale_deferral_synthesize_tasks(conn, slug, deferral) == 0
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="supersede_stale_deferral_synthesize_tasks")


def test_mark_parent_media_enqueue_shortfall_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_shortfall"
    url = "https://blog.substack.com/p/shortfall"
    parent_id = _parent_fixture(conn, slug, with_deferral=True)
    deferral = get_parent_media_deferral(conn, slug)
    assert deferral is not None
    conn.execute(
        "UPDATE parent_media_deferral SET media_changed = 0 WHERE parent_slug = ?",
        (slug,),
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    _mark_parent_media_enqueue_shortfall(
        conn, slug, sibling_n=0, expected_total=int(deferral["expected_total"]) + 1,
    )
    conn.commit()
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="_mark_parent_media_enqueue_shortfall")


def test_reconcile_parent_media_deferral_on_new_sibling_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_reconcile"
    parent_id = _parent_fixture(conn, slug, with_deferral=True)
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    queue_ingest(
        conn,
        "https://substack-post-media.s3.amazonaws.com/public/images/x.png",
        "substack_image",
        "img_x",
        parent_slug=slug,
        media_kind="image",
    )
    _reconcile_parent_media_deferral_on_new_sibling(conn, slug)
    conn.commit()
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="_reconcile_parent_media_deferral_on_new_sibling")


def test_clear_parent_media_changed_if_satisfied_fsm(tmp_path):
    conn = _open_conn(tmp_path)
    slug = "rt_clear_mc"
    url = "https://blog.substack.com/p/clear"
    parent_id = _parent_fixture(conn, slug, with_deferral=True)
    queue_ingest(
        conn,
        "https://substack-post-media.s3.amazonaws.com/public/images/y.png",
        "substack_image",
        "img_y",
        parent_slug=slug,
        media_kind="image",
    )
    conn.execute(
        "UPDATE parent_media_deferral SET expected_total = 1, media_changed = 1 WHERE parent_slug = ?",
        (slug,),
    )
    conn.execute(
        "UPDATE ingest_queue SET status = 'dispatched' WHERE parent_slug = ?",
        (slug,),
    )
    conn.commit()
    before = _classify_parent_row(conn, parent_id)
    _clear_parent_media_changed_if_satisfied(conn, slug)
    conn.commit()
    after = _classify_parent_row(conn, parent_id)
    _assert_fsm_edge(before, after, op="_clear_parent_media_changed_if_satisfied")


def test_runtime_cas_op_inventory_matches_tests():
    """Guardrail: new db CAS helpers should get a runtime FSM test or update RUNTIME_CAS_OPS."""
    assert len(RUNTIME_CAS_OPS) >= 12
