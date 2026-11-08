"""Table-driven tests for agent.deferral_fsm and deferral CAS parity in agent.db."""

from __future__ import annotations

import pytest

from agent.core.resume import retry_task
from agent.db import (
    AGENT_SCHEMA,
    complete_task,
    create_task,
    ensure_parent_media_deferral,
    init_db,
    may_reset_parent_task,
    requeue_task_for_retry_cas,
    resume_tasks_by_reason,
    supersede_deferred_parent_task,
    task_deferred_retry_preserves_embed_step,
    task_in_deferred_media_cycle,
)
from agent.deferral_fsm import (
    ALLOWED_TRANSITIONS,
    SUPERSEDE_DEFERRED_PARENT_REASON,
    DeferralState,
    classify_deferral_state,
    deferral_cycle_cmp,
    embed_step_output_terminal_success,
    is_deferred_media_pause,
    is_superseded_parent_error,
    is_synthesize_content_type,
)


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("awaiting media: 2 images + 1 videos", True),
        ("media gate — not ready", True),
        ("media_deferred:my-slug", True),
        ("firecrawl rate limit — retry after cooldown", False),
        ("", False),
        (None, False),
    ],
)
def test_is_deferred_media_pause_table(reason, expected):
    assert is_deferred_media_pause(reason) is expected


@pytest.mark.parametrize(
    "output,terminal_success",
    [
        ("no embedded media", True),
        ("media ready — deferred to batch\nextra", True),
        ("awaiting media: 1 images + 0 videos", False),
        (None, False),
        ("", False),
    ],
)
def test_embed_step_terminal_success_table(output, terminal_success):
    assert embed_step_output_terminal_success(output) is terminal_success


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("substack_text_synthesize", True),
        ("substack_text", False),
        ("", False),
    ],
)
def test_is_synthesize_content_type_table(content_type, expected):
    assert is_synthesize_content_type(content_type) is expected


@pytest.mark.parametrize(
    "error,expected",
    [
        ("superseded: stale owner", True),
        (SUPERSEDE_DEFERRED_PARENT_REASON, True),
        (None, False),
        ("transient", False),
    ],
)
def test_is_superseded_parent_error_table(error, expected):
    assert is_superseded_parent_error(error) is expected


def test_deferral_cycle_cmp_is_stable():
    assert deferral_cycle_cmp({"updated_at": "2020-01-01"}) == ">="


@pytest.mark.parametrize(
    "status,error,in_cycle,has_owner,expected",
    [
        ("paused", "superseded: x", False, False, DeferralState.SUPERSEDED),
        ("completed", SUPERSEDE_DEFERRED_PARENT_REASON, False, False, DeferralState.SUPERSEDED),
        ("done", None, False, False, DeferralState.DONE),
        ("completed", None, False, False, DeferralState.DONE),
        ("failed", "network", False, False, DeferralState.FAILED),
        ("in_progress", None, False, True, DeferralState.SYNTHESIZE_OWNED),
        ("paused", "awaiting media", True, False, DeferralState.AWAITING_MEDIA),
        ("pending", None, False, False, DeferralState.IDLE),
    ],
)
def test_classify_deferral_state_table(status, error, in_cycle, has_owner, expected):
    assert (
        classify_deferral_state(
            status=status,
            error=error,
            in_deferred_cycle=in_cycle,
            has_synthesize_owner=has_owner,
        )
        is expected
    )


# Documented deferral-state edges that db CAS paths must respect.
_CAS_DEFERRAL_EDGES: tuple[tuple[DeferralState, DeferralState], ...] = (
    (DeferralState.IDLE, DeferralState.AWAITING_MEDIA),
    (DeferralState.IDLE, DeferralState.FAILED),
    (DeferralState.IDLE, DeferralState.DONE),
    (DeferralState.IDLE, DeferralState.SUPERSEDED),
    (DeferralState.AWAITING_MEDIA, DeferralState.SYNTHESIZE_OWNED),
    (DeferralState.AWAITING_MEDIA, DeferralState.FAILED),
    (DeferralState.AWAITING_MEDIA, DeferralState.SUPERSEDED),
    (DeferralState.AWAITING_MEDIA, DeferralState.DONE),
    (DeferralState.AWAITING_MEDIA, DeferralState.IDLE),
    (DeferralState.SYNTHESIZE_OWNED, DeferralState.DONE),
    (DeferralState.SYNTHESIZE_OWNED, DeferralState.SUPERSEDED),
)


@pytest.mark.parametrize("src,dst", _CAS_DEFERRAL_EDGES)
def test_allowed_transitions_covers_cas_edges(src, dst):
    assert dst in ALLOWED_TRANSITIONS[src]


@pytest.mark.parametrize(
    "terminal",
    [DeferralState.FAILED, DeferralState.SUPERSEDED, DeferralState.DONE],
)
def test_terminal_deferral_states_have_no_outbound_edges(terminal):
    assert ALLOWED_TRANSITIONS[terminal] == ()


def test_requeue_task_for_retry_cas_refuses_superseded(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "web", "cas_slug", ["only"], status="failed",
    )
    conn.execute(
        "UPDATE tasks SET error = 'superseded: stale owner' WHERE id = ?", (task_id,),
    )
    conn.commit()
    with conn:
        assert requeue_task_for_retry_cas(conn, task_id) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == "failed"


def test_complete_task_cas_refuses_superseded_terminal(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "sup_cas"
    url = "https://blog.substack.com/p/cas"
    parent_id = create_task(
        conn, url, "substack_text", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.commit()
    assert supersede_deferred_parent_task(conn, slug) == 1
    assert complete_task(conn, parent_id) is False


@pytest.mark.parametrize(
    "status,should_move",
    [
        ("paused", True),
        ("pending", True),
        ("in_progress", True),
        ("completed", False),
        ("failed", False),
    ],
)
def test_supersede_deferred_parent_task_cas_from_statuses(tmp_path, status, should_move):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = f"sup_from_{status}"
    parent_id = create_task(
        conn,
        "https://blog.substack.com/p/x",
        "substack_text",
        slug,
        ["synthesize_note"],
        status=status,
    )
    conn.commit()
    count = supersede_deferred_parent_task(conn, slug)
    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?", (parent_id,),
    ).fetchone()
    if should_move:
        assert count == 1
        assert row[0] == "completed"
        assert row[1] == SUPERSEDE_DEFERRED_PARENT_REASON
    else:
        assert count == 0


def test_retry_terminal_success_embed_rewinds_preserves_false(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "terminal_embed_retry"
    url = "https://blog.substack.com/p/terminal-embed"
    ensure_parent_media_deferral(conn, slug, url, "substack_text", 1)
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    embed_id = conn.execute(
        "SELECT id FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE tasks SET error = ? WHERE id = ?",
        ("batch synthesis dispatched or completed — parent must not re-enter synthesis", task_id),
    )
    conn.execute(
        """
        UPDATE task_steps SET status = 'done', output = ?
        WHERE task_id = ? AND name IN ('substack_cleanup', 'substack_embedded_media')
        """,
        (f"cleaned\nmedia_deferred:{slug}", task_id),
    )
    conn.execute(
        "UPDATE task_steps SET output = ? WHERE id = ?",
        ("media ready — deferred to batch", embed_id),
    )
    conn.commit()
    assert task_deferred_retry_preserves_embed_step(conn, task_id, slug) is False
    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status FROM task_steps WHERE id = ?", (embed_id,),
    ).fetchone()
    assert embed[0] == "pending"


def test_retry_paused_embed_preserves_embed_step(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "preserve_embed"
    url = "https://blog.substack.com/p/preserve"
    task_id = create_task(
        conn,
        url,
        "substack_text",
        slug,
        ["substack_embedded_media", "synthesize_note"],
        status="paused",
    )
    conn.execute(
        "UPDATE task_steps SET status = 'paused', output = ? WHERE task_id = ? AND name = 'substack_embedded_media'",
        ("awaiting media: 1 images + 0 videos", task_id),
    )
    conn.commit()
    assert task_deferred_retry_preserves_embed_step(conn, task_id, slug) is True
    assert retry_task(conn, task_id) is True
    embed = conn.execute(
        "SELECT status, output FROM task_steps WHERE task_id = ? AND name = 'substack_embedded_media'",
        (task_id,),
    ).fetchone()
    assert embed[0] == "paused"
    assert embed[1] is not None


def test_resume_skips_synthesize_owner(tmp_path):
    from agent.db import has_active_synthesize_task

    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    slug = "resume_syn_block"
    pause_reason = "firecrawl rate limit — retry after cooldown"
    parent_id = create_task(
        conn, "https://blog.substack.com/p/p", "substack_text", slug,
        ["synthesize_note"], status="paused",
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (pause_reason, parent_id))
    syn_id = create_task(
        conn, "https://batch", "substack_text_synthesize", slug,
        ["synthesize_note"], status="in_progress",
    )
    conn.commit()
    assert has_active_synthesize_task(conn, slug)
    assert may_reset_parent_task(conn, parent_id, slug, error=pause_reason) is False
    assert resume_tasks_by_reason(conn, "%firecrawl%") == 0
    assert retry_task(conn, parent_id) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (parent_id,)).fetchone()[0] == "paused"
    assert syn_id


def test_retry_task_refuses_superseded_before_cas(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "web", "race_slug", ["s0"], status="failed",
    )
    conn.execute("UPDATE tasks SET error = 'superseded: orphan' WHERE id = ?", (task_id,))
    conn.commit()
    assert retry_task(conn, task_id) is False


def test_resume_tasks_by_reason_cas_skips_superseded(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    reason = "cooldown — retry later"
    task_id = create_task(
        conn, "https://example.com", "web", "resume_race", ["only"], status="paused",
    )
    conn.execute("UPDATE tasks SET error = ? WHERE id = ?", (reason, task_id))
    conn.commit()
    conn.execute(
        "UPDATE tasks SET error = 'superseded: bulk resume race' WHERE id = ?", (task_id,),
    )
    conn.commit()
    assert resume_tasks_by_reason(conn, f"%{reason[:8]}%") == 0


def test_requeue_on_done_status_fails(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "web", "done_slug", ["only"], status="done",
    )
    conn.commit()
    with conn:
        assert requeue_task_for_retry_cas(conn, task_id) is False


def test_task_in_deferred_media_cycle_honors_error_marker(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    task_id = create_task(
        conn, "https://example.com", "substack_text", "slug", ["embed"], status="paused",
    )
    assert task_in_deferred_media_cycle(
        conn, task_id, slug="slug", error="media gate — blocked",
    )


def test_import_leaf_has_no_db_dependency():
    import importlib
    import sys

    mods = {k: v for k, v in sys.modules.items() if k.startswith("agent.")}
    for name in list(mods):
        if name != "agent.deferral_fsm":
            del sys.modules[name]
    importlib.import_module("agent.deferral_fsm")
    assert "agent.db" not in sys.modules
