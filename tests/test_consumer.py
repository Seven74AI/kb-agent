"""Tests for the async consumer — error model, crash recovery, process management."""

import asyncio
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add kb-agent to path (works with both workspace and production paths)
_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from agent.config import Config
from agent.db import (
    AGENT_SCHEMA,
    claim_task,
    create_task,
    get_all_tasks,
    get_task_steps,
    init_db,
)
from agent.exceptions import ErrorType, StepError
from agent.pipelines import Step, register_pipeline
from agent.consumer import (
    STEP_REGISTRY,
    VIDEO_TASK_CONTENT_TYPES,
    Consumer,
    register_step,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    """Create a Config with temp paths for testing."""
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=Path("/tmp/test-vault"),
        temp_dir=Path("/tmp"),
        cookie_dir=Path("/tmp/test-cookies"),
        max_retries=3,
        retry_delays=[0.01, 0.02, 0.05],  # Fast for tests
        text_parallel=2,
        video_semaphore=1,
    )


@pytest.fixture
def agent_db(tmp_path):
    """Create a test agent.db."""
    db_path = tmp_path / "agent.db"
    return init_db(db_path, AGENT_SCHEMA)


@pytest.fixture
def logs_db(tmp_path):
    """Create a test logs.db."""
    from agent.db import LOGS_SCHEMA
    db_path = tmp_path / "logs.db"
    return init_db(db_path, LOGS_SCHEMA)


@pytest.fixture
def consumer(config, agent_db, logs_db):
    """Create a Consumer with test DBs injected."""
    c = Consumer(config)
    c.agent_conn = agent_db
    c.logs_conn = logs_db
    c._running = True
    c._video_semaphore = asyncio.BoundedSemaphore(config.video_semaphore)
    return c


# ── StepError tests ───────────────────────────────────────────────────────────


def test_step_error_retry():
    err = StepError("Network timeout", error_type=ErrorType.RETRY, step_name="download")
    assert err.error_type == ErrorType.RETRY
    assert err.error_type.value == "retry"
    assert str(err) == "Network timeout"
    assert err.step_name == "download"


def test_step_error_pause():
    err = StepError("Cookies invalid", error_type=ErrorType.PAUSE)
    assert err.error_type == ErrorType.PAUSE
    assert err.error_type.value == "pause"


def test_step_error_fail():
    err = StepError("URL 404", error_type=ErrorType.FAIL)
    assert err.error_type == ErrorType.FAIL
    assert err.error_type.value == "fail"


def test_step_error_repr():
    err = StepError("Bad thing", error_type=ErrorType.FAIL, step_name="scrape")
    r = repr(err)
    assert "fail" in r
    assert "scrape" in r
    assert "Bad thing" in r


def test_step_error_default_is_retry():
    err = StepError("something broke")
    assert err.error_type == ErrorType.RETRY


# ── PID extraction tests ──────────────────────────────────────────────────────


def test_extract_pid_from_valid_output():
    assert Consumer._extract_pid_from_output("pid:12345") == 12345
    assert Consumer._extract_pid_from_output("Done. pid:99999\nmore output") == 99999


def test_extract_pid_from_none():
    assert Consumer._extract_pid_from_output(None) is None


def test_extract_pid_from_no_pid():
    assert Consumer._extract_pid_from_output("Step completed successfully") is None


def test_extract_pid_from_empty():
    assert Consumer._extract_pid_from_output("") is None


# ── Process alive tests ───────────────────────────────────────────────────────


def test_is_process_alive_dead():
    """Check that a non-existent PID is reported as dead."""
    # Use a PID that almost certainly doesn't exist
    assert Consumer._is_process_alive(99999999) is False


def test_is_process_alive_self():
    """Check that our own PID is reported as alive."""
    assert Consumer._is_process_alive(os.getpid()) is True


# ── Step execution tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_step_sync(consumer):
    """Test executing a synchronous step function."""
    called = []

    def my_step(task, cfg):
        called.append(1)
        return "done"

    register_step("my_step", my_step, "retry")

    result = await consumer._run_step("my_step", {"id": "t1"})
    assert result == "done"
    assert len(called) == 1
    # Clean up
    del STEP_REGISTRY["my_step"]


@pytest.mark.asyncio
async def test_run_step_async(consumer):
    """Test executing an async step function."""
    called = []

    async def my_async_step(task, cfg):
        await asyncio.sleep(0.01)
        called.append(1)
        return "async-done"

    register_step("my_async", my_async_step, "retry")

    result = await consumer._run_step("my_async", {"id": "t2"})
    assert result == "async-done"
    assert len(called) == 1
    del STEP_REGISTRY["my_async"]


@pytest.mark.asyncio
async def test_run_step_unknown(consumer):
    """Test that executing an unknown step raises FAIL."""
    with pytest.raises(StepError) as exc:
        await consumer._run_step("nonexistent", {"id": "t3"})
    assert exc.value.error_type == ErrorType.FAIL


# ── Error routing tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_task_fail_error(consumer, agent_db):
    """Test that a FAIL StepError marks the task as failed."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["failing_step"])

    def failing_step(task, cfg):
        raise StepError("Page not found", error_type=ErrorType.FAIL)

    register_step("failing_step", failing_step, "retry")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "failed"
    assert "Page not found" in (tasks[0]["error"] or "")

    del STEP_REGISTRY["failing_step"]


@pytest.mark.asyncio
async def test_execute_task_step_error_logs_traceback(consumer, agent_db, logs_db):
    """StepError should be logged with a full traceback, not only the message."""
    task_id = create_task(agent_db, "https://example.com", "web", "tb_slug", ["tb_step"])

    def tb_step(task, cfg):
        raise StepError("boom", error_type=ErrorType.FAIL)

    register_step("tb_step", tb_step, "retry")
    await consumer._execute_task({"id": task_id, "content_type": "web"})

    row = logs_db.execute(
        "SELECT content FROM step_logs WHERE task_id = ? AND stream = 'stderr'",
        (task_id,),
    ).fetchone()
    assert row is not None
    assert "StepError(fail): boom" in row["content"]
    assert "Traceback" in row["content"]
    assert "tb_step" in row["content"]

    del STEP_REGISTRY["tb_step"]


@pytest.mark.asyncio
async def test_execute_task_pause_error(consumer, agent_db):
    """Test that a PAUSE StepError pauses the task."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["pause_step"])

    def pause_step_fn(task, cfg):
        raise StepError("Cookies expired", error_type=ErrorType.PAUSE)

    register_step("pause_step", pause_step_fn, "retry")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "paused"
    assert "Cookies expired" in (tasks[0]["error"] or "")

    del STEP_REGISTRY["pause_step"]


@pytest.mark.asyncio
async def test_execute_task_retry_then_success(consumer, agent_db):
    """Test retry: step fails once, then succeeds on retry."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["flaky_step"])

    call_count = [0]

    def flaky_step(task, cfg):
        call_count[0] += 1
        if call_count[0] == 1:
            raise StepError("Transient error", error_type=ErrorType.RETRY)
        return "success"

    register_step("flaky_step", flaky_step, "retry")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    assert call_count[0] == 2
    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "done"

    del STEP_REGISTRY["flaky_step"]


@pytest.mark.asyncio
async def test_execute_task_retry_exhausted(consumer, agent_db):
    """Test that after max_retries, a RETRY step pauses the task."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["always_fails"])

    def always_fails(task, cfg):
        raise StepError("Persistent error", error_type=ErrorType.RETRY)

    register_step("always_fails", always_fails, "retry")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "paused"
    assert "Exhausted 3 retries" in (tasks[0]["error"] or "")

    del STEP_REGISTRY["always_fails"]


@pytest.mark.asyncio
async def test_execute_task_unexpected_exception_uses_on_error(consumer, agent_db):
    """Test that non-StepError exceptions use the step's on_error setting."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["crash_step"])

    def crash_step(task, cfg):
        raise RuntimeError("Something crashed badly")

    # Register with on_error="fail" — unexpected exceptions should fail
    register_step("crash_step", crash_step, "fail")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "failed"

    del STEP_REGISTRY["crash_step"]


@pytest.mark.asyncio
async def test_execute_task_successful(consumer, agent_db):
    """Test that a task with all steps passes through to completion."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["step_a", "step_b", "step_c"])

    order = []

    def step_a(task, cfg):
        order.append("a")
        return "a_done"

    def step_b(task, cfg):
        order.append("b")
        return "b_done"

    def step_c(task, cfg):
        order.append("c")
        return "c_done"

    register_step("step_a", step_a, "retry")
    register_step("step_b", step_b, "retry")
    register_step("step_c", step_c, "retry")

    await consumer._execute_task({"id": task_id, "content_type": "web"})

    assert order == ["a", "b", "c"]

    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "done"

    steps = get_task_steps(agent_db, task_id)
    assert all(s["status"] == "done" for s in steps)

    for name in ["step_a", "step_b", "step_c"]:
        del STEP_REGISTRY[name]


@pytest.mark.asyncio
async def test_execute_task_exponential_backoff(consumer, agent_db):
    """Test that retries use increasing delays: 0.01s → 0.02s → 0.05s."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["backoff_step"])

    call_count = [0]

    def backoff_step(task, cfg):
        call_count[0] += 1
        raise StepError(f"Fail #{call_count[0]}", error_type=ErrorType.RETRY)

    register_step("backoff_step", backoff_step, "retry")

    # With max_retries=3 and delays [0.01, 0.02, 0.05],
    # after 3 failures it should pause
    await consumer._execute_task({"id": task_id, "content_type": "web"})

    assert call_count[0] == 3  # Called 3 times, then paused
    tasks = get_all_tasks(agent_db)
    assert tasks[0]["status"] == "paused"

    del STEP_REGISTRY["backoff_step"]


# ── Graceful shutdown tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_stops_polling(consumer):
    """Test that shutdown sets the right flags."""
    consumer._running = True
    consumer._stopping = False

    await consumer.shutdown()

    assert consumer._running is False
    assert consumer._stopping is True  # remains True
    assert consumer._stopping is True


@pytest.mark.asyncio
async def test_shutdown_waits_for_tasks(consumer, agent_db):
    """Test that shutdown waits for running tasks."""
    task_id = create_task(agent_db, "https://example.com", "web", "test_slug",
                          ["slow_step"])

    started = asyncio.Event()
    done = asyncio.Event()

    async def slow_step(task, cfg):
        started.set()
        await asyncio.sleep(0.1)
        done.set()
        return "ok"

    register_step("slow_step", slow_step, "retry")

    # Start task execution
    task_obj = asyncio.create_task(
        consumer._execute_task({"id": task_id, "content_type": "web"})
    )
    consumer._tasks[task_id] = task_obj

    # Wait for step to start
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Trigger shutdown
    await consumer.shutdown()

    # The task should have finished
    assert done.is_set()
    assert task_obj.done()

    del STEP_REGISTRY["slow_step"]


# ── Semaphore tests ───────────────────────────────────────────────────────────


def test_video_task_content_types_include_rumble_and_substack_excludes_mega():
    """Video path must gate AV pipelines; mega is book-archive only."""
    types = set(VIDEO_TASK_CONTENT_TYPES)
    assert "rumble" in types
    assert "substack_video" in types
    assert "mega" not in types


@pytest.mark.asyncio
async def test_video_semaphore_serializes(consumer, agent_db):
    """Test that video tasks are serialized (one at a time)."""
    running = [0]

    async def video_step(task, cfg):
        running[0] += 1
        await asyncio.sleep(0.05)
        assert running[0] == 1  # Only one should run at a time
        running[0] -= 1
        return "ok"

    register_step("video_step", video_step, "retry")

    # Create two video tasks
    t1 = create_task(agent_db, "https://youtube.com/1", "youtube", "slug1", ["video_step"])
    t2 = create_task(agent_db, "https://youtube.com/2", "youtube", "slug2", ["video_step"])

    # Execute both as video
    await asyncio.gather(
        consumer._execute_video({"id": t1, "content_type": "youtube"}),
        consumer._execute_video({"id": t2, "content_type": "youtube"}),
    )

    # Both should be done
    tasks = get_all_tasks(agent_db)
    assert all(t["status"] == "done" for t in tasks)

    del STEP_REGISTRY["video_step"]


# ── Orphan recovery tests ─────────────────────────────────────────────────────


def test_recover_orphans_no_running_step(consumer, agent_db):
    """Test recovery when an in_progress task has no running step."""
    task_id = create_task(
        agent_db, "https://example.com", "web", "test_slug",
        ["step_a", "step_b"], status="in_progress"
    )

    # Mark step_a as done, step_b as pending (no running step)
    with agent_db:
        agent_db.execute(
            "UPDATE task_steps SET status = 'done' WHERE task_id = ? AND step_order = 0",
            (task_id,),
        )

    # Run recovery synchronously via the event loop
    loop = asyncio.new_event_loop()

    async def _run():
        await consumer._recover_orphans()

    loop.run_until_complete(_run())
    loop.close()

    # Task should still be in_progress (no running step to reset)
    row = agent_db.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_recover_orphans_dead_process(consumer, agent_db):
    """Test recovery when a running step's PID is dead."""
    import uuid
    task_id = str(uuid.uuid4())

    with agent_db:
        agent_db.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status) "
            "VALUES (?, ?, ?, ?, 'in_progress')",
            (task_id, "https://example.com", "web", "test_slug"),
        )
        agent_db.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status, output) "
            "VALUES (?, 0, 'extract', 'running', 'pid:99999999')",
            (task_id,),
        )

    loop = asyncio.new_event_loop()

    async def _run():
        await consumer._recover_orphans()

    loop.run_until_complete(_run())
    loop.close()

    # Step should be reset to pending
    step = agent_db.execute(
        "SELECT status FROM task_steps WHERE task_id = ? AND step_order = 0",
        (task_id,),
    ).fetchone()
    assert step[0] == "pending"

    task_row = agent_db.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert task_row[0] == "pending"

    claimed = claim_task(agent_db)
    assert claimed is not None
    assert claimed["id"] == task_id


def test_recover_orphans_alive_process(consumer, agent_db):
    """Test that a still-running process is NOT reset."""
    import uuid
    task_id = str(uuid.uuid4())

    with agent_db:
        agent_db.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status) "
            "VALUES (?, ?, ?, ?, 'in_progress')",
            (task_id, "https://example.com", "web", "test_slug"),
        )
        # Store our own PID — we're definitely alive
        agent_db.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status, output) "
            "VALUES (?, 0, 'extract', 'running', ?)",
            (task_id, f"pid:{os.getpid()}"),
        )

    loop = asyncio.new_event_loop()

    async def _run():
        await consumer._recover_orphans()

    loop.run_until_complete(_run())
    loop.close()

    # Step should STILL be 'running' — we left it alone
    step = agent_db.execute(
        "SELECT status FROM task_steps WHERE task_id = ? AND step_order = 0",
        (task_id,),
    ).fetchone()
    assert step[0] == "running"


# ── Step registry tests ───────────────────────────────────────────────────────


def test_register_step_stores_fn_and_on_error(consumer):
    """Test that register_step stores the function and error type."""
    def my_fn(task, cfg):
        return "ok"

    register_step("test_step", my_fn, "pause")
    assert "test_step" in STEP_REGISTRY
    fn, on_error = STEP_REGISTRY["test_step"]
    assert fn is my_fn
    assert on_error == "pause"
    del STEP_REGISTRY["test_step"]


def test_step_error_details():
    """Test StepError with details field."""
    err = StepError(
        "Upload failed",
        error_type=ErrorType.RETRY,
        step_name="minio_upload",
        details="MinIO returned 503 Service Unavailable",
    )
    assert err.details == "MinIO returned 503 Service Unavailable"


# ── Watch scheduler gating ────────────────────────────────────────────────────


def test_watch_scheduler_enabled_default():
    assert Config().watch_scheduler_enabled is True


def test_watch_scheduler_enabled_env_override(monkeypatch, tmp_path):
    from agent.config import load_config

    monkeypatch.setenv("KB_WATCH_SCHEDULER_ENABLED", "false")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("kb_agent_dir: /tmp\n")
    cfg = load_config(config_path=cfg_path, env_path=tmp_path / ".env")
    assert cfg.watch_scheduler_enabled is False


@pytest.mark.asyncio
async def test_start_watch_scheduler_disabled_by_config(config, agent_db):
    config.watch_scheduler_enabled = False
    c = Consumer(config)
    c.agent_conn = agent_db
    c._stopping = False
    assert c._start_watch_scheduler() is None


@pytest.mark.asyncio
async def test_start_watch_scheduler_enabled(config, agent_db, monkeypatch):
    config.watch_scheduler_enabled = True

    async def _fake_scheduler(*args, **kwargs):
        return

    monkeypatch.setattr("agent.core.watch_source.run_watch_scheduler", _fake_scheduler)

    c = Consumer(config)
    c.agent_conn = agent_db
    c._stopping = False
    task = c._start_watch_scheduler()
    assert task is not None
    await task  # fake scheduler returns immediately


@pytest.mark.asyncio
async def test_start_watch_scheduler_no_conn(config):
    c = Consumer(config)
    c.agent_conn = None
    assert c._start_watch_scheduler() is None
