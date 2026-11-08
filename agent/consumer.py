"""Async consumer — claims tasks and executes pipeline steps.

Runs in a background thread with its own asyncio event loop.
Flask runs sync in the main thread.

Error model (three buckets):
- RETRY: transient errors — exponential backoff 5s→15s→60s→180s→PAUSE
- PAUSE: operator intervention needed (cookies, IP flag, infra down)
- FAIL: unrecoverable (URL deleted, 404, unsupported content)

Process management:
- No timeouts on steps (HTTP calls have built-in ~30s timeouts)
- PR_SET_PDEATHSIG in canonical scripts kills children when parent dies
- proc.returncode is None → alive; int → exited
- Hang detection via dashboard staleness (last output timestamp)
"""

import asyncio
import inspect
import logging
import os
import re
import sqlite3
import traceback
from typing import Any

logger = logging.getLogger(__name__)
detail_logger = logging.getLogger(__name__ + ".detail")

from agent.config import Config
from agent.db import (
    AGENT_SCHEMA, LOGS_SCHEMA,
    _persist_visible_error,
    claim_task,
    complete_task,
    fail_task,
    get_next_step,
    task_execution_aborted,
    init_db,
    pause_task,
    update_step,
)
from agent.exceptions import ErrorType, StepError
from agent.notify import notify_task
from agent.streaming import log_progress

# ── Pipeline step registry ───────────────────────────────────────────────────

# Maps step_name → (fn, on_error)
# Populated at startup from pipelines/*.py modules
STEP_REGISTRY: dict[str, tuple[Any, str]] = {}

def _sanitize_step_log_message(message: str) -> str:
    """Scrub text before persisting to step_logs (URL redaction + visible-message policy)."""
    return _persist_visible_error(message)


def _format_stdout_traceback(exc: BaseException, *, limit: int = 5) -> str:
    """Scrubbed exception summary for process stdout (not step_logs or tasks.error)."""
    formatted = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit)
    lines: list[str] = []
    for line in formatted:
        stripped = line.rstrip()
        if not stripped or stripped.startswith("Traceback"):
            continue
        clean = _persist_visible_error(stripped)
        if clean:
            lines.append(clean)
    return "\n".join(lines)

# Content types routed through _execute_video (video semaphore — pyannote RAM).
VIDEO_TASK_CONTENT_TYPES = (
    "youtube",
    "bunny_stream",
    "instagram_reel",
    "mega_video",
    "rumble",
    "substack_video",
    "threads_video",
)


def register_step(name: str, fn, on_error: str = "retry"):
    """Register a pipeline step function with its error handling mode.

    Args:
        name: Step name (must match Step.name in pipeline definition).
        fn: Callable (sync or async) that executes the step.
        on_error: Error mode — 'retry', 'pause', or 'fail'.
    """
    STEP_REGISTRY[name] = (fn, on_error)


def recover_orphan_tasks(conn: sqlite3.Connection, *, log_prefix: str = "[consumer]") -> None:
    """Reset in_progress tasks whose running step has no live subprocess PID."""
    orphans = conn.execute(
        "SELECT id, content_type, slug FROM tasks WHERE status = 'in_progress'"
    ).fetchall()

    for task_id, content_type, slug in orphans:
        step = conn.execute(
            "SELECT id, step_order, name, output FROM task_steps "
            "WHERE task_id = ? AND status = 'running' "
            "ORDER BY step_order LIMIT 1",
            (task_id,),
        ).fetchone()

        if not step:
            print(
                f"{log_prefix} Orphan task {task_id[:8]} — "
                f"no running step, resetting to pending"
            )
            conn.execute(
                "UPDATE tasks SET status = 'pending' WHERE id = ?",
                (task_id,),
            )
            conn.commit()
            continue

        step_id, step_order, step_name, step_output = step
        pid = Consumer._extract_pid_from_output(step_output)
        if pid is not None and Consumer._is_process_alive(pid):
            print(
                f"{log_prefix} Orphan task {task_id[:8]} — "
                f"step '{step_name}' PID {pid} still alive, leaving"
            )
            continue

        update_step(conn, step_id, "pending")
        conn.execute(
            "UPDATE tasks SET status = 'pending' WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        print(
            f"{log_prefix} Recovered orphan task {task_id[:8]} — "
            f"resetting step '{step_name}' and task to pending"
        )


class Consumer:
    """Background consumer that polls for tasks and executes pipeline steps.

    Concurrency:
    - Video tasks: BoundedSemaphore(1) — one at a time (pyannote RAM)
    - Text tasks: parallel up to config.text_parallel
    """

    def __init__(self, config: Config):
        self.config = config
        self.agent_conn: sqlite3.Connection | None = None
        self.logs_conn: sqlite3.Connection | None = None
        self._running = False
        self._stopping = False          # True after shutdown initiated
        self._video_semaphore: asyncio.BoundedSemaphore | None = None
        self._active_text: int = 0
        self._tasks: dict[str, asyncio.Task] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def startup(self):
        """Initialize databases, register pipelines, recover orphans."""
        db_dir = self.config.kb_agent_dir / "data"
        db_dir.mkdir(parents=True, exist_ok=True)

        self.agent_conn = init_db(db_dir / "agent.db", AGENT_SCHEMA)
        self.logs_conn = init_db(db_dir / "logs.db", LOGS_SCHEMA)

        self._video_semaphore = asyncio.BoundedSemaphore(self.config.video_semaphore)

        # Import pipeline modules to register steps
        from agent.pipelines import import_all_pipelines, PIPELINES

        import_all_pipelines()

        for steps in PIPELINES.values():
            for step in steps:
                register_step(step.name, step.fn, step.on_error)

        # ── Crash recovery: orphan detection (run.py may have run this already) ──
        if not os.environ.get("KB_AGENT_ORPHAN_RECOVERY_DONE"):
            await self._recover_orphans()

        self._running = True
        self._watch_task: asyncio.Task | None = self._start_watch_scheduler()

    def _start_watch_scheduler(self) -> asyncio.Task | None:
        """Start the in-process watch scheduler, unless disabled by config.

        Returns the created task, or ``None`` when the scheduler is disabled
        (e.g. cron drives polling via ``scripts/watch_sources_poll.py``, so the
        consumer must not poll too — otherwise both would enqueue duplicate
        tasks for the same URL).
        """
        if self.agent_conn is None:
            return None
        if not self.config.watch_scheduler_enabled:
            print(
                "[consumer] watch scheduler disabled (watch_scheduler_enabled=false); "
                "cron polling via scripts/watch_sources_poll.py is assumed"
            )
            return None
        from agent.core.watch_source import default_fetch_html, run_watch_scheduler, submit_routed

        return asyncio.create_task(
            run_watch_scheduler(
                self.config,
                fetch_html=default_fetch_html,
                submit_fn=submit_routed,
                stopping=lambda: self._stopping,
            ),
            name="watch-source-scheduler",
        )

    async def _recover_orphans(self):
        """Find in_progress tasks from a previous crash and resume them."""
        if not self.agent_conn:
            return
        recover_orphan_tasks(self.agent_conn, log_prefix="[consumer]")

    @staticmethod
    def _extract_pid_from_output(output: str | None) -> int | None:
        """Extract a PID from step output if it was stored.

        Format: 'pid:12345' somewhere in the output string.
        Returns None if no PID found.
        """
        if not output:
            return None
        m = re.search(r'pid:(\d+)', output)
        return int(m.group(1)) if m else None

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if a process is still running via /proc/{pid}.

        /proc/{pid} exists → process is alive (or zombie — check stat).
        """
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
                # State char is field 3 (0-indexed after 'comm)')
                # 'Z' = zombie (dead for practical purposes)
                state_start = stat.rfind(')') + 2
                state = stat[state_start] if state_start < len(stat) else ''
                return state != 'Z'
        except (FileNotFoundError, PermissionError):
            return False

    async def shutdown(self):
        """Graceful shutdown — stop polling, THEN wait for running steps.

        1. Set _stopping to stop claiming new tasks.
        2. Set _running = False to exit the polling loop.
        3. Wait for all active tasks to finish.
        4. Children die via PR_SET_PDEATHSIG in canonical scripts.
        """
        self._stopping = True
        self._running = False

        watch_task = getattr(self, "_watch_task", None)
        if watch_task is not None and not watch_task.done():
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass

        if self._tasks:
            print(
                f"[consumer] Shutting down — waiting for "
                f"{len(self._tasks)} active tasks..."
            )
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        print("[consumer] Shutdown complete")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Main polling loop. Claims tasks and executes them.

        IMPORTANT: Does NOT call startup() — the caller MUST initialize
        the consumer via startup() before calling run(). See run.py:_run_consumer()
        for the correct sequence: startup() first, then run().
        """
        while self._running:
            task = claim_task(self.agent_conn) if self.agent_conn else None

            if task:
                task_id = task["id"]
                content_type = task.get("content_type", "web")

                if content_type in VIDEO_TASK_CONTENT_TYPES:
                    # Video tasks — semaphore-controlled
                    task_obj = asyncio.create_task(self._execute_video(task))
                else:
                    # Text tasks — parallel up to limit
                    while self._active_text >= self.config.text_parallel:
                        await asyncio.sleep(1)
                    self._active_text += 1
                    task_obj = asyncio.create_task(self._execute_text(task))

                self._tasks[task_id] = task_obj

                # Clean up completed tasks
                done = [tid for tid, t in self._tasks.items() if t.done()]
                for tid in done:
                    del self._tasks[tid]

            await asyncio.sleep(5)  # Poll interval

    async def _execute_video(self, task: dict):
        """Execute a video pipeline task (sequential, semaphore-gated)."""
        if self._video_semaphore is None:
            await self._execute_task(task)
            return
        async with self._video_semaphore:
            await self._execute_task(task)

    async def _execute_text(self, task: dict):
        """Execute a text pipeline task (parallel)."""
        try:
            await self._execute_task(task)
        finally:
            self._active_text -= 1

    # ── Task execution ────────────────────────────────────────────────────

    def _log_error(self, task_id: str, step_name: str, message: str) -> None:
        """Persist an error to step_logs (stream='stderr') for dashboard/debug visibility."""
        if self.logs_conn is None:
            return
        try:
            log_progress(
                self.logs_conn,
                task_id,
                step_name,
                _sanitize_step_log_message(message),
                stream="stderr",
            )
        except Exception:  # noqa: BLE001 — logging must never break error handling
            pass

    def _alert_failure(self, task: dict, status: str, msg: str) -> None:
        """Best-effort Telegram alert for a task failure/pause."""
        try:
            notify_task(
                self.config,
                task.get("slug") or task["id"][:8],
                task.get("content_type", ""),
                status,
                msg,
            )
        except Exception:  # noqa: BLE001 — alerting must never break the consumer
            pass

    async def _execute_task(self, task: dict):
        """Execute all steps for a single task.

        Steps execute in order. On error, routes to RETRY/PAUSE/FAIL
        based on the step's on_error setting or the StepError's error_type.

        RETRY: exponential backoff 5s→15s→60s→180s→PAUSE after max_retries.
        PAUSE: store reason, operator resumes via dashboard.
        FAIL: mark failed, stop processing this task.
        """
        task_id = task["id"]
        retries = 0

        while True:
            if task_execution_aborted(self.agent_conn, task_id):
                return
            step = get_next_step(self.agent_conn, task_id)
            if step is None:
                # All steps done
                if not task_execution_aborted(self.agent_conn, task_id):
                    complete_task(self.agent_conn, task_id)
                return

            step_id = step["id"]
            step_name = step["name"]

            # Mark step as running
            update_step(self.agent_conn, step_id, "running")

            try:
                output = await self._run_step(step_name, task)
                # Store PID in output if we launched a subprocess
                update_step(self.agent_conn, step_id, "done", str(output))
                retries = 0  # Reset on success

            except StepError as e:
                error_type = e.error_type.value  # 'retry', 'pause', 'fail'
                error_msg = _persist_visible_error(str(e))
                logger.error(
                    "[consumer] Step '%s' StepError (%s): %s",
                    step_name,
                    error_type,
                    error_msg,
                )
                detail_logger.error(
                    "[consumer] Step '%s' StepError (%s) (detail): %s",
                    step_name,
                    error_type,
                    error_msg,
                )
                tb_summary = _format_stdout_traceback(e)
                print(
                    f"[consumer] Step '{step_name}' traceback:\n{tb_summary}",
                    flush=True,
                )
                self._log_error(
                    task_id,
                    step_name,
                    f"StepError({error_type}): {error_msg}",
                )

                if error_type == "fail":
                    print(f"[consumer] Step '{step_name}' FAILED: {error_msg}")
                    fail_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "failed", error_msg)
                    self._alert_failure(task, "failed", error_msg)
                    return

                elif error_type == "pause":
                    print(f"[consumer] Step '{step_name}' PAUSED: {error_msg}")
                    pause_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "paused", error_msg)
                    self._alert_failure(task, "paused", error_msg)
                    return

                else:  # retry
                    retries += 1
                    if retries >= self.config.max_retries:
                        print(
                            f"[consumer] Step '{step_name}' exhausted retries "
                            f"({retries}/{self.config.max_retries}) → PAUSING"
                        )
                        pause_msg = _persist_visible_error(
                            f"Exhausted {retries} retries: {error_msg}"
                        )
                        pause_task(self.agent_conn, task_id, pause_msg)
                        update_step(self.agent_conn, step_id, "paused", pause_msg)
                        self._alert_failure(task, "paused", pause_msg)
                        return

                    delay = self.config.retry_delays[
                        min(retries - 1, len(self.config.retry_delays) - 1)
                    ]
                    print(
                        f"[consumer] Step '{step_name}' RETRY "
                        f"({retries}/{self.config.max_retries}, {delay}s): {error_msg}"
                    )
                    update_step(self.agent_conn, step_id, "pending", error_msg)
                    await asyncio.sleep(delay)

            except Exception as e:
                error_msg = _persist_visible_error(f"{type(e).__name__}: {e}")
                logger.error(
                    "[consumer] Step '%s' unhandled exception: %s",
                    step_name,
                    error_msg,
                )
                detail_logger.error(
                    "[consumer] Step '%s' unhandled exception (detail): %s",
                    step_name,
                    error_msg,
                )
                tb_summary = _format_stdout_traceback(e)
                print(
                    f"[consumer] Step '{step_name}' unhandled traceback:\n{tb_summary}",
                    flush=True,
                )
                # Fall back to step's on_error setting
                _, default_on_error = STEP_REGISTRY.get(step_name, (None, "retry"))
                self._log_error(
                    task_id,
                    step_name,
                    f"Unhandled exception: {error_msg}",
                )

                if default_on_error == "fail":
                    print(f"[consumer] Step '{step_name}' FAILED (unexpected): {error_msg}")
                    fail_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "failed", error_msg)
                    self._alert_failure(task, "failed", error_msg)
                    return

                elif default_on_error == "pause":
                    print(f"[consumer] Step '{step_name}' PAUSED (unexpected): {error_msg}")
                    pause_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "paused", error_msg)
                    self._alert_failure(task, "paused", error_msg)
                    return

                else:  # default retry
                    retries += 1
                    if retries >= self.config.max_retries:
                        print(
                            f"[consumer] Step '{step_name}' exhausted retries "
                            f"({retries}/{self.config.max_retries}) → PAUSING"
                        )
                        pause_msg = _persist_visible_error(
                            f"Exhausted {retries} retries: {error_msg}"
                        )
                        pause_task(self.agent_conn, task_id, pause_msg)
                        update_step(self.agent_conn, step_id, "paused", pause_msg)
                        self._alert_failure(task, "paused", pause_msg)
                        return

                    delay = self.config.retry_delays[
                        min(retries - 1, len(self.config.retry_delays) - 1)
                    ]
                    print(
                        f"[consumer] Step '{step_name}' RETRY "
                        f"({retries}/{self.config.max_retries}, {delay}s): {error_msg}"
                    )
                    update_step(self.agent_conn, step_id, "pending", error_msg)
                    await asyncio.sleep(delay)

    async def _run_step(self, step_name: str, task: dict) -> Any:
        """Execute a single pipeline step, supporting both sync and async functions.

        Resolves step from the task's content_type pipeline registry FIRST
        (to avoid namespace collisions between pipelines with shared step names
        like 'synthesize_note', 'quality_gate', 'minio_upload', 'git_push').
        Falls back to global STEP_REGISTRY for backward compatibility.

        For mechanical steps that launch subprocesses, step functions should:
        - Launch the subprocess via asyncio.create_subprocess_exec
        - Stream output to logs.db via agent.streaming.stream_to_logs
        - Return the result (optionally with 'pid:<N>' for crash recovery)

        The PR_SET_PDEATHSIG guarantee: canonical scripts (diarize.py,
        transcribe.py) call `libc.prctl(1, signal.SIGKILL)` so kernel
        auto-kills children when the consumer process dies. No orphan
        subprocesses survive a consumer crash.
        """
        # Resolve step from the task's pipeline first (avoids namespace collisions)
        from agent.pipelines import get_pipeline
        content_type = task.get("content_type", "web")
        pipeline = get_pipeline(content_type)
        if pipeline is None:
            raise StepError(
                f"No pipeline registered for content_type={content_type!r}",
                error_type=ErrorType.FAIL,
                step_name=step_name,
            )
        entry = None
        if pipeline:
            for s in pipeline:
                if s.name == step_name:
                    entry = (s.fn, s.on_error)
                    break
        if entry is None:
            # Fall back to global registry (backward compat)
            entry = STEP_REGISTRY.get(step_name)
        if entry is None:
            raise StepError(
                f"Unknown step: {step_name} (content_type={content_type})",
                error_type=ErrorType.FAIL,
                step_name=step_name,
            )

        fn, _ = entry

        # Inject streaming context so steps can write progress to logs.db.
        # _task_id, _step_name, _logs_conn are available in task dict.
        # Steps import agent.streaming.log_progress() to write progress lines.
        task_id = task["id"]
        task["_task_id"] = task_id
        task["_step_name"] = step_name
        task["_logs_conn"] = self.logs_conn  # thread-safe: check_same_thread=False

        if inspect.iscoroutinefunction(fn):
            return await fn(task, self.config)
        else:
            # Sync function — run in default executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn, task, self.config)
