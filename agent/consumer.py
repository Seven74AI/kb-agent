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
import os
import sqlite3
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.db import (
    AGENT_SCHEMA, LOGS_SCHEMA,
    claim_task, complete_task, fail_task, get_next_step,
    init_db, pause_task, update_step,
)
from agent.exceptions import ErrorType, StepError

# ── Pipeline step registry ───────────────────────────────────────────────────

# Maps step_name → (fn, on_error)
# Populated at startup from pipelines/*.py modules
STEP_REGISTRY: dict[str, tuple[Any, str]] = {}


def register_step(name: str, fn, on_error: str = "retry"):
    """Register a pipeline step function with its error handling mode.

    Args:
        name: Step name (must match Step.name in pipeline definition).
        fn: Callable (sync or async) that executes the step.
        on_error: Error mode — 'retry', 'pause', or 'fail'.
    """
    STEP_REGISTRY[name] = (fn, on_error)


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
        from agent.pipelines import web  # noqa: F401
        from agent.pipelines import youtube  # noqa: F401
        from agent.pipelines import threads  # noqa: F401
        from agent.pipelines import instagram  # noqa: F401
        from agent.pipelines import books  # noqa: F401
        from agent.pipelines import PIPELINES

        for steps in PIPELINES.values():
            for step in steps:
                register_step(step.name, step.fn, step.on_error)

        # ── Crash recovery: orphan detection ──
        await self._recover_orphans()

        self._running = True

    async def _recover_orphans(self):
        """Find in_progress tasks from a previous crash and resume them.

        For each in_progress task:
        - Find the 'running' step (the one that was executing at crash time).
        - If step had a subprocess PID, check /proc/{pid} — if alive, leave it.
        - If dead or no PID, reset step to 'pending' for re-execution.
        """
        if not self.agent_conn:
            return

        orphans = self.agent_conn.execute(
            "SELECT id, content_type, slug FROM tasks WHERE status = 'in_progress'"
        ).fetchall()

        for task_id, content_type, slug in orphans:
            step = self.agent_conn.execute(
                "SELECT id, step_order, name, output FROM task_steps "
                "WHERE task_id = ? AND status = 'running' "
                "ORDER BY step_order LIMIT 1",
                (task_id,),
            ).fetchone()

            if not step:
                # Task was claimed but no step started — reset to pending for re-claim
                print(
                    f"[consumer] Orphan task {task_id[:8]} — "
                    f"no running step, resetting to pending"
                )
                self.agent_conn.execute(
                    "UPDATE tasks SET status = 'pending' WHERE id = ?",
                    (task_id,)
                )
                self.agent_conn.commit()
                continue

            step_id, step_order, step_name, step_output = step

            # Check if a subprocess PID is stored in output
            pid = self._extract_pid_from_output(step_output)
            if pid is not None and self._is_process_alive(pid):
                print(
                    f"[consumer] Orphan task {task_id[:8]} — "
                    f"step '{step_name}' PID {pid} still alive, leaving"
                )
                # Leave it — consumer will pick up when step finishes
                continue

            # Dead process or no PID — reset to pending
            update_step(self.agent_conn, step_id, "pending")
            print(
                f"[consumer] Recovered orphan task {task_id[:8]} — "
                f"resetting step '{step_name}'"
            )

    @staticmethod
    def _extract_pid_from_output(output: str | None) -> int | None:
        """Extract a PID from step output if it was stored.

        Format: 'pid:12345' somewhere in the output string.
        Returns None if no PID found.
        """
        if not output:
            return None
        import re
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

                if content_type in ("youtube", "instagram_reel", "mega", "threads_video"):
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
            step = get_next_step(self.agent_conn, task_id)
            if step is None:
                # All steps done
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
                last_error_type = None

            except StepError as e:
                error_type = e.error_type.value  # 'retry', 'pause', 'fail'
                error_msg = str(e)

                if error_type == "fail":
                    print(f"[consumer] Step '{step_name}' FAILED: {error_msg}")
                    fail_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "failed", error_msg)
                    return

                elif error_type == "pause":
                    print(f"[consumer] Step '{step_name}' PAUSED: {error_msg}")
                    pause_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "paused", error_msg)
                    return

                else:  # retry
                    retries += 1
                    if retries >= self.config.max_retries:
                        print(
                            f"[consumer] Step '{step_name}' exhausted retries "
                            f"({retries}/{self.config.max_retries}) → PAUSING"
                        )
                        pause_msg = f"Exhausted {retries} retries: {error_msg}"
                        pause_task(self.agent_conn, task_id, pause_msg)
                        update_step(self.agent_conn, step_id, "paused", pause_msg)
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
                # Unclassified exception — fall back to step's on_error setting
                _, default_on_error = STEP_REGISTRY.get(step_name, (None, "retry"))
                error_msg = f"{type(e).__name__}: {e}"

                if default_on_error == "fail":
                    print(f"[consumer] Step '{step_name}' FAILED (unexpected): {error_msg}")
                    fail_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "failed", error_msg)
                    return

                elif default_on_error == "pause":
                    print(f"[consumer] Step '{step_name}' PAUSED (unexpected): {error_msg}")
                    pause_task(self.agent_conn, task_id, error_msg)
                    update_step(self.agent_conn, step_id, "paused", error_msg)
                    return

                else:  # default retry
                    retries += 1
                    if retries >= self.config.max_retries:
                        print(
                            f"[consumer] Step '{step_name}' exhausted retries "
                            f"({retries}/{self.config.max_retries}) → PAUSING"
                        )
                        pause_msg = f"Exhausted {retries} retries: {error_msg}"
                        pause_task(self.agent_conn, task_id, pause_msg)
                        update_step(self.agent_conn, step_id, "paused", pause_msg)
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
