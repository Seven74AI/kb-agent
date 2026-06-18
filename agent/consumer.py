"""Async consumer — claims tasks and executes pipeline steps.

Runs in a background thread with its own asyncio event loop.
Flask runs sync in the main thread.
"""

import asyncio
import sqlite3
import signal
import time
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.db import (
    AGENT_SCHEMA, LOGS_SCHEMA,
    claim_task, complete_task, fail_task, get_next_step,
    init_db, pause_task, update_step,
)

# ── Pipeline step registry ───────────────────────────────────────────────────

# Populated at startup from pipelines/*.py modules
STEP_REGISTRY: dict[str, Any] = {}


def register_step(name: str, fn):
    """Register a pipeline step function."""
    STEP_REGISTRY[name] = fn


class Consumer:
    """Background consumer that polls for tasks and executes pipeline steps.

    Concurrency:
    - Video tasks: BoundedSemaphore(1) — one at a time
    - Text tasks: No semaphore, parallel (up to text_parallel)
    """

    def __init__(self, config: Config):
        self.config = config
        self.agent_conn: sqlite3.Connection | None = None
        self.logs_conn: sqlite3.Connection | None = None
        self._running = False
        self._video_semaphore: asyncio.BoundedSemaphore | None = None
        self._active_text: int = 0
        self._tasks: dict[str, asyncio.Task] = {}

    async def startup(self):
        """Initialize databases, register pipelines, recover orphans."""
        db_dir = self.config.kb_agent_dir / "data"
        db_dir.mkdir(parents=True, exist_ok=True)

        self.agent_conn = init_db(db_dir / "agent.db", AGENT_SCHEMA)
        self.logs_conn = init_db(db_dir / "logs.db", LOGS_SCHEMA)

        self._video_semaphore = asyncio.BoundedSemaphore(self.config.video_semaphore)

        # Import pipeline modules to register steps
        from agent.pipelines import web  # noqa: F401

        # ── Crash recovery: orphan detection ──
        await self._recover_orphans()

        self._running = True

    async def _recover_orphans(self):
        """Find in_progress tasks from a previous crash and resume them."""
        if not self.agent_conn:
            return

        orphans = self.agent_conn.execute(
            "SELECT id, content_type, slug FROM tasks WHERE status = 'in_progress'"
        ).fetchall()

        for task_id, content_type, slug in orphans:
            # Find the running step
            step = self.agent_conn.execute(
                "SELECT id, step_order, name FROM task_steps "
                "WHERE task_id = ? AND status = 'running' "
                "ORDER BY step_order LIMIT 1",
                (task_id,),
            ).fetchone()

            if step:
                # Re-run the step that was running at crash time
                update_step(self.agent_conn, step[0], "pending")
                print(f"[consumer] Recovered orphan task {task_id[:8]} — resetting step '{step[2]}'")
            else:
                print(f"[consumer] Orphan task {task_id[:8]} — no running step, resuming next pending")

    async def shutdown(self):
        """Graceful shutdown — wait for running steps to finish."""
        self._running = False
        if self._tasks:
            print(f"[consumer] Shutting down — waiting for {len(self._tasks)} active tasks...")
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        print("[consumer] Shutdown complete")

    async def run(self):
        """Main polling loop. Claims tasks and executes them."""
        await self.startup()

        while self._running:
            task = claim_task(self.agent_conn) if self.agent_conn else None

            if task:
                task_id = task["id"]
                content_type = task.get("content_type", "web")

                if content_type in ("youtube", "instagram_reel", "mega"):
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
        async with self._video_semaphore:
            await self._execute_task(task)

    async def _execute_text(self, task: dict):
        """Execute a text pipeline task (parallel)."""
        try:
            await self._execute_task(task)
        finally:
            self._active_text -= 1

    async def _execute_task(self, task: dict):
        """Execute all steps for a single task."""
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
                fn = STEP_REGISTRY.get(step_name)
                if fn is None:
                    raise ValueError(f"Unknown step: {step_name}")

                # Execute the step (synchronous for now, async when needed)
                output = fn(task, self.config)
                update_step(self.agent_conn, step_id, "done", str(output))
                retries = 0  # Reset on success

            except Exception as e:
                retries += 1
                if retries >= self.config.max_retries:
                    fail_task(self.agent_conn, task_id, str(e))
                    return

                delay = self.config.retry_delays[min(retries - 1, len(self.config.retry_delays) - 1)]
                print(f"[consumer] Step '{step_name}' failed (retry {retries}/{self.config.max_retries}): {e}")
                update_step(self.agent_conn, step_id, "pending", str(e))
                await asyncio.sleep(delay)


def start_consumer(config: Config) -> Consumer:
    """Create and start the consumer in a background thread.

    Returns the Consumer instance for shutdown control.
    """
    consumer = Consumer(config)

    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(consumer.run())

    import threading
    thread = threading.Thread(target=_run, daemon=True, name="kb-consumer")
    thread.start()

    # Install signal handlers on the main thread
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(consumer.shutdown(), loop=loop)
            ))
        except ValueError:
            pass  # Not in main thread

    return consumer
