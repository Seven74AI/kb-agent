"""Streaming — real-time subprocess output to logs.db and async execution helpers.

Core utilities:
- stream_to_logs: Reads subprocess stdout/stderr line-by-line, writes to logs.db.
- stream_both: Streams both stdout and stderr simultaneously.
- run_with_streaming: Launches an async subprocess with live streaming.

Process management (see CONTEXT.md):
- No timeouts on steps — HTTP calls have built-in ~30s timeouts.
- readline() streaming — NOT communicate() (which buffers all output).
- proc.returncode is None → process alive; int → exited.
- PR_SET_PDEATHSIG in canonical scripts kills children when parent dies.
"""

import asyncio
import sqlite3

from agent.notify import sanitize_visible_message


async def stream_to_logs(
    proc: asyncio.subprocess.Process,
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
    stream: str = "stdout",
):
    """Stream subprocess stdout/stderr line-by-line into logs.db.

    Reads lines as they appear, writing each to the step_logs table.
    This enables real-time progress visibility on the dashboard.

    Args:
        proc: The asyncio subprocess to monitor.
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Name of the current pipeline step.
        stream: 'stdout' or 'stderr'.
    """
    stream_obj = proc.stdout if stream == "stdout" else proc.stderr
    if stream_obj is None:
        return

    while True:
        line = await stream_obj.readline()
        if not line:
            break

        text = line.decode("utf-8", errors="replace").rstrip("\n")
        if not text.strip():
            continue
        text = sanitize_visible_message(text)

        with logs_conn:
            logs_conn.execute(
                "INSERT INTO step_logs (task_id, step_name, stream, content) "
                "VALUES (?, ?, ?, ?)",
                (task_id, step_name, stream, text),
            )


async def stream_both(
    proc: asyncio.subprocess.Process,
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
):
    """Stream both stdout and stderr simultaneously.

    Args:
        proc: The asyncio subprocess.
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Step name for log entries.
    """
    tasks = []
    if proc.stdout:
        tasks.append(stream_to_logs(proc, logs_conn, task_id, step_name, "stdout"))
    if proc.stderr:
        tasks.append(stream_to_logs(proc, logs_conn, task_id, step_name, "stderr"))

    if tasks:
        await asyncio.gather(*tasks)


async def run_with_streaming(
    *args: str,
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
    **kwargs,
) -> asyncio.subprocess.Process:
    """Launch an async subprocess with stdout/stderr streaming to logs.db.

    Wraps asyncio.create_subprocess_exec, sets up streaming tasks,
    and returns the process handle. The caller MUST await proc.wait()
    after this returns.

    Args:
        *args: Command and arguments (e.g., 'curl', '-s', url).
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Step name for log entries.
        **kwargs: Passed through to asyncio.create_subprocess_exec
                  (e.g., cwd, env).

    Returns:
        The asyncio.subprocess.Process handle with streaming active.

    Example:
        proc = await run_with_streaming(
            'yt-dlp', url,
            logs_conn=logs_conn, task_id=task_id, step_name='download',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise StepError("Download failed", error_type=ErrorType.RETRY)
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )

    # Start streaming tasks (they'll end when the pipes close on process exit)
    asyncio.create_task(
        stream_both(proc, logs_conn, task_id, step_name)
    )

    return proc


def log_progress(
    logs_conn: sqlite3.Connection,
    task_id: str,
    step_name: str,
    message: str,
    stream: str = "stdout",
) -> None:
    """Synchronous progress logger — safe to call from any thread.

    Write a progress message to step_logs for dashboard visibility.
    Designed for sync step functions running in run_in_executor threads
    (logs_conn must have check_same_thread=False, set by init_db for logs.db).

    Args:
        logs_conn: SQLite connection to logs.db.
        task_id: The task UUID.
        step_name: Name of the current pipeline step.
        message: Progress message to log.
        stream: 'stdout' or 'stderr' (default: stdout).
    """
    message = sanitize_visible_message(message)
    with logs_conn:
        logs_conn.execute(
            "INSERT INTO step_logs (task_id, step_name, stream, content) "
            "VALUES (?, ?, ?, ?)",
            (task_id, step_name, stream, message),
        )
