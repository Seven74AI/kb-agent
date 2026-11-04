"""Database initialization — SQLite schemas for agent.db and logs.db."""

import sqlite3
from pathlib import Path


AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    slug TEXT,
    error TEXT,
    retries INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS task_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    step_order INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    output TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
    source_url, content_type, status, error, content=tasks, content_rowid=rowid
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    cadence INTEGER NOT NULL,
    last_seen TEXT,
    last_content_hash TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""

LOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS step_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    stream TEXT NOT NULL DEFAULT 'stdout',
    content TEXT NOT NULL,
    logged_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_archives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
    content, task_id, step_name, content=step_logs, content_rowid=rowid
);
"""


def init_db(db_path: Path, schema: str) -> sqlite3.Connection:
    """Initialize a SQLite database with the given schema.

    Creates parent directories if needed, enables WAL mode,
    and runs the schema DDL.

    Args:
        db_path: Path to the SQLite database file.
        schema: DDL SQL to execute (CREATE TABLE IF NOT EXISTS ...).

    Returns:
        An open sqlite3.Connection in WAL mode.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False enables cross-thread access (safe with WAL mode).
    # Needed for step functions running in run_in_executor threads to write
    # to logs.db via streaming.log_progress().
    if db_path.name == "logs.db":
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(schema)
    conn.commit()
    return conn


def create_task(conn: sqlite3.Connection, url: str, content_type: str, slug: str,
                steps: list[str], status: str = "pending") -> str:
    """Create a new task with its pipeline steps.

    Args:
        conn: Agent database connection.
        url: Source URL for the task.
        content_type: Detected content type (youtube, web, book, etc.).
        slug: URL-derived slug for naming output files.
        steps: Ordered list of step names for this pipeline.
        status: Initial task status (default 'pending').

    Returns:
        The new task UUID.
    """
    import uuid
    task_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status) VALUES (?, ?, ?, ?, ?)",
            (task_id, url, content_type, slug, status),
        )
        for i, step_name in enumerate(steps):
            conn.execute(
                "INSERT INTO task_steps (task_id, step_order, name, status) VALUES (?, ?, ?, 'pending')",
                (task_id, i, step_name),
            )
    return task_id


def claim_task(conn: sqlite3.Connection) -> dict | None:
    """Atomically claim the next pending task.

    Uses UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING * —
    a single query that cannot race with other consumers.

    Args:
        conn: Agent database connection.

    Returns:
        Task dict if a pending task was claimed, None otherwise.
    """
    with conn:
        row = conn.execute("""
            UPDATE tasks SET status = 'in_progress', updated_at = datetime('now')
            WHERE id = (SELECT id FROM tasks WHERE status = 'pending' LIMIT 1)
            RETURNING *
        """).fetchone()
    if row is None:
        return None
    return dict(row)


def get_next_step(conn: sqlite3.Connection, task_id: str) -> dict | None:
    """Get the next pending step for a task, ordered by step_order.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.

    Returns:
        Step dict if a pending step exists, None otherwise.
    """
    row = conn.execute("""
        SELECT * FROM task_steps
        WHERE task_id = ? AND status = 'pending'
        ORDER BY step_order LIMIT 1
    """, (task_id,)).fetchone()
    return dict(row) if row else None


def update_step(conn: sqlite3.Connection, step_id: int, status: str, output: str | None = None):
    """Update a task step's status and optional output.

    Args:
        conn: Agent database connection.
        step_id: The step's integer ID.
        status: New status ('running', 'done', 'failed', 'paused').
        output: Optional output text from the step.
    """
    with conn:
        conn.execute(
            "UPDATE task_steps SET status = ?, output = ?, finished_at = datetime('now') WHERE id = ?",
            (status, output, step_id),
        )


def complete_task(conn: sqlite3.Connection, task_id: str):
    """Mark a task as done.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.
    """
    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'done', updated_at = datetime('now') WHERE id = ?",
            (task_id,),
        )


def fail_task(conn: sqlite3.Connection, task_id: str, error: str):
    """Mark a task as failed with an error message.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.
        error: Human-readable error description.
    """
    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'failed', error = ?, updated_at = datetime('now') WHERE id = ?",
            (error, task_id),
        )


def pause_task(conn: sqlite3.Connection, task_id: str, reason: str):
    """Pause a task that needs operator intervention.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.
        reason: Why the task is paused (e.g., 'cookies invalid').
    """
    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'paused', error = ?, updated_at = datetime('now') WHERE id = ?",
            (reason, task_id),
        )


def get_last_log_line(conn: sqlite3.Connection, task_id: str) -> str | None:
    """Get the most recent log line for a task from logs.db.

    Args:
        conn: Logs database connection.
        task_id: The task UUID.

    Returns:
        The most recent log content, or None if no logs exist.
    """
    row = conn.execute(
        "SELECT content FROM step_logs WHERE task_id = ? "
        "ORDER BY logged_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row[0] if row else None


def resume_tasks_by_reason(conn: sqlite3.Connection, reason_pattern: str) -> int:
    """Resume all paused tasks matching a reason pattern.

    Resets retries to 0 and sets status back to 'pending'.

    Args:
        conn: Agent database connection.
        reason_pattern: SQL LIKE pattern for the error/reason field
                        (e.g., '%cookies%', '%firecrawl%').

    Returns:
        Number of tasks resumed.
    """
    with conn:
        cursor = conn.execute(
            "UPDATE tasks SET status = 'pending', retries = 0, "
            "error = NULL, updated_at = datetime('now') "
            "WHERE status = 'paused' AND error LIKE ?",
            (reason_pattern,),
        )
        return cursor.rowcount


def get_task_steps(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    """Get all steps for a task, ordered by step_order.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.

    Returns:
        List of step dicts with keys: id, step_order, name, status, output,
        started_at, finished_at.
    """
    rows = conn.execute(
        "SELECT id, step_order, name, status, output, started_at, finished_at "
        "FROM task_steps WHERE task_id = ? ORDER BY step_order",
        (task_id,),
    ).fetchall()
    return [{k: r[k] for k in r.keys()} for r in rows]


def get_all_tasks(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    """Get all tasks ordered by creation time, newest first.

    Args:
        conn: Agent database connection.
        limit: Maximum number of tasks to return.

    Returns:
        List of task dicts.
    """
    rows = conn.execute(
        "SELECT id, source_url, content_type, status, error, slug, "
        "retries, created_at, updated_at "
        "FROM tasks ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{k: r[k] for k in r.keys()} for r in rows]


def archive_llm_call(conn: sqlite3.Connection, task_id: str, role: str,
                     prompt: str, response: str, model: str,
                     tokens_in: int | None = None, tokens_out: int | None = None):
    """Archive a full LLM prompt + response pair in logs.db.

    Args:
        conn: Logs database connection.
        task_id: The task UUID this call belongs to.
        role: LLM role (synthesis, boundary, review, see_also).
        prompt: Full system + user prompt sent to the LLM.
        response: Full LLM response content.
        model: Model identifier string.
        tokens_in: Input token count if available.
        tokens_out: Output token count if available.
    """
    with conn:
        conn.execute(
            """INSERT INTO llm_archives (task_id, role, prompt, response, model, tokens_in, tokens_out)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (task_id, role, prompt, response, model, tokens_in, tokens_out),
        )
