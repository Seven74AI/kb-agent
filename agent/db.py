"""Database initialization — SQLite schemas for agent.db and logs.db."""

import json
import re
import sqlite3
from pathlib import Path

# Max ingest-queue retries for failed embedded/standalone media (``recover_failed_media``).
INGEST_MAX_MEDIA_ATTEMPTS = 3


AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    slug TEXT,
    tags TEXT,
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

CREATE TABLE IF NOT EXISTS watch_sources (
    id TEXT PRIMARY KEY,
    page_url TEXT NOT NULL UNIQUE,
    item_pattern TEXT NOT NULL,
    item_url_template TEXT NOT NULL,
    cadence_seconds INTEGER NOT NULL DEFAULT 86400,
    last_seen_json TEXT NOT NULL DEFAULT '[]',
    last_polled_at TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_queue (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    slug TEXT NOT NULL,
    tags TEXT,
    parent_slug TEXT NOT NULL DEFAULT '',
    media_kind TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    dispatched_at TEXT,
    updated_at TEXT,
    UNIQUE(url, parent_slug)
);

CREATE TABLE IF NOT EXISTS batch_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    total INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    dispatched INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    summary TEXT
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


def normalize_tags(tags) -> list[str]:
    """Normalize tags input (str/comma/JSON/list) to a deduped list of clean strings."""
    if not tags:
        return []
    if isinstance(tags, str):
        s = tags.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                tags = json.loads(s)
            except json.JSONDecodeError:
                tags = [t.strip() for t in s.strip("[]").split(",") if t.strip()]
        else:
            tags = [t.strip() for t in s.split(",") if t.strip()]
    out = []
    for t in tags:
        t = str(t).strip()
        if t and t not in out:
            out.append(t)
    return out


def serialize_tags(tags) -> str | None:
    """Encode tags to a JSON array string for storage (None when empty)."""
    tags = normalize_tags(tags)
    return json.dumps(tags) if tags else None


_MIGRATE_TABLES = ("tasks", "ingest_queue")


def _column_defs_from_schema(schema: str) -> dict[str, dict[str, str]]:
    """Parse CREATE TABLE column definitions from agent DDL (non-virtual tables)."""
    result: dict[str, dict[str, str]] = {}
    for match in re.finditer(
        r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\);",
        schema,
        re.DOTALL | re.IGNORECASE,
    ):
        table = match.group(1)
        if table.endswith("_fts"):
            continue
        cols: dict[str, str] = {}
        for raw_line in match.group(2).split("\n"):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(("PRIMARY ", "FOREIGN ", "UNIQUE", "CHECK ", "CONSTRAINT ")):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                cols[parts[0]] = parts[1]
        result[table] = cols
    return result


_SCHEMA_COLUMNS = _column_defs_from_schema(AGENT_SCHEMA)


def _ingest_parent_slug(parent_slug: str | None) -> str:
    """Normalize parent_slug for ingest_queue (NULL/None → standalone sentinel '')."""
    return parent_slug or ""


def _migrate_ingest_queue_url_parent_unique(conn: sqlite3.Connection) -> None:
    """Rebuild ingest_queue when uniqueness is url-only (legacy schema)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "ingest_queue" not in tables:
        return
    ddl_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ingest_queue'"
    ).fetchone()
    if ddl_row is None:
        return
    ddl = ddl_row[0] or ""
    if "UNIQUE(url, parent_slug)" in ddl.replace(" ", ""):
        return
    conn.executescript(
        """
        CREATE TABLE ingest_queue_new (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            content_type TEXT NOT NULL,
            slug TEXT NOT NULL,
            tags TEXT,
            parent_slug TEXT NOT NULL DEFAULT '',
            media_kind TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            dispatched_at TEXT,
            updated_at TEXT,
            UNIQUE(url, parent_slug)
        );
        INSERT INTO ingest_queue_new (
            id, url, content_type, slug, tags, parent_slug, media_kind,
            status, error, attempts, created_at, dispatched_at, updated_at
        )
        SELECT id, url, content_type, slug, tags,
               COALESCE(parent_slug, ''), media_kind,
               status, error, attempts, created_at, dispatched_at, updated_at
        FROM ingest_queue;
        DROP TABLE ingest_queue;
        ALTER TABLE ingest_queue_new RENAME TO ingest_queue;
        """
    )
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns missing on pre-existing databases (idempotent, schema-safe).

    ``init_db`` is shared between the agent schema and the logs schema, so guard
    each table's migration on the table actually existing.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for table in _MIGRATE_TABLES:
        if table not in tables:
            continue
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, col_def in _SCHEMA_COLUMNS.get(table, {}).items():
            if col == "id" or col in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
    _migrate_ingest_queue_url_parent_unique(conn)
    conn.commit()


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
    _migrate(conn)
    return conn


def create_task(conn: sqlite3.Connection, url: str, content_type: str, slug: str,
                steps: list[str], status: str = "pending", tags=None) -> str:
    """Create a new task with its pipeline steps.

    Args:
        conn: Agent database connection.
        url: Source URL for the task.
        content_type: Detected content type (youtube, web, book, etc.).
        slug: URL-derived slug for naming output files.
        steps: Ordered list of step names for this pipeline.
        status: Initial task status (default 'pending').
        tags: Optional user-specified tags (str/comma/JSON/list) added to the note.

    Returns:
        The new task UUID.
    """
    import uuid
    task_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status, tags) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, url, content_type, slug, status, serialize_tags(tags)),
        )
        for i, step_name in enumerate(steps):
            conn.execute(
                "INSERT INTO task_steps (task_id, step_order, name, status) VALUES (?, ?, ?, 'pending')",
                (task_id, i, step_name),
            )
    return task_id


def queue_ingest(conn: sqlite3.Connection, url: str, content_type: str, slug: str,
                 status: str = "queued", tags=None, parent_slug=None, media_kind=None,
                 max_attempts: int = INGEST_MAX_MEDIA_ATTEMPTS) -> tuple[str, bool]:
    """Insert a link into the ingest queue (dedup by URL + parent_slug).

    Returns ``(ingest_id, created_new)``. The same URL queued for a different
    ``parent_slug`` creates a new row; the same pair only updates tags.

    ``parent_slug`` + ``media_kind`` link a media item (embedded image/video)
    back to the parent article it belongs to, so the nightly batch can append
    its reading/transcript to the parent's raw markdown and fire the parent's
    deferred synthesis once every sibling is done.
    """
    conn.row_factory = sqlite3.Row
    parent_key = _ingest_parent_slug(parent_slug)
    row = conn.execute(
        "SELECT id, status, attempts FROM ingest_queue WHERE url = ? AND COALESCE(parent_slug, '') = ?",
        (url, parent_key),
    ).fetchone()
    if row:
        serialized = serialize_tags(tags)
        with conn:
            if serialized:
                conn.execute(
                    "UPDATE ingest_queue SET tags = ? WHERE url = ? AND COALESCE(parent_slug, '') = ?",
                    (serialized, url, parent_key),
                )
            if row["status"] == "failed" and row["attempts"] < max_attempts:
                conn.execute(
                    "UPDATE ingest_queue SET status='queued', error=NULL, "
                    "updated_at=datetime('now') WHERE url = ? AND COALESCE(parent_slug, '') = ?",
                    (url, parent_key),
                )
        return row["id"], False
    import uuid
    ingest_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO ingest_queue (id, url, content_type, slug, status, tags, parent_slug, media_kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ingest_id, url, content_type, slug, status, serialize_tags(tags), parent_key, media_kind),
        )
    return ingest_id, True


def count_pending_media(conn: sqlite3.Connection, parent_slug: str) -> int:
    """Count media items not yet successfully processed (dispatched) for a parent.

    ``queued``, in-flight, AND ``failed`` items all count — a parent synthesizes
    ONLY once every sibling media item is ``dispatched``. A failed item therefore
    blocks synthesis until it is retried to success or removed.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ingest_queue WHERE COALESCE(parent_slug, '') = ? "
        "AND COALESCE(parent_slug, '') != '' AND status != 'dispatched'",
        (_ingest_parent_slug(parent_slug),),
    ).fetchone()
    return row["n"]


def is_synthesize_content_type(content_type: str) -> bool:
    """True when *content_type* is a deferred synthesize-only pipeline variant."""
    return (content_type or "").endswith("_synthesize")


_ACTIVE_TASK_STATUSES = (
    "'pending', 'in_progress', 'paused', 'done', 'completed'"
)


def find_active_task(
    conn: sqlite3.Connection, slug: str, content_type: str,
) -> str | None:
    """Return a task id when ``dispatch_synthesize`` must not enqueue another.

    Cross-run idempotency: any successful or in-flight synthesis for the slug
    is treated as a duplicate — including terminal ``done`` / ``completed``.
    For synthesize pipelines (``*_synthesize``), idempotency is slug-wide so
    ``substack_text_synthesize`` and ``substack_image_synthesize`` collide and
    a later parent retry cannot race a batch-dispatched synthesize task.
    Only ``failed`` is omitted so reconciliation or a later batch run may
    dispatch a retry; a done synthesis must never run twice.
    """
    conn.row_factory = sqlite3.Row
    if is_synthesize_content_type(content_type):
        row = conn.execute(
            "SELECT id FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
            f"AND status IN ({_ACTIVE_TASK_STATUSES}) "
            "LIMIT 1",
            (slug,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM tasks WHERE slug = ? AND content_type = ? "
            f"AND status IN ({_ACTIVE_TASK_STATUSES}) "
            "LIMIT 1",
            (slug, content_type),
        ).fetchone()
    return row["id"] if row else None


def has_active_synthesize_task(conn: sqlite3.Connection, slug: str) -> bool:
    """True when a deferred synthesize task for *slug* is active (not ``failed``)."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        f"AND status IN ({_ACTIVE_TASK_STATUSES}) "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    return row is not None


def recover_failed_media(conn: sqlite3.Connection, max_attempts: int = INGEST_MAX_MEDIA_ATTEMPTS) -> int:
    """Requeue failed media items for retry, up to ``max_attempts`` each.

    Returns the number requeued. After ``max_attempts`` retries an item stays
    ``failed`` and keeps blocking its parent's synthesis — the operator must
    remove it or fix the underlying failure.
    """
    with conn:
        cur = conn.execute(
            "UPDATE ingest_queue SET status='queued', error=NULL, "
            "attempts = attempts + 1, updated_at = datetime('now') "
            "WHERE status='failed' AND attempts < ?",
            (max_attempts,),
        )
        return cur.rowcount


def list_ingest_by_status(conn: sqlite3.Connection, status: str) -> list[dict]:
    """List ingest-queue rows in a given status, oldest first."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM ingest_queue WHERE status = ? ORDER BY created_at", (status,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_ingest_queue(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    """List all ingest-queue rows (any status), newest first — for the dashboard."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM ingest_queue ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def record_batch_run(conn: sqlite3.Connection) -> int:
    """Insert a new batch run (status='running') and return its id."""
    with conn:
        return conn.execute("INSERT INTO batch_runs (status) VALUES ('running')").lastrowid


def finish_batch_run(conn: sqlite3.Connection, run_id: int, status: str,
                     total: int, failed: int, dispatched: int, summary: str | None = None) -> None:
    """Mark a batch run complete with its outcome."""
    with conn:
        conn.execute(
            "UPDATE batch_runs SET status=?, total=?, failed=?, dispatched=?, "
            "summary=?, finished_at=datetime('now') WHERE id=?",
            (status, total, failed, dispatched, summary, run_id),
        )


def get_last_batch_run(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent batch run (for the dashboard)."""
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM batch_runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def update_ingest_status(conn: sqlite3.Connection, ingest_id: str, status: str,
                         error: str | None = None) -> None:
    """Transition an ingest-queue row to a new status (touches updated_at)."""
    with conn:
        if status == "dispatched":
            conn.execute(
                "UPDATE ingest_queue SET status = ?, error = ?, "
                "dispatched_at = datetime('now'), updated_at = datetime('now') WHERE id = ?",
                (status, error, ingest_id),
            )
        else:
            conn.execute(
                "UPDATE ingest_queue SET status = ?, error = ?, updated_at = datetime('now') WHERE id = ?",
                (status, error, ingest_id),
            )


def remove_ingest(conn: sqlite3.Connection, ingest_id: str) -> bool:
    """Delete an ingest-queue row (the dashboard's escape hatch). Returns True if removed."""
    with conn:
        return conn.execute("DELETE FROM ingest_queue WHERE id = ?", (ingest_id,)).rowcount > 0


def claim_ingest_items(conn: sqlite3.Connection) -> list[dict]:
    """Atomically claim every queued ingest item (queued -> downloading) and return them.

    One UPDATE ... RETURNING is a single SQLite write, so concurrent batch runs
    each claim a disjoint set — a second run grabs only what the first didn't.
    """
    conn.row_factory = sqlite3.Row
    with conn:
        rows = conn.execute(
            "UPDATE ingest_queue SET status='downloading', updated_at=datetime('now') "
            "WHERE status='queued' RETURNING *"
        ).fetchall()
    return [dict(r) for r in rows]


def recover_stale_ingest(
    conn: sqlite3.Connection,
    max_age_minutes: int = 120,
    max_attempts: int = INGEST_MAX_MEDIA_ATTEMPTS,
) -> int:
    """Reset in-flight items abandoned by a crashed batch back to queued.

    A crash mid-run leaves items in downloading/staged/transcribing/dispatching.
    Anything stuck in those states past max_age_minutes is stale. Each recovery
    increments ``attempts``; items that exceed ``max_attempts`` are marked
    ``failed`` instead of requeued forever.
    """
    stale_filter = (
        "status IN ('downloading','staged','transcribing','dispatching') "
        "AND updated_at IS NOT NULL AND updated_at < datetime('now', ?)"
    )
    age_param = (f"-{max_age_minutes} minutes",)
    with conn:
        conn.execute(
            "UPDATE ingest_queue SET status='failed', "
            "error='stale batch recovery exceeded max attempts', "
            "attempts = attempts + 1, updated_at = datetime('now') "
            f"WHERE {stale_filter} AND attempts + 1 >= ?",
            age_param + (max_attempts,),
        )
        cur = conn.execute(
            "UPDATE ingest_queue SET status='queued', "
            "error='recovered from stale batch run', "
            "attempts = attempts + 1, updated_at = datetime('now') "
            f"WHERE {stale_filter} AND attempts + 1 < ?",
            age_param + (max_attempts,),
        )
        return cur.rowcount


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
            WHERE id = (
                SELECT id FROM tasks WHERE status = 'pending'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
            )
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
    terminal = status in ("done", "failed", "paused")
    with conn:
        if status == "running":
            conn.execute(
                "UPDATE task_steps SET status = ?, output = ?, "
                "started_at = datetime('now'), finished_at = NULL WHERE id = ?",
                (status, output, step_id),
            )
        elif terminal:
            conn.execute(
                "UPDATE task_steps SET status = ?, output = ?, "
                "finished_at = datetime('now') WHERE id = ?",
                (status, output, step_id),
            )
        else:
            conn.execute(
                "UPDATE task_steps SET status = ?, output = ?, finished_at = NULL WHERE id = ?",
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

    Resets retries to 0 and sets status back to 'pending'. Tasks paused for
    deferred Substack media (error contains ``awaiting media``) are left paused
    so the consumer does not skip past ``substack_embedded_media`` to synthesis.

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
            "WHERE status = 'paused' AND error LIKE ? "
            "AND error NOT LIKE '%awaiting media%'",
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
