"""Database initialization — SQLite schemas for agent.db and logs.db."""

import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from agent.deferral_fsm import (
    REQUEUE_SUPERSEDE_GUARD,
    SUPERSEDE_DEFERRED_PARENT_REASON,
    SYNTHESIZE_IN_FLIGHT_STATUSES,
    deferral_cycle_cmp,
    embed_step_output_terminal_success,
    is_deferred_media_pause,
    is_superseded_parent_error,
    is_synthesize_content_type,
)
from agent.redact import url_for_display

_ALLOWED_INGEST_SCHEMES = frozenset({"http", "https"})

# Max ingest-queue retries for failed embedded/standalone media (``recover_failed_media``).
INGEST_MAX_MEDIA_ATTEMPTS = 3

_EMBEDDED_MEDIA_AWAITING_RE = re.compile(
    r"awaiting media:\s*(\d+)\s*images\s*\+\s*(\d+)\s*videos",
    re.IGNORECASE,
)

def _substack_cleanup_output_signals_deferred_media(
    conn: sqlite3.Connection,
    task_id: str,
    slug: str | None = None,
) -> bool:
    """True when ``substack_cleanup`` output carries a durable media-deferral marker."""
    row = conn.execute(
        """
        SELECT output FROM task_steps
        WHERE task_id = ? AND name = 'substack_cleanup'
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if not row or not row[0]:
        return False
    output = row[0]
    if slug and f"media_deferred:{slug}" in output:
        return True
    return is_deferred_media_pause(output)


def task_deferred_retry_preserves_embed_step(
    conn: sqlite3.Connection,
    task_id: str,
    slug: str | None,
    *,
    error: str | None = None,
) -> bool:
    """True when ``retry_task`` must not rewind ``substack_embedded_media``.

    Unlike :func:`task_in_deferred_media_cycle`, stale deferral rows and terminal-
    success embed output do not preserve the embed step — only open embed pause
    states (failed/paused step, or embed output matching a deferral pause marker).
    """
    if slug:
        deferral = get_parent_media_deferral(conn, slug)
        if deferral is not None and int(deferral.get("media_changed") or 0):
            return False
        if count_pending_media(conn, slug) > 0:
            return False

    row = conn.execute(
        """
        SELECT status, output FROM task_steps
        WHERE task_id = ? AND name = 'substack_embedded_media'
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        return False
    status, output = row[0], row[1]

    if embed_step_output_terminal_success(output):
        return False

    if status in ("failed", "paused"):
        return True
    if output and is_deferred_media_pause(output):
        return True
    return False


def task_in_deferred_media_cycle(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    slug: str | None = None,
    error: str | None = None,
) -> bool:
    """True when the parent task must stay in the deferred-media gate (retry/resume).

    Every durable deferral signal must be visible here: ``tasks.error``,
    ``parent_media_deferral``, ``substack_embedded_media`` step output, and the
    ``substack_cleanup`` ``media_deferred:`` marker from
    :func:`agent.pipelines.web._persist_parent_media_deferral`.
    """
    if is_deferred_media_pause(error):
        return True
    if slug and get_parent_media_deferral(conn, slug) is not None:
        return True
    row = conn.execute(
        """
        SELECT output FROM task_steps
        WHERE task_id = ? AND name = 'substack_embedded_media'
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row and is_deferred_media_pause(row[0]):
        return True
    if _substack_cleanup_output_signals_deferred_media(conn, task_id, slug):
        return True
    return False


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
    media_appended INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS parent_media_deferral (
    parent_slug TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    expected_total INTEGER NOT NULL,
    media_changed INTEGER NOT NULL DEFAULT 0,
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
    if "UNIQUE(url,parent_slug)" in ddl.replace(" ", ""):
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
            media_appended INTEGER NOT NULL DEFAULT 0,
            UNIQUE(url, parent_slug)
        );
        INSERT INTO ingest_queue_new (
            id, url, content_type, slug, tags, parent_slug, media_kind,
            status, error, attempts, created_at, dispatched_at, updated_at,
            media_appended
        )
        SELECT id, url, content_type, slug, tags,
               COALESCE(parent_slug, ''), media_kind,
               status, error, attempts, created_at, dispatched_at, updated_at,
               0
        FROM ingest_queue;
        DROP TABLE ingest_queue;
        ALTER TABLE ingest_queue_new RENAME TO ingest_queue;
        """
    )
    conn.commit()


_ACTIVE_SYNTHESIZE_SLUG_INDEX = "idx_tasks_one_active_synthesize_per_slug"
_ACTIVE_DELEGATE_SLUG_CONTENT_TYPE_INDEX = "idx_tasks_one_active_delegate_per_slug_ct"
_DELEGATE_IN_FLIGHT_SQL_STATUSES = SYNTHESIZE_IN_FLIGHT_STATUSES
_DELEGATE_CHILD_CONTENT_TYPES = frozenset({
    "book_epub",
    "book_mobi",
    "book_pdf",
    "book_scanned",
    "book_degraded",
})


def _migrate_dedupe_in_flight_synthesize_owners(conn: sqlite3.Connection) -> None:
    """Fail duplicate in-flight *_synthesize rows per slug before the unique index."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "tasks" not in tables:
        return
    slugs = conn.execute(
        f"""
        SELECT slug FROM tasks
        WHERE content_type GLOB '*_synthesize'
          AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES})
        GROUP BY slug HAVING COUNT(*) > 1
        """
    ).fetchall()
    supersede_error = "superseded: duplicate active synthesize owner"
    for row in slugs:
        slug = row[0]
        keep = conn.execute(
            f"""
            SELECT id FROM tasks
            WHERE slug = ? AND content_type GLOB '*_synthesize'
              AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES})
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (slug,),
        ).fetchone()
        if keep is None:
            continue
        conn.execute(
            f"""
            UPDATE tasks
            SET status = 'failed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type GLOB '*_synthesize'
              AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES})
              AND id != ?
            """,
            (supersede_error, slug, keep[0]),
        )


def is_delegate_content_type(content_type: str) -> bool:
    """True for mega/batch delegate child pipelines (one in-flight owner per slug+type)."""
    return (content_type or "") in _DELEGATE_CHILD_CONTENT_TYPES


def _delegate_content_type_sql_in() -> str:
    quoted = ", ".join(f"'{ct}'" for ct in sorted(_DELEGATE_CHILD_CONTENT_TYPES))
    return quoted


def _migrate_dedupe_in_flight_delegate_owners(conn: sqlite3.Connection) -> None:
    """Fail duplicate in-flight delegate child rows per (slug, content_type) before the index."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "tasks" not in tables:
        return
    ct_in = _delegate_content_type_sql_in()
    pairs = conn.execute(
        f"""
        SELECT slug, content_type FROM tasks
        WHERE content_type IN ({ct_in})
          AND status IN ({_DELEGATE_IN_FLIGHT_SQL_STATUSES})
        GROUP BY slug, content_type HAVING COUNT(*) > 1
        """
    ).fetchall()
    supersede_error = "superseded: duplicate active delegate child"
    for slug, content_type in pairs:
        keep = conn.execute(
            f"""
            SELECT id FROM tasks
            WHERE slug = ? AND content_type = ?
              AND status IN ({_DELEGATE_IN_FLIGHT_SQL_STATUSES})
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (slug, content_type),
        ).fetchone()
        if keep is None:
            continue
        conn.execute(
            f"""
            UPDATE tasks
            SET status = 'failed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type = ?
              AND status IN ({_DELEGATE_IN_FLIGHT_SQL_STATUSES})
              AND id != ?
            """,
            (supersede_error, slug, content_type, keep[0]),
        )


def _migrate_active_delegate_slug_content_type_unique(conn: sqlite3.Connection) -> None:
    """Enforce exactly one in-flight delegate child per (slug, content_type)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "tasks" not in tables:
        return
    existing = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?",
        (_ACTIVE_DELEGATE_SLUG_CONTENT_TYPE_INDEX,),
    ).fetchone()
    if existing:
        return
    _migrate_dedupe_in_flight_delegate_owners(conn)
    ct_in = _delegate_content_type_sql_in()
    conn.execute(
        f"""
        CREATE UNIQUE INDEX {_ACTIVE_DELEGATE_SLUG_CONTENT_TYPE_INDEX}
        ON tasks(slug, content_type)
        WHERE content_type IN ({ct_in})
          AND status IN ({_DELEGATE_IN_FLIGHT_SQL_STATUSES})
        """
    )


def _migrate_active_synthesize_slug_unique(conn: sqlite3.Connection) -> None:
    """Enforce exactly one in-flight *_synthesize task row per slug (DB invariant)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "tasks" not in tables:
        return
    existing = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?",
        (_ACTIVE_SYNTHESIZE_SLUG_INDEX,),
    ).fetchone()
    if existing:
        return
    _migrate_dedupe_in_flight_synthesize_owners(conn)
    conn.execute(
        f"""
        CREATE UNIQUE INDEX {_ACTIVE_SYNTHESIZE_SLUG_INDEX}
        ON tasks(slug)
        WHERE content_type GLOB '*_synthesize'
          AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES})
        """
    )


def _migrate_step_names_minio_to_storage(conn: sqlite3.Connection) -> None:
    """Rename legacy storage-era task_step names to storage-agnostic names (idempotent)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "task_steps" not in tables:
        return
    legacy_upload = "min" + "io_upload"
    legacy_ig = "ig_" + legacy_upload
    legacy_threads = "threads_" + legacy_upload
    conn.execute(
        "UPDATE task_steps SET name='storage_upload' WHERE name=?",
        (legacy_upload,),
    )
    conn.execute(
        "UPDATE task_steps SET name='ig_storage_upload' WHERE name=?",
        (legacy_ig,),
    )
    conn.execute(
        "UPDATE task_steps SET name='threads_storage_upload' WHERE name=?",
        (legacy_threads,),
    )


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
    _migrate_active_synthesize_slug_unique(conn)
    _migrate_active_delegate_slug_content_type_unique(conn)
    _migrate_step_names_minio_to_storage(conn)
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


def _insert_task_with_steps(
    conn: sqlite3.Connection,
    task_id: str,
    url: str,
    content_type: str,
    slug: str,
    steps: list[str],
    *,
    status: str = "pending",
    tags=None,
) -> None:
    conn.execute(
        "INSERT INTO tasks (id, source_url, content_type, slug, status, tags) VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, url, content_type, slug, status, serialize_tags(tags)),
    )
    for i, step_name in enumerate(steps):
        conn.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status) VALUES (?, ?, ?, 'pending')",
            (task_id, i, step_name),
        )


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
        _insert_task_with_steps(
            conn, task_id, url, content_type, slug, steps, status=status, tags=tags,
        )
    return task_id


def create_synthesize_task(
    conn: sqlite3.Connection,
    url: str,
    content_type: str,
    slug: str,
    steps: list[str],
    tags=None,
    *,
    temp_dir: Path | None = None,
) -> tuple[str, bool]:
    """Atomically ensure at most one in-flight *_synthesize owner per slug.

    Uses ``BEGIN IMMEDIATE`` plus a partial unique index on in-flight synthesize
    rows. ``find_active_task`` semantics (deferral generation, oldest-row
    preference) are preserved.

    Returns:
        ``(task_id, created_new)``. When ``created_new`` is False, *task_id* is
        the existing owner and dispatch is a duplicate.
    """
    import uuid

    if not is_synthesize_content_type(content_type):
        return create_task(conn, url, content_type, slug, steps, tags=tags), True

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing_id = find_active_task(
            conn, slug, content_type, temp_dir=temp_dir,
        )
        if existing_id:
            conn.commit()
            return existing_id, False

        task_id = str(uuid.uuid4())
        _insert_task_with_steps(
            conn, task_id, url, content_type, slug, steps, status="pending", tags=tags,
        )
        conn.commit()
        return task_id, True
    except sqlite3.IntegrityError:
        conn.rollback()
        existing_id = find_active_task(
            conn, slug, content_type, temp_dir=temp_dir,
        )
        if existing_id:
            return existing_id, False
        raise


def _reset_delegate_child_task(
    conn: sqlite3.Connection, task_id: str, file_path: str,
) -> None:
    """Re-queue a delegate child for ingestion (failed terminal or source generation change)."""
    conn.execute(
        """
        UPDATE task_steps
        SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
        WHERE task_id = ?
        """,
        (task_id,),
    )
    conn.execute(
        """
        UPDATE tasks
        SET source_url = ?, status = 'pending', error = NULL, retries = 0,
            updated_at = datetime('now')
        WHERE id = ?
        """,
        (file_path, task_id),
    )


def _delegate_existing_or_reset(
    conn: sqlite3.Connection,
    existing_id: str,
    file_path: str,
) -> str:
    row = conn.execute(
        "SELECT status, source_url FROM tasks WHERE id = ?",
        (existing_id,),
    ).fetchone()
    status = row[0]
    src = row[1]
    same_source = Path(src or "").resolve() == Path(file_path).resolve()
    if status in ("done", "completed"):
        if same_source:
            return existing_id
        _reset_delegate_child_task(conn, existing_id, file_path)
        return existing_id
    if not same_source:
        conn.execute(
            "UPDATE tasks SET source_url = ?, updated_at = datetime('now') WHERE id = ?",
            (file_path, existing_id),
        )
    return existing_id


def create_delegate_task(
    conn: sqlite3.Connection,
    file_path: str,
    content_type: str,
    slug: str,
    steps: list[str],
    tags=None,
) -> str:
    """Atomically resolve a delegate child task with at most one in-flight owner.

    Uses ``BEGIN IMMEDIATE`` and a partial unique index on in-flight delegate
    child rows for ``(slug, content_type)``. Preserves failed reuse and source
    generation change handling for book pipeline delegates.
    """
    import uuid

    if not is_delegate_content_type(content_type):
        return create_task(conn, file_path, content_type, slug, steps, tags=tags)

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing_id = find_active_task(conn, slug, content_type)
        if existing_id:
            task_id = _delegate_existing_or_reset(conn, existing_id, file_path)
            conn.commit()
            return task_id

        failed_row = conn.execute(
            "SELECT id FROM tasks WHERE slug = ? AND content_type = ? AND status = 'failed' "
            "ORDER BY created_at ASC, id ASC LIMIT 1",
            (slug, content_type),
        ).fetchone()
        if failed_row:
            task_id = failed_row[0]
            _reset_delegate_child_task(conn, task_id, file_path)
            conn.commit()
            return task_id

        task_id = str(uuid.uuid4())
        _insert_task_with_steps(
            conn, task_id, file_path, content_type, slug, steps, status="pending", tags=tags,
        )
        conn.commit()
        return task_id
    except sqlite3.IntegrityError:
        conn.rollback()
        existing_id = find_active_task(conn, slug, content_type)
        if existing_id:
            return _delegate_existing_or_reset(conn, existing_id, file_path)
        raise


def _reset_dispatched_parent_siblings_if_deferral_cycle(
    conn: sqlite3.Connection, parent_slug: str,
) -> None:
    """Re-queue dispatched siblings when the parent deferral cycle is open (``media_changed``)."""
    if not parent_slug:
        return
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None or not int(deferral.get("media_changed") or 0):
        return
    supersede_stale_deferral_synthesize_tasks(conn, parent_slug, deferral)
    with conn:
        conn.execute(
            """
            UPDATE ingest_queue
            SET status = 'queued', media_appended = 0, error = NULL,
                dispatched_at = NULL, updated_at = datetime('now')
            WHERE parent_slug = ? AND status = 'dispatched'
            """,
            (parent_slug,),
        )


def _reject_non_http_ingest_url(url: str) -> None:
    """Ingest queue URLs are rendered as dashboard hrefs — only http(s) may be stored."""
    scheme = urlparse((url or "").strip()).scheme.lower()
    if scheme not in _ALLOWED_INGEST_SCHEMES:
        raise ValueError("URL must use http or https")


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
    _reject_non_http_ingest_url(url)
    conn.row_factory = sqlite3.Row
    parent_key = _ingest_parent_slug(parent_slug)
    if parent_key:
        _reset_dispatched_parent_siblings_if_deferral_cycle(conn, parent_slug)
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
                    "attempts = attempts + 1, updated_at=datetime('now') "
                    "WHERE url = ? AND COALESCE(parent_slug, '') = ?",
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
    if parent_slug:
        _reconcile_parent_media_deferral_on_new_sibling(conn, parent_slug)
    return ingest_id, True


def _embedded_media_high_water_hint(
    conn: sqlite3.Connection,
    parent_slug: str,
    temp_dir: Path | None,
) -> int:
    """Largest persisted embedded-media total from manifest or parent task-step output."""
    if not parent_slug:
        return 0
    hints: list[int] = []
    if temp_dir is not None:
        manifest_path = temp_dir / f"{parent_slug}_media.json"
        if manifest_path.is_file():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                total = data.get("total")
                if total is not None:
                    hints.append(int(total))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
    rows = conn.execute(
        """
        SELECT ts.output FROM task_steps ts
        JOIN tasks t ON t.id = ts.task_id
        WHERE t.slug = ? AND ts.name = 'substack_embedded_media' AND ts.output IS NOT NULL
        """,
        (parent_slug,),
    ).fetchall()
    for row in rows:
        out = row[0] if not hasattr(row, "keys") else row["output"]
        if not out:
            continue
        match = _EMBEDDED_MEDIA_AWAITING_RE.search(out)
        if match:
            hints.append(int(match.group(1)) + int(match.group(2)))
    return max(hints) if hints else 0


def _lookup_parent_article_metadata(
    conn: sqlite3.Connection,
    parent_slug: str,
) -> tuple[str | None, str | None]:
    """Resolve source_url and content_type for a parent article slug."""
    if not parent_slug:
        return None, None
    conn.row_factory = sqlite3.Row
    parent_row = conn.execute(
        "SELECT source_url, content_type FROM tasks "
        "WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
        "ORDER BY created_at ASC, id ASC LIMIT 1",
        (parent_slug,),
    ).fetchone()
    source_url = None
    content_type = None
    if parent_row:
        source_url = parent_row["source_url"]
        content_type = parent_row["content_type"]
    if not source_url or not content_type:
        syn_row = conn.execute(
            "SELECT source_url, content_type FROM tasks "
            "WHERE slug = ? AND content_type GLOB '*_synthesize' "
            "ORDER BY created_at ASC, id ASC LIMIT 1",
            (parent_slug,),
        ).fetchone()
        if syn_row:
            source_url = source_url or syn_row["source_url"]
            syn_ct = syn_row["content_type"]
            if syn_ct and is_synthesize_content_type(syn_ct):
                content_type = content_type or syn_ct[: -len("_synthesize")]
            else:
                content_type = content_type or syn_ct
    return source_url, content_type


def _mark_parent_media_enqueue_shortfall(
    conn: sqlite3.Connection,
    parent_slug: str,
    sibling_n: int,
    expected_total: int,
    *,
    expected_grew: bool = False,
) -> None:
    """Set-only shortfall marker: raise ``media_changed`` when siblings lag or expected grew.

    Invariant: never clear ``media_changed`` here — only
    :func:`_clear_parent_media_changed_if_satisfied` may consume the flag after raw verification.
    """
    if not (sibling_n < expected_total or expected_grew):
        return
    with conn:
        conn.execute(
            """
            UPDATE parent_media_deferral
            SET media_changed = MAX(media_changed, 1), updated_at = datetime('now')
            WHERE parent_slug = ?
            """,
            (parent_slug,),
        )


def _clear_parent_media_changed_if_satisfied(
    conn: sqlite3.Connection,
    parent_slug: str,
    *,
    raw_path: Path | None = None,
) -> None:
    """Clear consumed ``media_changed`` once sibling count and landed raw match expected_total."""
    if not parent_slug:
        return
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None or not int(deferral.get("media_changed") or 0):
        return
    sibling_n = _parent_sibling_media_count(conn, parent_slug)
    expected = int(deferral["expected_total"])
    if sibling_n < expected:
        return
    if not parent_media_complete(conn, parent_slug, raw_path=raw_path):
        return
    with conn:
        conn.execute(
            """
            UPDATE parent_media_deferral
            SET media_changed = 0
            WHERE parent_slug = ?
            """,
            (parent_slug,),
        )


def _reconcile_parent_media_deferral_on_new_sibling(
    conn: sqlite3.Connection,
    parent_slug: str,
    *,
    temp_dir: Path | None = None,
) -> None:
    """Grow expected_total when a sibling ingest row appears outside embedded_media / manifest sync."""
    if not parent_slug:
        return
    sibling_n = _parent_sibling_media_count(conn, parent_slug)
    high_water = _embedded_media_high_water_hint(conn, parent_slug, temp_dir)
    deferral = get_parent_media_deferral(conn, parent_slug)
    prior_expected = int(deferral["expected_total"]) if deferral else 0
    expected = max(prior_expected, sibling_n, high_water)
    if expected == 0:
        return
    source_url = deferral["source_url"] if deferral else None
    content_type = deferral["content_type"] if deferral else None
    if not source_url or not content_type:
        source_url, content_type = _lookup_parent_article_metadata(conn, parent_slug)
    if not source_url or not content_type:
        return
    grew = expected > prior_expected
    ensure_parent_media_deferral(
        conn, parent_slug, source_url, content_type, expected,
    )
    _mark_parent_media_enqueue_shortfall(
        conn, parent_slug, sibling_n, expected, expected_grew=grew,
    )


def _reconcile_parent_media_deferral_on_remove_sibling(
    conn: sqlite3.Connection,
    parent_slug: str,
    *,
    temp_dir: Path | None = None,
) -> None:
    """Lower expected_total after a sibling ingest row is deleted (monotonic down only).

    Recomputes from the remaining sibling COUNT(*), capped by extracted/manifest
    high-water so expected_total never exceeds what embedding already committed.
    """
    if not parent_slug:
        return
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        return
    prior_expected = int(deferral["expected_total"])
    sibling_n = _parent_sibling_media_count(conn, parent_slug)
    high_water = _embedded_media_high_water_hint(conn, parent_slug, temp_dir)
    reconciled = min(prior_expected, sibling_n)
    if high_water > 0:
        reconciled = min(reconciled, high_water)
    if reconciled >= prior_expected:
        return
    with conn:
        conn.execute(
            """
            UPDATE parent_media_deferral
            SET expected_total = ?, updated_at = datetime('now')
            WHERE parent_slug = ?
            """,
            (reconciled, parent_slug),
        )
    _mark_parent_media_enqueue_shortfall(
        conn, parent_slug, sibling_n, reconciled,
    )


def _parent_sibling_media_count(conn: sqlite3.Connection, parent_slug: str) -> int:
    """Count ingest-queue sibling rows linked to *parent_slug* (any status)."""
    if not parent_slug:
        return 0
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
        (parent_slug,),
    ).fetchone()
    return int(row["n"] if hasattr(row, "keys") else row[0])


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


def normalize_deferral_parent_content_type(content_type: str) -> str:
    """Map legacy bare ``substack`` parent rows to ``substack_text`` for deferral/dispatch."""
    ct = (content_type or "").strip()
    if ct == "substack":
        return "substack_text"
    return ct


_ACTIVE_TASK_STATUSES = (
    "'pending', 'in_progress', 'paused', 'done', 'completed'"
)


def find_active_task(
    conn: sqlite3.Connection,
    slug: str,
    content_type: str,
    *,
    temp_dir: Path | None = None,
) -> str | None:
    """Return a task id when ``dispatch_synthesize`` must not enqueue another.

    In-flight work is always deduplicated. For ``*_synthesize``, terminal
    ``done`` / ``completed`` rows count only when they belong to the current
    deferral generation; stale terminals are superseded so a reconciled parent
    may dispatch synthesis again. Idempotency is slug-wide across synthesize
    pipelines. Only ``failed`` is omitted from duplicate detection.
    """
    conn.row_factory = sqlite3.Row
    if is_synthesize_content_type(content_type):
        row = conn.execute(
            "SELECT id, status FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
            f"AND status IN ({_ACTIVE_TASK_STATUSES}) "
            "ORDER BY created_at ASC, id ASC "
            "LIMIT 1",
            (slug,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, status FROM tasks WHERE slug = ? AND content_type = ? "
            f"AND status IN ({_ACTIVE_TASK_STATUSES}) "
            "ORDER BY created_at ASC, id ASC "
            "LIMIT 1",
            (slug, content_type),
        ).fetchone()
    if row is None:
        return None
    task_id = row["id"] if hasattr(row, "keys") else row[0]
    status = row["status"] if hasattr(row, "keys") else row[1]
    if is_synthesize_content_type(content_type):
        deferral = get_parent_media_deferral(conn, slug)
        if deferral is None and _parent_sibling_media_count(conn, slug) > 0:
            _supersede_synthesize_tasks_orphan_deferral(conn, slug)
            return None
        if deferral is not None:
            if status in ("done", "completed"):
                raw_path = (temp_dir / f"{slug}_raw.md") if temp_dir is not None else None
                _clear_parent_media_changed_if_satisfied(
                    conn, slug, raw_path=raw_path,
                )
                deferral = get_parent_media_deferral(conn, slug)
                fresh = _deferral_synthesize_terminal_fresh(conn, slug, deferral)
                if int(deferral.get("media_changed") or 0) or not fresh:
                    supersede_stale_deferral_synthesize_tasks(conn, slug, deferral)
                    return None
            elif status in ("pending", "in_progress", "paused"):
                if not _deferral_synthesize_task_fresh(conn, task_id, deferral):
                    supersede_stale_deferral_synthesize_tasks(conn, slug, deferral)
                    return None
    return task_id


def may_reset_parent_task(
    conn: sqlite3.Connection,
    task_id: str,
    slug: str | None,
    *,
    error: str | None = None,
    content_type: str | None = None,
) -> bool:
    """Shared gate for parent re-queue paths (orphan recovery, retry, bulk resume).

    Returns False when the row is terminal-superseded or an in-flight synthesize owner
    blocks resetting a non-synthesize parent for the same slug.
    """
    if error is None or content_type is None:
        row = conn.execute(
            "SELECT error, content_type FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return False
        if error is None:
            error = row[0]
        if content_type is None:
            content_type = row[1]
    if is_superseded_parent_error(error):
        return False
    if slug and content_type and not is_synthesize_content_type(content_type):
        if has_active_synthesize_task(conn, slug):
            return False
    return True


def requeue_task_for_retry_cas(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    allowed_statuses: tuple[str, ...] = ("failed", "paused"),
) -> bool:
    """Compare-and-swap task re-queue; refuses superseded terminal rows."""
    if not allowed_statuses:
        return False
    status_sql = ", ".join(f"'{s}'" for s in allowed_statuses)
    cur = conn.execute(
        f"""
        UPDATE tasks
        SET status = 'pending', error = NULL, retries = 0, updated_at = datetime('now')
        WHERE id = ?
          AND status IN ({status_sql})
          {REQUEUE_SUPERSEDE_GUARD}
        """,
        (task_id,),
    )
    return cur.rowcount == 1


def has_active_synthesize_task(conn: sqlite3.Connection, slug: str) -> bool:
    """True when a generation-fresh deferred synthesize task is in flight (not terminal)."""
    conn.row_factory = sqlite3.Row
    deferral = get_parent_media_deferral(conn, slug)
    if deferral is not None:
        anchor = deferral["updated_at"]
        op = deferral_cycle_cmp(deferral)
        row = conn.execute(
            f"SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
            f"AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES}) "
            f"AND datetime(updated_at) {op} datetime(?) "
            "ORDER BY created_at ASC, id ASC "
            "LIMIT 1",
            (slug, anchor),
        ).fetchone()
        return row is not None
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        f"AND status IN ({SYNTHESIZE_IN_FLIGHT_STATUSES}) "
        "ORDER BY created_at ASC, id ASC "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    return row is not None


def has_completed_synthesize_task(conn: sqlite3.Connection, slug: str) -> bool:
    """True when a generation-fresh *_synthesize task finished with ``done`` / ``completed``."""
    deferral = get_parent_media_deferral(conn, slug)
    if deferral is not None:
        return _deferral_synthesize_terminal_fresh(conn, slug, deferral)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') {REQUEUE_SUPERSEDE_GUARD} "
        f"{_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    return row is not None


_TERMINAL_PROBE_ORDER = "ORDER BY created_at ASC, id ASC"


def _deferral_terminal_parent_fresh(conn: sqlite3.Connection, slug: str, deferral: dict) -> bool:
    """Inline parent synthesis finished in the current deferral generation."""
    conn.row_factory = sqlite3.Row
    anchor = deferral["updated_at"]
    op = deferral_cycle_cmp(deferral)
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') AND datetime(updated_at) {op} datetime(?) "
        f"{REQUEUE_SUPERSEDE_GUARD} {_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug, anchor),
    ).fetchone()
    return row is not None


def _deferral_synthesize_terminal_fresh(conn: sqlite3.Connection, slug: str, deferral: dict) -> bool:
    """Batch *_synthesize finished in the current deferral generation."""
    conn.row_factory = sqlite3.Row
    anchor = deferral["updated_at"]
    op = deferral_cycle_cmp(deferral)
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') AND datetime(updated_at) {op} datetime(?) "
        f"{REQUEUE_SUPERSEDE_GUARD} {_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug, anchor),
    ).fetchone()
    return row is not None


def _deferral_synthesize_task_fresh(
    conn: sqlite3.Connection, task_id: str, deferral: dict,
) -> bool:
    """True when a *_synthesize task row belongs to the current deferral generation."""
    conn.row_factory = sqlite3.Row
    anchor = deferral["updated_at"]
    op = deferral_cycle_cmp(deferral)
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE id = ? AND datetime(updated_at) {op} datetime(?)",
        (task_id, anchor),
    ).fetchone()
    return row is not None


_MEDIA_LANDING_ANCHOR_ROW_FILTER = (
    "(status = 'dispatched' OR (dispatched_at IS NOT NULL AND status = 'failed'))"
)


def _deferral_media_landing_anchor(
    conn: sqlite3.Connection, slug: str, deferral: dict,
) -> str:
    """Latest sibling ``dispatched_at`` or deferral ``updated_at``, whichever is later."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"SELECT MAX(dispatched_at) AS latest FROM ingest_queue "
        f"WHERE parent_slug = ? AND {_MEDIA_LANDING_ANCHOR_ROW_FILTER}",
        (slug,),
    ).fetchone()
    latest = None
    if row is not None:
        latest = row["latest"] if hasattr(row, "keys") else row[0]
    anchor = deferral["updated_at"]
    if not latest:
        return anchor
    picked = conn.execute(
        "SELECT CASE WHEN datetime(?) >= datetime(?) THEN ? ELSE ? END",
        (latest, anchor, latest, anchor),
    ).fetchone()
    return picked[0] if picked else anchor


def _parent_terminal_after_anchor(conn: sqlite3.Connection, slug: str, anchor: str) -> bool:
    conn.row_factory = sqlite3.Row
    op = deferral_cycle_cmp({})
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') AND datetime(updated_at) {op} datetime(?) "
        f"{REQUEUE_SUPERSEDE_GUARD} {_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug, anchor),
    ).fetchone()
    return row is not None


def _synthesize_terminal_after_anchor(conn: sqlite3.Connection, slug: str, anchor: str) -> bool:
    conn.row_factory = sqlite3.Row
    op = deferral_cycle_cmp({})
    row = conn.execute(
        f"SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') AND datetime(updated_at) {op} datetime(?) "
        f"{REQUEUE_SUPERSEDE_GUARD} {_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug, anchor),
    ).fetchone()
    return row is not None


def _has_stale_deferral_synthesize_terminal(
    conn: sqlite3.Connection, slug: str, anchor: str,
) -> bool:
    """Terminal *_synthesize from before media landed for this cycle (vault shortcut hazard)."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        "AND status IN ('done', 'completed') AND datetime(updated_at) <= datetime(?) "
        f"{REQUEUE_SUPERSEDE_GUARD} {_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug, anchor),
    ).fetchone()
    return row is not None


def _supersede_synthesize_tasks_orphan_deferral(
    conn: sqlite3.Connection, slug: str,
) -> int:
    """Fail *_synthesize rows when siblings exist but the deferral record is missing."""
    if not slug:
        return 0
    with conn:
        cur = conn.execute(
            """
            UPDATE tasks SET status = 'failed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type GLOB '*_synthesize'
            AND status IN ('pending', 'in_progress', 'paused', 'done', 'completed')
            """,
            ("superseded: parent deferral record missing", slug),
        )
        return cur.rowcount


def supersede_stale_deferral_synthesize_tasks(
    conn: sqlite3.Connection, slug: str, deferral: dict,
) -> int:
    """Fail prior-generation *_synthesize rows (any status) so batch may re-dispatch."""
    if not slug or deferral is None:
        return 0
    anchor = deferral["updated_at"]
    with conn:
        cur = conn.execute(
            """
            UPDATE tasks SET status = 'failed', error = ?, updated_at = datetime('now')
            WHERE slug = ? AND content_type GLOB '*_synthesize'
            AND status IN ('pending', 'in_progress', 'paused', 'done', 'completed')
            AND datetime(updated_at) < datetime(?)
            """,
            ("superseded by deferral media cycle", slug, anchor),
        )
        return cur.rowcount


def has_synthesize_task_signal(conn: sqlite3.Connection, slug: str) -> bool:
    """True when generation-fresh batch synthesis is in flight or finished for the deferral cycle."""
    return has_active_synthesize_task(conn, slug) or has_completed_synthesize_task(conn, slug)


def _slug_has_media_manifest(temp_dir: Path | None, slug: str) -> bool:
    if temp_dir is None:
        return False
    return (temp_dir / f"{slug}_media.json").is_file()


def _slug_has_embedded_media_context(
    conn: sqlite3.Connection, slug: str, temp_dir: Path | None,
) -> bool:
    """True when embedded siblings or an on-disk media manifest imply deferred batch synthesis."""
    return (
        _parent_sibling_media_count(conn, slug) > 0
        or _slug_has_media_manifest(temp_dir, slug)
    )


def _non_superseded_parent_terminal(conn: sqlite3.Connection, slug: str) -> bool:
    """Terminal non-synthesize parent row that was not superseded by batch synthesis."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
        f"AND status IN ('done', 'completed') {REQUEUE_SUPERSEDE_GUARD} "
        f"{_TERMINAL_PROBE_ORDER} "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    return row is not None


def _embedded_media_landing_anchor(conn: sqlite3.Connection, slug: str) -> str | None:
    """Latest sibling ``dispatched_at`` when embedded media rows exist."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"SELECT MAX(dispatched_at) AS latest FROM ingest_queue "
        f"WHERE parent_slug = ? AND {_MEDIA_LANDING_ANCHOR_ROW_FILTER}",
        (slug,),
    ).fetchone()
    if row is None:
        return None
    latest = row["latest"] if hasattr(row, "keys") else row[0]
    return latest or None


def _embedded_context_synthesis_complete(
    conn: sqlite3.Connection,
    slug: str,
    *,
    temp_dir: Path | None = None,
) -> bool:
    """Generation-fresh completion when embedded media implies batch-owned synthesis."""
    anchor = _embedded_media_landing_anchor(conn, slug)
    if anchor:
        if _synthesize_terminal_after_anchor(conn, slug, anchor):
            return True
        if _parent_terminal_after_anchor(conn, slug, anchor):
            return True
        return False
    if _slug_has_embedded_media_context(conn, slug, temp_dir):
        return False
    if has_completed_synthesize_task(conn, slug):
        return True
    return False


def _parent_synthesis_media_complete(
    conn: sqlite3.Connection,
    slug: str,
    *,
    deferral: dict | None,
    raw_path: Path | None,
    temp_dir: Path | None = None,
) -> bool:
    """Fail-closed gate: every sibling dispatched and media verifiably landed."""
    if count_pending_media(conn, slug) > 0:
        return False
    if deferral is not None:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (slug,),
        ).fetchone()
        actual = row["n"] if hasattr(row, "keys") else row[0]
        if actual != deferral["expected_total"]:
            return False
        if raw_path is None:
            return False
        return parent_media_complete(conn, slug, raw_path=raw_path)
    if _parent_sibling_media_count(conn, slug) > 0:
        if raw_path is None:
            return False
        return parent_media_complete(conn, slug, raw_path=raw_path)
    if _slug_has_media_manifest(temp_dir, slug):
        if temp_dir is not None:
            bootstrap_parent_media_deferral_from_manifest(conn, slug, temp_dir)
        deferral = get_parent_media_deferral(conn, slug)
        if deferral is None:
            return False
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (slug,),
        ).fetchone()
        actual = row["n"] if hasattr(row, "keys") else row[0]
        if actual != deferral["expected_total"]:
            return False
        if raw_path is None:
            return False
        return parent_media_complete(conn, slug, raw_path=raw_path)
    return True


def has_completed_parent_synthesis(
    conn: sqlite3.Connection,
    slug: str,
    *,
    vault_path: Path | None = None,
    temp_dir: Path | None = None,
) -> bool:
    """True when synthesis for *slug* already finished for the active deferral generation.

    Used to make parent inline synthesis and batch ``*_synthesize`` dispatch
    mutually exclusive. Terminal parent rows superseded by batch, stale
    ``*_synthesize`` tasks, and vault notes from prior cycles do not count.

    Monotonic invariant: pending or unverified media always yields False before
    any terminal-task or vault freshness check.
    """
    conn.row_factory = sqlite3.Row
    raw_path = (temp_dir / f"{slug}_raw.md") if temp_dir is not None else None
    deferral = get_parent_media_deferral(conn, slug)
    embedded = _slug_has_embedded_media_context(conn, slug, temp_dir)

    if deferral is not None or embedded:
        if not _parent_synthesis_media_complete(
            conn,
            slug,
            deferral=deferral,
            raw_path=raw_path,
            temp_dir=temp_dir,
        ):
            return False

        _clear_parent_media_changed_if_satisfied(conn, slug, raw_path=raw_path)
        deferral = get_parent_media_deferral(conn, slug)

        if deferral is not None:
            if int(deferral.get("media_changed") or 0):
                return False
            media_anchor = _deferral_media_landing_anchor(conn, slug, deferral)
            if _deferral_synthesize_terminal_fresh(conn, slug, deferral):
                return True
            if (
                _deferral_terminal_parent_fresh(conn, slug, deferral)
                and _parent_terminal_after_anchor(conn, slug, media_anchor)
            ):
                return True
            if vault_path is None:
                return False
            note = vault_path / "Knowledge base" / "articles" / f"{slug}.md"
            if not note.is_file():
                return False
            if _parent_terminal_after_anchor(conn, slug, media_anchor):
                return True
            if _synthesize_terminal_after_anchor(conn, slug, media_anchor):
                return True
            if _has_stale_deferral_synthesize_terminal(conn, slug, media_anchor):
                return False
            if int(deferral.get("expected_total") or 0) == 0:
                if raw_path is not None and raw_path.is_file():
                    if note.stat().st_mtime >= raw_path.stat().st_mtime:
                        return True
            return False

        return _embedded_context_synthesis_complete(conn, slug, temp_dir=temp_dir)

    if _non_superseded_parent_terminal(conn, slug):
        return True
    if has_completed_synthesize_task(conn, slug):
        return True
    return False


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


def _persist_visible_error(message: str) -> str:
    """Scrub durable task/ingest error text (boundary4: URLs + credentials, no display path scrub)."""
    from agent.persist import scrub_durable_text

    return scrub_durable_text(message, policy="boundary4")


def finish_batch_run(conn: sqlite3.Connection, run_id: int, status: str,
                     total: int, failed: int, dispatched: int, summary: str | None = None) -> None:
    """Mark a batch run complete with its outcome."""
    if summary is not None:
        summary = _persist_visible_error(summary)
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


def set_ingest_media_appended(conn: sqlite3.Connection, ingest_id: str) -> None:
    """Record that this ingest row's media block was appended to the parent raw file."""
    with conn:
        conn.execute(
            "UPDATE ingest_queue SET media_appended = 1, updated_at = datetime('now') WHERE id = ?",
            (ingest_id,),
        )


def media_append_marker(ingest_id: str) -> str:
    """HTML comment embedded in parent raw markdown for crash-safe append idempotency."""
    return f"<!-- kb-ingest-media:{ingest_id} -->"


def _sibling_media_block_has_body(text: str, ingest_id: str, sibling_ids: list[str]) -> bool:
    """True when the raw segment for *ingest_id* includes a section heading and body."""
    marker = media_append_marker(ingest_id)
    pos = text.find(marker)
    if pos < 0:
        return False
    start = 0
    for sid in sibling_ids:
        if sid == ingest_id:
            break
        prior = media_append_marker(sid)
        prior_pos = text.find(prior)
        if prior_pos >= 0 and prior_pos < pos:
            start = prior_pos + len(prior)
    segment = text[start:pos].strip()
    return bool(segment) and "##" in segment


def _sibling_media_block_bounds(
    text: str, ingest_id: str, sibling_ids: list[str],
) -> tuple[int, int] | None:
    """Inclusive-exclusive slice ``[start:end)`` covering one sibling's media block and marker."""
    marker = media_append_marker(ingest_id)
    pos = text.find(marker)
    if pos < 0:
        return None
    start = 0
    for sid in sibling_ids:
        if sid == ingest_id:
            break
        prior = media_append_marker(sid)
        prior_pos = text.find(prior)
        if prior_pos >= 0 and prior_pos < pos:
            start = prior_pos + len(prior)
    end = pos + len(marker)
    while end < len(text) and text[end] in "\n\r":
        end += 1
    while start > 0 and text[start - 1] in "\n\r":
        start -= 1
    return start, end


def _strip_sibling_media_block(text: str, ingest_id: str, sibling_ids: list[str]) -> str:
    """Remove a marker-delimited sibling media block (heading, body, marker) from raw text."""
    bounds = _sibling_media_block_bounds(text, ingest_id, sibling_ids)
    if bounds is None:
        return text
    start, end = bounds
    return text[:start] + text[end:]


def _ordered_sibling_ids_for_ingest(conn: sqlite3.Connection, ingest_id: str) -> list[str]:
    row = conn.execute(
        "SELECT parent_slug FROM ingest_queue WHERE id = ?", (ingest_id,),
    ).fetchone()
    if row is None:
        return []
    parent_slug = row["parent_slug"] if hasattr(row, "keys") else row[0]
    if not parent_slug:
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id FROM ingest_queue WHERE parent_slug = ? ORDER BY created_at ASC, id ASC",
        (parent_slug,),
    ).fetchall()
    return [r["id"] if hasattr(r, "keys") else r[0] for r in rows]


def parent_media_complete(
    conn: sqlite3.Connection,
    parent_slug: str,
    *,
    raw_path: Path | None = None,
    allow_db_only: bool = False,
) -> bool:
    """True when every deferred sibling is dispatched and its append marker is on disk."""
    if not parent_slug:
        return False
    if count_pending_media(conn, parent_slug) > 0:
        return False
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is None:
        return False
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
        (parent_slug,),
    ).fetchone()
    actual = row["n"] if hasattr(row, "keys") else row[0]
    expected = deferral["expected_total"]
    if actual != expected:
        return False
    if expected == 0 and actual == 0:
        return True
    rows = conn.execute(
        "SELECT id, media_appended FROM ingest_queue WHERE parent_slug = ?",
        (parent_slug,),
    ).fetchall()
    for sibling in rows:
        if not bool(
            sibling["media_appended"] if hasattr(sibling, "keys") else sibling[1]
        ):
            return False
    if raw_path is None:
        if expected > 0 and not allow_db_only:
            return False
        return True
    if not raw_path.is_file():
        return False
    text = raw_path.read_text(encoding="utf-8")
    sibling_ids = [
        sibling["id"] if hasattr(sibling, "keys") else sibling[0] for sibling in rows
    ]
    for sibling in rows:
        sid = sibling["id"] if hasattr(sibling, "keys") else sibling[0]
        if media_append_marker(sid) not in text:
            return False
        if not _sibling_media_block_has_body(text, sid, sibling_ids):
            return False
    return True


def parent_deferral_synthesize_execute_blocked(
    conn: sqlite3.Connection,
    slug: str,
    *,
    raw_path: Path | None,
) -> str | None:
    """Execute-time gate for deferred ``*_synthesize`` tasks.

    Synthesis must not run until sibling media is fully dispatched and landed in
    the parent raw bundle. Returns a human-readable pause reason when blocked;
    ``None`` when media is complete or this is standalone synthesis (no siblings).

    Fail closed when the deferral row is missing but siblings may still be
    incomplete (partial DB loss, manual task, race before batch bootstrap).
    """
    temp_dir = raw_path.parent if raw_path is not None else None
    deferral = get_parent_media_deferral(conn, slug)
    if deferral is None:
        if temp_dir is not None:
            bootstrap_parent_media_deferral_from_manifest(conn, slug, temp_dir)
        deferral = get_parent_media_deferral(conn, slug)
    if deferral is None:
        bootstrap_parent_media_deferral_from_siblings(conn, slug, temp_dir)
        deferral = get_parent_media_deferral(conn, slug)
    sibling_n = _parent_sibling_media_count(conn, slug)
    if deferral is None:
        if sibling_n == 0:
            return None
        if count_pending_media(conn, slug) > 0:
            return "deferred synthesis paused — sibling media still pending"
        return "deferred synthesis paused — parent media not complete at execute time"
    if count_pending_media(conn, slug) > 0:
        return "deferred synthesis paused — sibling media still pending"
    if not parent_media_complete(conn, slug, raw_path=raw_path):
        return "deferred synthesis paused — parent media not complete at execute time"
    return None


def append_parent_media_atomic(
    conn: sqlite3.Connection,
    ingest_id: str,
    raw_path: Path,
    section: str,
    body: str,
) -> bool:
    """Append once; persist raw content to disk before recording ``media_appended``."""
    marker = media_append_marker(ingest_id)
    row = conn.execute(
        "SELECT media_appended FROM ingest_queue WHERE id = ?", (ingest_id,)
    ).fetchone()
    if row is None:
        return False
    if not raw_path.exists():
        return False
    from agent.persist import scrub_durable_text

    safe_body = scrub_durable_text(body, policy="boundary4").strip()
    text = raw_path.read_text(encoding="utf-8")
    media_flag = bool(row["media_appended"] if hasattr(row, "keys") else row[0])
    sibling_ids = _ordered_sibling_ids_for_ingest(conn, ingest_id)
    if marker in text and media_flag:
        return True
    if marker in text and not media_flag:
        text = _strip_sibling_media_block(text, ingest_id, sibling_ids)
    if media_flag:
        section_present = f"## {section}" in text and safe_body in text
        if section_present:
            pending_text = text + f"\n\n{marker}\n"
            from agent.persist import persist_durable_text

            persist_durable_text(raw_path, pending_text, policy="boundary4")
            return True
        with conn:
            conn.execute(
                "UPDATE ingest_queue SET media_appended = 0, updated_at = datetime('now') WHERE id = ?",
                (ingest_id,),
            )
        media_flag = False
    block = f"\n\n## {section}\n\n{safe_body}\n\n{marker}\n"
    pending_text = text + block
    from agent.persist import persist_durable_text

    persist_durable_text(raw_path, pending_text, policy="boundary4")
    set_ingest_media_appended(conn, ingest_id)
    parent_row = conn.execute(
        "SELECT parent_slug FROM ingest_queue WHERE id = ?", (ingest_id,),
    ).fetchone()
    if parent_row is not None:
        parent_slug = parent_row["parent_slug"] if hasattr(parent_row, "keys") else parent_row[0]
        if parent_slug:
            _clear_parent_media_changed_if_satisfied(
                conn, parent_slug, raw_path=raw_path,
            )
    return True


def update_ingest_status(conn: sqlite3.Connection, ingest_id: str, status: str,
                         error: str | None = None) -> None:
    """Transition an ingest-queue row to a new status (touches updated_at)."""
    if error is not None:
        error = _persist_visible_error(error)
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


def get_parent_media_deferral(conn: sqlite3.Connection, parent_slug: str) -> dict | None:
    """Return the persisted deferral gate for a parent article, if any."""
    if not parent_slug:
        return None
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM parent_media_deferral WHERE parent_slug = ?",
        (parent_slug,),
    ).fetchone()
    return dict(row) if row else None


def ensure_parent_media_deferral(
    conn: sqlite3.Connection,
    parent_slug: str,
    source_url: str,
    content_type: str,
    expected_total: int,
    *,
    media_changed: int | None = None,
) -> None:
    """Persist the deferral manifest total (never lowered; may grow on conflict)."""
    if not parent_slug:
        return
    content_type = normalize_deferral_parent_content_type(content_type)
    prior = get_parent_media_deferral(conn, parent_slug)
    prior_total = int(prior["expected_total"]) if prior else 0
    mc_insert = 1 if media_changed else 0
    with conn:
        conn.execute(
            """
            INSERT INTO parent_media_deferral
                (parent_slug, source_url, content_type, expected_total, media_changed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(parent_slug) DO UPDATE SET
                content_type = excluded.content_type,
                expected_total = MAX(parent_media_deferral.expected_total, excluded.expected_total),
                updated_at = CASE
                    WHEN excluded.expected_total > parent_media_deferral.expected_total
                    THEN datetime('now')
                    ELSE parent_media_deferral.updated_at
                END,
                media_changed = CASE
                    WHEN excluded.media_changed = 1 THEN 1
                    WHEN excluded.expected_total > parent_media_deferral.expected_total
                    THEN 1
                    ELSE parent_media_deferral.media_changed
                END
            """,
            (parent_slug, source_url, content_type, int(expected_total), mc_insert),
        )
    deferral = get_parent_media_deferral(conn, parent_slug)
    if (
        prior is not None
        and deferral is not None
        and int(deferral["expected_total"]) > prior_total
    ):
        supersede_stale_deferral_synthesize_tasks(conn, parent_slug, deferral)
        with conn:
            conn.execute(
                """
                UPDATE ingest_queue
                SET status = 'queued', media_appended = 0, error = NULL,
                    dispatched_at = NULL, updated_at = datetime('now')
                WHERE parent_slug = ? AND status = 'dispatched'
                """,
                (parent_slug,),
            )


def bootstrap_parent_media_deferral_from_siblings(
    conn: sqlite3.Connection,
    parent_slug: str,
    temp_dir: Path | None = None,
) -> None:
    """Seed deferral from sibling ingest rows when manifest bootstrap did not run."""
    if not parent_slug or get_parent_media_deferral(conn, parent_slug):
        return
    if temp_dir is not None:
        bootstrap_parent_media_deferral_from_manifest(conn, parent_slug, temp_dir)
        if get_parent_media_deferral(conn, parent_slug):
            return
    sibling_n = _parent_sibling_media_count(conn, parent_slug)
    high_water = _embedded_media_high_water_hint(conn, parent_slug, temp_dir)
    if sibling_n == 0 and high_water == 0:
        return
    expected = max(sibling_n, high_water)
    source_url, content_type = _lookup_parent_article_metadata(conn, parent_slug)
    if not source_url or not content_type:
        return
    ensure_parent_media_deferral(
        conn, parent_slug, source_url, content_type, expected,
    )
    _mark_parent_media_enqueue_shortfall(
        conn, parent_slug, sibling_n, expected,
    )


def bootstrap_parent_media_deferral_from_manifest(
    conn: sqlite3.Connection,
    parent_slug: str,
    temp_dir: Path,
) -> None:
    """Seed the deferral record from on-disk manifest when the DB row is still absent."""
    if not parent_slug or get_parent_media_deferral(conn, parent_slug):
        return
    manifest_path = temp_dir / f"{parent_slug}_media.json"
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return
    total = data.get("total")
    if total is None:
        return
    source_url = data.get("url")
    content_type = data.get("content_type")
    if not source_url or not content_type:
        parent_row = conn.execute(
            "SELECT source_url, content_type FROM tasks "
            "WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
            "ORDER BY created_at ASC, id ASC LIMIT 1",
            (parent_slug,),
        ).fetchone()
        if parent_row:
            source_url = source_url or (
                parent_row["source_url"] if hasattr(parent_row, "keys") else parent_row[0]
            )
            content_type = content_type or (
                parent_row["content_type"] if hasattr(parent_row, "keys") else parent_row[1]
            )
    if not source_url or not content_type:
        return
    source_url = url_for_display(source_url)
    manifest_media_changed = 1 if data.get("media_changed") else None
    ensure_parent_media_deferral(
        conn,
        parent_slug,
        source_url,
        content_type,
        int(total),
        media_changed=manifest_media_changed,
    )


def mark_parent_media_changed(conn: sqlite3.Connection, parent_slug: str) -> None:
    """Record that embedded media set changed (e.g. dashboard removed a sibling)."""
    if not parent_slug:
        return
    with conn:
        conn.execute(
            """
            UPDATE parent_media_deferral
            SET media_changed = 1, updated_at = datetime('now')
            WHERE parent_slug = ?
            """,
            (parent_slug,),
        )
    deferral = get_parent_media_deferral(conn, parent_slug)
    if deferral is not None:
        supersede_stale_deferral_synthesize_tasks(conn, parent_slug, deferral)
    _reset_dispatched_parent_siblings_if_deferral_cycle(conn, parent_slug)


def remove_ingest(
    conn: sqlite3.Connection,
    ingest_id: str,
    *,
    temp_dir: Path | None = None,
) -> str | None:
    """Delete an ingest-queue row (dashboard escape hatch).

    Returns the removed row's ``parent_slug`` when deleted (non-empty parents only),
    else ``None``. Reconciles deferral ``expected_total`` down to remaining siblings.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT parent_slug FROM ingest_queue WHERE id = ?", (ingest_id,),
    ).fetchone()
    if row is None:
        return None
    parent_slug = row["parent_slug"] or ""
    with conn:
        deleted = conn.execute(
            "DELETE FROM ingest_queue WHERE id = ?", (ingest_id,),
        ).rowcount
    if not deleted:
        return None
    if parent_slug:
        if temp_dir is not None:
            bootstrap_parent_media_deferral_from_manifest(conn, parent_slug, temp_dir)
        _reconcile_parent_media_deferral_on_remove_sibling(
            conn, parent_slug, temp_dir=temp_dir,
        )
        mark_parent_media_changed(conn, parent_slug)
    return parent_slug


def claim_ingest_items(conn: sqlite3.Connection) -> list[dict]:
    """Atomically claim every queued ingest item (queued -> downloading) and return them.

    One UPDATE ... RETURNING is a single SQLite write, so concurrent batch runs
    each claim a disjoint set — a second run grabs only what the first didn't.
    """
    conn.row_factory = sqlite3.Row
    with conn:
        rows = conn.execute(
            "UPDATE ingest_queue SET status='downloading', updated_at=datetime('now') "
            "WHERE id IN ("
            "  SELECT id FROM ingest_queue WHERE status='queued' "
            "  ORDER BY created_at ASC, id ASC"
            ") "
            "RETURNING *"
        ).fetchall()
    claimed = [dict(r) for r in rows]
    claimed.sort(key=lambda r: (r["created_at"], r["id"]))
    return claimed


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
    """Get the next step the consumer should run, ordered by ``step_order``.

    Invariant: any earlier non-success step is a barrier — the consumer must not
    run a later ``pending`` step while an earlier step is still ``paused`` or
    ``failed`` (deferred media gate, cookies, Firecrawl, paywall, or any other).

    Args:
        conn: Agent database connection.
        task_id: The task UUID.

    Returns:
        Step dict for the next runnable or blocking step, or ``None`` when every
        step is terminal (no pending and no paused).
    """
    pending_row = conn.execute(
        """
        SELECT * FROM task_steps
        WHERE task_id = ? AND status = 'pending'
        ORDER BY step_order LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    pending_order = pending_row["step_order"] if pending_row is not None else None

    paused_rows = conn.execute(
        """
        SELECT * FROM task_steps
        WHERE task_id = ? AND status = 'paused'
        ORDER BY step_order
        """,
        (task_id,),
    ).fetchall()
    for paused in paused_rows:
        if pending_order is None or paused["step_order"] < pending_order:
            return dict(paused)

    failed_rows = conn.execute(
        """
        SELECT * FROM task_steps
        WHERE task_id = ? AND status = 'failed'
        ORDER BY step_order
        """,
        (task_id,),
    ).fetchall()
    for failed in failed_rows:
        if pending_order is None or failed["step_order"] < pending_order:
            return dict(failed)

    if pending_row is None:
        return None
    return dict(pending_row)


def update_step(conn: sqlite3.Connection, step_id: int, status: str, output: str | None = None):
    """Update a task step's status and optional output.

    Args:
        conn: Agent database connection.
        step_id: The step's integer ID.
        status: New status ('running', 'done', 'failed', 'paused').
        output: Optional output text from the step.
    """
    if output is not None:
        output = _persist_visible_error(output)
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


def task_execution_aborted(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when a superseded or terminal task must not run further consumer steps."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status, error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return True
    status = row["status"] if hasattr(row, "keys") else row[0]
    err = row["error"] if hasattr(row, "keys") else row[1]
    if is_superseded_parent_error(err):
        return True
    return status in ("failed", "completed", "done")


def supersede_deferred_parent_task(
    conn: sqlite3.Connection, parent_slug: str,
) -> int:
    """Mark the original parent ingestion task terminal after batch synthesis.

    Compare-and-swap: only active non-synthesize rows transition; superseded or
    already-terminal parents are left unchanged so ``complete_task`` cannot revive
    them.
    """
    if not parent_slug:
        return 0
    with conn:
        cur = conn.execute(
            f"""
            UPDATE tasks SET status = 'completed', error = ?, updated_at = datetime('now')
            WHERE slug = ?
              AND content_type NOT GLOB '*_synthesize'
              AND status IN ('paused', 'pending', 'in_progress')
              {REQUEUE_SUPERSEDE_GUARD}
            """,
            (SUPERSEDE_DEFERRED_PARENT_REASON, parent_slug),
        )
        return cur.rowcount


def complete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Mark a task as done when it is still runnable (compare-and-swap).

    Superseded or already-terminal rows are left unchanged so stale consumers
    cannot mint generation-fresh owners via ``updated_at``.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.

    Returns:
        True when the row transitioned to ``done``, else False.
    """
    with conn:
        cur = conn.execute(
            f"""
            UPDATE tasks SET status = 'done', updated_at = datetime('now')
            WHERE id = ?
              AND status IN ('pending', 'in_progress')
              {REQUEUE_SUPERSEDE_GUARD}
            """,
            (task_id,),
        )
        return cur.rowcount == 1


def fail_task(conn: sqlite3.Connection, task_id: str, error: str):
    """Mark a task as failed with an error message.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.
        error: Human-readable error description.
    """
    error = _persist_visible_error(error)
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
    reason = _persist_visible_error(reason)
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


def reset_task_steps_for_retry(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    slug: str | None,
    error: str | None,
) -> None:
    """Reset failed/paused/running steps when re-queueing a task.

    Shared by :func:`agent.core.resume.retry_task` and
    :func:`resume_tasks_by_reason` so bulk and single-task resume share the
    same deferral-preserving step rewind policy.
    """
    in_cycle = task_in_deferred_media_cycle(
        conn, task_id, slug=slug, error=error,
    )
    if in_cycle:
        embed = conn.execute(
            """
            SELECT step_order FROM task_steps
            WHERE task_id = ? AND name = 'substack_embedded_media'
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if embed is not None:
            preserve_embed = task_deferred_retry_preserves_embed_step(
                conn, task_id, slug, error=error,
            )
            step_filter = "step_order > ?" if preserve_embed else "step_order >= ?"
            if preserve_embed:
                status_clause = "status IN ('failed', 'paused', 'running')"
            else:
                status_clause = (
                    "(status IN ('failed', 'paused', 'running') "
                    "OR (name = 'substack_embedded_media' AND status = 'done'))"
                )
            conn.execute(
                f"""
                UPDATE task_steps
                SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
                WHERE task_id = ? AND {step_filter}
                AND {status_clause}
                """,
                (task_id, embed[0]),
            )
            return
    conn.execute(
        """
        UPDATE task_steps
        SET status = 'pending', output = NULL, started_at = NULL, finished_at = NULL
        WHERE task_id = ? AND status IN ('failed', 'paused', 'running')
        """,
        (task_id,),
    )


def _task_blocks_bulk_resume(
    conn: sqlite3.Connection,
    task_id: str,
    slug: str | None,
    error: str | None,
) -> bool:
    """Deferral gate for bulk resume: active media cycle only, not stale deferral rows."""
    if is_deferred_media_pause(error):
        return True
    if slug:
        deferral = get_parent_media_deferral(conn, slug)
        if deferral is not None:
            if count_pending_media(conn, slug) > 0 or int(deferral.get("media_changed") or 0):
                return True
    row = conn.execute(
        """
        SELECT output FROM task_steps
        WHERE task_id = ? AND name = 'substack_embedded_media'
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row and is_deferred_media_pause(row[0]):
        return True
    if _substack_cleanup_output_signals_deferred_media(conn, task_id, slug):
        return True
    return False


def resume_tasks_by_reason(conn: sqlite3.Connection, reason_pattern: str) -> int:
    """Resume all paused tasks matching a reason pattern.

    Resets retries to 0, clears task error, and sets status to ``pending``.
    Step reset matches :func:`agent.core.resume.retry_task` (failed/paused/running
    steps rewound). Skips tasks whose slug has an in-flight ``*_synthesize`` sibling
    and tasks blocked by :func:`_task_blocks_bulk_resume` (active deferral cycle or
    durable pause markers). Stale ``parent_media_deferral`` rows alone do not block.

    Args:
        conn: Agent database connection.
        reason_pattern: SQL LIKE pattern for the error/reason field
                        (e.g., '%cookies%', '%firecrawl%').

    Returns:
        Number of tasks resumed.
    """
    rows = conn.execute(
        "SELECT id, slug, error FROM tasks WHERE status = 'paused' AND error LIKE ?",
        (reason_pattern,),
    ).fetchall()
    resumed = 0
    with conn:
        for task_id, slug, error in rows:
            if not may_reset_parent_task(
                conn, task_id, slug, error=error,
            ):
                continue
            if _task_blocks_bulk_resume(conn, task_id, slug=slug, error=error):
                continue
            if not requeue_task_for_retry_cas(conn, task_id, allowed_statuses=("paused",)):
                continue
            reset_task_steps_for_retry(
                conn, task_id, slug=slug, error=error,
            )
            resumed += 1
    return resumed


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


_LLM_ARCHIVE_RAW_ENV = "KB_LLM_ARCHIVE_RAW"


def llm_archive_store_raw() -> bool:
    """Opt-in via KB_LLM_ARCHIVE_RAW=1 — default is redacted storage at rest."""
    return os.getenv(_LLM_ARCHIVE_RAW_ENV, "").strip().lower() in ("1", "true", "yes")


def redact_llm_archive_text(text: str, *, store_raw: bool | None = None) -> str:
    """Scrub paths and token-shaped secrets before persisting llm_archives rows."""
    if store_raw if store_raw is not None else llm_archive_store_raw():
        return text or ""
    from agent.redact import redact_llm_archive_text_content

    return redact_llm_archive_text_content(text or "")


def archive_llm_call(conn: sqlite3.Connection, task_id: str, role: str,
                     prompt: str, response: str, model: str,
                     tokens_in: int | None = None, tokens_out: int | None = None,
                     *, store_raw: bool | None = None):
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
        store_raw: When True, skip redaction (default: env KB_LLM_ARCHIVE_RAW).
    """
    prompt_row = redact_llm_archive_text(prompt, store_raw=store_raw)
    response_row = redact_llm_archive_text(response, store_raw=store_raw)
    with conn:
        conn.execute(
            """INSERT INTO llm_archives (task_id, role, prompt, response, model, tokens_in, tokens_out)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (task_id, role, prompt_row, response_row, model, tokens_in, tokens_out),
        )
