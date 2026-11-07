"""Database initialization — SQLite schemas for agent.db and logs.db."""

import json
import os
import re
import sqlite3
from pathlib import Path

# Max ingest-queue retries for failed embedded/standalone media (``recover_failed_media``).
INGEST_MAX_MEDIA_ATTEMPTS = 3

# Substrings in task/step pause reasons that mean deferred media still blocks synthesis.
DEFERRED_MEDIA_PAUSE_MARKERS: tuple[str, ...] = (
    "awaiting media",
    "deferred to batch synthesis",
    "batch synthesis dispatched or completed",
    "media_deferred:",
)


def is_deferred_media_pause(reason: str | None) -> bool:
    """True when a pause reason signals the parent must not pass ``synthesize_note``."""
    if not reason:
        return False
    low = reason.lower()
    return any(marker in low for marker in DEFERRED_MEDIA_PAUSE_MARKERS)


def _deferred_media_resume_sql_exclusion() -> tuple[str, tuple[str, ...]]:
    """SQL fragment and params that exclude deferred-media pauses from bulk resume."""
    clauses = [
        "LOWER(COALESCE(error, '')) NOT LIKE ?"
        for _ in DEFERRED_MEDIA_PAUSE_MARKERS
    ]
    params = tuple(f"%{marker}%" for marker in DEFERRED_MEDIA_PAUSE_MARKERS)
    return " AND ".join(clauses), params


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


def has_completed_synthesize_task(conn: sqlite3.Connection, slug: str) -> bool:
    """True when a deferred *_synthesize task for *slug* finished with ``done`` / ``completed``."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type GLOB '*_synthesize' "
        "AND status IN ('done', 'completed') "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    return row is not None


def has_synthesize_task_signal(conn: sqlite3.Connection, slug: str) -> bool:
    """True when batch synthesis is queued, in flight, or finished (any non-``failed`` *_synthesize)."""
    return has_active_synthesize_task(conn, slug) or has_completed_synthesize_task(conn, slug)


def has_completed_parent_synthesis(
    conn: sqlite3.Connection,
    slug: str,
    *,
    vault_path: Path | None = None,
    temp_dir: Path | None = None,
) -> bool:
    """True when synthesis for *slug* already finished on the parent or batch path.

    Used to make parent inline synthesis and batch ``*_synthesize`` dispatch
    mutually exclusive: once any owner completes, the other path must stand down.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE slug = ? AND content_type NOT GLOB '*_synthesize' "
        "AND status IN ('done', 'completed') "
        "LIMIT 1",
        (slug,),
    ).fetchone()
    if row is not None:
        return True
    if has_completed_synthesize_task(conn, slug):
        return True
    if vault_path is not None:
        note = vault_path / "Knowledge base" / "articles" / f"{slug}.md"
        if not note.is_file():
            return False
        if count_pending_media(conn, slug) > 0:
            return False
        deferral = get_parent_media_deferral(conn, slug)
        if deferral is None:
            return False
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE parent_slug = ?",
            (slug,),
        ).fetchone()
        actual = row["n"] if hasattr(row, "keys") else row[0]
        if actual != deferral["expected_total"]:
            return False
        if temp_dir is not None:
            raw_path = temp_dir / f"{slug}_raw.md"
            if not parent_media_complete(conn, slug, raw_path=raw_path):
                return False
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


def ingest_media_appended(conn: sqlite3.Connection, ingest_id: str) -> bool:
    """True when embedded media body was already written to the parent raw markdown."""
    row = conn.execute(
        "SELECT media_appended FROM ingest_queue WHERE id = ?", (ingest_id,)
    ).fetchone()
    if row is None:
        return False
    val = row["media_appended"] if hasattr(row, "keys") else row[0]
    return bool(val)


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


def parent_media_complete(
    conn: sqlite3.Connection,
    parent_slug: str,
    *,
    raw_path: Path | None = None,
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
        return True
    if not raw_path.is_file():
        return False
    text = raw_path.read_text(encoding="utf-8")
    for sibling in rows:
        if media_append_marker(sibling["id"]) not in text:
            return False
    return True


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
    text = raw_path.read_text(encoding="utf-8")
    media_flag = bool(row["media_appended"] if hasattr(row, "keys") else row[0])
    if marker in text:
        if not media_flag:
            set_ingest_media_appended(conn, ingest_id)
        return True
    if media_flag:
        pass
    block = f"\n\n## {section}\n\n{body.strip()}\n\n{marker}\n"
    pending_text = text + block
    tmp_path = raw_path.with_name(f"{raw_path.name}.append-{ingest_id}.tmp")
    tmp_path.write_text(pending_text, encoding="utf-8")
    os.replace(tmp_path, raw_path)
    set_ingest_media_appended(conn, ingest_id)
    return True


def update_ingest_status(conn: sqlite3.Connection, ingest_id: str, status: str,
                         error: str | None = None) -> None:
    """Transition an ingest-queue row to a new status (touches updated_at)."""
    if error is not None:
        from agent.notify import sanitize_visible_message

        error = sanitize_visible_message(error)
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
) -> None:
    """Persist the deferral manifest total (never lowered; may grow on conflict)."""
    if not parent_slug:
        return
    with conn:
        conn.execute(
            """
            INSERT INTO parent_media_deferral
                (parent_slug, source_url, content_type, expected_total)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(parent_slug) DO UPDATE SET
                expected_total = MAX(expected_total, excluded.expected_total)
            """,
            (parent_slug, source_url, content_type, int(expected_total)),
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
            "ORDER BY created_at LIMIT 1",
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
    ensure_parent_media_deferral(
        conn, parent_slug, source_url, content_type, int(total),
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


def remove_ingest(
    conn: sqlite3.Connection,
    ingest_id: str,
    *,
    temp_dir: Path | None = None,
) -> str | None:
    """Delete an ingest-queue row (dashboard escape hatch).

    Returns the removed row's ``parent_slug`` when deleted (non-empty parents only),
    else ``None``. Refreshes the deferral gate — does not shrink ``expected_total``.
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

    A paused ``synthesize_note`` with a deferred-media reason blocks later pending
    steps so the consumer cannot skip ahead to ``quality_gate`` / ``minio_upload``.

    Args:
        conn: Agent database connection.
        task_id: The task UUID.

    Returns:
        Step dict if a pending step exists, None otherwise.
    """
    pending_row = conn.execute(
        """
        SELECT * FROM task_steps
        WHERE task_id = ? AND status = 'pending'
        ORDER BY step_order LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    synthesize_paused = conn.execute(
        """
        SELECT * FROM task_steps
        WHERE task_id = ? AND name = 'synthesize_note' AND status = 'paused'
        ORDER BY step_order LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if synthesize_paused is not None:
        task_row = conn.execute(
            "SELECT error FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        task_error = task_row[0] if task_row else None
        step_output = synthesize_paused["output"]
        if is_deferred_media_pause(task_error) or is_deferred_media_pause(step_output):
            if pending_row is None or (
                synthesize_paused["step_order"] < pending_row["step_order"]
            ):
                return dict(synthesize_paused)
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
    deferred Substack media (structured deferral pause markers) are left paused
    so the consumer does not skip past ``substack_embedded_media`` / ``synthesize_note``.

    Args:
        conn: Agent database connection.
        reason_pattern: SQL LIKE pattern for the error/reason field
                        (e.g., '%cookies%', '%firecrawl%').

    Returns:
        Number of tasks resumed.
    """
    exclusion, ex_params = _deferred_media_resume_sql_exclusion()
    with conn:
        cursor = conn.execute(
            "UPDATE tasks SET status = 'pending', retries = 0, "
            "error = NULL, updated_at = datetime('now') "
            f"WHERE status = 'paused' AND error LIKE ? AND {exclusion}",
            (reason_pattern, *ex_params),
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


_LLM_ARCHIVE_RAW_ENV = "KB_LLM_ARCHIVE_RAW"
_LLM_ARCHIVE_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*\S+"
)
_LLM_ARCHIVE_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)


def llm_archive_store_raw() -> bool:
    """Opt-in via KB_LLM_ARCHIVE_RAW=1 — default is redacted storage at rest."""
    return os.getenv(_LLM_ARCHIVE_RAW_ENV, "").strip().lower() in ("1", "true", "yes")


def redact_llm_archive_text(text: str, *, store_raw: bool | None = None) -> str:
    """Scrub paths and token-shaped secrets before persisting llm_archives rows."""
    if store_raw if store_raw is not None else llm_archive_store_raw():
        return text or ""
    from agent.notify import sanitize_visible_message

    out = sanitize_visible_message(text or "")
    out = _LLM_ARCHIVE_BEARER_RE.sub("Bearer [redacted]", out)
    out = _LLM_ARCHIVE_SECRET_RE.sub("[credential redacted]", out)
    return out


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
