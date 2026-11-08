#!/usr/bin/env python3
"""Log rotation — archive step_logs + llm_archives rows older than 30 days to MinIO.

Monthly cron job (0 0 1 * *):
    python3 /root/kb-agent/scripts/rotate_logs.py

Workflow:
    1. Dump logs.db rows (step_logs + llm_archives) older than 30 days → JSON lines
    2. Gzip the JSON
    3. Upload to MinIO: kb-agent-archive/logs/YYYY-MM.json.gz
    4. DELETE old rows from both tables + VACUUM

Retention: indefinite on MinIO, 30 days active in DB.
"""

import gzip
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.notify import sanitize_visible_message

_OBVIOUS_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*\S+"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DB = Path("/root/kb-agent/data/logs.db")
MINIO_BUCKET = "kb-agent-archive"
MINIO_PREFIX = "logs"
RETENTION_DAYS = 30
BATCH_SIZE = 500


def _redact_obvious_secrets(text: str) -> str:
    if not text:
        return text
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _OBVIOUS_SECRET_RE.sub("[credential redacted]", text)
    return text


def _sanitize_llm_text(text: str) -> str:
    return _redact_obvious_secrets(sanitize_visible_message(text or ""))


def sanitize_archive_record(record: dict) -> dict:
    """Redact sensitive fields before writing archived log rows to MinIO."""
    out = dict(record)
    table = out.get("_table")
    if table == "step_logs" and out.get("content"):
        out["content"] = sanitize_visible_message(out["content"])
    elif table == "llm_archives":
        for key in ("prompt", "response"):
            if out.get(key):
                out[key] = _sanitize_llm_text(out[key])
    return out


def dump_rows(conn: sqlite3.Connection, table: str, columns: list[str],
              date_col: str, cutoff: str) -> tuple[list[dict], list[int]]:
    """Dump rows older than cutoff from a table.

    Args:
        conn: SQLite connection to logs.db.
        table: Table name (step_logs or llm_archives).
        columns: Column names to select.
        date_col: Name of the timestamp column to filter on.
        cutoff: ISO-format cutoff datetime string.

    Returns:
        (records, ids) — list of dicts with _table field, list of integer IDs.
    """
    col_list = ", ".join(columns)
    rows = conn.execute(
        f"SELECT id, {col_list} FROM {table} WHERE {date_col} < ? ORDER BY {date_col} ASC",
        (cutoff,),
    ).fetchall()

    records = []
    ids = []
    for row in rows:
        record = {"_table": table, "id": row["id"]}
        for col in columns:
            record[col] = row[col]
        records.append(record)
        ids.append(row["id"])

    return records, ids


def delete_batched(conn: sqlite3.Connection, table: str, ids: list[int]) -> int:
    """Delete rows by ID in batches.

    Args:
        conn: SQLite connection.
        table: Table name.
        ids: List of integer IDs to delete.

    Returns:
        Total number of rows deleted.
    """
    total = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE id IN ({placeholders})",
            batch,
        )
        total += cursor.rowcount
    return total


def rebuild_fts(conn: sqlite3.Connection):
    """Rebuild all FTS5 indexes in logs.db."""
    # logs_fts covers step_logs content
    conn.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")


def main():
    """Run the full log rotation cycle."""
    if not LOGS_DB.exists():
        print(
            "[rotate_logs] No logs.db at",
            sanitize_visible_message(str(LOGS_DB)),
            "— nothing to rotate",
        )
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    archive_month = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).strftime(
        "%Y-%m"
    )

    print(f"[rotate_logs] Rotating logs older than {cutoff} → "
          f"{MINIO_BUCKET}/{MINIO_PREFIX}/{archive_month}.json.gz")

    # ── 1. Dump old rows from both tables ──
    conn = sqlite3.connect(str(LOGS_DB))
    conn.row_factory = sqlite3.Row

    # step_logs: id, task_id, step_name, stream, content, logged_at
    step_records, step_ids = dump_rows(
        conn, "step_logs",
        columns=["task_id", "step_name", "stream", "content", "logged_at"],
        date_col="logged_at", cutoff=cutoff,
    )

    # llm_archives: id, task_id, role, prompt, response, model, tokens_in, tokens_out, created_at
    llm_records, llm_ids = dump_rows(
        conn, "llm_archives",
        columns=["task_id", "role", "prompt", "response", "model",
                 "tokens_in", "tokens_out", "created_at"],
        date_col="created_at", cutoff=cutoff,
    )

    total_rows = len(step_records) + len(llm_records)
    print(f"[rotate_logs] Found {len(step_records)} step_logs + "
          f"{len(llm_records)} llm_archives = {total_rows} total rows to archive")

    if total_rows == 0:
        conn.close()
        print("[rotate_logs] Nothing to rotate — done")
        return 0

    # ── 2. Write JSON lines (step_logs first, then llm_archives) + gzip ──
    tmp_json = Path("/tmp") / f"kb-agent-logs-{archive_month}.json"
    tmp_gz = Path("/tmp") / f"kb-agent-logs-{archive_month}.json.gz"

    try:
        with open(tmp_json, "w") as f:
            for record in step_records + llm_records:
                safe = sanitize_archive_record(record)
                f.write(json.dumps(safe, ensure_ascii=False) + "\n")

        # Gzip
        with open(tmp_json, "rb") as f_in:
            with gzip.open(tmp_gz, "wb", compresslevel=9) as f_out:
                f_out.write(f_in.read())

        archive_size = tmp_gz.stat().st_size
        print(f"[rotate_logs] Archived {total_rows} rows → {archive_size:,} bytes compressed")

    except Exception as e:
        print(
            f"[rotate_logs] ERROR during dump/gzip: {sanitize_visible_message(str(e))}",
            file=sys.stderr,
        )
        conn.close()
        return 1

    # ── 3. Upload to MinIO ──
    minio_path = f"{MINIO_BUCKET}/{MINIO_PREFIX}/{archive_month}.json.gz"
    result = subprocess.run(
        ["mc", "cp", str(tmp_gz), f"minio/{minio_path}"],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        print(
            "[rotate_logs] MinIO upload FAILED:",
            sanitize_visible_message((result.stderr or "").strip()),
            file=sys.stderr,
        )
        # Clean up temp files but DON'T delete rows — retry next run
        tmp_json.unlink(missing_ok=True)
        tmp_gz.unlink(missing_ok=True)
        conn.close()
        return 1

    print(f"[rotate_logs] Uploaded to minio/{minio_path}")

    # ── 4. DELETE old rows from both tables + VACUUM ──
    deleted_steps = delete_batched(conn, "step_logs", step_ids)
    deleted_llm = delete_batched(conn, "llm_archives", llm_ids)

    rebuild_fts(conn)
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    # ── 5. Clean up temp files ──
    tmp_json.unlink(missing_ok=True)
    tmp_gz.unlink(missing_ok=True)

    print(f"[rotate_logs] Deleted {deleted_steps} step_logs + {deleted_llm} llm_archives "
          f"+ VACUUM — done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
