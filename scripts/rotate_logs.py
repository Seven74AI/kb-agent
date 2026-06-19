#!/usr/bin/env python3
"""Log rotation — archive step_logs rows older than 30 days to MinIO.

Monthly cron job (0 0 1 * *):
    python3 /root/kb-agent/scripts/rotate_logs.py

Workflow:
    1. Dump logs.db rows older than 30 days → JSON lines
    2. Gzip the JSON
    3. Upload to MinIO: kb-agent-archive/logs/YYYY-MM.json.gz
    4. DELETE old rows + VACUUM

Retention: indefinite on MinIO, 30 days active in DB.
"""

import gzip
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DB = Path("/root/kb-agent/data/logs.db")
MINIO_BUCKET = "kb-agent-archive"
MINIO_PREFIX = "logs"
RETENTION_DAYS = 30


def main():
    """Run the full log rotation cycle."""
    if not LOGS_DB.exists():
        print(f"[rotate_logs] No logs.db at {LOGS_DB} — nothing to rotate")
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    archive_month = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).strftime(
        "%Y-%m"
    )

    print(f"[rotate_logs] Rotating logs older than {cutoff} → {MINIO_BUCKET}/{MINIO_PREFIX}/{archive_month}.json.gz")

    # ── 1. Dump old rows to JSON ──
    conn = sqlite3.connect(str(LOGS_DB))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, task_id, step_name, stream, content, logged_at "
        "FROM step_logs WHERE logged_at < ? "
        "ORDER BY logged_at ASC",
        (cutoff,),
    ).fetchall()

    row_count = len(rows)
    print(f"[rotate_logs] Found {row_count} rows to archive")

    if row_count == 0:
        conn.close()
        print("[rotate_logs] Nothing to rotate — done")
        return 0

    # ── 2. Write JSON lines + gzip ──
    tmp_json = Path("/tmp") / f"kb-agent-logs-{archive_month}.json"
    tmp_gz = Path("/tmp") / f"kb-agent-logs-{archive_month}.json.gz"

    try:
        with open(tmp_json, "w") as f:
            for row in rows:
                record = {
                    "id": row["id"],
                    "task_id": row["task_id"],
                    "step_name": row["step_name"],
                    "stream": row["stream"],
                    "content": row["content"],
                    "logged_at": row["logged_at"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Gzip
        with open(tmp_json, "rb") as f_in:
            with gzip.open(tmp_gz, "wb", compresslevel=9) as f_out:
                f_out.write(f_in.read())

        archive_size = tmp_gz.stat().st_size
        print(f"[rotate_logs] Archived {row_count} rows → {archive_size:,} bytes compressed")

    except Exception as e:
        print(f"[rotate_logs] ERROR during dump/gzip: {e}", file=sys.stderr)
        conn.close()
        return 1

    # ── 3. Upload to MinIO ──
    minio_path = f"{MINIO_BUCKET}/{MINIO_PREFIX}/{archive_month}.json.gz"
    result = subprocess.run(
        ["mc", "cp", str(tmp_gz), f"minio/{minio_path}"],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        print(f"[rotate_logs] MinIO upload FAILED: {result.stderr.strip()}", file=sys.stderr)
        # Clean up temp files but DON'T delete rows — retry next run
        tmp_json.unlink(missing_ok=True)
        tmp_gz.unlink(missing_ok=True)
        conn.close()
        return 1

    print(f"[rotate_logs] Uploaded to minio/{minio_path}")

    # ── 4. DELETE old rows + VACUUM ──
    ids = [str(row["id"]) for row in rows]
    # Delete in batches to avoid huge IN() clause
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        conn.execute(
            f"DELETE FROM step_logs WHERE id IN ({placeholders})",
            batch,
        )

    # Rebuild FTS index
    conn.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    # ── 5. Clean up temp files ──
    tmp_json.unlink(missing_ok=True)
    tmp_gz.unlink(missing_ok=True)

    print(f"[rotate_logs] Deleted {row_count} rows + VACUUM — done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
