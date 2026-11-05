#!/usr/bin/env python3
"""Morning digest — Telegram summary of the overnight batch collector run.

Runs at 08:00 via cron. Reads the latest batch_runs row finished since
yesterday 08:00 UTC; if it failed or crashed, sends a Telegram digest.
Silent on success, so a clean night produces no message.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Isolate from the Hermes runtime's py3.14 PYTHONPATH (same as batch_ingest.py).
os.environ.pop("PYTHONPATH", None)

_bootstrap_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bootstrap_root))
# The cron runs a COPY under ~/.hermes/scripts/, so parents[1] is wrong there.
sys.path.insert(0, "/root/kb-agent")

from agent.config import load_config  # noqa: E402
from agent.db import AGENT_SCHEMA, init_db  # noqa: E402
from agent.notify import notify_batch  # noqa: E402

sys.path.insert(0, str(load_config().kb_agent_dir))


def _since_morning() -> str:
    """Timestamp of the most recent 08:00 UTC (cutoff for 'last night's run')."""
    now = datetime.now(timezone.utc)
    morning = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now < morning:
        morning -= timedelta(days=1)
    return morning.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    config = load_config()
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    cutoff = _since_morning()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM batch_runs WHERE finished_at >= ? OR started_at >= ? "
        "ORDER BY id DESC LIMIT 1",
        (cutoff, cutoff),
    ).fetchone()
    conn.close()

    if row is None:
        return  # no batch run since the cutoff — nothing to report
    run = dict(row)
    if run["status"] in ("failed", "crashed"):
        notify_batch(
            config,
            failed=run["failed"],
            total=run["total"],
            dispatched=run["dispatched"],
            causes=[run["summary"]] if run["summary"] else [],
        )
        print(f"Sent morning digest: {run['status']} ({run['failed']}/{run['total']} failed)")


if __name__ == "__main__":
    main()
