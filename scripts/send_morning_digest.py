#!/usr/bin/env python3
"""Morning digest — Telegram summary of the overnight batch collector run.

Runs at 08:00 via cron. Invariant: alert on ANY abnormal batch_runs row since
the cutoff (failed, crashed, or still ``running`` past the stale window), not
only the newest row. Silent when every matching run finished successfully.
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

# Batch runs left in ``running`` longer than this are treated as abandoned/crashed.
STALE_RUNNING_HOURS = 6

_BATCH_RUNS_SINCE_CUTOFF = (
    "SELECT * FROM batch_runs WHERE finished_at >= ? OR started_at >= ? "
    "ORDER BY id ASC"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _since_morning() -> str:
    """Timestamp of the most recent 08:00 UTC (cutoff for 'last night's run')."""
    now = _utc_now()
    morning = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now < morning:
        morning -= timedelta(days=1)
    return morning.strftime("%Y-%m-%d %H:%M:%S")


def _parse_db_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _is_stale_running(run: dict, now: datetime, stale_hours: int) -> bool:
    if run["status"] != "running":
        return False
    started = _parse_db_ts(run["started_at"])
    return (now - started) > timedelta(hours=stale_hours)


def _collect_abnormal_runs(
    runs: list[dict], now: datetime, stale_hours: int = STALE_RUNNING_HOURS
) -> list[dict]:
    """Return every run that should appear in the morning digest (not only the latest)."""
    abnormal: list[dict] = []
    for run in runs:
        if run["status"] in ("failed", "crashed"):
            abnormal.append(run)
        elif _is_stale_running(run, now, stale_hours):
            abnormal.append(run)
    return abnormal


def _aggregate_for_notify(abnormal: list[dict]) -> tuple[int, int, int, list[str]]:
    total_failed = sum(int(r.get("failed") or 0) for r in abnormal)
    total_items = sum(int(r.get("total") or 0) for r in abnormal)
    total_dispatched = sum(int(r.get("dispatched") or 0) for r in abnormal)
    causes: list[str] = []
    for run in abnormal:
        if run["status"] in ("failed", "crashed"):
            summary = run.get("summary")
            if summary:
                causes.append(summary)
            else:
                causes.append(
                    f"batch run #{run['id']} {run['status']} since {run['started_at']}"
                )
        elif run["status"] == "running":
            causes.append(
                f"batch run #{run['id']} still running since {run['started_at']}"
            )
    return total_failed, total_items, total_dispatched, causes


def main() -> None:
    config = load_config()
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    cutoff = _since_morning()
    conn.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in conn.execute(_BATCH_RUNS_SINCE_CUTOFF, (cutoff, cutoff)).fetchall()
    ]
    conn.close()

    if not rows:
        return

    now = _utc_now()
    abnormal = _collect_abnormal_runs(rows, now)
    if not abnormal:
        return

    failed, total, dispatched, causes = _aggregate_for_notify(abnormal)
    notify_batch(config, failed=failed, total=total, dispatched=dispatched, causes=causes)
    statuses = ", ".join(sorted({r["status"] for r in abnormal}))
    print(f"Sent morning digest: {statuses} ({failed}/{total} failed, {len(abnormal)} run(s))")


if __name__ == "__main__":
    main()
