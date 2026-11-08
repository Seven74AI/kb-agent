#!/usr/bin/env python3
"""One-shot poll of all due watch sources (for cron).

Example:
  python3 scripts/watch_sources_poll.py

When cron drives polling, disable the consumer's in-process scheduler to avoid
two pollers enqueuing duplicate tasks — set in config.yaml:

  watch_scheduler_enabled: false

(or export KB_WATCH_SCHEDULER_ENABLED=false for the consumer process).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config
from agent.core.watch_source import default_fetch_html, poll_all_due_sources
from agent.db import AGENT_SCHEMA, init_db
from agent.notify import sanitize_visible_message
from agent.redact import url_for_display


def main() -> int:
    config = load_config()
    db_path = config.kb_agent_dir / "data" / "agent.db"
    conn = init_db(db_path, AGENT_SCHEMA)
    try:
        results = poll_all_due_sources(conn, config, fetch_html=default_fetch_html)
    finally:
        conn.close()

    if not results:
        print("No due watch sources or no new items.", file=sys.stderr)
        return 0

    for r in results:
        if "error" in r:
            safe_url = url_for_display(r.get("page_url") or "")
            safe_err = sanitize_visible_message(r.get("error") or "")
            print(f"{safe_url}: ERROR {safe_err}", file=sys.stderr)
            continue
        new = r.get("new_items") or []
        failed = [s for s in r.get("submissions", []) if s.get("status") == "error"]
        if failed:
            print(
                f"{r['page_url']}: {len(new) - len(failed)}/{len(new)} submitted; "
                f"{len(failed)} failed (retry next run)",
                file=sys.stderr,
            )
        else:
            print(f"{r['page_url']}: submitted {len(new)} new item(s): {', '.join(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
