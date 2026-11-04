#!/usr/bin/env python3
"""Nightly batch collector — dispatch queued ingest links to the task pipeline.

Reads the durable ingest_queue (status='queued') and dispatches each link:
- Non-video (web/books/substack/...) -> full pipeline task (submit_url).
- Video with a synthesize-only variant (bunny_stream_synthesize) -> synthesize-only
  task (transcript already batch-transcribed + archived by the per-type collector,
  e.g. tdah_daily_check.py).

Runs as a --no-agent cron. Idempotent: a link is marked 'dispatched' only after
its task is created, so a crash re-runs cleanly next night.
"""
import sys

sys.path.insert(0, "/root/kb-agent")

from agent.config import load_config  # noqa: E402
from agent.db import (  # noqa: E402
    AGENT_SCHEMA,
    init_db,
    list_ingest_by_status,
    update_ingest_status,
)
from agent.core.pipelines import PIPELINE_STEPS  # noqa: E402
from agent.core.submit import dispatch_synthesize, submit_url  # noqa: E402

VIDEO_TYPES = {"youtube", "bunny_stream", "instagram_reel", "threads_video"}


def main() -> None:
    config = load_config()
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)

    queued = list_ingest_by_status(conn, "queued")
    if not queued:
        print("No queued ingest links.")
        return

    print(f"Queued: {len(queued)} link(s)")
    dispatched, skipped = [], []

    for q in queued:
        url, ct, slug = q["url"], q["content_type"], q["slug"]

        if ct in VIDEO_TYPES:
            synth_ct = f"{ct}_synthesize"
            if synth_ct not in PIPELINE_STEPS:
                skipped.append(f"{ct}:{slug}")
                print(f"  [skip] {slug} ({ct}) — no synthesize-only path yet")
                continue
            r = dispatch_synthesize(url, config, conn, content_type=synth_ct, slug=slug)
        else:
            r = submit_url(url, config, conn, ignore_dedup=True)

        status = r.get("status")
        if status in ("pending", "paused", "skipped"):
            update_ingest_status(conn, q["id"], "dispatched")
            dispatched.append(f"{slug}->{r.get('task_id', status)}")
            print(f"  [dispatch] {slug} ({ct}) -> {r.get('task_id', status)}")
        else:
            update_ingest_status(conn, q["id"], "failed", error=str(r.get("error") or status))
            skipped.append(f"{slug}:{status}")
            print(f"  [failed] {slug} ({ct}) -> {status}")

    conn.close()
    print(f"DONE: {len(dispatched)} dispatched, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
