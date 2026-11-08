#!/usr/bin/env python3
"""KB Agent — main entry point.

Usage: python3 /root/kb-agent/run.py

Lifecycle:
    1. Init databases (agent.db + logs.db)
    2. Recover orphan tasks from previous crash
    3. Start Flask webhook (main thread)
    4. Start async consumer (background thread)
    5. Wait for signals (SIGINT/SIGTERM) → graceful shutdown

Architecture:
    Flask runs sync in the main thread — handles POST /submit + GET /dashboard.
    Consumer runs async in a background thread — claims tasks and executes
    pipeline steps via the pipeline registry.

Signal handling:
    SIGINT (Ctrl+C)  → graceful shutdown
    SIGTERM → graceful shutdown
    Child processes are killed via PR_SET_PDEATHSIG (set by canonical scripts).
"""

import asyncio
import os
import signal
import sys
import threading
from pathlib import Path

# Add kb-agent to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import Config, load_config
from agent.consumer import Consumer, recover_orphan_tasks
from agent.db import AGENT_SCHEMA, LOGS_SCHEMA, init_db
from agent.webhook import app


def _recover_orphans(config: Config):
    """Detect and recover orphan tasks from a previous crash (once per process)."""
    db_path = config.kb_agent_dir / "data" / "agent.db"
    if not db_path.exists():
        return

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    recover_orphan_tasks(conn, log_prefix="[run]")
    conn.close()
    os.environ["KB_AGENT_ORPHAN_RECOVERY_DONE"] = "1"


def _run_consumer(config: Config, loop: asyncio.AbstractEventLoop):
    """Run the async consumer in a background thread.

    This is the target for the consumer thread. It sets up the event loop,
    initializes the consumer, and runs it until shutdown is requested.
    """
    asyncio.set_event_loop(loop)
    consumer = Consumer(config)

    try:
        # Run startup (DB init, pipeline registration, orphan recovery)
        loop.run_until_complete(consumer.startup())

        print(f"[run] Consumer started — polling for tasks")
        print(f"[run] Webhook: http://{config.webhook_host}:{config.webhook_port}/dashboard")

        # Run the main polling loop
        loop.run_until_complete(consumer.run())
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[run] Consumer error: {e}", file=sys.stderr)
    finally:
        # Graceful shutdown
        print("[run] Shutting down consumer...")
        loop.run_until_complete(consumer.shutdown())
        print("[run] Consumer stopped")


def main():
    """Main entry point — init DBs, recover orphans, start Flask + consumer."""
    config = load_config()

    print(f"[run] KB Agent v0.1.0 starting")
    print(f"[run] Config: provider={config.llm_provider}, model={config.llm_model}")

    # ── 1. Init databases ──
    db_dir = config.kb_agent_dir / "data"
    db_dir.mkdir(parents=True, exist_ok=True)

    agent_conn = init_db(db_dir / "agent.db", AGENT_SCHEMA)
    logs_conn = init_db(db_dir / "logs.db", LOGS_SCHEMA)
    agent_conn.close()
    logs_conn.close()

    print(f"[run] Databases initialized: {db_dir}")

    # ── 2. Recover orphan tasks ──
    _recover_orphans(config)

    # ── 3. Start consumer in background thread ──
    consumer_loop = asyncio.new_event_loop()
    consumer_thread = threading.Thread(
        target=_run_consumer,
        args=(config, consumer_loop),
        daemon=True,
        name="kb-agent-consumer",
    )
    consumer_thread.start()

    # ── 4. Signal handlers ──
    shutdown_flag = {"shutdown": False}

    def _shutdown(signum, frame):
        if shutdown_flag["shutdown"]:
            print("[run] Force exit")
            os._exit(1)
        shutdown_flag["shutdown"] = True
        print(f"[run] Received signal {signum} — graceful shutdown")

        # Break app.run() blocking loop.
        # Werkzeug dev server only handles SIGINT natively (KeyboardInterrupt).
        # In production (gunicorn/waitress) the WSGI server handles SIGTERM,
        # but with the dev server we must raise KeyboardInterrupt ourselves.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 5. Start Flask (main thread, blocking) ──
    try:
        app.run(
            host=config.webhook_host,
            port=config.webhook_port,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass

    # ── 6. Cleanup ──
    print("[run] Flask stopped — waiting for consumer...")
    consumer_loop.call_soon_threadsafe(consumer_loop.stop)
    consumer_thread.join(timeout=30)
    print("[run] Shutdown complete")


if __name__ == "__main__":
    main()
