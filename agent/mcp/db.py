"""SQLite connections for the MCP adapter (stateless: open per tool call)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.config import Config
from agent.db import AGENT_SCHEMA, LOGS_SCHEMA, init_db


def agent_db_path(config: Config) -> Path:
    return config.kb_agent_dir / "data" / "agent.db"


def logs_db_path(config: Config) -> Path:
    return config.kb_agent_dir / "data" / "logs.db"


def open_agent_conn(config: Config) -> sqlite3.Connection:
    db_path = agent_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(AGENT_SCHEMA)
    return conn


def open_logs_conn(config: Config) -> sqlite3.Connection:
    db_path = logs_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(LOGS_SCHEMA)
    return conn


def ensure_databases(config: Config) -> None:
    """Create agent.db and logs.db if missing (same as run.py startup)."""
    agent = open_agent_conn(config)
    agent.close()
    logs = open_logs_conn(config)
    logs.close()
