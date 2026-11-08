"""Regression: run.py orphan recovery matches consumer (reset in_progress → pending)."""

import uuid

import pytest

from agent.config import Config
from agent.db import AGENT_SCHEMA, init_db
from run import _recover_orphans


@pytest.fixture
def run_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    init_db(data_dir / "agent.db", AGENT_SCHEMA)
    return Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")


def test_run_recover_orphans_resets_in_progress_without_running_step(run_config):
    conn = init_db(run_config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    task_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO tasks (id, source_url, content_type, slug, status) "
            "VALUES (?, ?, ?, ?, 'in_progress')",
            (task_id, "https://example.com", "web", "run_orphan_slug"),
        )
        conn.execute(
            "INSERT INTO task_steps (task_id, step_order, name, status) "
            "VALUES (?, 0, 'step_a', 'done'), (?, 1, 'step_b', 'pending')",
            (task_id, task_id),
        )
    conn.close()

    _recover_orphans(run_config)

    conn = init_db(run_config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    assert row[0] == "pending"
