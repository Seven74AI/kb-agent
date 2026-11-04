"""Tests for the cron one-shot watch-sources poller exit semantics."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from scripts.watch_sources_poll import main  # noqa: E402


class _FakeConfig:
    def __init__(self, kb_agent_dir: Path):
        self.kb_agent_dir = kb_agent_dir


class _FakeConn:
    def close(self) -> None:
        pass


@patch("scripts.watch_sources_poll.poll_all_due_sources")
@patch("scripts.watch_sources_poll.init_db")
@patch("scripts.watch_sources_poll.load_config")
def test_main_no_results_returns_0(mock_load, mock_init, mock_poll, tmp_path):
    mock_load.return_value = _FakeConfig(tmp_path)
    mock_init.return_value = _FakeConn()
    mock_poll.return_value = []

    assert main() == 0


@patch("scripts.watch_sources_poll.poll_all_due_sources")
@patch("scripts.watch_sources_poll.init_db")
@patch("scripts.watch_sources_poll.load_config")
def test_main_fetch_error_returns_1(mock_load, mock_init, mock_poll, tmp_path):
    mock_load.return_value = _FakeConfig(tmp_path)
    mock_init.return_value = _FakeConn()
    mock_poll.return_value = [{"page_url": "https://x.example/", "error": "boom"}]

    assert main() == 1


@patch("scripts.watch_sources_poll.poll_all_due_sources")
@patch("scripts.watch_sources_poll.init_db")
@patch("scripts.watch_sources_poll.load_config")
def test_main_submit_failure_returns_1(mock_load, mock_init, mock_poll, tmp_path):
    mock_load.return_value = _FakeConfig(tmp_path)
    mock_init.return_value = _FakeConn()
    mock_poll.return_value = [
        {
            "page_url": "https://x.example/",
            "new_items": ["a", "b"],
            "submissions": [
                {"status": "pending"},
                {"status": "error", "error": "boom"},
            ],
        }
    ]

    assert main() == 1


@patch("scripts.watch_sources_poll.poll_all_due_sources")
@patch("scripts.watch_sources_poll.init_db")
@patch("scripts.watch_sources_poll.load_config")
def test_main_success_returns_0(mock_load, mock_init, mock_poll, tmp_path):
    mock_load.return_value = _FakeConfig(tmp_path)
    mock_init.return_value = _FakeConn()
    mock_poll.return_value = [
        {
            "page_url": "https://x.example/",
            "new_items": ["a"],
            "submissions": [{"status": "pending"}],
        }
    ]

    assert main() == 0
