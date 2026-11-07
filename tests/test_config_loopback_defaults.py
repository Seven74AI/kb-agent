"""config.yaml binds webhook and MCP to loopback by default."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_loopback_hosts():
    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f) or {}
    assert raw.get("webhook_host") == "127.0.0.1"
    assert raw.get("mcp_host") == "127.0.0.1"
