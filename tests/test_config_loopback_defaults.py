"""config.yaml and load_config() default webhook/MCP to loopback and MCP port 8002."""

from pathlib import Path

import yaml

from agent.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_loopback_hosts():
    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f) or {}
    assert raw.get("webhook_host") == "127.0.0.1"
    assert raw.get("mcp_host") == "127.0.0.1"


def test_load_config_code_fallbacks_loopback_and_mcp_port(tmp_path):
    minimal = tmp_path / "config.yaml"
    minimal.write_text("kb_agent_dir: {}\n".format(tmp_path), encoding="utf-8")
    cfg = load_config(config_path=minimal, env_path=tmp_path / "missing.env")
    assert cfg.webhook_host == "127.0.0.1"
    assert cfg.mcp_host == "127.0.0.1"
    assert cfg.mcp_port == 8002
