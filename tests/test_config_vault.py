"""Tests for main vault config pointer (retire A/B kb-agent-vault)."""

from pathlib import Path

import pytest
import yaml

from agent.config import (
    DEFAULT_VAULT_PATH,
    RETIRED_AB_VAULT_PATH,
    load_config,
)


REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_repo_config_yaml_points_at_main_vault():
    raw = yaml.safe_load(REPO_CONFIG.read_text()) or {}
    assert Path(raw["vault_path"]) == DEFAULT_VAULT_PATH
    assert raw["vault_path"] != str(RETIRED_AB_VAULT_PATH)


def test_load_config_default_vault_is_main_vault(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("temp_dir: /tmp\n")
    cfg = load_config(config_path=config_path, env_path=tmp_path / "missing.env")
    assert cfg.vault_path == DEFAULT_VAULT_PATH


def test_load_config_vault_path_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-vault"
    monkeypatch.setenv("KB_VAULT_PATH", str(override))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"vault_path: {RETIRED_AB_VAULT_PATH}\n")
    cfg = load_config(config_path=config_path, env_path=tmp_path / "missing.env")
    assert cfg.vault_path == override


def test_load_config_obsidian_vault_path_env_override(tmp_path, monkeypatch):
    override = tmp_path / "hermes-vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(override))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("temp_dir: /tmp\n")
    cfg = load_config(config_path=config_path, env_path=tmp_path / "missing.env")
    assert cfg.vault_path == override


def test_load_config_rejects_retired_ab_vault(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"vault_path: {RETIRED_AB_VAULT_PATH}\n")
    with pytest.raises(ValueError, match="retired A/B vault"):
        load_config(config_path=config_path, env_path=tmp_path / "missing.env")
