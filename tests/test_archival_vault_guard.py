"""Ensure Git push refuses the retired A/B vault."""

from pathlib import Path

import pytest

from agent.archival import step_git_push
from agent.config import Config, RETIRED_AB_VAULT_PATH
from agent.exceptions import StepError


def test_git_push_refuses_retired_ab_vault(tmp_path):
    config = Config(vault_path=RETIRED_AB_VAULT_PATH, temp_dir=tmp_path / "tmp")
    task = {"id": "abc", "slug": "test-slug", "content_type": "web"}

    with pytest.raises(StepError, match="retired A/B vault"):
        step_git_push(task, config)
