"""Content-type-scoped step resolution (STEP_REGISTRY last-wins bug)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from agent.consumer import STEP_REGISTRY, _resolve_step_entry, register_step
from agent.exceptions import ErrorType, StepError
from agent.pipelines import PIPELINES, Step, register_pipeline


@pytest.fixture
def isolated_step_resolution():
    """Snapshot and restore PIPELINES / STEP_REGISTRY around mutation tests."""
    pipelines_before = dict(PIPELINES)
    registry_before = dict(STEP_REGISTRY)
    yield
    PIPELINES.clear()
    PIPELINES.update(pipelines_before)
    STEP_REGISTRY.clear()
    STEP_REGISTRY.update(registry_before)


def test_resolve_step_entry_content_type_scoped_on_error(isolated_step_resolution):
    """Shared step names resolve per pipeline, not global last-wins registry."""
    fn_retry = MagicMock()
    fn_fail = MagicMock()

    register_pipeline(
        "test_content_a",
        [Step(name="shared", fn=fn_retry, on_error="retry")],
    )
    register_pipeline(
        "test_content_b",
        [Step(name="shared", fn=fn_fail, on_error="fail")],
    )
    # Global registry overwritten as if a later pipeline registered last
    register_step("shared", fn_fail, "fail")

    entry_a = _resolve_step_entry("test_content_a", "shared")
    entry_b = _resolve_step_entry("test_content_b", "shared")

    assert entry_a == (fn_retry, "retry")
    assert entry_b == (fn_fail, "fail")


def test_resolve_step_entry_unknown_step_returns_none(isolated_step_resolution):
    register_pipeline("test_only", [Step(name="only", fn=MagicMock(), on_error="pause")])

    assert _resolve_step_entry("test_only", "missing") is None


def test_resolve_step_entry_missing_pipeline_raises(isolated_step_resolution):
    with pytest.raises(StepError) as exc_info:
        _resolve_step_entry("no_such_pipeline", "any")

    assert exc_info.value.args[0] == "No pipeline registered for content_type='no_such_pipeline'"
    assert exc_info.value.error_type == ErrorType.FAIL
    assert exc_info.value.step_name == "any"
