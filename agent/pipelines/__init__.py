"""Pipeline registry — maps content types to ordered step lists.

Each pipeline is a list of Step objects defined in submodules.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Step:
    """A single pipeline step definition."""
    name: str
    fn: Callable
    on_error: str = "retry"  # retry | pause | fail


# Pipeline registry: content_type → list[Step]
PIPELINES: dict[str, list[Step]] = {}


def register_pipeline(content_type: str, steps: list[Step]):
    """Register a pipeline for a content type."""
    PIPELINES[content_type] = steps


def get_pipeline(content_type: str) -> list[Step] | None:
    """Get the pipeline steps for a content type.

    Falls back to 'web' pipeline if content_type not registered.
    """
    return PIPELINES.get(content_type, PIPELINES.get("web"))


def list_pipelines() -> list[str]:
    """List all registered content types."""
    return list(PIPELINES.keys())
