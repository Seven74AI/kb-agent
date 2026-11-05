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


def import_all_pipelines() -> None:
    """Import every pipeline submodule so register_pipeline() populates PIPELINES.

    The consumer calls this at startup; standalone scripts (the batch collector)
    rely on get_pipeline()'s auto-import. Importing a module runs its top-level
    register_pipeline() calls.
    """
    from . import (  # noqa: F401
        books,
        bunny_stream,
        instagram,
        mega_video,
        threads,
        web,
        wikispooks,
        youtube,
    )


def get_pipeline(content_type: str) -> list[Step] | None:
    """Get the pipeline steps for a content type.

    Falls back to 'web' pipeline if content_type not registered. Auto-imports
    the pipeline modules on first call so callers that never import them (e.g.
    the standalone batch collector) still get a populated registry.
    """
    if not PIPELINES:
        import_all_pipelines()
    return PIPELINES.get(content_type, PIPELINES.get("web"))


def list_pipelines() -> list[str]:
    """List all registered content types."""
    return list(PIPELINES.keys())


def get_pipeline_names(content_type: str, platform: str = "") -> list[str]:
    """Resolve step names for a content type, with subtype→platform→web fallback.

    Replaces ``core.pipelines.steps_for_content_type`` — derives names from the
    Step-object registry so there is a single source of truth. Auto-imports the
    pipeline modules on first call (same as :func:`get_pipeline`).
    """
    if not PIPELINES:
        import_all_pipelines()
    for key in (content_type, platform):
        if key and key in PIPELINES:
            return [s.name for s in PIPELINES[key]]
    return [s.name for s in (PIPELINES.get("web") or [])]
