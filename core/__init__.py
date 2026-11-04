"""kb-agent core — adapter-agnostic domain API (no Flask/MCP imports).

Public entry points mirror the MCP tool surface defined in docs/prd-kb-mcp-consolidation.md.
Flask (agent.webhook) and the future MCP server become thin adapters over these functions.
"""

from core.detect import detect_platform, detect_type, make_detect_type_fn
from core.notes import get_note, search_notes
from core.pipelines import PIPELINE_STEPS, resolve_pipeline_steps
from core.resume import normalize_resume_reason_pattern, resume_task, retry_task
from core.status import (
    compute_staleness,
    format_elapsed,
    get_last_log_line,
    get_task,
    list_tasks,
)
from core.submit import SubmitDependencies, submit_url
from core.types import (
    DetectTypeResult,
    NoteContent,
    NoteSearchHit,
    NoteSearchResult,
    ResumeResult,
    RetryResult,
    SubmitOutcome,
    SubmitResult,
    TaskListResult,
    TaskRecord,
    TaskStepRecord,
)
from core.util import slugify

__all__ = [
    "PIPELINE_STEPS",
    "DetectTypeResult",
    "NoteContent",
    "NoteSearchHit",
    "NoteSearchResult",
    "ResumeResult",
    "RetryResult",
    "SubmitDependencies",
    "SubmitOutcome",
    "SubmitResult",
    "TaskListResult",
    "TaskRecord",
    "TaskStepRecord",
    "compute_staleness",
    "detect_platform",
    "detect_type",
    "format_elapsed",
    "get_last_log_line",
    "get_note",
    "get_task",
    "list_tasks",
    "make_detect_type_fn",
    "normalize_resume_reason_pattern",
    "resolve_pipeline_steps",
    "resume_task",
    "retry_task",
    "search_notes",
    "slugify",
    "submit_url",
]
