"""Shared types for the kb-agent core API (adapter-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SubmitOutcome(str, Enum):
    """Result of enqueueing a URL for processing."""

    CREATED = "pending"
    SKIPPED = "skipped"
    PAUSED = "paused"


@dataclass(frozen=True)
class DetectTypeResult:
    """Platform + content subtype for a URL."""

    platform: str
    content_type: str


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of submit_url."""

    outcome: SubmitOutcome
    platform: str
    task_id: str | None = None
    content_type: str | None = None
    slug: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize for HTTP/MCP adapters."""
        data: dict[str, Any] = {
            "status": self.outcome.value,
            "platform": self.platform,
        }
        if self.task_id is not None:
            data["task_id"] = self.task_id
        if self.content_type is not None:
            data["content_type"] = self.content_type
        if self.slug is not None:
            data["slug"] = self.slug
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class TaskStepRecord:
    id: int
    step_order: int
    name: str
    status: str
    output: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class TaskRecord:
    id: str
    source_url: str
    content_type: str
    status: str
    slug: str | None
    error: str | None
    retries: int
    created_at: str
    updated_at: str | None
    steps: list[TaskStepRecord] = field(default_factory=list)
    last_log: str | None = None
    elapsed: str | None = None
    staleness: str | None = None
    is_stale: bool = False


@dataclass(frozen=True)
class TaskListResult:
    tasks: list[TaskRecord]
    counts: dict[str, int]


@dataclass(frozen=True)
class ResumeResult:
    resumed: int
    reason_pattern: str


@dataclass(frozen=True)
class RetryResult:
    task_id: str
    retried: bool
    previous_status: str | None = None


@dataclass(frozen=True)
class NoteSearchHit:
    path: str
    slug: str
    snippet: str | None = None


@dataclass(frozen=True)
class NoteSearchResult:
    query: str
    hits: list[NoteSearchHit]


@dataclass(frozen=True)
class NoteContent:
    path: str
    slug: str
    body: str
    frontmatter_source_url: str | None = None
