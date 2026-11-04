"""URL platform and content-type detection (no Flask/MCP)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from core.types import DetectTypeResult

# Regex platform table — canonical copy for core; agent.detection will re-import.
PLATFORM_PATTERNS: list[tuple[str, str]] = [
    (
        r"youtube\.com/watch|youtu\.be/|youtube\.com/shorts|"
        r"m\.youtube\.com|youtube\.com/embed|music\.youtube\.com",
        "youtube",
    ),
    (r"threads\.(net|com)/@", "threads"),
    (r"instagram\.com/reel/", "instagram_reel"),
    (r"instagram\.com/p/", "instagram_post"),
    (r"substack\.com/", "substack"),
    (r"mega\.nz/", "mega"),
    (r"player\.mediadelivery\.net/", "bunny_stream"),
    (r"terragrandir\.com/", "bunny_stream"),
    (r"wikispooks\.org/", "wikispooks"),
    (r"\.(epub|pdf|mobi|azw3)$", "book"),
]


def detect_platform(url: str) -> str:
    """Classify a URL by platform using regex patterns (deterministic, no I/O)."""
    for pattern, platform in PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return platform
    return "web"


class ContentSubtypeDetector(Protocol):
    """Pluggable subtype resolver (Jev + legacy fetch lives in agent.detection today)."""

    def __call__(self, url: str, platform: str) -> str: ...


def detect_type(
    url: str,
    *,
    subtype_detector: ContentSubtypeDetector | None = None,
) -> DetectTypeResult:
    """Detect platform and content subtype for a URL.

    Args:
        url: Source URL.
        subtype_detector: Optional callable ``(url, platform) -> content_type``.
            When omitted, ``content_type`` equals ``platform`` (platform-only probe).

    Returns:
        DetectTypeResult with platform and content_type.
    """
    platform = detect_platform(url)
    if subtype_detector is None:
        content_type = platform
    else:
        content_type = subtype_detector(url, platform)
    return DetectTypeResult(platform=platform, content_type=content_type)


def make_detect_type_fn(
    subtype_detector: ContentSubtypeDetector,
) -> Callable[[str], DetectTypeResult]:
    """Bind a subtype detector for MCP ``detect_type`` / CLI use."""

    def _detect(url: str) -> DetectTypeResult:
        return detect_type(url, subtype_detector=subtype_detector)

    return _detect
