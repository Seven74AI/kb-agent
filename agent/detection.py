"""Platform detection and pre-flight health checks.

Deterministic URL pattern matching (no ML, no external calls except health checks).
"""

import re
import subprocess
from pathlib import Path


# ── URL Pattern Matching ────────────────────────────────────────────────────

PLATFORM_PATTERNS: list[tuple[str, str]] = [
    # YouTube — 6 URL patterns
    (r"youtube\.com/watch|youtu\.be/|youtube\.com/shorts|"
     r"m\.youtube\.com|youtube\.com/embed|music\.youtube\.com", "youtube"),
    # Threads
    (r"threads\.(net|com)/@", "threads"),
    # Instagram
    (r"instagram\.com/reel/", "instagram_reel"),
    (r"instagram\.com/p/", "instagram_post"),
    # Substack
    (r"substack\.com/", "substack"),
    # Mega
    (r"mega\.nz/", "mega"),
    # Books (direct upload by file extension)
    (r"\.(epub|pdf|mobi|azw3)$", "book"),
]


def detect_platform(url: str) -> str:
    """Classify a URL by platform using regex patterns.

    Order matters — YouTube is checked before generic web patterns.
    Returns 'web' if no specific platform matches.

    Args:
        url: The URL to classify.

    Returns:
        Platform string: youtube, threads, instagram_reel, instagram_post,
        substack, mega, book, or web.
    """
    for pattern, platform in PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return platform
    return "web"


# ── Content Subtype Detection ────────────────────────────────────────────────

def detect_content_subtype(url: str, platform: str) -> str:
    """Determine the content subtype for platforms with variants.

    Args:
        url: The source URL.
        platform: Detected platform from detect_platform().

    Returns:
        Content subtype string. For most platforms, just returns the platform.
        - threads → 'text' or 'video' (based on video_versions in page JSON)
        - substack → 'text', 'image', or 'video'
        - mega → 'book_batch', 'video', or 'unknown'
    """
    if platform == "threads":
        return _detect_threads_subtype(url)
    elif platform == "substack":
        return _detect_substack_subtype(url)
    elif platform == "mega":
        return _detect_mega_subtype(url)
    else:
        # Single content type — no subtype needed
        return platform


def _detect_threads_subtype(url: str) -> str:
    """Check a Threads URL for video_versions in the page JSON.

    Uses cookies + Googlebot UA for reliable scraping.
    Returns 'threads_video' if video_versions is present, 'threads_text' otherwise.

    Handles media_type=19 (Instagram video cross-post) — checks video_versions
    regardless of outer media type.
    """
    try:
        cookie_path = Path("/root/.hermes/cookies/threads_cookies.txt")
        cmd = [
            "curl", "-s", "--max-time", "15",
            "-H", "User-Agent: Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.204 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "-H", "Accept: text/html,application/xhtml+xml",
        ]
        if cookie_path.exists():
            cmd.extend(["-b", str(cookie_path)])
        cmd.append(url)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20,
        )
        if 'video_versions' in result.stdout:
            return "threads_video"
    except (subprocess.TimeoutExpired, Exception):
        pass
    return "threads_text"


def _detect_substack_subtype(url: str) -> str:
    """Check a Substack URL for content type via preloads JSON.

    Falls back to SSR HTML if preloads unavailable.
    Returns 'text', 'image', or 'video'.
    """
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        html = result.stdout
        if "/api/v1/video/upload/" in html:
            return "video"
        # Look for SSR body tag hints
        if '<body class="image-post' in html or 'image-gallery' in html.lower():
            return "image"
    except Exception:
        pass
    return "text"


def _detect_mega_subtype(url: str) -> str:
    """Check a Mega URL for content type (currently stub — always 'unknown').

    Full implementation requires megadl metadata inspection.
    """
    # Stub — full implementation in Phase 1 (Mega pipeline)
    return "unknown"


# ── Pre-Flight Health Checks ─────────────────────────────────────────────────

def preflight_health(platform: str) -> tuple[bool, str]:
    """Run platform-specific health checks before queuing a task.

    Args:
        platform: The detected platform from detect_platform().

    Returns:
        (ok, reason) tuple. ok=True means system is healthy for this platform.
        ok=False means the task should be paused with the given reason.
    """
    if platform == "youtube":
        return _check_youtube()
    elif platform in ("instagram_reel", "instagram_post"):
        return _check_instagram()
    elif platform == "threads":
        return _check_threads()
    elif platform in ("substack", "web"):
        return _check_firecrawl()
    elif platform == "mega":
        return _check_mega()
    elif platform == "book":
        return True, ""
    return True, ""


def _check_youtube() -> tuple[bool, str]:
    """Tier 1: Check YouTube cookies (LOGIN_INFO + SAPISID + 3PSID ≥ 3)."""
    cookie_file = Path("/root/kb-agent/cookies/yt_cookies.txt")
    if not cookie_file.exists():
        cookie_file = Path("/root/.hermes/cookies/yt_cookies.txt")
    if not cookie_file.exists():
        return False, "YouTube cookies missing — no yt_cookies.txt found"

    try:
        content = cookie_file.read_text()
        patterns = ["LOGIN_INFO", "SAPISID", "__Secure-3PSID"]
        found = sum(1 for p in patterns if p in content)
        if found < 3:
            return False, f"YouTube cookies degraded — {found}/3 required fields present"
        return True, ""
    except Exception as e:
        return False, f"YouTube cookie check failed: {e}"


def _check_instagram() -> tuple[bool, str]:
    """Check Instagram sessionid cookie presence."""
    cookie_file = Path("/root/kb-agent/cookies/ig_cookies.txt")
    if not cookie_file.exists():
        cookie_file = Path("/root/.hermes/cookies/ig_cookies.txt")
    if not cookie_file.exists():
        return False, "Instagram cookies missing — no ig_cookies.txt found"

    try:
        content = cookie_file.read_text()
        if "sessionid" not in content:
            return False, "Instagram cookie missing sessionid"
        return True, ""
    except Exception as e:
        return False, f"Instagram cookie check failed: {e}"


def _check_threads() -> tuple[bool, str]:
    """Check Threads cookies file exists."""
    cookie_file = Path("/root/kb-agent/cookies/threads_cookies.txt")
    if not cookie_file.exists():
        cookie_file = Path("/root/.hermes/cookies/threads_cookies.txt")
    if not cookie_file.exists():
        return False, "Threads cookies missing — no threads_cookies.txt found"
    return True, ""


def _check_firecrawl() -> tuple[bool, str]:
    """Check Firecrawl health via /scrape endpoint."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "10", "http://localhost:3002/v2/scrape"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, f"Firecrawl unreachable (exit {result.returncode})"
        # Firecrawl returns a valid JSON response even on /scrape with no body
        return True, ""
    except Exception as e:
        return False, f"Firecrawl health check failed: {e}"


def _check_mega() -> tuple[bool, str]:
    """Check megadl availability."""
    try:
        result = subprocess.run(
            ["which", "megadl"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False, "megadl not installed"
        return True, ""
    except Exception as e:
        return False, f"megadl check failed: {e}"


# ── Deduplication ────────────────────────────────────────────────────────────

def dedup_check(url: str, vault_path: Path) -> bool:
    """Check if a URL already exists in the Obsidian vault.

    Greps all markdown notes in the vault for the source_url.

    Args:
        url: The source URL to check.
        vault_path: Path to the Obsidian vault root.

    Returns:
        True if the URL already exists in the vault, False otherwise.
    """
    try:
        result = subprocess.run(
            ["grep", "-rl", f"source_url: {url}", str(vault_path / "Knowledge base")],
            capture_output=True, text=True, timeout=30,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False
