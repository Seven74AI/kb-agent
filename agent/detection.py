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
    # Bunny Stream (TDAH summit / terragrandir, mediadelivery.net)
    (r"player\.mediadelivery\.net/", "bunny_stream"),
    (r"terragrandir\.com/", "bunny_stream"),
    # Wikispooks
    (r"wikispooks\.org/", "wikispooks"),
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
        substack, mega, wikispooks, book, or web.
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
        - threads -> 'threads_text' or 'threads_video' (based on video_versions in page JSON)
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
    """Check a Substack URL for content type via preloads JSON + SSR HTML.

    Two-layer detection:
    Layer 1: Parse preloads JSON for attachments type:video or type:image.
    Layer 2: SSR HTML fallback — grep for Video player aria-label or image-post.

    Returns 'substack_text', 'substack_video', or 'substack_image'.
    """
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        html = result.stdout

        # ── Layer 1: Preloads JSON ──
        import json as _json
        preloads_match = re.search(
            r'window\.__preloads\s*=\s*(\{.+?\});\s*</script>',
            html, re.DOTALL,
        )
        if preloads_match:
            try:
                preloads = _json.loads(preloads_match.group(1))
                attachments = preloads.get("attachments", [])
                for att in attachments:
                    att_type = att.get("type", "")
                    if att_type == "video":
                        return "substack_video"
                    elif att_type == "image":
                        return "substack_image"
            except (_json.JSONDecodeError, KeyError, TypeError):
                pass  # Preloads parse failed — fall through to Layer 2

        # ── Layer 2: SSR HTML fallback ──
        if "/api/v1/video/upload/" in html:
            return "substack_video"
        if 'aria-label="Video player"' in html:
            return "substack_video"

        if '<body class="image-post' in html or 'image-gallery' in html.lower():
            return "substack_image"
    except Exception:
        pass
    return "substack_text"


# ── Mega.nz Detection ────────────────────────────────────────────────────────

# Video extensions recognized for Mega video classification
MEGA_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".flv"}
# Book extensions recognized for Mega book classification
MEGA_BOOK_EXTENSIONS = {".epub", ".pdf", ".mobi", ".azw3", ".djvu", ".cbr", ".cbz"}
# Archive extensions that may contain book batches
MEGA_ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}


def _detect_mega_subtype(url: str) -> str:
    """Check a Mega URL for content type via megadl metadata.

    For /file/ URLs: queries filename via megadl --print-names,
    classifies by extension.
    For /folder/ URLs: samples first 10 filenames, classifies
    by majority extension.

    Returns:
        'book_batch' — zip/rar/7z archive or majority book extensions
        'book' — single book file (.epub/.pdf/.mobi/.azw3)
        'mega_video' — video file (.mp4/.mkv/.webm)
        'unknown' — could not determine or mixed content
    """
    try:
        result = subprocess.run(
            ["megadl", "--print-names", url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return "unknown"

        filenames = [
            line.strip() for line in result.stdout.splitlines()
            if line.strip()
        ]
        if not filenames:
            return "unknown"

        return _classify_mega_filenames(filenames)
    except (subprocess.TimeoutExpired, Exception):
        return "unknown"


def _classify_mega_filenames(filenames: list[str]) -> str:
    """Classify a list of Mega filenames into a content subtype.

    Rules (in priority order):
    1. Any single .zip/.rar/.7z → 'book_batch' (likely a book archive)
    2. Majority video extensions → 'mega_video'
    3. Majority book extensions → 'book_batch' (multiple books)
    4. Single book → 'book'
    5. Unknown / mixed → 'unknown'
    """
    extensions: dict[str, int] = {}
    for fname in filenames:
        # Get lowercase extension
        ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        extensions[ext] = extensions.get(ext, 0) + 1

    total = sum(extensions.values())
    if total == 0:
        return "unknown"

    # Check for archive files first (priority)
    archive_count = sum(
        n for ext, n in extensions.items()
        if ext in MEGA_ARCHIVE_EXTENSIONS
    )
    if archive_count > 0:
        return "book_batch"

    # Check for video files
    video_count = sum(
        n for ext, n in extensions.items()
        if ext in MEGA_VIDEO_EXTENSIONS
    )
    if video_count > 0 and video_count > total * 0.5:
        return "mega_video"

    # Check for book files
    book_count = sum(
        n for ext, n in extensions.items()
        if ext in MEGA_BOOK_EXTENSIONS
    )

    if total == 1 and book_count == 1:
        return "book"
    elif book_count > total * 0.5:
        return "book_batch"

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
    elif platform in ("substack", "web", "wikispooks"):
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
