"""Configuration loader — typed config from config.yaml + .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root, derived from this file's location — works in CI and any clone.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Production Obsidian vault (Seven74AI/obsidian-vault) on the VPS.
DEFAULT_VAULT_PATH = Path("/root/Documents/Obsidian Vault")
# Retired kb-agent A/B comparison vault — must not receive new writes.
RETIRED_AB_VAULT_PATH = Path("/root/kb-agent-vault")


@dataclass
class Config:
    """KB Agent configuration, loaded from config.yaml with .env overrides."""

    # ── Paths ──
    kb_agent_dir: Path = REPO_ROOT
    vault_path: Path = DEFAULT_VAULT_PATH
    cookie_dir: Path = REPO_ROOT / "cookies"
    temp_dir: Path = Path("/tmp")

    # ── Concurrency ──
    video_semaphore: int = 1
    text_parallel: int = 3

    # ── Watch sources ──
    # In-process background scheduler for recurring page watches. Disable when
    # cron drives polling (scripts/watch_sources_poll.py) to avoid dual pollers.
    watch_scheduler_enabled: bool = True

    # ── Retry ──
    retry_delays: list[int] = field(default_factory=lambda: [5, 15, 60, 180])
    max_retries: int = 5

    # ── Quality gates ──
    book_min_lines: int = 100
    book_min_quotes: int = 4
    book_min_size: int = 8000
    book_min_fact_checks: int = 2
    article_min_quotes: int = 4
    article_min_size: int = 5000
    article_min_fact_checks: int = 2
    youtube_min_concepts: int = 3
    youtube_min_key_points: int = 8
    youtube_min_density: int = 100
    threads_video_min_density: int = 100
    threads_video_min_key_points: int = 3
    threads_text_min_quotes: int = 2
    threads_text_min_size: int = 2000
    ig_reel_min_concepts: int = 2
    ig_reel_min_key_points: int = 3
    ig_carousel_min_size: int = 2000

    # ── LLM ──
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""           # DeepSeek API key
    edgee_api_key: str = ""         # Edgee API key (separate from DeepSeek)

    # ── Object storage (S3-compatible via mc) ──
    storage_bucket: str = "knowledge-base"
    storage_alias: str = "tigris"

    # ── Firecrawl ──
    firecrawl_url: str = "http://localhost:3002/v2/scrape"

    # ── Webhook ──
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 5000

    # ── Cloudflare Access (browser auth for the dashboard + its forms) ──
    # Both empty by default → the browser path fails closed until the operator
    # sets CF_ACCESS_TEAM + CF_ACCESS_AUD (the Zero Trust team subdomain and the
    # application's AUD tag). The live team subdomain is visible in the Access
    # login redirect (e.g. https://<team>.cloudflareaccess.com).
    cloudflare_access_team: str = ""
    cloudflare_access_aud: str = ""

    # ── MCP (stateless streamable HTTP) ──
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8002
    mcp_path: str = "/mcp"

    # ── Transcription ──
    transcription_backend: str = "cpu"  # cpu | modal
    modal_cli: str = "/root/modal/venv/bin/modal"

    # ── Notifications (Telegram) ──
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


def _env_or_default(raw_value, env_key: str, default, convert=None):
    """Merge env var into a config value — env takes priority."""
    env_val = os.getenv(env_key)
    if env_val is not None:
        if convert:
            return convert(env_val)
        return env_val
    return raw_value if raw_value is not None else default


def _as_bool(value) -> bool:
    """Coerce an env string to bool (true/1/yes/on → True)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")
def _resolve_vault_path(raw: dict) -> Path:
    """Resolve vault_path from env, YAML, or default; block the retired A/B vault."""
    env_path = os.getenv("KB_VAULT_PATH") or os.getenv("OBSIDIAN_VAULT_PATH")
    if env_path is not None:
        path = Path(env_path)
    else:
        path = Path(raw.get("vault_path", str(DEFAULT_VAULT_PATH)))

    try:
        resolved = path.resolve()
        retired = RETIRED_AB_VAULT_PATH.resolve()
    except (OSError, RuntimeError):
        resolved = path
        retired = RETIRED_AB_VAULT_PATH

    if resolved == retired:
        raise ValueError(
            f"vault_path {path} is the retired A/B vault; point kb-agent at "
            f"{DEFAULT_VAULT_PATH} (Seven74AI/obsidian-vault) instead"
        )
    return path


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    """Load configuration from YAML file and .env, returning a typed Config object.

    Args:
        config_path: Path to config.yaml (default: /root/kb-agent/config.yaml)
        env_path: Path to .env file (default: /root/kb-agent/.env)

    Returns:
        Config dataclass with all values resolved.
    """
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    if env_path is None:
        env_path = REPO_ROOT / ".env"

    # Load .env if it exists
    try:
        env_exists = env_path.exists()
    except (PermissionError, OSError):
        env_exists = False
    if env_exists:
        load_dotenv(env_path)

    # Load YAML config
    raw: dict = {}
    try:
        config_exists = config_path.exists()
    except (PermissionError, OSError):
        config_exists = False
    if config_exists:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # Flatten quality subsection
    quality = raw.get("quality", {})

    _legacy_bucket_yaml_key = "min" + "io_bucket"
    yaml_storage_bucket = raw.get("storage_bucket") or raw.get(_legacy_bucket_yaml_key)

    return Config(
        # Paths
        kb_agent_dir=Path(raw.get("kb_agent_dir", str(REPO_ROOT))),
        vault_path=_resolve_vault_path(raw),
        cookie_dir=Path(raw.get("cookie_dir", str(REPO_ROOT / "cookies"))),
        temp_dir=Path(raw.get("temp_dir", "/tmp")),
        # Concurrency
        video_semaphore=int(_env_or_default(raw.get("video_semaphore"), "KB_VIDEO_SEMAPHORE", 1)),
        text_parallel=int(_env_or_default(raw.get("text_parallel"), "KB_TEXT_PARALLEL", 3)),
        # Watch sources
        watch_scheduler_enabled=_env_or_default(
            raw.get("watch_scheduler_enabled"), "KB_WATCH_SCHEDULER_ENABLED", True, convert=_as_bool
        ),
        # Retry
        retry_delays=raw.get("retry_delays", [5, 15, 60, 180]),
        max_retries=int(_env_or_default(raw.get("max_retries"), "KB_MAX_RETRIES", 5)),
        # Quality
        book_min_lines=int(quality.get("book_min_lines", 100)),
        book_min_quotes=int(quality.get("book_min_quotes", 4)),
        book_min_size=int(quality.get("book_min_size", 8000)),
        book_min_fact_checks=int(quality.get("book_min_fact_checks", 2)),
        article_min_quotes=int(quality.get("article_min_quotes", 4)),
        article_min_size=int(quality.get("article_min_size", 5000)),
        article_min_fact_checks=int(quality.get("article_min_fact_checks", 2)),
        youtube_min_concepts=int(quality.get("youtube_min_concepts", 3)),
        youtube_min_key_points=int(quality.get("youtube_min_key_points", 8)),
        youtube_min_density=int(quality.get("youtube_min_density", 100)),
        threads_video_min_density=int(quality.get("threads_video_min_density", 100)),
        threads_text_min_quotes=int(quality.get("threads_text_min_quotes", 2)),
        threads_text_min_size=int(quality.get("threads_text_min_size", 2000)),
        ig_reel_min_concepts=int(quality.get("ig_reel_min_concepts", 2)),
        ig_reel_min_key_points=int(quality.get("ig_reel_min_key_points", 3)),
        ig_carousel_min_size=int(quality.get("ig_carousel_min_size", 2000)),
        # LLM
        llm_provider=str(_env_or_default(raw.get("llm_provider"), "KB_LLM_PROVIDER", "deepseek")),
        llm_model=str(_env_or_default(raw.get("llm_model"), "KB_LLM_MODEL", "deepseek-v4-pro")),
        llm_base_url=str(_env_or_default(raw.get("llm_base_url"), "KB_LLM_BASE_URL", "https://api.deepseek.com/v1")),
        llm_api_key=str(_env_or_default(raw.get("llm_api_key"), "DEEPSEEK_API_KEY", "")),
        edgee_api_key=str(_env_or_default(raw.get("edgee_api_key"), "EDGEE_API_KEY", "")),
        # Object storage
        storage_bucket=str(
            _env_or_default(
                yaml_storage_bucket,
                "STORAGE_BUCKET",
                os.environ.get("MINIO_BUCKET", "knowledge-base"),
            )
        ),
        storage_alias=str(_env_or_default(raw.get("storage_alias"), "STORAGE_ALIAS", "tigris")),
        # Firecrawl
        firecrawl_url=str(_env_or_default(raw.get("firecrawl_url"), "FIRECRAWL_URL", "http://localhost:3002/v2/scrape")),
        # Webhook
        webhook_host=str(raw.get("webhook_host", "127.0.0.1")),
        webhook_port=int(_env_or_default(raw.get("webhook_port"), "KB_WEBHOOK_PORT", 5000)),
        cloudflare_access_team=str(
            _env_or_default(raw.get("cloudflare_access_team"), "CF_ACCESS_TEAM", "")
        ),
        cloudflare_access_aud=str(
            _env_or_default(raw.get("cloudflare_access_aud"), "CF_ACCESS_AUD", "")
        ),
        mcp_host=str(raw.get("mcp_host", "127.0.0.1")),
        mcp_port=int(_env_or_default(raw.get("mcp_port"), "KB_MCP_PORT", 8002)),
        mcp_path=str(raw.get("mcp_path", "/mcp")),
        # Transcription
        transcription_backend=str(
            _env_or_default(raw.get("transcription_backend"), "KB_TRANSCRIPTION_BACKEND", "cpu")
        ),
        modal_cli=str(_env_or_default(raw.get("modal_cli"), "KB_MODAL_CLI", "/root/modal/venv/bin/modal")),
        # Notifications (Telegram)
        telegram_bot_token=str(_env_or_default(raw.get("telegram_bot_token"), "KB_TELEGRAM_BOT_TOKEN", "")),
        telegram_chat_id=str(_env_or_default(raw.get("telegram_chat_id"), "KB_TELEGRAM_CHAT_ID", "")),
    )
