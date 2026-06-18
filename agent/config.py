"""Configuration loader — typed config from config.yaml + .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """KB Agent configuration, loaded from config.yaml with .env overrides."""

    # ── Paths ──
    kb_agent_dir: Path = Path("/root/kb-agent")
    vault_path: Path = Path("/root/kb-agent-vault")
    cookie_dir: Path = Path("/root/kb-agent/cookies")
    temp_dir: Path = Path("/tmp")

    # ── Concurrency ──
    video_semaphore: int = 1
    text_parallel: int = 3

    # ── Retry ──
    retry_delays: list[int] = field(default_factory=lambda: [5, 15, 60, 180])
    max_retries: int = 5

    # ── Quality gates ──
    book_min_lines: int = 100
    book_min_quotes: int = 4
    book_min_size: int = 8000
    article_min_quotes: int = 4
    article_min_size: int = 5000
    youtube_min_concepts: int = 3
    youtube_min_key_points: int = 8

    # ── LLM ──
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""

    # ── MinIO ──
    minio_endpoint: str = "http://localhost:9000"
    minio_bucket: str = "kb-agent-archive"
    minio_access_key: str = ""
    minio_secret_key: str = ""

    # ── Firecrawl ──
    firecrawl_url: str = "http://localhost:3002/v2/scrape"

    # ── Webhook ──
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 5000


def _env_or_default(raw_value, env_key: str, default, convert=None):
    """Merge env var into a config value — env takes priority."""
    env_val = os.getenv(env_key)
    if env_val is not None:
        if convert:
            return convert(env_val)
        return env_val
    return raw_value if raw_value is not None else default


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    """Load configuration from YAML file and .env, returning a typed Config object.

    Args:
        config_path: Path to config.yaml (default: /root/kb-agent/config.yaml)
        env_path: Path to .env file (default: /root/kb-agent/.env)

    Returns:
        Config dataclass with all values resolved.
    """
    if config_path is None:
        config_path = Path("/root/kb-agent/config.yaml")
    if env_path is None:
        env_path = Path("/root/kb-agent/.env")

    # Load .env if it exists
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML config
    raw: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # Flatten quality subsection
    quality = raw.get("quality", {})

    return Config(
        # Paths
        kb_agent_dir=Path(raw.get("kb_agent_dir", "/root/kb-agent")),
        vault_path=Path(raw.get("vault_path", "/root/kb-agent-vault")),
        cookie_dir=Path(raw.get("cookie_dir", "/root/kb-agent/cookies")),
        temp_dir=Path(raw.get("temp_dir", "/tmp")),
        # Concurrency
        video_semaphore=int(_env_or_default(raw.get("video_semaphore"), "KB_VIDEO_SEMAPHORE", 1)),
        text_parallel=int(_env_or_default(raw.get("text_parallel"), "KB_TEXT_PARALLEL", 3)),
        # Retry
        retry_delays=raw.get("retry_delays", [5, 15, 60, 180]),
        max_retries=int(_env_or_default(raw.get("max_retries"), "KB_MAX_RETRIES", 5)),
        # Quality
        book_min_lines=int(quality.get("book_min_lines", 100)),
        book_min_quotes=int(quality.get("book_min_quotes", 4)),
        book_min_size=int(quality.get("book_min_size", 8000)),
        article_min_quotes=int(quality.get("article_min_quotes", 4)),
        article_min_size=int(quality.get("article_min_size", 5000)),
        youtube_min_concepts=int(quality.get("youtube_min_concepts", 3)),
        youtube_min_key_points=int(quality.get("youtube_min_key_points", 8)),
        # LLM
        llm_provider=str(_env_or_default(raw.get("llm_provider"), "KB_LLM_PROVIDER", "deepseek")),
        llm_model=str(_env_or_default(raw.get("llm_model"), "KB_LLM_MODEL", "deepseek-v4-pro")),
        llm_base_url=str(_env_or_default(raw.get("llm_base_url"), "KB_LLM_BASE_URL", "https://api.deepseek.com/v1")),
        llm_api_key=str(_env_or_default(raw.get("llm_api_key"), "DEEPSEEK_API_KEY", "")),
        # MinIO
        minio_endpoint=str(_env_or_default(raw.get("minio_endpoint"), "MINIO_ENDPOINT", "http://localhost:9000")),
        minio_bucket=str(_env_or_default(raw.get("minio_bucket"), "MINIO_BUCKET", "kb-agent-archive")),
        minio_access_key=str(_env_or_default(raw.get("minio_access_key"), "MINIO_ACCESS_KEY", "")),
        minio_secret_key=str(_env_or_default(raw.get("minio_secret_key"), "MINIO_SECRET_KEY", "")),
        # Firecrawl
        firecrawl_url=str(_env_or_default(raw.get("firecrawl_url"), "FIRECRAWL_URL", "http://localhost:3002/v2/scrape")),
        # Webhook
        webhook_host=str(raw.get("webhook_host", "0.0.0.0")),
        webhook_port=int(_env_or_default(raw.get("webhook_port"), "KB_WEBHOOK_PORT", 5000)),
    )
