"""Archival steps — MinIO upload and Git push.

Deterministic shell commands for archiving source content and pushing notes.
Used by all pipelines (web, threads, instagram, books, youtube).
"""

import subprocess
from pathlib import Path

from agent.exceptions import ErrorType, StepError


# ── MinIO helpers ─────────────────────────────────────────────────────────────

# Mapping from content-type to MinIO bucket folder
CONTENT_TYPE_TO_MINIO_FOLDER: dict[str, str] = {
    "web": "articles",
    "substack": "articles",
    "instagram_post": "articles",
    "threads": "threads",
    "youtube": "videos",
    "instagram_reel": "reels",
    "book": "books",
    "book_epub": "books",
    "book_pdf": "books",
    "mega": "books",
}

# Mapping from content-type to vault subfolder
CONTENT_TYPE_TO_VAULT_FOLDER: dict[str, str] = {
    "web": "articles",
    "substack": "articles",
    "instagram_post": "articles",
    "threads": "articles",
    "youtube": "videos",
    "instagram_reel": "videos",
    "book": "books",
    "book_epub": "books",
    "book_pdf": "books",
}


def upload_file_to_minio(source_path: Path, dest: str, timeout: int = 60) -> str:
    """Upload a single file to MinIO using mc cp.

    Args:
        source_path: Absolute path to the file to upload.
        dest: MinIO destination path (e.g. "kb-agent-archive/articles/slug.md").
        timeout: Subprocess timeout in seconds.

    Returns:
        The Tailscale-accessible MinIO URL for the uploaded file.

    Raises:
        StepError(RETRY): If the upload fails.
        FileNotFoundError: If the source file does not exist.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    result = subprocess.run(
        ["mc", "cp", str(source_path), f"minio/{dest}"],
        capture_output=True, text=True, timeout=timeout,
    )

    if result.returncode != 0:
        raise StepError(
            f"MinIO upload failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="minio_upload",
        )

    tailscale_url = f"http://vmi3304846.tail5c02a1.ts.net:9000/{dest}"
    return tailscale_url


# ── Step: MinIO Upload ────────────────────────────────────────────────────────

def step_minio_upload(task: dict, config, source_path: Path | None = None) -> str:
    """Upload source content to MinIO for permanent archival.

    Args:
        task: Task dict with id, slug, content_type, source_url.
        config: Config dataclass.
        source_path: Optional explicit source file to upload. If None, defaults
                     to ``{config.temp_dir}/{slug}_raw.md``.

    Returns:
        MinIO object URL for the archived file.

    Raises:
        StepError: If MinIO upload fails.
        FileNotFoundError: If the source file does not exist.
    """
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "web")

    # Determine source file
    if source_path is None:
        source_path = Path(config.temp_dir) / f"{slug}_raw.md"

    folder = CONTENT_TYPE_TO_MINIO_FOLDER.get(content_type, "articles")
    dest = f"{config.minio_bucket}/{folder}/{slug}.md"

    return upload_file_to_minio(source_path, dest)


# ── Step: Git Push ────────────────────────────────────────────────────────────

def step_git_push(
    task: dict,
    config,
    vault_subfolder: str | None = None,
    commit_prefix: str = "add:",
) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Args:
        task: Task dict with id, slug, content_type.
        config: Config dataclass.
        vault_subfolder: Optional vault subfolder (e.g. "articles", "threads").
                         If None, derived from content_type via
                         CONTENT_TYPE_TO_VAULT_FOLDER.
        commit_prefix: Prefix for the commit message (e.g. "add threads:").

    Returns:
        The commit hash of the push.

    Raises:
        StepError(RETRY): If Git operations fail.
        FileNotFoundError: If the note does not exist.
    """
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "web")
    vault_path = config.vault_path

    if vault_subfolder is None:
        vault_subfolder = CONTENT_TYPE_TO_VAULT_FOLDER.get(content_type, "articles")

    note_path = vault_path / "Knowledge base" / vault_subfolder / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for Git push: {note_path}")

    commit_msg = f"{commit_prefix} {slug}"

    try:
        # Stage the note
        subprocess.run(
            ["git", "-C", str(vault_path), "add", str(note_path)],
            check=True, capture_output=True, text=True, timeout=10,
        )

        # Commit
        commit_result = subprocess.run(
            ["git", "-C", str(vault_path), "commit", "-m", commit_msg],
            check=True, capture_output=True, text=True, timeout=10,
        )

        # Push
        subprocess.run(
            ["git", "-C", str(vault_path), "push", "origin", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        )

        # Extract commit hash
        commit_hash = (
            commit_result.stdout.strip().split()[-1]
            if commit_result.stdout else "unknown"
        )
        return commit_hash

    except subprocess.CalledProcessError as e:
        raise StepError(
            f"Git push failed: {e.stderr.strip() if e.stderr else str(e)}",
            error_type=ErrorType.RETRY,
            step_name="git_push",
        ) from e


# ── Vault pull (pre-flight) ───────────────────────────────────────────────────

def step_git_pull_vault(config) -> str:
    """Pull the latest vault state before starting a task.

    Ensures See Also searches use the latest notes.

    Args:
        config: Config dataclass.

    Returns:
        Status message.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(config.vault_path), "pull"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Pull failed (continuing): {e.stderr.strip()}"
