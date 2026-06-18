"""Archival steps — MinIO upload and Git push.

Deterministic shell commands for archiving source content and pushing notes.
"""

import subprocess
from pathlib import Path


def step_minio_upload(task: dict, config) -> str:
    """Upload source content to MinIO for permanent archival.

    Args:
        task: Task dict with id, slug, content_type, source_url.
        config: Config dataclass.

    Returns:
        MinIO object URL for the archived file.

    Raises:
        RuntimeError: If MinIO upload fails.
    """
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "web")

    # Determine source file and destination bucket folder
    source_path = Path(config.temp_dir) / f"{slug}_raw.md"
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Map content type to MinIO folder
    folder_map = {
        "web": "articles",
        "substack": "articles",
        "instagram_post": "articles",
        "threads": "threads",
        "youtube": "videos",
        "instagram_reel": "reels",
        "book": "books",
        "mega": "books",
    }
    folder = folder_map.get(content_type, "articles")

    dest = f"{config.minio_bucket}/{folder}/{slug}.md"

    # Upload using mc (MinIO client)
    result = subprocess.run(
        ["mc", "cp", str(source_path), f"minio/{dest}"],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"MinIO upload failed: {result.stderr.strip()}")

    # Return the Tailscale-accessible URL
    tailscale_url = f"http://vmi3304846.tail5c02a1.ts.net:9000/{dest}"
    return tailscale_url


def step_git_push(task: dict, config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Args:
        task: Task dict with id, slug, content_type.
        config: Config dataclass.

    Returns:
        The commit hash of the push.

    Raises:
        RuntimeError: If Git operations fail.
    """
    slug = task.get("slug", task["id"][:8])
    content_type = task.get("content_type", "web")
    vault_path = config.vault_path

    # Map content type to vault folder
    folder_map = {
        "web": "articles",
        "substack": "articles",
        "instagram_post": "articles",
        "threads": "articles",
        "youtube": "videos",
        "instagram_reel": "videos",
        "book": "books",
    }
    folder = folder_map.get(content_type, "articles")

    note_path = vault_path / "Knowledge base" / folder / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for Git push: {note_path}")

    try:
        # Stage the note
        subprocess.run(
            ["git", "-C", str(vault_path), "add", str(note_path)],
            check=True, capture_output=True, text=True, timeout=10,
        )

        # Commit
        result = subprocess.run(
            ["git", "-C", str(vault_path), "commit", "-m", f"add: {slug}"],
            check=True, capture_output=True, text=True, timeout=10,
        )

        # Push
        subprocess.run(
            ["git", "-C", str(vault_path), "push"],
            check=True, capture_output=True, text=True, timeout=30,
        )

        # Extract commit hash
        commit_hash = result.stdout.strip().split()[-1] if result.stdout else "unknown"
        return commit_hash

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git push failed: {e.stderr.strip()}") from e


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
