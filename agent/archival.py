"""Archival steps — MinIO upload and Git push.

Deterministic shell commands for archiving source content and pushing notes.
Used by all pipelines (web, threads, instagram, books, youtube).
"""

import re
import subprocess
import threading
from pathlib import Path

from agent.config import RETIRED_AB_VAULT_PATH
from agent.db import normalize_tags
from agent.exceptions import ErrorType, StepError

# Serialize all Git operations on the shared vault repo. The consumer runs
# several tasks in parallel, each ending in git_push; without this, concurrent
# `git add` + `git commit` collide (one commits the other's staged note, the
# other then gets "nothing to commit" -> exit 1 -> spurious retry/pause).
_GIT_PUSH_LOCK = threading.Lock()


def append_note_tags(note_path: Path, tags) -> None:
    """Append user-specified tags to a note's frontmatter ``tags:`` list.

    Deterministic post-processing (not prompt-based): merges custom tags into the
    existing ``tags: [...]`` YAML line. Idempotent + deduped; no-op when there are
    no tags or no ``tags:`` line.
    """
    tags = normalize_tags(tags)
    if not tags:
        return
    note_path = Path(note_path)
    if not note_path.exists():
        return
    text = note_path.read_text(encoding="utf-8")

    def _merge(match: re.Match) -> str:
        indent = match.group(1)
        inner = match.group(2).strip()
        existing = [t.strip() for t in inner.split(",") if t.strip()]
        merged: list[str] = []
        for t in existing + tags:
            if t and t not in merged:
                merged.append(t)
        return f"{indent}tags: [{', '.join(merged)}]"

    new_text = re.sub(r"^([ \t]*)tags:\s*\[([^\]]*)\]", _merge, text, count=1, flags=re.MULTILINE)
    if new_text != text:
        note_path.write_text(new_text, encoding="utf-8")
        return

    # No tags: line — insert one right after the opening frontmatter fence.
    new_text = re.sub(r"(^---\s*\n)", r"\1" + f"tags: [{', '.join(tags)}]\n", text, count=1)
    if new_text != text:
        note_path.write_text(new_text, encoding="utf-8")


def write_source_files_block(note_path: Path, uploaded: list[str], config) -> None:
    """Rewrite a note's ``source_files:`` frontmatter from uploaded MinIO object paths."""
    if not uploaded:
        return
    note_path = Path(note_path)
    if not note_path.exists():
        return
    content = note_path.read_text(encoding="utf-8")
    lines = ["source_files:"]
    for u in uploaded:
        suffix = Path(u).suffix.lower().lstrip(".")
        kind = {
            "json": "transcript",
            "webm": "video",
            "mp4": "video",
            "mp3": "audio",
            "m4a": "audio",
            "txt": "text",
            "md": "raw",
            "epub": "epub",
            "pdf": "pdf",
        }.get(suffix, "raw")
        lines.append(f"  {kind}: {config.minio_tailscale_base}/{config.minio_bucket}/{u}")
    block = "\n".join(lines)
    new = re.sub(
        r"source_files:.*?(?=\n---|\n[a-zA-Z_]+:|\Z)",
        block, content, count=1, flags=re.DOTALL,
    )
    if new != content:
        note_path.write_text(new, encoding="utf-8")
        return

    # No source_files: line — insert after source_url: (or the opening fence).
    new = re.sub(r"(^[ \t]*source_url:.*\n)", r"\1" + block + "\n", content, count=1)
    if new == content:
        new = re.sub(r"(^---\s*\n)", r"\1" + block + "\n", content, count=1)
    if new != content:
        note_path.write_text(new, encoding="utf-8")


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


def upload_file_to_minio(source_path: Path, dest: str, timeout: int = 60, tailscale_base: str = "") -> str:
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

    tailscale_url = f"{tailscale_base}/{dest}"
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

    return upload_file_to_minio(source_path, dest, tailscale_base=config.minio_tailscale_base)


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

    try:
        if vault_path.resolve() == RETIRED_AB_VAULT_PATH.resolve():
            raise StepError(
                f"Refusing Git push to retired A/B vault {vault_path}",
                error_type=ErrorType.FAIL,
                step_name="git_push",
            )
    except OSError:
        pass

    if vault_subfolder is None:
        vault_subfolder = CONTENT_TYPE_TO_VAULT_FOLDER.get(content_type, "articles")

    note_path = vault_path / "Knowledge base" / vault_subfolder / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for Git push: {note_path}")

    commit_msg = f"{commit_prefix} {slug}"

    try:
        with _GIT_PUSH_LOCK:
            # Stage the note
            subprocess.run(
                ["git", "-C", str(vault_path), "add", str(note_path)],
                check=True, capture_output=True, text=True, timeout=10,
            )

            # Commit (idempotent: a prior retry or concurrent task may have
            # already committed the note — "nothing to commit" is success)
            commit_result = subprocess.run(
                ["git", "-C", str(vault_path), "commit", "-m", commit_msg],
                capture_output=True, text=True, timeout=10,
            )
            if commit_result.returncode != 0:
                out = (commit_result.stdout or "") + (commit_result.stderr or "")
                if "nothing to commit" not in out:
                    raise StepError(
                        f"Git commit failed: {out.strip()[:200]}",
                        error_type=ErrorType.RETRY,
                        step_name="git_push",
                    )
                commit_result = None

            # Push
            subprocess.run(
                ["git", "-C", str(vault_path), "push", "origin", "HEAD"],
                check=True, capture_output=True, text=True, timeout=30,
            )

        # Extract commit hash
        commit_hash = (
            commit_result.stdout.strip().split()[-1]
            if (commit_result and commit_result.stdout) else "unknown"
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
