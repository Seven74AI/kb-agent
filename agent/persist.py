"""Single choke point for durable text/JSON writes with explicit scrub policies.

Leaf module: imports only :mod:`agent.redact` and the stdlib (no db/core/detection).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from agent.redact import prepare_ocr_reading_for_write, prepare_vault_note_for_write

PersistPolicy = Literal["boundary4", "machine_local", "vault"]


def scrub_durable_text(text: str, *, policy: PersistPolicy) -> str:
    """Apply a durable-write scrub policy without writing to disk."""
    if policy == "machine_local":
        return text
    if policy == "boundary4":
        return prepare_ocr_reading_for_write(text)
    if policy == "vault":
        return prepare_vault_note_for_write(text)
    raise ValueError(f"unknown persist policy: {policy!r}")


def persist_durable_text(
    path: Path | str,
    text: str,
    *,
    policy: PersistPolicy,
    append: bool = False,
) -> Path:
    """Write *text* to *path* after applying *policy* (atomic replace, or append)."""
    path = Path(path)
    payload = scrub_durable_text(text, policy=policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
        return path
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def persist_durable_json(
    path: Path | str,
    obj: Any,
    *,
    policy: PersistPolicy,
    indent: int | None = 2,
) -> Path:
    """Serialize *obj* to JSON and persist with *policy* on the serialized string."""
    serialized = json.dumps(obj, indent=indent, ensure_ascii=False)
    return persist_durable_text(path, serialized, policy=policy)
