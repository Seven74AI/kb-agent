#!/usr/bin/env python3
"""#101: migrate note media URLs -> object keys (idempotent).

Strips the object storage host prefix from `source_files` (and any other media field)
so notes reference portable keys (videos/x.mp4) instead of a deployment-specific
host URL. Dry-run by default; pass --apply to write.

Re-runnable: after migration the prefix is gone, so a second pass is a no-op.
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from agent.redact import write_vault_note  # noqa: E402

_vault_env = (os.environ.get("KB_VAULT_PATH") or "").strip()
VAULT = Path(_vault_env) if _vault_env else Path()
KB = VAULT / "Knowledge base"

# media hosts whose "<host>/knowledge-base/" prefix is stripped to a bare key.
_PREFIX_DEFAULTS = (
    "http://127.0.0.1:9000/knowledge-base/",
    "https://127.0.0.1:9000/knowledge-base/",
)
_extra = [
    p.strip()
    for p in (os.environ.get("KB_MIGRATE_MEDIA_PREFIXES") or "").split(",")
    if p.strip()
]
PREFIXES = list(_PREFIX_DEFAULTS) + _extra

APPLY = "--apply" in sys.argv


def main() -> int:
    if not _vault_env or not KB.is_dir():
        print("KB_VAULT_PATH must point at an Obsidian vault (Knowledge base/ missing)", file=sys.stderr)
        return 1
    total = 0
    changed_notes = []
    for p in sorted(KB.rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        orig = text
        for prefix in PREFIXES:
            text = text.replace(prefix, "")
        if text != orig:
            total += 1
            changed_notes.append(str(p.relative_to(KB)))
            if APPLY:
                write_vault_note(p, text)
    for n in changed_notes:
        print(("migrated" if APPLY else "would-migrate"), n)
    print(f"\n{total} notes {'migrated' if APPLY else 'would be migrated'} (apply={APPLY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
