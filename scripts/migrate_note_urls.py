#!/usr/bin/env python3
"""#101: migrate note media URLs -> object keys (idempotent).

Strips the MinIO host prefix from `source_files` (and any other media field)
so notes reference portable keys (videos/x.mp4) instead of the Tailscale
host URL. Dry-run by default; pass --apply to write.

Re-runnable: after migration the prefix is gone, so a second pass is a no-op.
"""

import sys
from pathlib import Path

VAULT = Path("/root/Documents/Obsidian Vault")
KB = VAULT / "Knowledge base"

# media hosts whose "<host>/knowledge-base/" prefix is stripped to a bare key.
PREFIXES = [
    "http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/",
    "https://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/",
    "http://media.sevend.io/knowledge-base/",
    "https://media.sevend.io/knowledge-base/",
]

APPLY = "--apply" in sys.argv


def main() -> int:
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
                p.write_text(text, encoding="utf-8")
    for n in changed_notes:
        print(("migrated" if APPLY else "would-migrate"), n)
    print(f"\n{total} notes {'migrated' if APPLY else 'would be migrated'} (apply={APPLY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
