# 002 — Note frontmatter schema: `source_files` for archived sources

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Each pipeline generates note frontmatter independently, and the archive-locator
field drifted across content types:

- **Web, Instagram, Threads** — a flat `minio:` field (single archived URL).
- **YouTube, Bunny Stream** — a nested `source_files:` map (multiple archived files).

No ADR documented the split, and the two shapes were semantically identical: both
locate the **archived raw source** — a *derived* artifact produced by the
`minio_upload` step *after* LLM synthesis. This is distinct from `source` /
`source_url`, which locate the **original** source.

YouTube's `source_files:` shape is the more recent and general one (a source can
archive more than one file). It was adopted here as the canonical form.

## Decision

All note frontmatter uses a single canonical schema, with `source_files:` as the
archive locator:

```yaml
---
date: YYYY-MM-DD
source: <human-readable source id>   # "en.wikipedia.org" | "youtube.com (Channel)" | "Author — *Title* (book)"
source_url: <original URL or local path>
source_files:                       # archived raw source(s) — filled by the archival step, never the LLM
  <kind>: <minio URL>
confidence: <verified|plausible|emerging|debunked|untested>
tags: [tag1, tag2, ...]
---
```

`source_files` keys are semantic **file-kind** names:

| kind | content |
|---|---|
| `raw` | scraped/extracted raw source (web articles, social posts) |
| `text` | extracted full text (books) |
| `video` | video file |
| `audio` | audio file |
| `transcript` | transcript JSON |
| `epub` / `pdf` | source ebook files |

Rules:

1. `source_files` is always filled by the archival step (`*_minio_upload`), never
   by the LLM. The synthesis prompt emits a placeholder only.
2. Keys are semantic file kinds, **not** filename stems (a filename stem like
   `en.wikipedia.org_wiki_Statin` leaks the slug and is meaningless as a key).
3. `source` and `source_url` identify the original source; `source_files` locates
   the archived copy. They are not interchangeable.

## Consequences

- Web, Instagram, Threads: `minio:` → `source_files: {raw: <url>}`.
- Books: add the missing `source_url` and `source_files` fields.
- YouTube: `_update_note_source_files` emitted `{Path.stem}` keys; changed to
  semantic `video` / `audio` / `transcript` keys.
- Bunny Stream: already conforms; its extra `created` / `modified` fields are
  Obsidian-sync metadata and are left in place (out of scope).
- `templates/deep-treatment-template.md` and the `_fallback_template` in
  `web.py` updated to match.
