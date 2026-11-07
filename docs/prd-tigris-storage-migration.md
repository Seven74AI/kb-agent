## Problem Statement

The knowledge base stores all media — videos, transcripts, reels, audio, books — in a local MinIO instance on the VPS, exposed to the browser through a hardcoded Tailscale URL (e.g. `http://<TAILSCALE_HOST>:9000`, historically set via `MINIO_TAILSCALE_BASE`) baked into every note's `source_files` block. This couples the KB to the VPS and Tailscale, routes all video bandwidth through the VPS, and lets ~40GB of media consume the same disk that runs the pipeline and reader — which directly caused the disk-full outage during the VP9→H.264 re-encode.

## Solution

Migrate the `knowledge-base` MinIO bucket to a **private Tigris** bucket (S3-compatible, `fly.storage.tigris.dev`), matching the music-library's proven pattern. The pipeline writes object **keys** instead of absolute URLs; the reader signs short-lived **presigned URLs** on demand, so media streams from Tigris's CDN, the bucket stays private, and notes become storage-agnostic. Retire local MinIO once cutover is verified.

## User Stories

1. As a reader of kb.sevend.io, I want videos to stream from managed object storage, so playback no longer depends on the VPS or Tailscale being reachable.
2. As the pipeline operator, I want ingestion to write objects to Tigris instead of local MinIO, so the VPS disk stops filling with media.
3. As the vault maintainer, I want notes to reference media by object key rather than a hardcoded host, so notes stay portable across storage backends.
4. As a reader, I want video and transcript access to go through short-lived presigned URLs, so the bucket remains private while authorized readers can still load media.
5. As a reader watching a long talk, I want the video presigned URL to remain valid for the whole playback, so mid-playback range requests do not 403 when the URL expires.
6. As an operator, I want the migration to be additive and reversible until cutover is verified, so a failed migration can roll back by pointing the reader back at MinIO.
7. As the pipeline operator, I want a single, well-tested storage seam, so future backend changes (R2, B2, S3) require touching one module, not every pipeline.
8. As a maintainer, I want the local MinIO service and its ~40GB data directory removed after cutover, so the VPS disk is reclaimed.
9. As a user of the kba.sevend.io dashboard, I want uploads (cookies, etc.) to keep working, so nothing outside the media path regresses during the migration.

## Implementation Decisions

- **Vendor: Tigris** over Cloudflare R2 — $0.004/GB vs $0.015/GB, zero egress on both, globally-distributed, and already in use by music-library (harmonization).
- **Bucket: private, single bucket** `kb-knowledge-base` (Tigris), same prefix layout (`videos/`, `reels/`, `transcripts/`, `audio/`, `books/`, `articles/`, `threads/`, `instagram/`, `epubs/`, `pdfs/`).
- **Access: presigned URLs**, not public-read — re-hosted conference/YouTube content must not be publicly listable. The reader generates presigned GET URLs via the AWS SDK (boto3), `region=auto`, `endpoint=fly.storage.tigris.dev`, `force_path_style=False` (virtual-hosted).
- **Presigned TTL: 24h for video, 1h for transcripts.** music-library uses 1h for short audio tracks; KB videos are 30min–2h talks, so 1h would expire mid-playback and 403 range requests. Transcripts are single small fetches, 1h is fine.
- **Notes store keys, not URLs.** `write_source_files_block` writes `videos/<name>.mp4` / `transcripts/<name>.json` (no `MINIO_TAILSCALE_BASE` prefix). The reader resolves keys → presigned URLs at render time.
- **Single seam:** `agent/archival.py` (`upload_file_to_minio` + `write_source_files_block`) and the storage config. All pipelines already funnel through these two functions; no per-pipeline changes. `mc cp` gains a `tigris/kb-knowledge-base/<key>` alias in place of `minio/<key>`.
- **Config:** drop `MINIO_TAILSCALE_BASE`; replace `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` with Tigris credentials (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or a `tigris` mc alias).
- **Reader:** new `/media/<key>` route on the stream app (FastAPI) that signs a presigned URL and 302-redirects; the catalog API returns keys, and the frontend resolves them through `/media/<key>`.
- **One-time migration:** `mc mirror minio/knowledge-base tigris/kb-knowledge-base` (additive; source bucket untouched until cutover).
- **Note migration:** a script rewrites every note's `source_files.*` from the Tailscale URL to the bare key (or the reader tolerates the legacy full URL and extracts the key).
- **Cutover:** point the reader at Tigris, verify one video + one transcript resolve and play, then stop MinIO and delete `/data/minio`.

## Testing Decisions

- **Test external behavior, not implementation:** assert that a pipeline run ends with objects reachable via a resolvable key and a note whose `source_files` contains keys (no host prefix); assert the reader returns a redirect to a Tigris URL for a known key; assert a long video's presigned TTL is ≥ 24h.
- **Modules tested:** `agent/archival.py` (upload + source-files block), the reader's media route, and the note-URL migration script (idempotent: running twice is a no-op).
- **Prior art:** existing pipeline tests already mock `upload_file_to_minio` / `step_minio_upload`; the reader has no test suite — introduce a minimal one for the media route (mirror music-library's route tests for presigned URLs).
- **Migration script test:** a dry-run mode that reports the rewrite count without writing, run against the real vault before applying.

## Out of Scope

- Moving the KB's SQLite state (`state.db`, 2GB) to Tigris or a managed DB — separate concern (optional backup bucket is noted but not required here).
- Lifecycle/tiering policies on Tigris (moving cold objects to a cheaper tier) — can be added later.
- Changing the ingestion pipeline's content types, quality gates, or cookie flow.
- Migrating the music-library's storage or harmonizing its reader.

## Further Notes

- Rollback = repoint reader config back to MinIO (MinIO is not deleted until step 6 verified).
- The reader (stream app) lives outside the kb-agent repo; the presigned-URL route may be implemented there and its changes tracked in a separate repo/notes, with the kb-agent issue scoped to the pipeline seam + key-format change.
- The orphan `u2PzSPD-wVI` video is being re-fetched separately and is independent of this migration.
