# Runbook — Tigris storage cutover (#101)

Cut the KB over from local MinIO to Tigris. One-shot, ordered, with a clean
rollback. Do **not** run until the mirror is verified complete (bucket has
939 objects, ~34 GiB).

## Preconditions
- Mirror `mc mirror minio/knowledge-base tigris/kb-knowledge-base` complete.
- `mc du tigris/kb-knowledge-base` ≈ `mc du minio/knowledge-base` (939 objects).
- Reader `storage.py` refactor + pipeline `STORAGE_ALIAS` seam merged (done).

## Step 1 — migrate note URLs → keys
```bash
cd /root/kb-agent
python3 scripts/migrate_note_urls.py            # dry-run, expect ~461
python3 scripts/migrate_note_urls.py --apply    # write
cd "/root/Documents/Obsidian Vault"
git add -A && git commit -m "migrate media URLs -> object keys" && git push
```
Idempotent; a re-run reports 0.

## Step 2 — flip the pipeline to Tigris
Append to `/root/kb-agent/.env`:
```
STORAGE_ALIAS=tigris
MINIO_BUCKET=kb-knowledge-base
```
Restart the ingesters:
```bash
systemctl restart kb-agent kb-mcp
```

## Step 3 — flip the reader to Tigris
Append to `/root/stream/.env`:
```
AWS_ENDPOINT_URL_S3=https://fly.storage.tigris.dev
```
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BUCKET_NAME` are already
present.) Restart both reader instances:
```bash
systemctl restart kb-stream tdah-2026
```

## Step 4 — verify
```bash
# catalog lists (now from Tigris)
curl -s http://127.0.0.1:8001/api/catalog | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d),"entries")'
# presigned URL points at fly.storage.tigris.dev (not media.sevend.io)
curl -s http://127.0.0.1:8001/api/play/<slug> | python3 -m json.tool
# media route 302 + playback
curl -sI http://127.0.0.1:8001/media/videos/<slug>.mp4 | grep -i location
# download + epub range still work
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/api/download/<slug>
```
Acceptance: a video plays end-to-end and a transcript resolves, both signed by
Tigris.

## Step 5 — clean stale multipart uploads
The connection-drop retries may have left abandoned multipart uploads:
```bash
mc ls --incomplete tigris/kb-knowledge-base --recursive   # list
mc rm --incomplete tigris/kb-knowledge-base --recursive --force   # abort
```

## Step 6 — retire MinIO (deferred)
Keep MinIO for a soak period (recommend 7 days) as an instant rollback, then:
```bash
systemctl disable --now minio.service   # confirm service name first
```
Do not delete `/data/minio` until playback has been clean for the soak window.

## Rollback (any failure in steps 3–4)
```bash
# reader back to MinIO
sed -i '/^AWS_ENDPOINT_URL_S3=/d' /root/stream/.env
systemctl restart kb-stream tdah-2026
# pipeline back to MinIO
sed -i '/^STORAGE_ALIAS=tigris$/d; /^MINIO_BUCKET=kb-knowledge-base$/d' /root/kb-agent/.env
systemctl restart kb-agent kb-mcp
```
Notes already migrated to keys remain valid — the reader resolves keys against
MinIO's `knowledge-base` bucket (fallback `BUCKET` constant).

## Gotchas
- Tigris bucket is **`kb-knowledge-base`**, not `knowledge-base` (name was
  taken). Both the pipeline `MINIO_BUCKET` and reader `BUCKET_NAME` must carry
  the new name at cutover.
- Bucket-name mismatch = objects upload to a nonexistent bucket and 404 on
  read; verify `MINIO_BUCKET` before restarting ingesters.
- Video presign TTL is 24h (talks must not 403 mid-playback); transcripts 1h.
