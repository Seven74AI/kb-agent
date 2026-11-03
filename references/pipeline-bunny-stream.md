# Bunny Stream pipeline (Bunny CDN / mediadelivery.net)

Online course/summit sites (terragrandir.com and many others) host video behind the Bunny Stream player. Videos are NOT login-gated at the CDN layer — they need a `Referer` header, not cookies.

## Phase 0 — Detect the embed

Fetch the video's page and grep for the iframe:

```bash
curl -s -A "Mozilla/5.0" "$PAGE_URL" -o /tmp/page.html
grep -oE 'https://player\.mediadelivery\.net/embed/[0-9]+/[a-f0-9-]+' /tmp/page.html
```

Pattern: `player.mediadelivery.net/embed/{libraryId}/{videoId}`. Record both IDs.

## Phase 1 — Locate the stream URL

The playlist URL is NOT in the static HTML — the player builds it in JS. Fetch the embed page and grep for the CDN zone + `playlist.m3u8` (it also appears in the page's `content-src` CSP header, so `grep playlist.m3u8` finds it even when JS hides it):

```bash
curl -s -A "Mozilla/5.0" "https://player.mediadelivery.net/embed/$LIB/$VID" -o /tmp/embed.html
grep -oE 'https://vz-[a-z0-9-]+\.b-cdn\.net/[a-f0-9-]+/playlist\.m3u8' /tmp/embed.html
```

Master playlist: `https://vz-<zone>.b-cdn.net/{videoId}/playlist.m3u8`. Quality variants live at `{240|360|480|720}p/video.m3u8` relative to it.

The same embed page also exposes a plain `play_720p.mp4` and `captions/<lang>.vtt` — grep those too; the direct MP4 is the simplest download (Phase 2).

## Phase 2 — Download

**`Referer: https://player.mediadelivery.net/` is mandatory on every request** — the CDN returns 403 without it.

**Preferred — direct MP4 (`play_720p.mp4`).** No HLS, no ffmpeg/bsf needed:

```bash
curl -sL -H "Referer: https://player.mediadelivery.net/" \
  -o "$OUT.mp4" "https://vz-<zone>.b-cdn.net/$VID/play_720p.mp4"
```

**Fallback — HLS.** ffmpeg handles HLS directly (no yt-dlp needed):

```bash
# probe duration first
ffprobe -hide_banner -headers "Referer: https://player.mediadelivery.net/" \
  -v error -show_entries format=duration -of csv=p=0 \
  "https://vz-<zone>.b-cdn.net/$VID/720p/video.m3u8"

# download the 720p variant (remux to mp4)
# add -bsf:a aac_adtstoasc: HLS TS segments carry ADTS AAC, and the mp4 muxer
# needs the bitstream filter to remux cleanly (skip it → broken audio in some builds)
ffmpeg -hide_banner -headers "Referer: https://player.mediadelivery.net/" \
  -i "https://vz-<zone>.b-cdn.net/$VID/720p/video.m3u8" \
  -c copy -bsf:a aac_adtstoasc "$OUT.mp4"
```

Highest quality in the master playlist is typically 720p — no 1080p, even if a `play_1080p.mp4` path exists (403).

## Phase 3 — Native captions (verification aid)

Bunny Stream serves per-language subtitle files directly:

```bash
curl -sL -A "Mozilla/5.0" -H "Referer: https://player.mediadelivery.net/" -o captions.vtt \
  "https://vz-<zone>.b-cdn.net/$VID/captions/fr.vtt"
```

Captions can lag video availability — a 404 on `captions/<lang>.vtt` means not published yet, not a wrong URL. Re-check before relying on them.

Use native captions to cross-check the whisper transcript. They do NOT replace diarization (no speaker labels) — run pyannote + `large-v3` per `video-pipeline-global.md` regardless.

## Phase 4 — Archive + note

- Video → `knowledge-base/videos/<slug>.mp4` (same convention as YouTube)
- Transcript → `knowledge-base/transcripts/<slug>.json`
- Frontmatter `source_files:` → `video:` + `transcript:` subkeys
- Note prefix: use the site/title slug, content in source language

Rate limits and worker rules from `video-pipeline-global.md` apply unchanged (2 videos/session, diarize then transcribe sequentially).
