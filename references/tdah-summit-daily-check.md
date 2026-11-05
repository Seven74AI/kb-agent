# TDAH Summit — Daily Video Check (cron)

Automated daily check for new TDAH summit videos on terragrandir.com: detect → download → transcribe (batch Modal) → note ticket. Runs as a no-agent cron; only the note-writing step touches the LLM (via worker tickets).

## Components

- **Cron**: `hermes cron list` → "TDAH daily video check" (no-agent script, Telegram delivery).
- **Script**: `/root/.hermes/scripts/tdah_daily_check.py` — the whole pipeline.
- **Modal**: `/root/modal/kb_transcribe.py` — BATCH mode (one container, all WAVs), French TDAH context.
- **Template**: `templates/tdah-summit-note-template.md` — what the note ticket worker follows.

## Flow (per run)

1. **Detect** — fetch `https://www.terragrandir.com/conferences-tdah-adulte-2026/`, extract conference slugs with `/tdah-adulte-(j\d+-[a-z0-9-]+)/`, map to note slugs `tdah-adulte-2026-<slug>`, diff against MinIO `knowledge-base/videos/`. Strip the `.mp4` suffix from the `mc ls` output before comparing (the slug set has no extension).

2. **Download** — for each new slug, fetch the detail page and extract the Bunny UUID from `player.mediadelivery.net/embed/<libraryId>/<uuid>`:
   ```
   curl -sL -H "Referer: https://player.mediadelivery.net/" \
     -o <slug>.mp4 "https://vz-46d29c85-edb.b-cdn.net/<uuid>/play_720p.mp4"
   ```

3. **WAV** — `ffmpeg -i <slug>.mp4 -ac 1 -ar 16000 <slug>.wav` (16 kHz mono).

4. **Transcribe (one batch)** — `modal run kb_transcribe.py --audio-dir <wav_dir> --out-dir <out_dir>`. One container loads the models once and processes every WAV in the dir. Emits `<slug>.diarization.json` + `<slug>.transcript.json` per input. French forced via `language="fr"` + TDAH `hotwords`/`initial_prompt` (see `references/modal-gpu-transcription.md`).

5. **Merge + upload** — assign each transcript segment its diarization speaker (max overlap at segment midpoint), write `<slug>.json` (segments `{time, speaker, text}`), then `mc cp` the MP4 to `videos/` and the JSON to `transcripts/`.

6. **Note ticket** — one per video, `--assignee researcher-videos --skill knowledge-base --max-runtime 2h`, note-only (worker reads the uploaded transcript, writes the note per the TDAH template).

7. **Notify** — script stdout goes to Telegram; it is silent (stderr-only) on no-new days so the user is only pinged when videos are found.

## Re-arming next summit

Update `MAIN` (page URL), `CDN` (Bunny pull-zone host), and the cron schedule/repeat in the script. The detect/diff/download/Modal/ticket chain is reusable as-is. One batch Modal run per day is the right cost/robustness trade-off — batching only beats per-video runs by the per-run cold-start (~45s ≈ $0.04/video), negligible at the daily scale (0–6 videos).
