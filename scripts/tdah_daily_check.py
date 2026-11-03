#!/usr/bin/env python3
"""Daily TDAH summit check — detect new videos on terragrandir, download them,
transcribe in ONE batch Modal run, upload transcripts, and create one note ticket
per new video. Runs as a --no-agent cron; stdout is delivered to the user.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

MAIN = "https://www.terragrandir.com/conferences-tdah-adulte-2026/"
CDN = "https://vz-46d29c85-edb.b-cdn.net"
REF = "Referer: https://player.mediadelivery.net/"
MODAL = "/root/modal"
TMP = "/tmp/tdah_daily"
VB = "minio/knowledge-base/videos"       # mc alias 'minio'
TB = "minio/knowledge-base/transcripts"

UUID_RE = re.compile(
    r"player\.mediadelivery\.net/embed/\d+/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
)

# French TDAH domain context — passed explicitly to the (now language-agnostic) Modal spec.
LANG = "fr"
INITIAL_PROMPT = (
    "Conférence en français sur le TDAH (trouble du déficit de l'attention avec ou sans "
    "hyperactivité) chez l'adulte : diagnostic, dopamine, neurodéveloppement, hormones, sommeil."
)
HOTWORDS = (
    "TDAH,TDA,hyperactivité,impulsivité,inattention,neurodéveloppement,neuroatypie,"
    "neurodivergence,dopamine,noradrénaline,méthylphénidate,Ritaline,diagnostic,comorbidité,"
    "autisme,TSA,HPI,psychiatre,psychologue,neuropsychologie,attention,concentration,"
    "sommeil,procrastination,hormones,œstrogène,progestérone,ménopause,émotions"
)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")


def bunny_uuid(url_slug):
    html = fetch(f"https://www.terragrandir.com/{url_slug}/")
    m = UUID_RE.search(html)
    return m.group(1) if m else None


def merge_speakers(diar_path, trans_path):
    d = json.load(open(diar_path))["segments"]
    t = json.load(open(trans_path))["segments"]
    out = []
    for seg in t:
        mid = (seg["start"] + seg["end"]) / 2
        best, spk = -1, "SPEAKER_00"
        for s in d:
            ov = min(seg["end"], s["end"]) - max(seg["start"], s["start"])
            if ov > best:
                best, spk = ov, s["speaker"]
        out.append({"time": round(seg["start"], 1), "speaker": spk, "text": seg["text"]})
    return {"segments": out}


def note_body(slug):
    return f"""## Task
Write the KB note for the TDAH summit talk `{slug}`.

The video and the diarized French transcription are already downloaded and archived in MinIO. Read the transcription and write the note.

## Sources
- Transcription (speaker-attributed JSON, segments with `time`/`speaker`/`text`): `mc cat minio/knowledge-base/transcripts/{slug}.json`
- Video (already uploaded): `minio/knowledge-base/videos/{slug}.mp4`

## Rules
1. **French content AND French section labels.** The talk is French. Follow `templates/tdah-summit-note-template.md` (in the `knowledge-base` skill) — exact section order: Métadonnées → Résumé → Chapitres → Concepts clés → Points clés → Résumé par chapitre → Résumé par locuteur → Extractions utiles → Nuances & Limites → Transcription → Voir aussi.
2. Frontmatter: `date` = 2026 for Day-2 talks (inédites), 2025 for Day-1 replays; `tags` must include `sommet-tdah-2026`; `source_files` = the MinIO video + transcript URLs (full Tailscale FQDN `http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/...`).
3. Ground every claim in the transcript — no hallucination.

## Steps
1. `mc cat minio/knowledge-base/transcripts/{slug}.json` (read fully).
2. Write `Knowledge base/{slug}.md` following the template.
3. `cd "$OBSIDIAN_VAULT_PATH" && git add "Knowledge base/{slug}.md" && git commit -m "add: {slug}" && git push`
"""


def main():
    os.makedirs(TMP, exist_ok=True)

    # 1. detect new slugs from the main page
    html = fetch(MAIN)
    url_slugs = set(re.findall(r"/tdah-adulte-(j\d+-[a-z0-9-]+)/", html))
    note_slugs = {f"tdah-adulte-2026-{s}" for s in url_slugs}

    # 2. already-processed = MinIO videos
    r = sh(["mc", "ls", VB])
    existing = set(re.findall(r"(tdah-adulte-2026-j\d+-[a-z0-9-]+)\.mp4", r.stdout))
    new = sorted(note_slugs - existing)
    if not new:
        print("No new TDAH videos found.", file=sys.stderr)
        return
    print(f"Found {len(new)} new video(s): " + ", ".join(new))

    # 3. download each + extract 16kHz mono WAV
    wav_dir = os.path.join(TMP, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    downloaded = []
    for ns in new:
        uu = bunny_uuid(ns.replace("tdah-adulte-2026-", "tdah-adulte-"))
        if not uu:
            print(f"  [skip] {ns}: no Bunny UUID on detail page")
            continue
        mp4 = os.path.join(TMP, f"{ns}.mp4")
        sh(["curl", "-sL", "-H", REF, "-o", mp4, f"{CDN}/{uu}/play_720p.mp4"])
        if not os.path.exists(mp4) or os.path.getsize(mp4) < 100000:
            print(f"  [skip] {ns}: download failed")
            continue
        wav = os.path.join(wav_dir, f"{ns}.wav")
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-ac", "1", "-ar", "16000", wav])
        downloaded.append((ns, mp4))
        print(f"  downloaded {ns}")

    if not downloaded:
        print("Nothing downloadable.")
        return

    # 4. ONE Modal run for all WAVs
    out_dir = os.path.join(TMP, "out")
    os.makedirs(out_dir, exist_ok=True)
    r = sh([os.path.join(MODAL, "venv", "bin", "modal"), "run", "kb_transcribe.py",
            "--audio-dir", wav_dir, "--out-dir", out_dir,
            "--language", LANG, "--initial-prompt", INITIAL_PROMPT, "--hotwords", HOTWORDS],
            cwd=MODAL, timeout=3600)
    tail = (r.stdout or r.stderr)[-1500:]
    print("Modal output tail:\n" + tail)

    # 5. merge + upload + ticket
    made = []
    for ns, mp4 in downloaded:
        diar = os.path.join(out_dir, f"{ns}.diarization.json")
        trans = os.path.join(out_dir, f"{ns}.transcript.json")
        if not (os.path.exists(diar) and os.path.exists(trans)):
            print(f"  [warn] {ns}: missing Modal output")
            continue
        mpath = os.path.join(TMP, f"{ns}.json")
        json.dump(merge_speakers(diar, trans), open(mpath, "w"), ensure_ascii=False)
        sh(["mc", "cp", mp4, f"{VB}/{ns}.mp4"])
        sh(["mc", "cp", mpath, f"{TB}/{ns}.json"])

        bpath = os.path.join(TMP, f"{ns}.ticket.txt")
        open(bpath, "w").write(note_body(ns))
        r = sh(["hermes", "kanban", "--board", "knowledge-base", "create",
                f"KB: note {ns}", "--assignee", "researcher-videos",
                "--skill", "knowledge-base", "--max-runtime", "2h",
                "--body-file", bpath, "--json"])
        tid = re.search(r'"id"\s*:\s*"(t_[a-f0-9]+)"', r.stdout)
        made.append((ns, tid.group(1) if tid else "?"))
        print(f"  {ns}: uploaded + ticket {tid.group(1) if tid else '?'}")

    print("DONE: " + ", ".join(n for n, _ in made))


if __name__ == "__main__":
    main()
