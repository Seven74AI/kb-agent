#!/usr/bin/env python3
"""Daily TDAH summit check — detect new videos on terragrandir, download them,
transcribe in ONE batch Modal run, upload transcripts, and dispatch one synthesize task
per new video. Runs as a --no-agent cron; stdout is delivered to the user.
"""
import json
import os
import re
import subprocess
import sys
sys.path.insert(0, "/root/kb-agent")
from agent.config import load_config  # noqa: E402
from agent.db import AGENT_SCHEMA, init_db  # noqa: E402
from agent.core.submit import dispatch_synthesize  # noqa: E402
from agent.fetch_guard import _assert_fetch_url_allowed  # noqa: E402

MAIN = "https://www.terragrandir.com/conferences-tdah-adulte-2026/"
CDN = "https://vz-46d29c85-ed.b-cdn.net"
REF = "Referer: https://player.mediadelivery.net/"
MODAL = "/root/modal"
TMP = "/tmp/tdah_daily"
VB = "minio/kb-agent-archive/videos"       # mc alias 'minio'
TB = "minio/kb-agent-archive/transcripts"

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
    _assert_fetch_url_allowed(url)
    hdrs = headers or {"User-Agent": "Mozilla/5.0"}
    curl_headers = [
        arg
        for name, value in hdrs.items()
        for arg in ("-H", f"{name}: {value}")
    ]
    cmd = [
        "curl", "-s", "-L", "--max-redirs", "0", "--max-time", "40",
        *curl_headers,
        url,
    ]
    r = sh(cmd, timeout=45)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"curl failed ({r.returncode})")
    return r.stdout


def bunny_uuid(url_slug):
    html = fetch(f"https://www.terragrandir.com/{url_slug}/")
    m = UUID_RE.search(html)
    return m.group(1) if m else None


def _finalize_video_dispatch(page_url, config, conn, slug):
    """Dispatch synthesize for one uploaded video; return task_id or None on error."""
    rr = dispatch_synthesize(
        page_url,
        config,
        conn,
        content_type="bunny_stream_synthesize",
        slug=slug,
    )
    if rr.get("status") == "error":
        err = rr.get("error") or "dispatch_synthesize failed"
        print(f"  [warn] {slug}: dispatch failed ({err})")
        return None
    print(f"  {slug}: uploaded + task {rr.get('task_id', '?')}")
    return rr.get("task_id", "?")


def merge_speakers(diar_path, trans_path):
    d = json.load(open(diar_path))["segments"]
    t = json.load(open(trans_path))["segments"]
    out = []
    for seg in t:
        best, spk = -1, "SPEAKER_00"
        for s in d:
            ov = min(seg["end"], s["end"]) - max(seg["start"], s["start"])
            if ov > best:
                best, spk = ov, s["speaker"]
        out.append({"time": round(seg["start"], 1), "speaker": spk, "text": seg["text"]})
    return {"segments": out}


def main():
    os.makedirs(TMP, exist_ok=True)
    config = load_config()
    conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)

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
        cdn_url = f"{CDN}/{uu}/play_720p.mp4"
        _assert_fetch_url_allowed(cdn_url)
        sh(["curl", "-s", "--max-redirs", "0", "-H", REF, "-o", mp4, cdn_url])
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

    # 5. merge + upload + dispatch
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

        page_url = f"https://www.terragrandir.com/{ns.replace('tdah-adulte-2026-', 'tdah-adulte-')}/"
        task_id = _finalize_video_dispatch(page_url, config, conn, ns)
        if task_id is None:
            continue
        made.append((ns, task_id))

    print("DONE: " + ", ".join(n for n, _ in made))


if __name__ == "__main__":
    main()
