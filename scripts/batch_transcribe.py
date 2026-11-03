#!/usr/bin/env python3
"""General batch transcription via Modal — any language, one cold-start.

Usage:
    batch_transcribe.py <input_dir> [--out-dir DIR] [--language fr] [--hotwords "..."] [--initial-prompt "..."]

<input_dir> may contain videos (.mp4/.webm/.mkv/.mov) or already-extracted .wav files
(16 kHz mono). Extracts WAVs, runs ONE Modal batch (auto-detect language unless
--language is given), then merges diarization + transcription into a
speaker-attributed <name>.json per input.
"""
import json
import os
import shutil
import subprocess
import sys

MODAL = "/root/modal"
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("usage: batch_transcribe.py <input_dir> [--out-dir DIR] [--language fr] [--hotwords ...] [--initial-prompt ...]")
        return 1
    in_dir = args[0]
    out_dir = "."
    lang = hot = prompt = None
    i = 1
    while i < len(args):
        if args[i] == "--out-dir" and i + 1 < len(args):
            out_dir = args[i + 1]; i += 2
        elif args[i] == "--language" and i + 1 < len(args):
            lang = args[i + 1]; i += 2
        elif args[i] == "--hotwords" and i + 1 < len(args):
            hot = args[i + 1]; i += 2
        elif args[i] == "--initial-prompt" and i + 1 < len(args):
            prompt = args[i + 1]; i += 2
        else:
            i += 1
    os.makedirs(out_dir, exist_ok=True)

    wav_dir = os.path.join(out_dir, ".wavs")
    os.makedirs(wav_dir, exist_ok=True)
    names = []
    for f in sorted(os.listdir(in_dir)):
        p = os.path.join(in_dir, f)
        base = os.path.splitext(f)[0]
        if f.lower().endswith(VIDEO_EXTS):
            wav = os.path.join(wav_dir, base + ".wav")
            sh(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-ac", "1", "-ar", "16000", wav])
            names.append(base)
        elif f.lower().endswith(".wav"):
            shutil.copy(p, os.path.join(wav_dir, f))
            names.append(base)
    if not names:
        print("no videos or .wav files found in", in_dir)
        return 1
    print(f"batch: {len(names)} input(s) -> one Modal run")

    cmd = [os.path.join(MODAL, "venv", "bin", "modal"), "run", "kb_transcribe.py",
           "--audio-dir", wav_dir, "--out-dir", out_dir]
    if lang:
        cmd += ["--language", lang]
    if hot:
        cmd += ["--hotwords", hot]
    if prompt:
        cmd += ["--initial-prompt", prompt]
    r = sh(cmd, cwd=MODAL, timeout=3600)
    print((r.stdout or r.stderr)[-1500:])

    made = []
    for n in names:
        diar = os.path.join(out_dir, f"{n}.diarization.json")
        trans = os.path.join(out_dir, f"{n}.transcript.json")
        if not (os.path.exists(diar) and os.path.exists(trans)):
            print(f"[warn] {n}: missing Modal output")
            continue
        mpath = os.path.join(out_dir, f"{n}.json")
        json.dump(merge_speakers(diar, trans), open(mpath, "w"), ensure_ascii=False)
        made.append(n)
    print("MERGED: " + ", ".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
