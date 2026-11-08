"""KB transcription on Modal — pyannote diarization + faster-whisper large-v3 on GPU.

BATCH mode: one container loads the models ONCE and processes ALL WAVs in a
directory, so a daily batch of videos pays a single cold-start instead of one
per video. Each input WAV → <name>.diarization.json + <name>.transcript.json.

One-time setup (on any machine with the Modal CLI):
    pip install modal && modal token new
    modal secret create hf-token HF_TOKEN=<your huggingface token>   # pyannote gated models
    modal volume create kb-models                                     # persists model cache

Run a batch:
    modal run kb_transcribe.py --audio-dir ./wavs --out-dir ./out

Cost: A10 = $1.10/hr. One cold-start (~45s) + ~1-2 min/video.
"""

import json
import os
import time
import wave

import modal

app = modal.App("kb-transcribe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")          # torchcodec (pyannote audio I/O) needs libavutil.so
    .pip_install(
        "faster-whisper==1.2.1",
        "pyannote.audio==4.0.7",
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "torchcodec==0.9",
    )
)

# Persistent volume → models download once, survive every run
vol = modal.Volume.from_name("kb-models", create_if_missing=True)
CACHE = "/cache"

hf_secret = modal.Secret.from_name("hf-token")  # HF_TOKEN=...


def _sanitize_emit(message: object) -> str:
    """Route Modal stdout lines that may carry paths/URLs through notify scrub."""
    import sys
    from pathlib import Path

    for root in (Path(__file__).resolve().parent.parent, Path("/root/kb-agent")):
        if root.is_dir():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            break
    from agent.notify import sanitize_visible_message

    return sanitize_visible_message(str(message))


# Language/hotwords/prompt are passed per-batch via CLI flags (--language / --hotwords /
# --initial-prompt). Default None = auto-detect, no bias. The TDAH cron passes French + TDAH
# vocabulary explicitly.


@app.function(
    image=image,
    gpu="A10",
    volumes={CACHE: vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def transcribe_batch(audios: list, language: str = None, initial_prompt: str = None, hotwords: str = None) -> list:
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    os.environ["HF_HOME"] = CACHE  # route both model caches onto the volume

    # Load models ONCE — this is the whole point of batching.
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]
    )
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    results = []
    for item in audios:
        name = item["name"]
        wav = f"/tmp/{name}.wav"
        with open(wav, "wb") as f:
            f.write(item["bytes"])

        with wave.open(wav, "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            raw = w.readframes(n)
        # Pre-decode with the stdlib wave module and pass a waveform dict, so
        # pyannote never touches torchcodec's FFmpeg-backed audio decoder.
        waveform = (
            torch.frombuffer(raw, dtype=torch.int16)
            .reshape(-1, ch).T.contiguous().float() / 32768.0
        )
        if ch > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono

        t0 = time.time()
        diar_out = pipe({"waveform": waveform, "sample_rate": sr})
        diar = [
            {"start": round(t.start, 2), "end": round(t.end, 2), "speaker": s}
            for t, _, s in diar_out.speaker_diarization.itertracks(yield_label=True)
        ]

        segs, info = model.transcribe(
            wav,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            vad_filter=True,
            beam_size=5,
        )
        tr = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segs
        ]

        results.append({
            "name": name,
            "diarization": {"segments": diar},
            "transcript": {
                "language": info.language,
                "duration": round(info.duration, 1),
                "segments": tr,
            },
        })
        print(
            f"[{name}] diar {len(diar)} segs, trans {len(tr)} segs ({info.language}) "
            f"in {time.time() - t0:.0f}s",
            flush=True,
        )
    return results


@app.local_entrypoint()
def main(audio_dir: str = None, out_dir: str = ".", language: str = None, initial_prompt: str = None, hotwords: str = None):
    if not audio_dir:
        print("usage: modal run kb_transcribe.py --audio-dir <dir_of_wavs> [--out-dir ./out]")
        return
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for f in sorted(os.listdir(audio_dir)):
        if f.endswith(".wav"):
            with open(os.path.join(audio_dir, f), "rb") as fh:
                items.append({"name": f[:-4], "bytes": fh.read()})
    if not items:
        print("no .wav files found")
        return
    results = transcribe_batch.remote(items, language, initial_prompt, hotwords)
    from agent.redact import write_transcription_sidecar_json

    for r in results:
        name = r["name"]
        for k in ("diarization", "transcript"):
            p = os.path.join(out_dir, f"{name}.{k}.json")
            write_transcription_sidecar_json(p, r[k], indent=2)
            print(f"wrote {_sanitize_emit(p)}")
