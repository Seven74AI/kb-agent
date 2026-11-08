"""KB image reading on Modal — Qwen2.5-VL-7B OCR + description on GPU.

BATCH mode: one container loads the VLM ONCE and reads ALL images in a
directory, so a nightly batch pays a single cold-start. Each input image →
``<name>.ocr.json`` with a ``reading`` field (verbatim text + description).

Run a batch:
    modal run kb_ocr.py --image-dir ./imgs --out-dir ./out

Cost: A10 = $1.10/hr. One cold-start (~60s) + ~2-5s/image. Qwen2.5-VL is
Apache-2.0 and ungated — no HF token needed (the volume is still reused for the
model cache, so downloads happen once).
"""

import json
import os
import time

import modal

app = modal.App("kb-ocr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "torchvision",
        "transformers>=4.49,<5",
        "qwen-vl-utils",
        "accelerate",
        "pillow-heif",
        "pillow-avif-plugin",
    )
)

# Reuse the same persistent volume as transcription — model cache survives runs.
vol = modal.Volume.from_name("kb-models", create_if_missing=True)
CACHE = "/cache"

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# Cap decoded pixels before VLM preprocess (OOM / pixel bombs).
MAX_IMAGE_PIXELS = 4_194_304  # 2048 x 2048
MAX_NEW_OCR_TOKENS = 768


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


PROMPT = (
    "Transcribe every piece of text visible in this image, verbatim and complete, "
    "preserving order. Then, in a separate paragraph after the text, describe what "
    "the image shows (chart, data, document, screenshot, photo) and its key meaning."
)


@app.function(
    image=image,
    gpu="A10",
    volumes={CACHE: vol},
    timeout=60 * 60,
)
def read_images(images: list) -> list:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    from pillow_heif import register_heif_opener

    register_heif_opener()
    try:
        import pillow_avif  # noqa: F401 — registers AVIF support with Pillow
    except ImportError:
        pillow_avif = None

    os.environ["HF_HOME"] = CACHE  # route the model cache onto the volume

    # Load the model ONCE — this is the whole point of batching.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL)

    results = []
    for item in images:
        name = item["name"]
        try:
            path = f"/tmp/{name}"
            with open(path, "wb") as f:
                f.write(item["bytes"])

            from PIL import Image

            with Image.open(path) as img:
                w, h = img.size
                if w * h > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"image pixel count {w * h} exceeds cap {MAX_IMAGE_PIXELS}"
                    )

            t0 = time.time()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_OCR_TOKENS)
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            reading = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()
            truncated = len(trimmed[0]) >= MAX_NEW_OCR_TOKENS
            if truncated:
                results.append(
                    {
                        "name": name,
                        "truncated": True,
                        "error": f"OCR output truncated at {MAX_NEW_OCR_TOKENS} tokens",
                    }
                )
                print(f"[{name}] truncated at token cap", flush=True)
                continue

            results.append({"name": name, "reading": reading})
            print(f"[{name}] {len(reading)} chars in {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:
            safe = _sanitize_emit(exc)
            print(f"[{name}] failed: {safe}", flush=True)
            results.append({"name": name, "error": safe})
    return results


@app.local_entrypoint()
def main(image_dir: str = None, out_dir: str = "."):
    if not image_dir:
        print("usage: modal run kb_ocr.py --image-dir <dir> [--out-dir ./out]")
        return
    os.makedirs(out_dir, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".avif")
    items = []
    skipped = 0
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(exts):
            continue
        path = os.path.join(image_dir, f)
        size = os.path.getsize(path)
        if size > MAX_IMAGE_BYTES:
            print(f"skip {f}: {size} bytes exceeds cap {MAX_IMAGE_BYTES}", flush=True)
            skipped += 1
            continue
        with open(path, "rb") as fh:
            items.append({"name": f, "bytes": fh.read()})
    if skipped:
        print(f"skipped {skipped} oversized image(s)", flush=True)
    if not items:
        print("no images found in", _sanitize_emit(image_dir))
        return
    _sanitize_emit("")  # ensure repo on sys.path for agent.notify
    from agent.notify import ocr_error_json

    results = read_images.remote(items)
    for r in results:
        p = os.path.join(out_dir, f"{r['name']}.ocr.json")
        with open(p, "w") as f:
            if r.get("error"):
                json.dump(ocr_error_json(r["error"]), f, ensure_ascii=False, indent=2)
            else:
                from agent.redact import redact_http_urls_in_text

                json.dump(
                    {"reading": redact_http_urls_in_text(r["reading"])},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        print("wrote", _sanitize_emit(p))
