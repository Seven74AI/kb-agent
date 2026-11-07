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
    )
)

# Reuse the same persistent volume as transcription — model cache survives runs.
vol = modal.Volume.from_name("kb-models", create_if_missing=True)
CACHE = "/cache"

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

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

    os.environ["HF_HOME"] = CACHE  # route the model cache onto the volume

    # Load the model ONCE — this is the whole point of batching.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL)

    results = []
    for item in images:
        name = item["name"]
        path = f"/tmp/{name}"
        with open(path, "wb") as f:
            f.write(item["bytes"])

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
        generated_ids = model.generate(**inputs, max_new_tokens=768)
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        reading = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

        results.append({"name": name, "reading": reading})
        print(f"[{name}] {len(reading)} chars in {time.time() - t0:.0f}s", flush=True)
    return results


@app.local_entrypoint()
def main(image_dir: str = None, out_dir: str = "."):
    if not image_dir:
        print("usage: modal run kb_ocr.py --image-dir <dir> [--out-dir ./out]")
        return
    os.makedirs(out_dir, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".webp")
    items = []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith(exts):
            with open(os.path.join(image_dir, f), "rb") as fh:
                items.append({"name": f, "bytes": fh.read()})
    if not items:
        print("no images found in", image_dir)
        return
    results = read_images.remote(items)
    for r in results:
        p = os.path.join(out_dir, f"{r['name']}.ocr.json")
        with open(p, "w") as f:
            json.dump({"reading": r["reading"]}, f, ensure_ascii=False, indent=2)
        print(f"wrote {p}")
