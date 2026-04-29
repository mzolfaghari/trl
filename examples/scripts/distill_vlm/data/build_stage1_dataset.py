# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ──────────────────────────────────────────────────────────────────────────────
# Why this file exists:
#   Stage 1 of the VLM knowledge-distillation pipeline aligns the SmolVLM2
#   connector to a domain-specific caption distribution. This script generates
#   that supervision OFFLINE by running a domain teacher over a folder of
#   images and saving the resulting (image, caption) pairs as a HuggingFace
#   `Dataset` in the chat-message format expected by `trl.SFTTrainer` for VLMs.
#
#   The bundled medical example uses LLaVA-Med (microsoft/llava-med-v1.5-
#   mistral-7b) — for a different domain, swap `LLAVA_MED_MODEL_ID` and the
#   `CAPTION_PROMPT` below.
#
#   The teacher is loaded in 4-bit (bitsandbytes nf4) to fit on a single
#   consumer GPU. It is NOT loaded during the actual Stage 1 training run —
#   only its captions are consumed downstream.
#
# Output schema (one row per image):
#   {
#     "messages": [
#       {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": <prompt>}
#       ]},
#       {"role": "assistant", "content": [
#         {"type": "text", "text": <caption>}
#       ]}
#     ],
#     "images": [<PIL.Image>]
#   }
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


LLAVA_MED_MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"

CAPTION_PROMPT = (
    "Describe this medical image in clinical detail, including all visible "
    "findings, anatomical structures, and any abnormalities."
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger("build_stage1_dataset")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 1 captions with LLaVA-Med.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images (recursively scanned).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the resulting HuggingFace Dataset is saved (`save_to_disk`).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images per LLaVA-Med generation batch.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per caption.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of images to process (handy for smoke tests).",
    )
    return parser.parse_args()


def _discover_images(image_dir: Path) -> list[Path]:
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images with supported extensions found under {image_dir}.")
    return paths


def _load_teacher() -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(LLAVA_MED_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MED_MODEL_ID,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def _build_chat(prompt: str) -> str:
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def _generate_captions(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    images: list[PILImage.Image],
    max_new_tokens: int,
) -> list[str]:
    prompts = [_build_chat(CAPTION_PROMPT) for _ in images]
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    captions = []
    for text in decoded:
        # Keep only the assistant turn.
        captions.append(text.split("ASSISTANT:", maxsplit=1)[-1].strip())
    return captions


def _row(image: PILImage.Image, caption: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption},
                ],
            },
        ],
        "images": [image],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _discover_images(image_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    logger.info("Discovered %d images under %s", len(image_paths), image_dir)

    model, processor = _load_teacher()

    rows: list[dict] = []
    skipped: list[str] = []

    progress = tqdm(range(0, len(image_paths), args.batch_size), desc="LLaVA-Med captions")
    for start in progress:
        batch_paths = image_paths[start : start + args.batch_size]
        batch_images: list[PILImage.Image] = []
        loaded_paths: list[Path] = []
        for path in batch_paths:
            try:
                img = PILImage.open(path).convert("RGB")
                batch_images.append(img)
                loaded_paths.append(path)
            except Exception as err:
                logger.warning("Failed to load image %s: %s", path, err)
                skipped.append(str(path))

        if not batch_images:
            continue

        try:
            captions = _generate_captions(model, processor, batch_images, args.max_new_tokens)
        except Exception as err:
            logger.warning("Generation failed for batch starting at %s: %s", batch_paths[0], err)
            skipped.extend(str(p) for p in loaded_paths)
            continue

        for img, caption in zip(batch_images, captions, strict=True):
            rows.append(_row(img, caption))

    if not rows:
        raise RuntimeError("No captions generated — refusing to write an empty dataset.")

    features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": [
                        {"type": Value("string"), "text": Value("string")},
                    ],
                }
            ],
            "images": Sequence(Image()),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    dataset.save_to_disk(str(output_dir))
    logger.info("Wrote %d rows to %s (skipped %d images).", len(dataset), output_dir, len(skipped))


if __name__ == "__main__":
    main()
