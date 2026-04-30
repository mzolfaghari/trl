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
#   `build_stage1_dataset.py` runs LLaVA-Med (4-bit, ~7B params) and
#   `build_stage2_dataset.py` requires a labelled CSV + matching image folder.
#   Neither is appropriate when all you want is a 30-second smoke-test fixture
#   that exercises the trainers end-to-end. This script generates tiny
#   synthetic datasets (random RGB images + dummy captions/answers) in the
#   exact schema consumed by `train_stage1.py` and `train_stage2.py`.
#
#   Output layout:
#       <output_dir>/stage1   # consumed by train_stage1.py --dataset_dir
#       <output_dir>/stage2   # consumed by train_stage2.py --dataset_dir
#
# Example:
#       python examples/scripts/distill_vlm/data/build_synthetic_datasets.py \
#           --output_dir /tmp/sanity
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


CAPTION_PROMPT = "Describe this image in detail."
STAGE2_QUESTION = "What is in the image?"
STAGE2_LABELS = ["cat", "dog"]

logger = logging.getLogger("build_synthetic_datasets")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny synthetic datasets for the Stage 1 / Stage 2 smoke tests."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/sanity",
        help="Parent directory; subdirectories `stage1/` and `stage2/` are written underneath.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Rows per stage. The smoke tests cap at 50 anyway.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Side length of the generated square images.",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which stage(s) to build.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _random_image(rng: np.random.Generator, size: int) -> PILImage.Image:
    return PILImage.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def _build_stage1(output_dir: Path, num_samples: int, image_size: int, rng: np.random.Generator) -> None:
    rows = []
    for i in range(num_samples):
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "text": None},
                            {"type": "text", "text": CAPTION_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Synthetic caption for image #{i}."},
                        ],
                    },
                ],
                "images": [_random_image(rng, image_size)],
            }
        )

    features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": [{"type": Value("string"), "text": Value("string")}],
                }
            ],
            "images": Sequence(Image()),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    logger.info("Stage 1: wrote %d rows to %s", len(dataset), output_dir)


def _build_stage2(output_dir: Path, num_samples: int, image_size: int, rng: np.random.Generator) -> None:
    rows = []
    for i in range(num_samples):
        label = STAGE2_LABELS[i % len(STAGE2_LABELS)]
        answer = f"It is a {label}."
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "text": None},
                            {"type": "text", "text": STAGE2_QUESTION},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ],
                "images": [_random_image(rng, image_size)],
                "class_weight": 1.0,
                "ground_truth": answer,
                "label": label,
            }
        )

    dataset = Dataset.from_list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    logger.info("Stage 2: wrote %d rows to %s", len(dataset), output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.stage in ("1", "both"):
        _build_stage1(output_dir / "stage1", args.num_samples, args.image_size, rng)
    if args.stage in ("2", "both"):
        _build_stage2(output_dir / "stage2", args.num_samples, args.image_size, rng)


if __name__ == "__main__":
    main()
