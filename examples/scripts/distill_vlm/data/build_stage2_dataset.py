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
#   Stage 2 of the VLM knowledge-distillation pipeline expects a labelled,
#   class-balanced dataset of (image, question, answer, label) tuples. This
#   script reads a folder of images plus a CSV with columns
#   `image_path,label,question,answer` and produces a HuggingFace `Dataset`
#   in the chat-message format consumed by `DistillationTrainer`. Domain-
#   agnostic — the same schema applies whether the labels are radiology
#   findings, document categories, scientific concepts, etc.
#
#   It also computes a per-sample `class_weight = 1 / freq(label)` normalised
#   so the mean weight is 1. The Stage 2 trainer multiplies the KL term by
#   this weight, which counteracts class imbalance during distillation.
#
# Output schema:
#   {
#     "messages": [
#       {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": <question>}
#       ]},
#       {"role": "assistant", "content": [
#         {"type": "text", "text": <answer>}
#       ]}
#     ],
#     "images":       [<PIL.Image>],
#     "class_weight": <float>,
#     "ground_truth": <str>,
#     "label":        <str>
#   }
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from datasets import Dataset
from PIL import Image as PILImage
from tqdm import tqdm


logger = logging.getLogger("build_stage2_dataset")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage 2 VLM SFT/distillation dataset.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing the images referenced by the labels CSV.",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="CSV with columns: image_path, label, question, answer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the resulting HuggingFace Dataset (`save_to_disk`).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows (for smoke tests).",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    required = {"image_path", "label", "question", "answer"}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Labels CSV must contain columns {sorted(required)}, found {reader.fieldnames}."
            )
        return [row for row in reader]


def _compute_class_weights(rows: list[dict[str, str]]) -> dict[str, float]:
    """Inverse-frequency weights normalised so the mean weight across rows is 1."""
    counts = Counter(row["label"] for row in rows)
    inv = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = [inv[row["label"]] for row in rows]
    mean_weight = sum(sample_weights) / len(sample_weights)
    return {label: weight / mean_weight for label, weight in inv.items()}


def _row(image: PILImage.Image, question: str, answer: str, label: str, weight: float) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ],
        "images": [image],
        "class_weight": float(weight),
        "ground_truth": answer,
        "label": label,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(labels_csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No usable rows in {labels_csv}.")

    weights_by_label = _compute_class_weights(rows)
    logger.info(
        "Computed class weights for %d labels (min=%.3f, max=%.3f).",
        len(weights_by_label),
        min(weights_by_label.values()),
        max(weights_by_label.values()),
    )

    out_rows: list[dict] = []
    skipped: list[str] = []
    for row in tqdm(rows, desc="Loading images"):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = image_dir / image_path
        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as err:
            logger.warning("Failed to load %s: %s", image_path, err)
            skipped.append(str(image_path))
            continue

        out_rows.append(
            _row(
                image=image,
                question=row["question"],
                answer=row["answer"],
                label=row["label"],
                weight=weights_by_label[row["label"]],
            )
        )

    if not out_rows:
        raise RuntimeError("No rows survived image loading — refusing to write empty dataset.")

    dataset = Dataset.from_list(out_rows)
    dataset.save_to_disk(str(output_dir))
    logger.info("Wrote %d rows to %s (skipped %d).", len(dataset), output_dir, len(skipped))


if __name__ == "__main__":
    main()
