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
#   Stage 1 of the VLM knowledge-distillation pipeline trains ONLY the SmolVLM2
#   connector (the MLP that bridges vision features into the language model)
#   on captions produced by `data/build_stage1_dataset.py` (LLaVA-Med in the
#   bundled medical example). The vision encoder, the language backbone and
#   the LM head are all frozen so the connector can specialise to the target
#   image distribution without destabilising the rest of the model.
#
#   The trainer is the standard `trl.SFTTrainer` (no subclassing): SFTTrainer
#   already auto-selects a vision-language data collator when the model is a
#   VLM, which is exactly what we need.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging

import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl import SFTConfig, SFTTrainer


SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Instruct"

logger = logging.getLogger("train_stage1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 connector alignment for SmolVLM2.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/stage1",
        help="Path to the HuggingFace Dataset produced by `build_stage1_dataset.py`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/stage1",
        help="Directory where the Stage 1 checkpoint is written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=SMOLVLM_MODEL_ID,
        help="Student model ID. Defaults to the SmolVLM2 500M instruct checkpoint.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run on 50 samples for 2 optimizer steps to verify the pipeline end-to-end.",
    )
    return parser.parse_args()


def _freeze_all_except_connector(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every parameter except those under `model.model.connector`.

    The path ``model.model.connector`` matches the SmolVLM family (verified against
    `transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration`).
    Returns a `(trainable, total)` parameter-count tuple for logging.
    """
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        is_connector = name.startswith("model.connector.") or name == "model.connector"
        param.requires_grad = is_connector
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def _print_trainable_summary(model: torch.nn.Module, trainable: int, total: int) -> None:
    trainable_modules = sorted(
        {name.rsplit(".", 1)[0] for name, p in model.named_parameters() if p.requires_grad}
    )
    logger.info("Trainable modules (Stage 1):")
    for name in trainable_modules:
        logger.info("  - %s", name)
    pct = 100.0 * trainable / max(total, 1)
    logger.info("Trainable params: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", pct)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    )

    trainable, total = _freeze_all_except_connector(model)
    _print_trainable_summary(model, trainable, total)

    train_dataset = load_from_disk(args.dataset_dir)
    if args.smoke_test:
        n = min(50, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        logger.info("Smoke test active: training on %d samples for 2 steps.", n)

    sft_config_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=1e-3,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="epoch" if not args.smoke_test else "no",
        report_to="none",
    )
    if args.smoke_test:
        sft_config_kwargs["max_steps"] = 2
        sft_config_kwargs["num_train_epochs"] = 1

    training_args = SFTConfig(**sft_config_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Stage 1 checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
