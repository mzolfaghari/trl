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
#   Stage 2 of the VLM knowledge-distillation pipeline takes the connector-aligned
#   SmolVLM2 student from Stage 1 and finetunes it against a labelled VQA dataset,
#   distilling from a teacher served via vLLM at localhost:8000. The bundled
#   medical example uses a VILA-M3 7B teacher; other domains (legal, scientific,
#   …) only need to swap the data builders + teacher model ID.
#
#   Trainable / frozen split:
#     - vision_model : FROZEN
#     - connector    : full fine-tune (`modules_to_save` in LoRA config)
#     - text_model   : LoRA adapters (r=64) on q/k/v/o/gate/up/down projections
#
#   Loss:
#     L = alpha * SFT  +  beta_kl * KL  +  beta_rdist * RDist
#   computed by the VLM-extended `trl.experimental.distillation_vlm.DistillationTrainer`
#   (the upstream `trl.experimental.distillation` package is intentionally left
#   pristine; the VLM additions live in the parallel `distillation_vlm` package).
#
#   This script also defines `VLMDistillationCollator`, a thin VLM-aware data
#   collator that produces the (input_ids, labels, pixel_values, ...) dictionary
#   the patched DistillationTrainer expects, including the optional `class_weight`
#   field for class-frequency-weighted KL.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl.data_utils import prepare_multimodal_messages
from trl.experimental.distillation_vlm import DistillationConfig, DistillationTrainer


# Default teacher model ID for the medical example. Swap this constant (and
# point `--teacher_url` at the matching vLLM server) to retarget another domain.
# Verify the exact ID on https://huggingface.co — the medical VILA-M3 variants
# live under the Efficient-Large-Model and MONAI organizations.
VILA_M3_MODEL_ID = "Efficient-Large-Model/VILA-M3-7B"

logger = logging.getLogger("train_stage2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 VLM SFT + LoRA + KL + RDist distillation.")
    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default="checkpoints/stage1",
        help="Stage 1 student checkpoint produced by `train_stage1.py`.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/stage2",
        help="HuggingFace Dataset directory produced by `data/build_stage2_dataset.py`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/stage2",
        help="Where to save the Stage 2 LoRA adapters + connector weights.",
    )
    parser.add_argument(
        "--teacher_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the VILA-M3 vLLM server.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Train on 50 samples for 2 optimizer steps to validate the pipeline.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# VLM data collator
# ──────────────────────────────────────────────────────────────────────────────


class VLMDistillationCollator:
    """Per-sample VLM collator producing the tensors `DistillationTrainer` expects.

    For each example in the batch we render the prompt (everything but the last assistant turn)
    and the full message list separately via the processor's chat template. Both are tokenised
    *with images* so the processor's image-token expansion stays consistent. The completion token
    ids are then taken as the suffix of the full sequence and their positions are kept in
    `labels` while the prompt positions are set to `-100`.

    The output dict matches what the patched `_DistillationCollator` produces, plus
    `pixel_values` (and `pixel_attention_mask` when the processor emits it) and an optional
    per-sample `class_weight`.
    """

    def __init__(self, processor):
        self.processor = processor
        if getattr(processor.tokenizer, "pad_token_id", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def _process_one(self, messages, images):
        prepared = prepare_multimodal_messages(messages, images=list(images))
        prompt_only = prepared[:-1]

        full_text = self.processor.apply_chat_template(prepared, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_only, add_generation_prompt=True)

        full = self.processor(text=full_text, images=list(images), return_tensors="pt")
        prompt = self.processor(text=prompt_text, images=list(images), return_tensors="pt")

        full_ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        if prompt_len > full_ids.shape[0]:
            # Defensive: chat templates that don't strictly nest the prompt inside the full
            # text would break the suffix split. Fall back to no-completion training.
            prompt_len = full_ids.shape[0]

        labels = full_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": full_ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "prompts": prompt_ids,
            "prompt_attention_mask": prompt["attention_mask"][0],
            "pixel_values": full["pixel_values"][0],
        }
        if "pixel_attention_mask" in full:
            item["pixel_attention_mask"] = full["pixel_attention_mask"][0]
        return item

    @staticmethod
    def _left_pad_1d(tensors, pad_value):
        max_len = max(t.shape[0] for t in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, max_len - t.shape[0] :] = t
        return out

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        items = [self._process_one(ex["messages"], ex["images"]) for ex in examples]

        pad_id = self.processor.tokenizer.pad_token_id
        batch = {
            "input_ids": self._left_pad_1d([it["input_ids"] for it in items], pad_id),
            "attention_mask": self._left_pad_1d([it["attention_mask"] for it in items], 0),
            "labels": self._left_pad_1d([it["labels"] for it in items], -100),
            "prompts": self._left_pad_1d([it["prompts"] for it in items], pad_id),
            "prompt_attention_mask": self._left_pad_1d([it["prompt_attention_mask"] for it in items], 0),
        }

        # pixel_values shape varies across processors (e.g. SmolVLM2 produces (num_images, ...)).
        # Stack only when shapes line up across the batch — otherwise pad along the patch dim.
        pixel_tensors = [it["pixel_values"] for it in items]
        if all(t.shape == pixel_tensors[0].shape for t in pixel_tensors):
            batch["pixel_values"] = torch.stack(pixel_tensors)
        else:
            max_dim0 = max(t.shape[0] for t in pixel_tensors)
            padded = []
            for t in pixel_tensors:
                if t.shape[0] < max_dim0:
                    pad_shape = (max_dim0 - t.shape[0], *t.shape[1:])
                    t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                padded.append(t)
            batch["pixel_values"] = torch.stack(padded)

        if "pixel_attention_mask" in items[0]:
            mask_tensors = [it["pixel_attention_mask"] for it in items]
            if all(t.shape == mask_tensors[0].shape for t in mask_tensors):
                batch["pixel_attention_mask"] = torch.stack(mask_tensors)
            else:
                max_dim0 = max(t.shape[0] for t in mask_tensors)
                padded = []
                for t in mask_tensors:
                    if t.shape[0] < max_dim0:
                        pad_shape = (max_dim0 - t.shape[0], *t.shape[1:])
                        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                    padded.append(t)
                batch["pixel_attention_mask"] = torch.stack(padded)

        if "class_weight" in examples[0]:
            batch["class_weight"] = torch.tensor(
                [float(ex["class_weight"]) for ex in examples], dtype=torch.float32
            )

        return batch


# ──────────────────────────────────────────────────────────────────────────────
# Freezing
# ──────────────────────────────────────────────────────────────────────────────


def _freeze_vision_encoder(model: torch.nn.Module) -> None:
    """Freeze every parameter under `vision_model`, including any LoRA adapters PEFT may have
    attached to its attention projections (since `q_proj` etc. exist there too)."""
    frozen = 0
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False
            frozen += param.numel()
    logger.info("Frozen vision_model parameters: %s", f"{frozen:,}")


def _print_trainable_summary(model: torch.nn.Module) -> None:
    trainable = total = 0
    trainable_modules: set[str] = set()
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_modules.add(name.rsplit(".", 1)[0])
    logger.info("Trainable modules (Stage 2):")
    for name in sorted(trainable_modules):
        logger.info("  - %s", name)
    pct = 100.0 * trainable / max(total, 1)
    logger.info("Trainable params: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", pct)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    processor = AutoProcessor.from_pretrained(args.stage1_ckpt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.stage1_ckpt,
        dtype=torch.bfloat16,
    )

    # ── PEFT: LoRA adapters on the LLM projections + full fine-tune of the connector ──
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["connector"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    _freeze_vision_encoder(model)
    _print_trainable_summary(model)

    train_dataset = load_from_disk(args.dataset_dir)
    if args.smoke_test:
        n = min(50, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        logger.info("Smoke test active: training on %d samples for 2 steps.", n)

    # Note on `dataset_kwargs={"skip_prepare_dataset": True}`: this option only exists on
    # `SFTConfig`. `DistillationConfig` doesn't have it because `DistillationTrainer` already
    # consumes raw rows via its custom data collator (`remove_unused_columns` is also forced to
    # `False` inside the trainer). So we simply omit `dataset_kwargs` here — the effect is the
    # same as enabling `skip_prepare_dataset`.
    distill_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        bf16=True,
        max_length=None,
        max_prompt_length=None,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Distillation knobs
        use_teacher_server=True,
        teacher_model_server_url=args.teacher_url,
        temperature=2.0,
        beta=0.0,  # forward KL
        lmbda=0.0,  # pure off-policy (no on-policy generation)
        loss_top_k=50,
        # qvac multi-loss
        use_rdist=True,
        alpha=0.5,
        beta_kl=0.3,
        beta_rdist=0.2,
        # Logging / saving
        logging_steps=1,
        save_strategy="epoch" if not args.smoke_test else "no",
        report_to="none",
    )
    if args.smoke_test:
        distill_kwargs["max_steps"] = 2
        distill_kwargs["num_train_epochs"] = 1

    training_args = DistillationConfig(**distill_kwargs)

    collator = VLMDistillationCollator(processor=processor)

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=processor,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Stage 2 checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
