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

"""
SFT for SmolVLM2-500M on the Nemotron-Image-Training-v3 VQA dataset.

NOTE: this is NVIDIA's *Nemotron-Image-Training-v3* VLM dataset (image + Q/A
JSONL), NOT NVIDIA's *Nemotron-3 LLM* (which is what `sft_nemotron_3.py`
covers — a text-only Mamba model trained on a different multilingual dataset).
Same word, completely unrelated.

Mirrors the hyperparameters and lazy JSONL-offset loading strategy from
`/home/reza/projects/nanoVLM/nemotron_finetune.py`, but adapted to use TRL's
`SFTTrainer` and the SmolVLM2 vision-language model.

Dataset records are single-turn VQA: {config, image, question, answer}.
Images are resolved on-the-fly via `NemotronImageResolver` from the nanoVLM repo
(supports loose files, COCO/GQA subdirs, GeomVerse, and WebDataset shards).

Launch (single node, 8 GPUs):

    accelerate launch \\
        --config_file examples/accelerate_configs/deepspeed_zero3.yaml \\
        examples/scripts/sft_smolvlm2_nemotron.py \\
        --train_jsonl /home/Changdae/edge_vlm/nemotron_curation/results/manifests/nemotron_full_train.jsonl \\
        --nemotron_root /home/pengfei/datasets/Nemotron-Image-Training-v3 \\
        --output_dir checkpoints/smolvlm2-500m-nemotron \\
        --model_name_or_path HuggingFaceTB/SmolVLM2-500M-Video-Instruct
"""

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, TrainerCallback

from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

# Self-contained Nemotron data plumbing: byte-offset JSONL index with optional
# node-local /dev/shm copy, timeout-aware image resolver, WDS tar-index warmup.
# See examples/scripts/nemotron_data/__init__.py for origin and details.
from nemotron_data import (
    DEFAULT_ROOT as NEMOTRON_DEFAULT_ROOT,
    IndexedJsonlDataset,
    JsonlIndex,
    NemotronImageResolver,
    warmup_media_indexes,
)


@dataclass
class NemotronScriptArguments:
    """Arguments specific to the Nemotron JSONL data path."""

    train_jsonl: str = field(
        default="/home/Changdae/edge_vlm/nemotron_curation/results/manifests/nemotron_full_train.jsonl",
        metadata={"help": "Path to the Nemotron VQA JSONL manifest."},
    )
    nemotron_root: str = field(
        default=str(NEMOTRON_DEFAULT_ROOT),
        metadata={"help": "Root directory of Nemotron-Image-Training-v3 media."},
    )
    val_frac: float = field(
        default=0.01,
        metadata={"help": "Fraction of records held out for evaluation."},
    )
    smoke_limit: int = field(
        default=0,
        metadata={"help": "If > 0, truncate dataset to this many records (smoke test)."},
    )
    node_local_jsonl: bool = field(
        default=False,
        metadata={"help": "Copy the JSONL to /dev/shm or /tmp on each node before training. "
                          "Avoids NFS contention from 8 ranks/node hammering the same file."},
    )
    configs_stats_json: str = field(
        default="",
        metadata={"help": "Optional precomputed stats JSON with a `configs` list "
                          "(matches Pengfei's nemotron_*_stats.json schema). When set, "
                          "those configs are used for media-index warmup; otherwise the "
                          "warmup scans the first 10000 rows of the JSONL."},
    )
    lr_connector: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the modality projector (`connector`)."},
    )
    lr_text: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for the language model (`text_model`)."},
    )
    lr_vision: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for the vision encoder (`vision_model`)."},
    )


class NemotronVQAMessagesDataset(Dataset):
    """Yield TRL-format VQA examples backed by Pengfei's IndexedJsonlDataset.

    Each item returns conversational prompt-completion fields:
    `{"prompt": [...], "completion": [...], "images": [PIL.Image]}`.
    This allows TRL's VLM collator to apply completion-only masking (train on
    assistant answer tokens only), which mirrors Pengfei's objective.

    If an image fails to resolve (corrupted, missing, unreadable archive entry,
    NFS hang past 120s), we advance to the next index and try again, up to
    `_MAX_RESOLVE_RETRIES`. Returning `None` would crash the default vision
    collator, so we silently substitute a neighboring row — same effect as
    nanoVLM's `_safe_collate` filter.
    """

    _MAX_RESOLVE_RETRIES = 10

    def __init__(self, records: IndexedJsonlDataset, resolver: NemotronImageResolver):
        self.records = records
        self.resolver = resolver

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        n = len(self.records)
        for offset in range(self._MAX_RESOLVE_RETRIES):
            j = (idx + offset) % n
            rec = self.records[j]
            img = self.resolver.open_image(rec["config"], rec["image"])
            if img is not None:
                prompt = [{"role": "user", "content": rec["question"]}]
                completion = [{"role": "assistant", "content": rec["answer"]}]
                return {"prompt": prompt, "completion": completion, "images": [img]}
        raise RuntimeError(
            f"Failed to resolve any image in {self._MAX_RESOLVE_RETRIES} rows starting at idx={idx}; "
            f"check NemotronImageResolver root and JSONL contents."
        )


class StepNumberCallback(TrainerCallback):
    """Inject the current `global_step` into the logs dict so it appears in printed
    training logs alongside `loss`, `epoch`, etc. HF Trainer logs `epoch` by default
    but not `step`, which makes resume-from-checkpoint diagnostics harder."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Rebuild in place so `step` appears first in the printed dict.
        ordered = {"step": state.global_step, **logs}
        logs.clear()
        logs.update(ordered)


class BestCheckpointCallback(TrainerCallback):
    """Save a `best/` snapshot whenever the tracked eval metric improves.

    HF Trainer with `load_best_model_at_end=True` restores the best at the
    *end* of training but does not maintain a separate `best/` directory
    during the run. This callback fills that gap (matches Pengfei's pattern
    in nanoVLM/Nemotron_finetune/finetune.py) so the best-so-far checkpoint
    is always on disk, even if the job is killed mid-training.

    Wire up after constructing the trainer:

        best_cb = BestCheckpointCallback()
        trainer = GroupedLRSFTTrainer(..., callbacks=[best_cb])
        best_cb.trainer = trainer
    """

    def __init__(self, metric_name: str = "eval_loss", greater_is_better: bool = False):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_value: float | None = None
        self.trainer = None  # set by caller after the trainer is constructed

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return
        current = float(metrics[self.metric_name])
        improved = self.best_value is None or (
            current > self.best_value if self.greater_is_better else current < self.best_value
        )
        if not improved:
            return
        self.best_value = current
        best_dir = os.path.join(args.output_dir, "best")
        # save_model handles DDP-unwrap and rank-0 gating internally.
        self.trainer.save_model(best_dir)
        if state.is_world_process_zero:
            with open(os.path.join(best_dir, "best_metric.json"), "w") as f:
                json.dump(
                    {"metric": self.metric_name, "value": current, "step": state.global_step},
                    f,
                    indent=2,
                )
            print(
                f"[best] new {self.metric_name}={current:.4f} at step {state.global_step} -> {best_dir}",
                flush=True,
            )


class GroupedLRSFTTrainer(SFTTrainer):
    """SFTTrainer with per-submodule learning rates (connector / text / vision).

    The HF Trainer applies a single `learning_rate` to every parameter; the
    nanoVLM recipe instead trains the modality projector ~2x the LM and ~10x
    the vision tower. Override `create_optimizer` to build matching param
    groups before the base scheduler attaches.
    """

    def __init__(self, *args, lr_connector: float, lr_text: float, lr_vision: float, **kwargs):
        self._lr_connector = lr_connector
        self._lr_text = lr_text
        self._lr_vision = lr_vision
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model_wrapped if self.model_wrapped is not self.model else self.model

        connector_params, text_params, vision_params, other_params = [], [], [], []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if "connector" in name:
                connector_params.append(param)
            elif "text_model" in name or "language_model" in name:
                text_params.append(param)
            elif "vision_model" in name or "vision_tower" in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        wd = self.args.weight_decay
        param_groups = []
        if connector_params:
            param_groups.append({"params": connector_params, "lr": self._lr_connector, "weight_decay": wd, "name": "connector"})
        if text_params:
            param_groups.append({"params": text_params, "lr": self._lr_text, "weight_decay": wd, "name": "text"})
        if vision_params:
            param_groups.append({"params": vision_params, "lr": self._lr_vision, "weight_decay": wd, "name": "vision"})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.args.learning_rate, "weight_decay": wd, "name": "other"})

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        # nanoVLM uses AdamW betas=(0.9, 0.95); pass through SFTConfig adam_beta1/2.
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


def maybe_init_dist() -> None:
    """Initialize the torch.distributed process group early.

    HF Trainer / accelerate normally initializes the process group inside
    `SFTTrainer.__init__`. We need it earlier because `JsonlIndex` uses
    `dist.barrier()` to coordinate cache-build between master and non-master
    ranks, and the dataset is constructed before the trainer. Pattern
    borrowed verbatim from Pengfei's `nanoVLM/Nemotron_finetune/finetune.py`.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Single-process run (e.g., smoke test); nothing to initialize.
        return
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _is_master() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _split_train_val_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[n_val:], indices[:n_val]


def _collect_configs(index: JsonlIndex, stats_json: str, sample_n: int = 10000) -> set[str]:
    """Return the set of `config` names referenced by training data.

    Prefers a precomputed stats JSON (matches Pengfei's `nemotron_*_stats.json`
    schema, key `configs`). Falls back to scanning the first `sample_n` rows.
    """
    if stats_json and os.path.isfile(stats_json):
        with open(stats_json) as f:
            return set(json.load(f).get("configs", []))
    configs: set[str] = set()
    for i in range(min(sample_n, len(index))):
        configs.add(index[i]["config"])
    return configs


if __name__ == "__main__":
    parser = TrlParser((NemotronScriptArguments, SFTConfig, ModelConfig))
    nemotron_args, training_args, model_args = parser.parse_args_and_config()

    # Initialize the process group before constructing JsonlIndex so its
    # `dist.barrier()` can coordinate cache-build across ranks. HF Trainer
    # detects an already-initialized group and reuses it.
    maybe_init_dist()

    # VLM canonical (mirrors examples/scripts/sft_vlm.py): disable truncation.
    # SmolVLM2 tiles images into many <image> placeholders; truncating at a
    # fixed max_length can slice into the placeholder region and break the
    # processor's image-token-count invariant. None lets every sample through.
    training_args.max_length = None
    # For this one-turn VQA format, prompt-completion masking is equivalent to
    # answer-only supervision and matches Pengfei's manual label masking.
    training_args.completion_only_loss = True

    # The custom dataset returns pre-built {messages, images} dicts, so block
    # SFTTrainer's HF-Hub-shaped dataset prep and keep all columns.
    training_args.remove_unused_columns = False
    if training_args.dataset_kwargs is None:
        training_args.dataset_kwargs = {}
    training_args.dataset_kwargs["skip_prepare_dataset"] = True

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Dataset (Pengfei's pattern: JsonlIndex + media-index warmup)
    ################
    if _is_master():
        print(f"[jsonl] indexing {nemotron_args.train_jsonl} (node_local={nemotron_args.node_local_jsonl})", flush=True)
        os.makedirs(training_args.output_dir, exist_ok=True)
    _barrier()
    # Write the offsets cache under the run's output_dir so it's user-writable
    # (Pengfei's default — next to the source JSONL — fails when the JSONL
    # lives in a read-only shared dir like /home/pengfei/...).
    jsonl_index = JsonlIndex(
        nemotron_args.train_jsonl,
        node_local=nemotron_args.node_local_jsonl,
        cache_dir=training_args.output_dir,
    )
    if _is_master():
        print(f"[jsonl] {len(jsonl_index)} records", flush=True)

    train_idx, val_idx = _split_train_val_indices(
        len(jsonl_index), nemotron_args.val_frac, training_args.seed
    )
    if nemotron_args.smoke_limit > 0:
        train_idx = train_idx[: nemotron_args.smoke_limit]
    if _is_master():
        print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

    # Pre-build WebDataset tar indexes and GeomVerse index on rank 0 only.
    # Without this, each rank lazily indexes tars on first hit during the first
    # epoch — a thundering-herd NFS scan that can take hours.
    configs = _collect_configs(jsonl_index, nemotron_args.configs_stats_json)
    if _is_master():
        print(f"[warmup] building media indexes for {len(configs)} configs ...", flush=True)
        warmup_media_indexes(nemotron_args.nemotron_root, configs)
        print("[warmup] media indexes ready", flush=True)
    _barrier()

    resolver = NemotronImageResolver(nemotron_args.nemotron_root)
    train_records = IndexedJsonlDataset(jsonl_index, train_idx)
    val_records = IndexedJsonlDataset(jsonl_index, val_idx)
    train_dataset = NemotronVQAMessagesDataset(train_records, resolver)
    eval_dataset = NemotronVQAMessagesDataset(val_records, resolver) if training_args.eval_strategy != "no" else None

    ################
    # Training
    ################
    best_cb = BestCheckpointCallback(
        metric_name=training_args.metric_for_best_model or "eval_loss",
        greater_is_better=training_args.greater_is_better,
    )
    trainer = GroupedLRSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[StepNumberCallback(), best_cb],
        lr_connector=nemotron_args.lr_connector,
        lr_text=nemotron_args.lr_text,
        lr_vision=nemotron_args.lr_vision,
    )
    best_cb.trainer = trainer

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name="nemotron-tier1-vqa")
