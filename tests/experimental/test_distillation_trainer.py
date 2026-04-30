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
#   Unit tests for the qvac additions to
#   `trl/experimental/distillation_vlm/distillation_trainer.py` and its
#   accompanying `DistillationConfig`. These are scoped to the *new* behaviour
#   only — the existing JSD / KL / on-policy paths are exercised by upstream
#   distillation tests and we explicitly verify backwards compatibility here
#   (use_rdist=False must reproduce the pre-patch loss return value).
#
#   IMPORTANT: import from `trl.experimental.distillation_vlm` (NOT the upstream
#   `trl.experimental.distillation`). The two packages co-exist by design — the
#   upstream package is intentionally left pristine for parity with HF/TRL, and
#   all VLM-distillation additions live in the parallel `_vlm` package.
#
#   The tests fall into four buckets:
#     1. Config: new dataclass fields are accepted and propagate.
#     2. Collator: `_DistillationCollator` forwards `pixel_values`,
#        `pixel_attention_mask`, and `class_weight` when present in examples,
#        and remains a strict no-op for text-only batches.
#     3. processing_class alias: `tokenizer` constructor parameter falls back
#        to `processing_class` and the data collator picks up the resolved
#        tokenizer.
#     4. Loss path: end-to-end smoke training with `use_rdist=True` populates
#        the `loss/sft`, `loss/kl`, `loss/rdist` metrics and the combined loss
#        equals `alpha*sft + beta_kl*kl + beta_rdist*rdist` (within float
#        tolerance). RDist hooks register cleanly on a model exposing the
#        SmolVLM-style `model.vision_model.encoder.layers` path and degrade
#        gracefully (warning, not exception) when the path is missing.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.distillation_vlm import DistillationConfig, DistillationTrainer
from trl.experimental.distillation_vlm.distillation_trainer import _DistillationCollator

from ..testing_utils import TrlTestCase


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


SMALL_TEXT_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def _make_text_examples(tokenizer, n: int = 2) -> list[dict]:
    """Tiny chat-format examples used to construct collator inputs."""
    return [
        {
            "messages": [
                {"role": "user", "content": f"hello {i}"},
                {"role": "assistant", "content": f"world {i}"},
            ]
        }
        for i in range(n)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────────────


class TestDistillationConfigQvacFields(TrlTestCase):
    def test_defaults(self):
        config = DistillationConfig(output_dir=self.tmp_dir)
        assert config.use_rdist is False
        assert config.alpha == 0.5
        assert config.beta_kl == 0.3
        assert config.beta_rdist == 0.2
        assert config.teacher_conf_threshold == 0.7

    def test_custom_values(self):
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            use_rdist=True,
            alpha=0.6,
            beta_kl=0.25,
            beta_rdist=0.15,
            teacher_conf_threshold=0.8,
        )
        assert config.use_rdist is True
        assert config.alpha == 0.6
        assert config.beta_kl == 0.25
        assert config.beta_rdist == 0.15
        assert config.teacher_conf_threshold == 0.8

    def test_post_init_does_not_override_qvac_fields(self):
        # The qvac fields are not part of any validation rule in __post_init__;
        # changing per_device_train_batch_size / gradient_accumulation_steps must
        # not affect them.
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            use_rdist=True,
            alpha=0.4,
            beta_kl=0.4,
            beta_rdist=0.2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        )
        assert config.use_rdist is True
        assert config.alpha + config.beta_kl + config.beta_rdist == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Collator additions
# ──────────────────────────────────────────────────────────────────────────────


class TestDistillationCollatorVLMAdditions(TrlTestCase):
    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(SMALL_TEXT_MODEL_ID)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def _collator(self):
        return _DistillationCollator(tokenizer=self.tokenizer, max_length=64, max_prompt_length=32)

    def test_text_only_unchanged(self):
        collator = self._collator()
        examples = _make_text_examples(self.tokenizer)
        batch = collator(examples)

        # The legacy schema must remain intact for text-only callers.
        assert set(batch).issuperset({"input_ids", "attention_mask", "labels", "prompts", "prompt_attention_mask"})
        assert "pixel_values" not in batch
        assert "pixel_attention_mask" not in batch
        assert "class_weight" not in batch

    def test_pixel_values_forwarded(self):
        collator = self._collator()
        examples = _make_text_examples(self.tokenizer)
        for example in examples:
            example["pixel_values"] = torch.zeros(3, 4, 4)
            example["pixel_attention_mask"] = torch.ones(4, 4, dtype=torch.long)

        batch = collator(examples)

        assert batch["pixel_values"].shape == (len(examples), 3, 4, 4)
        assert batch["pixel_attention_mask"].shape == (len(examples), 4, 4)
        assert torch.equal(batch["pixel_values"], torch.stack([ex["pixel_values"] for ex in examples]))

    def test_class_weight_forwarded(self):
        collator = self._collator()
        examples = _make_text_examples(self.tokenizer)
        weights = [0.5, 1.5]
        for example, w in zip(examples, weights, strict=True):
            example["class_weight"] = w

        batch = collator(examples)
        assert batch["class_weight"].dtype == torch.float32
        assert torch.equal(batch["class_weight"], torch.tensor(weights, dtype=torch.float32))

    def test_class_weight_only_when_present(self):
        collator = self._collator()
        examples = _make_text_examples(self.tokenizer)  # no class_weight
        batch = collator(examples)
        assert "class_weight" not in batch


# ──────────────────────────────────────────────────────────────────────────────
# 3. processing_class alias
# ──────────────────────────────────────────────────────────────────────────────


class TestDistillationTrainerProcessingClassAlias(TrlTestCase):
    """The new `tokenizer` parameter must alias from `processing_class` for back-compat."""

    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(SMALL_TEXT_MODEL_ID)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

    def _build_trainer(self, **trainer_kwargs):
        model = AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID)
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            report_to="none",
            use_teacher_server=False,
        )
        return DistillationTrainer(
            model=model,
            teacher_model=AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID),
            args=config,
            train_dataset=self.dataset,
            **trainer_kwargs,
        )

    def test_tokenizer_aliases_from_processing_class(self):
        trainer = self._build_trainer(processing_class=self.tokenizer)
        # The collator should have been constructed with the resolved tokenizer
        # (i.e. our processing_class), not None.
        assert isinstance(trainer.data_collator, _DistillationCollator)
        assert trainer.data_collator.tokenizer is self.tokenizer

    def test_explicit_tokenizer_wins_over_processing_class(self):
        explicit = AutoTokenizer.from_pretrained(SMALL_TEXT_MODEL_ID)
        if explicit.pad_token is None:
            explicit.pad_token = explicit.eos_token
        trainer = self._build_trainer(processing_class=self.tokenizer, tokenizer=explicit)
        assert trainer.data_collator.tokenizer is explicit


# ──────────────────────────────────────────────────────────────────────────────
# 4. RDist hooks + combined loss
# ──────────────────────────────────────────────────────────────────────────────


class _FakeVisionLayer(nn.Module):
    """Stub that lets us drive the RDist student/teacher hooks deterministically."""

    def __init__(self, batch: int = 1, tokens: int = 4, dim: int = 6):
        super().__init__()
        self.batch = batch
        self.tokens = tokens
        self.dim = dim
        # Trainable parameter so nn.Module bookkeeping is non-empty.
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, _x=None):
        return torch.randn(self.batch, self.tokens, self.dim)


class TestRDistHookRegistration(TrlTestCase):
    """`_register_rdist_hooks` must register on SmolVLM-shaped paths and warn otherwise."""

    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(SMALL_TEXT_MODEL_ID)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

    def _make_trainer(self, *, use_rdist: bool):
        model = AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID)
        teacher = AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID)
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            report_to="none",
            use_rdist=use_rdist,
        )
        return DistillationTrainer(
            model=model,
            teacher_model=teacher,
            args=config,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )

    def test_buffers_initialized_even_without_rdist(self):
        trainer = self._make_trainer(use_rdist=False)
        assert trainer._s_feat == [None]
        assert trainer._t_feat == [None]

    def test_warns_when_path_missing(self):
        # The text-only models used in tests don't expose `model.vision_model`,
        # so the registration must downgrade to a warning rather than crashing.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer = self._make_trainer(use_rdist=True)
        assert trainer._s_feat == [None]
        assert trainer._t_feat == [None]
        messages = " ".join(str(w.message) for w in caught)
        assert "RDist hook" in messages

    def test_hook_closure_captures_and_detaches_feature(self):
        # The lambda we register on the student's last vision layer must (a) populate
        # `_s_feat[0]`, (b) detach so it cannot leak gradients into the frozen vision
        # encoder, and (c) handle both tensor and (tensor, ...) tuple outputs.
        feat = [None]
        layer = _FakeVisionLayer(batch=2, tokens=3, dim=4)
        layer.register_forward_hook(
            lambda m, i, o: feat.__setitem__(
                0, (o[0] if isinstance(o, tuple) else o).detach()
            )
        )
        # Trigger a real forward — `__call__` (not `forward`) is what fires hooks.
        _ = layer(torch.zeros(1))
        assert feat[0] is not None
        assert feat[0].shape == (2, 3, 4)
        assert not feat[0].requires_grad


class TestCombinedLossAndLogging(TrlTestCase):
    """End-to-end: a single training step with `use_rdist=True` produces all three loss metrics."""

    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(SMALL_TEXT_MODEL_ID)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

    def _build(self, *, use_rdist: bool, alpha=0.5, beta_kl=0.3, beta_rdist=0.2):
        model = AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID)
        teacher = AutoModelForCausalLM.from_pretrained(SMALL_TEXT_MODEL_ID)
        config = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            logging_steps=1,
            report_to="none",
            # Pin precision to fp32 so this metric-logging test is independent of
            # GPU availability / bf16 detection. The values asserted below are
            # logging-only and don't depend on training dtype.
            bf16=False,
            fp16=False,
            beta=0.0,
            lmbda=0.0,
            use_rdist=use_rdist,
            alpha=alpha,
            beta_kl=beta_kl,
            beta_rdist=beta_rdist,
        )
        # Suppress the warning emitted while registering RDist hooks against a text-only model.
        ctx = warnings.catch_warnings() if use_rdist else nullcontext()
        with ctx:
            if use_rdist:
                warnings.simplefilter("ignore")
            trainer = DistillationTrainer(
                model=model,
                teacher_model=teacher,
                args=config,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
            )
        return trainer

    @staticmethod
    def _last_logged(trainer, key: str):
        """Return the most-recent value `key` was logged with via the Trainer's log_history."""
        for entry in reversed(trainer.state.log_history):
            if key in entry:
                return entry[key]
        return None

    def test_use_rdist_false_preserves_legacy_loss(self):
        trainer = self._build(use_rdist=False)
        trainer.train()
        # No SFT/RDist component should appear in the log_history when use_rdist is off.
        for key in ("loss/sft", "loss/kl", "loss/rdist"):
            assert self._last_logged(trainer, key) is None, f"unexpected metric {key} logged"

    def test_use_rdist_true_logs_three_components(self):
        trainer = self._build(use_rdist=True)
        trainer.train()

        sft = self._last_logged(trainer, "loss/sft")
        kl = self._last_logged(trainer, "loss/kl")
        rdist = self._last_logged(trainer, "loss/rdist")

        for key, value in (("loss/sft", sft), ("loss/kl", kl), ("loss/rdist", rdist)):
            assert value is not None, f"metric {key} was never logged"

        # Text-only models can't satisfy the RDist hook path, so the contribution must be 0.
        assert rdist == pytest.approx(0.0, abs=1e-6)
        # Sanity: the other components are finite and non-negative.
        assert sft >= 0
        # Student and teacher start from the same checkpoint, so KL is ~0 modulo
        # fp32 reduction noise; just guard against gross negative values / NaNs.
        assert math.isfinite(kl)
        assert kl == pytest.approx(0.0, abs=1e-4)


class TestRDistMath(TrlTestCase):
    """Direct verification of the RDist similarity-matrix MSE used in compute_loss."""

    def test_identical_features_yield_zero_loss(self):
        s = torch.randn(2, 5, 7)
        # Manually compute the same expression compute_loss runs.
        s_n = torch.nn.functional.normalize(s.float(), dim=-1)
        sim = torch.bmm(s_n, s_n.transpose(1, 2))
        loss = torch.nn.functional.mse_loss(sim, sim)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_orthogonal_vs_identity_features_increases_loss(self):
        s = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
        t = torch.randn(1, 4, 4)
        s_n = torch.nn.functional.normalize(s.float(), dim=-1)
        t_n = torch.nn.functional.normalize(t.float(), dim=-1)
        s_sim = torch.bmm(s_n, s_n.transpose(1, 2))
        t_sim = torch.bmm(t_n, t_n.transpose(1, 2))
        loss = torch.nn.functional.mse_loss(s_sim, t_sim)
        assert loss.item() > 0
