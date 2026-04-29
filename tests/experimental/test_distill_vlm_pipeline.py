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
#   Unit tests for the VLM distillation example scripts under
#   `examples/scripts/distill_vlm/`. The bundled example targets medical
#   imaging (LLaVA-Med + VILA-M3 teachers); these tests are domain-agnostic
#   and exercise only the structural helpers + collator. The scripts are not
#   a Python package, so we load them as modules via
#   `importlib.util.spec_from_file_location` and exercise their freeze helpers,
#   dataset row builders, and the Stage 2 VLM data collator end-to-end without
#   ever touching the real (heavyweight) teacher or SmolVLM2-500M checkpoints.
#
#   Coverage:
#     - Stage 1 dataset row schema (`build_stage1_dataset._row`).
#     - Stage 1 freeze helper (`train_stage1._freeze_all_except_connector`).
#     - Stage 2 class-weight computation and row schema
#       (`build_stage2_dataset._compute_class_weights`, `_row`).
#     - Stage 2 vision-encoder freeze
#       (`train_stage2._freeze_vision_encoder`).
#     - Stage 2 VLM data collator
#       (`train_stage2.VLMDistillationCollator`) returns the dict shape
#       the patched `DistillationTrainer` expects, including
#       `pixel_values`, `prompts`, `prompt_attention_mask`, `labels` (with
#       `-100` masked prompt) and the optional `class_weight`.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage
from transformers import AutoProcessor

from ..testing_utils import TrlTestCase, require_vision


# ──────────────────────────────────────────────────────────────────────────────
# Module loaders
# ──────────────────────────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL_VLM_DIR = REPO_ROOT / "examples" / "scripts" / "distill_vlm"


def _load_script(relative_path: str, module_name: str):
    """Load a script as a uniquely-named module without touching `sys.path` globally."""
    full_path = DISTILL_VLM_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, str(full_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {full_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage1_dataset_module():
    return _load_script("data/build_stage1_dataset.py", "_test_build_stage1_dataset")


@pytest.fixture(scope="module")
def stage1_train_module():
    return _load_script("train_stage1.py", "_test_train_stage1")


@pytest.fixture(scope="module")
def stage2_dataset_module():
    return _load_script("data/build_stage2_dataset.py", "_test_build_stage2_dataset")


@pytest.fixture(scope="module")
def stage2_train_module():
    return _load_script("train_stage2.py", "_test_train_stage2")


# ──────────────────────────────────────────────────────────────────────────────
# Tiny models we use to exercise freeze helpers (kept local so the tests stay
# offline and fast)
# ──────────────────────────────────────────────────────────────────────────────


def _make_smolvlm_like() -> nn.Module:
    """Module shaped like `SmolVLMForConditionalGeneration` for freeze-helper tests.

    We mirror the canonical attribute path:
        model.model.vision_model
        model.model.connector
        model.model.text_model
        model.lm_head
    so the helpers in `train_stage{1,2}.py` find the same names they look for at runtime.
    """

    class Connector(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

    class VisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = VisionModel()
            self.connector = Connector()
            self.text_model = TextModel()

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(8, 32)

    return Outer()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: dataset schema
# ──────────────────────────────────────────────────────────────────────────────


@require_vision
class TestStage1DatasetRowSchema(TrlTestCase):
    def test_row_matches_expected_schema(self, stage1_dataset_module):
        image = PILImage.new("RGB", (16, 16), color=(127, 200, 33))
        caption = "Mock caption."
        row = stage1_dataset_module._row(image, caption)

        assert set(row) == {"messages", "images"}
        assert row["images"] == [image]

        messages = row["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # User turn must have one image placeholder followed by the canonical caption prompt.
        user_content = messages[0]["content"]
        types = [block["type"] for block in user_content]
        assert types == ["image", "text"]
        assert user_content[1]["text"] == stage1_dataset_module.CAPTION_PROMPT

        # Assistant turn must be a single text block holding the caption.
        assistant_content = messages[1]["content"]
        assert assistant_content == [{"type": "text", "text": caption}]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: freeze helper
# ──────────────────────────────────────────────────────────────────────────────


class TestStage1Freezing(TrlTestCase):
    def test_only_connector_remains_trainable(self, stage1_train_module):
        model = _make_smolvlm_like()
        trainable, total = stage1_train_module._freeze_all_except_connector(model)

        for name, param in model.named_parameters():
            if name.startswith("model.connector"):
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

        # The trainable count must match the connector parameter count exactly.
        connector_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.connector"))
        assert trainable == connector_params
        assert total == sum(p.numel() for p in model.parameters())
        assert 0 < trainable < total


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: class weights + dataset row
# ──────────────────────────────────────────────────────────────────────────────


class TestStage2ClassWeights(TrlTestCase):
    def test_inverse_frequency_normalized_to_unit_mean(self, stage2_dataset_module):
        rows = (
            [{"label": "a"}] * 4
            + [{"label": "b"}] * 1
            + [{"label": "c"}] * 5
        )
        weights = stage2_dataset_module._compute_class_weights(rows)

        # Per-sample weights (the implementation normalises by their mean, not the
        # mean of unique-class weights).
        sample_weights = [weights[r["label"]] for r in rows]
        mean_w = sum(sample_weights) / len(sample_weights)
        assert mean_w == pytest.approx(1.0, abs=1e-6)

        # Inverse-frequency ordering must hold: rarer label → larger weight.
        assert weights["b"] > weights["a"] > weights["c"]

    def test_class_weight_values_match_closed_form(self, stage2_dataset_module):
        # Closed-form sanity check on a small fixed sample.
        rows = [{"label": x} for x in ["a", "a", "b", "c", "c", "c"]]
        weights = stage2_dataset_module._compute_class_weights(rows)
        # Per-sample inverse frequency (a: 1/2, b: 1, c: 1/3) has mean 0.5 → divide by 0.5.
        assert weights["a"] == pytest.approx(1.0)
        assert weights["b"] == pytest.approx(2.0)
        assert weights["c"] == pytest.approx(2.0 / 3.0)

    def test_row_schema(self, stage2_dataset_module):
        image = PILImage.new("RGB", (8, 8), color=(0, 0, 0))
        row = stage2_dataset_module._row(
            image=image,
            question="What is shown?",
            answer="A pneumothorax.",
            label="pneumothorax",
            weight=1.25,
        )

        assert set(row) == {"messages", "images", "class_weight", "ground_truth", "label"}
        assert row["class_weight"] == pytest.approx(1.25)
        assert row["ground_truth"] == "A pneumothorax."
        assert row["label"] == "pneumothorax"
        assert row["images"] == [image]

        msgs = row["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0] == {"type": "image"}
        assert msgs[0]["content"][1] == {"type": "text", "text": "What is shown?"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == [{"type": "text", "text": "A pneumothorax."}]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: freeze helper
# ──────────────────────────────────────────────────────────────────────────────


class TestStage2Freezing(TrlTestCase):
    def test_vision_encoder_only_is_frozen(self, stage2_train_module):
        model = _make_smolvlm_like()
        # Pretend the model is fully trainable to start (mimics post-PEFT state).
        for p in model.parameters():
            p.requires_grad = True

        stage2_train_module._freeze_vision_encoder(model)

        for name, param in model.named_parameters():
            if "vision_model" in name:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: VLM data collator
# ──────────────────────────────────────────────────────────────────────────────


@require_vision
class TestStage2Collator(TrlTestCase):
    """Exercises the VLM distillation collator end-to-end with a tiny SmolVLM processor."""

    @classmethod
    def setup_class(cls):
        cls.processor = AutoProcessor.from_pretrained(
            "trl-internal-testing/tiny-SmolVLMForConditionalGeneration"
        )

    def _example(self, question: str, answer: str, *, with_class_weight: bool = False) -> dict:
        image = PILImage.new("RGB", (32, 32), color=(120, 120, 120))
        ex = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ],
            "images": [image],
        }
        if with_class_weight:
            ex["class_weight"] = 0.75
        return ex

    def test_returns_expected_keys_and_shapes(self, stage2_train_module):
        collator = stage2_train_module.VLMDistillationCollator(self.processor)
        batch = collator([self._example("q1", "a1"), self._example("question two?", "answer two")])

        for key in ("input_ids", "attention_mask", "labels", "prompts", "prompt_attention_mask", "pixel_values"):
            assert key in batch, f"missing collator output key: {key}"

        assert batch["input_ids"].dim() == 2
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["pixel_values"].shape[0] == 2  # batch dim

        # `class_weight` is opt-in and must not be present unless examples provide it.
        assert "class_weight" not in batch

    def test_labels_mask_prompt_with_minus_100(self, stage2_train_module):
        collator = stage2_train_module.VLMDistillationCollator(self.processor)
        batch = collator([self._example("q", "ans")])

        labels = batch["labels"]
        attention = batch["attention_mask"]

        # At least one position must be -100 (the masked prompt) and at least one must be a real id
        # (the masked-out completion target).
        assert (labels == -100).any()
        non_pad_real_labels = ((labels != -100) & (attention.bool())).sum().item()
        assert non_pad_real_labels > 0, "expected at least one un-masked completion token in labels"

    def test_class_weight_passthrough(self, stage2_train_module):
        collator = stage2_train_module.VLMDistillationCollator(self.processor)
        batch = collator(
            [
                self._example("q1", "a1", with_class_weight=True),
                self._example("q2", "a2", with_class_weight=True),
            ]
        )
        assert "class_weight" in batch
        assert batch["class_weight"].dtype == torch.float32
        assert torch.equal(batch["class_weight"], torch.tensor([0.75, 0.75], dtype=torch.float32))
