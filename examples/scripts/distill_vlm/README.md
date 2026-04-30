<!--
Why this file exists:
  Operator-facing runbook for the VLM knowledge-distillation pipeline
  implemented in this directory. The pipeline is domain-agnostic; the bundled
  example targets medical imaging (LLaVA-Med + VILA-M3 teachers) but the same
  scripts apply to any image-text domain after swapping the data builders and
  teacher model IDs. Lists every command in the order they should be run, plus
  a smoke-test path that validates the whole loop on tiny inputs before
  launching long training jobs.
-->

# VLM Knowledge Distillation (Stages 1 & 2)

Two-stage knowledge-distillation pipeline that compresses a large VLM teacher into the
500M-parameter SmolVLM2 student. **Stage 1** aligns the SmolVLM2 connector to the target
image distribution. **Stage 2** does multimodal SFT + LoRA on the language backbone, with
a class-frequency-weighted KL distillation term and an optional RDist relation-distillation
loss.

The bundled example targets **medical imaging** (LLaVA-Med Stage 1 teacher, VILA-M3 Stage 2
teacher). To retarget another domain (legal, scientific, …) you only need to swap:

* `data/build_stage1_dataset.py` — change `LLAVA_MED_MODEL_ID` and `CAPTION_PROMPT`.
* `data/build_stage2_dataset.py` — domain-agnostic; supply your own labels CSV.
* `train_stage2.py` — update `VILA_M3_MODEL_ID` (or just point `--teacher_url` at any
  vLLM-served teacher).

| Component | Path |
| --- | --- |
| Student model (default) | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (image+video; the only 500M SmolVLM2 image-instruct on the Hub) |
| Student model (recommended for the larger ISIC run) | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| Stage 1 teacher (offline, medical example) | `microsoft/llava-med-v1.5-mistral-7b` |
| Stage 2 teacher (vLLM server, medical example) | `Efficient-Large-Model/VILA-M3-7B` (verify on the Hub) |
| Trainer modifications | `trl/experimental/distillation_vlm/distillation_{config,trainer}.py` |
| Scripts | this folder |
| Accelerate config | `examples/accelerate_configs/distill_vlm_zero2.yaml` |

## 0. Install the fork as editable

```bash
pip install -e ".[dev]"
```

This makes any change to `trl/` apply immediately without a reinstall.

## 1. Generate the Stage 1 caption dataset (offline teacher)

```bash
python examples/scripts/distill_vlm/data/build_stage1_dataset.py \
  --image_dir /path/to/images \
  --output_dir data/stage1
```

In the medical example LLaVA-Med is loaded in 4-bit (bitsandbytes nf4 / bf16 compute) so a
single 24 GB GPU is enough. The output is a HuggingFace `Dataset` with one (image, caption)
pair per row in the `messages` schema consumed by `trl.SFTTrainer`.

## 2. Smoke-test Stage 1 (50 samples, 2 steps)

```bash
python examples/scripts/distill_vlm/train_stage1.py \
  --dataset_dir data/stage1 \
  --output_dir checkpoints/stage1 \
  --smoke_test
```

The script prints the trainable-module summary before training; it should list **only**
modules under `model.connector`.

## 3. Full Stage 1 training

```bash
accelerate launch \
  --config_file examples/accelerate_configs/distill_vlm_zero2.yaml \
  examples/scripts/distill_vlm/train_stage1.py \
    --dataset_dir data/stage1 \
    --output_dir checkpoints/stage1
```

After training completes, `checkpoints/stage1` is loadable with
`AutoModelForImageTextToText.from_pretrained("checkpoints/stage1")`.

## 4. Build the Stage 2 dataset

Provide a CSV with columns `image_path,label,question,answer`.

```bash
python examples/scripts/distill_vlm/data/build_stage2_dataset.py \
  --image_dir /path/to/images \
  --labels_csv /path/to/labels.csv \
  --output_dir data/stage2
```

Per-sample inverse-frequency `class_weight` values are computed inline.

## 5. Start the teacher (separate terminal, separate GPU)

Use `trl vllm-serve` (not the upstream `vllm serve`) so the custom `/get_sequence_logprobs/`
endpoint that `DistillationTrainer` requires is exposed:

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model Efficient-Large-Model/VILA-M3-7B \
  --port 8000 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.9
```

Multimodal teachers are supported: `VLMDistillationCollator` retains the raw PIL images
per-sample and the trainer forwards them on the `images` field of the request, which the
server attaches as `multi_modal_data` on each vLLM prompt. Text-only teachers continue to work
unchanged — when `images` is absent the request is byte-identical to the pre-VLM wire format.

> Verify the exact model ID on https://huggingface.co before running. The placeholder
> constant `VILA_M3_MODEL_ID` in `train_stage2.py` should be updated if the canonical ID
> differs (or if you are retargeting another domain).

## 6. Smoke-test Stage 2 (50 samples, 2 steps, real teacher)

```bash
python examples/scripts/distill_vlm/train_stage2.py \
  --stage1_ckpt checkpoints/stage1 \
  --dataset_dir data/stage2 \
  --output_dir checkpoints/stage2 \
  --smoke_test
```

The trainer logs three loss components every step:

```
loss/sft   – cross-entropy against ground-truth answers
loss/kl    – temperature-scaled KL against the teacher's top-50 logprobs
loss/rdist – pairwise visual-token similarity gap (zero unless the teacher
             exposes a local vision encoder; see notes below)
```

## 7. Full Stage 2 training

```bash
accelerate launch \
  --config_file examples/accelerate_configs/distill_vlm_zero2.yaml \
  examples/scripts/distill_vlm/train_stage2.py \
    --stage1_ckpt checkpoints/stage1 \
    --dataset_dir data/stage2 \
    --output_dir checkpoints/stage2
```

## SLURM smoke tests

The `slurm/` subdirectory ships ready-to-edit batch scripts. Copy them, replace the
`CHANGE_ME` placeholders in the `#SBATCH` headers (account, partition) and the
environment-activation lines for your cluster (conda/venv/module), then submit.

### Pre-flight (no GPU, ~30 s) — sanity-check the trainer changes first

Run these unit tests on a login node before burning a GPU allocation:

```bash
pip install -e ".[dev]"
pytest -q \
  tests/experimental/test_distillation_trainer.py \
  tests/experimental/test_distill_vlm_pipeline.py
```

These cover the `DistillationConfig` fields, the `_DistillationCollator` VLM
passthrough, the `processing_class` alias, the RDist hook registration, the
freezing logic for both stages, and the class-weight computation. If any of
these fail, fix locally before submitting a job.

### Synthetic smoke-test datasets (skip LLaVA-Med + the labels CSV)

The bundled builders need a real teacher (Stage 1) or a labelled CSV +
matching image folder (Stage 2). When you only want to verify the trainers
end-to-end, generate tiny random-image fixtures instead:

```bash
python examples/scripts/distill_vlm/data/build_synthetic_datasets.py \
  --output_dir /tmp/sanity
```

Writes 50-row HuggingFace `Dataset`s to `/tmp/sanity/stage1` and
`/tmp/sanity/stage2` in the exact schema consumed by `train_stage1.py` and
`train_stage2.py`. Use `--stage 1` or `--stage 2` to build only one.

### Stage 1 smoke test — single GPU, ~5–15 min

```bash
sbatch examples/scripts/distill_vlm/slurm/stage1_smoke.sbatch
```

What it does:

1. Allocates 1 node, 1 GPU.
2. Loads `data/stage1`, takes the first 50 rows, runs 2 optimizer steps.
3. Prints the trainable-parameter summary (must show only `model.connector`).
4. Writes nothing to disk (`save_strategy="no"` under `--smoke_test`).

Expected output: 2 `loss=…` log lines + a non-zero `train_loss` summary, no
crash, no `CUDA out of memory`.

### Stage 2 smoke test — 2 GPUs on one node, ~15–30 min

```bash
sbatch examples/scripts/distill_vlm/slurm/stage2_smoke.sbatch
```

What it does:

1. Allocates 1 node, 2 GPUs.
2. GPU 1 starts the teacher vLLM server in the background, waits up to 7.5 min
   for `http://localhost:8000/v1/models` to respond.
3. GPU 0 runs `train_stage2.py --smoke_test` (50 samples, 2 steps).
4. The teacher process is killed automatically when the script exits (`trap`).

Expected output: every step logs three loss components — `loss/sft`, `loss/kl`,
`loss/rdist`. `loss/rdist` will be `0.0` because the teacher is remote (see the
*RDist + remote teacher* note below). `loss/sft` and `loss/kl` should be
finite positive floats.

### Common overrides

Both scripts read environment variables before falling back to defaults, so you
can keep one `.sbatch` and parameterise it from the submit line:

```bash
PROJECT_DIR=/scratch/$USER/trl \
DATA_DIR=/scratch/$USER/data/stage1 \
OUT_DIR=/scratch/$USER/ckpts/stage1_smoke \
sbatch examples/scripts/distill_vlm/slurm/stage1_smoke.sbatch
```

### Multi-GPU full training

For the full (non-smoke) runs, replace `python …` with `accelerate launch` and
bump `--gres=gpu:N` + `#SBATCH --cpus-per-task` accordingly. The
`accelerate_configs/distill_vlm_zero2.yaml` config defaults to `num_processes=1`;
override it on the launch line:

```bash
accelerate launch \
  --config_file examples/accelerate_configs/distill_vlm_zero2.yaml \
  --num_processes $SLURM_GPUS_ON_NODE \
  examples/scripts/distill_vlm/train_stage1.py \
    --dataset_dir "$DATA_DIR" --output_dir "$OUT_DIR"
```

### Interactive debugging (faster iteration than `sbatch`)

```bash
salloc --account=CHANGE_ME --partition=CHANGE_ME \
  --gres=gpu:1 --cpus-per-task=8 --time=00:30:00
srun --pty bash
# then, on the compute node:
python examples/scripts/distill_vlm/train_stage1.py \
  --dataset_dir data/stage1 --output_dir /tmp/stage1_smoke --smoke_test
```

## Notes

* **RDist + remote teacher.** The provided `_register_rdist_hooks` registers the student hook
  unconditionally and the teacher hook only when `self.teacher_model is not None`. With a vLLM
  teacher served over HTTP there is no local teacher module to hook, so `_t_feat[0]` stays
  `None` and `loss/rdist` evaluates to `0`. To enable real RDist, load a small local teacher
  vision encoder alongside the vLLM client and adjust the hook path printed by:
  ```python
  list(dict(trainer.teacher_model.named_modules()).keys())[:60]
  ```

* **Frozen vision encoder + detached student feature.** Per spec, the student RDist hook
  detaches the captured feature, so RDist contributes zero gradient while `vision_model` stays
  frozen. The hook is left in place so Stage 3+ (which unfreezes parts of the vision encoder)
  can drop the `.detach()` and benefit from the alignment objective without re-plumbing.

* **Why a separate `distillation_vlm` package?** All VLM extensions live in
  `trl/experimental/distillation_vlm/`, leaving `trl/experimental/distillation/` byte-identical
  to upstream. This keeps a future upstream-PR diff minimal and avoids polluting upstream's
  text-only distillation behaviour with VLM-specific plumbing.
