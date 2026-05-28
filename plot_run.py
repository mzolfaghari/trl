"""
Visualize a training run from SLURM/HF trainer logs.

This script is adapted from nanoVLM's `plot_run.py` and extended for TRL-style
logs that print Python dict metrics, for example:

  {'step': 500, 'loss': 6.6265, ...}
  {'step': 500, 'eval_loss': 6.4032, ...}

Usage:
    # Search logs/ recursively for <job_id>*.out
    python plot_run.py <job_id>

    # Restrict search root
    python plot_run.py <job_id> --log_dir logs/sft_smolvlm2_nemotron

    # Use explicit path
    python plot_run.py <job_id> --log_path logs/sft_smolvlm2_nemotron/58469_4294967294.out

    # Validation loss only
    python plot_run.py <job_id> --val_only

Output:
    experiments/<job_id>/loss_curve.png
    experiments/<job_id>/val_loss.png  (with --val_only)
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

try:
    import seaborn as sns

    sns.set_theme(style="darkgrid", font_scale=1.05)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


COLORS = {
    "train": "#4C72B0",
    "val": "#C44E52",
    "lr": "#55A868",
    "entropy": "#8172B2",
    "acc": "#DD8452",
    "tokens": "#937860",
}

dark = not HAS_SEABORN


def ema(values: list[float], alpha: float = 0.15) -> list[float]:
    """Exponential moving average used only for plotting."""
    out: list[float] = []
    running = None
    for value in values:
        running = value if running is None else alpha * value + (1 - alpha) * running
        out.append(running)
    return out


def style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color="white" if dark else "#222")
    ax.set_xlabel(xlabel, fontsize=10, color="white" if dark else "#444")
    ax.set_ylabel(ylabel, fontsize=10, color="white" if dark else "#444")
    if dark:
        ax.set_facecolor("#2c2c2e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#555")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.25)


def _parse_metric_dict(line: str):
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None

    try:
        payload = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None
    return payload


def _maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_log(path: Path) -> dict:
    """Parse TRL/HF trainer logs and return train/eval metrics by step."""
    train = []
    val = []

    # Optional fallback for non-dict val log format.
    fallback_val_re = re.compile(r"Step:\s*(\d+),\s*Val Loss:\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)")

    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            metrics = _parse_metric_dict(line)
            if metrics is not None:
                step = _maybe_int(metrics.get("step"))
                if step is None:
                    continue

                if "loss" in metrics:
                    loss = _maybe_float(metrics.get("loss"))
                    if loss is not None:
                        train.append(
                            {
                                "step": step,
                                "loss": loss,
                                "learning_rate": _maybe_float(metrics.get("learning_rate")),
                                "entropy": _maybe_float(metrics.get("entropy")),
                                "num_tokens": _maybe_float(metrics.get("num_tokens")),
                                "mean_token_accuracy": _maybe_float(metrics.get("mean_token_accuracy")),
                                "epoch": _maybe_float(metrics.get("epoch")),
                            }
                        )
                    continue

                if "eval_loss" in metrics:
                    eval_loss = _maybe_float(metrics.get("eval_loss"))
                    if eval_loss is not None:
                        val.append(
                            {
                                "step": step,
                                "eval_loss": eval_loss,
                                "eval_entropy": _maybe_float(metrics.get("eval_entropy")),
                                "eval_num_tokens": _maybe_float(metrics.get("eval_num_tokens")),
                                "eval_mean_token_accuracy": _maybe_float(metrics.get("eval_mean_token_accuracy")),
                                "epoch": _maybe_float(metrics.get("epoch")),
                            }
                        )
                    continue

            match = fallback_val_re.search(line)
            if match:
                val.append(
                    {
                        "step": int(match.group(1)),
                        "eval_loss": float(match.group(2)),
                        "eval_entropy": None,
                        "eval_num_tokens": None,
                        "eval_mean_token_accuracy": None,
                        "epoch": None,
                    }
                )

    return {"train": train, "val": val}


def find_log(job_id: str, log_dir: str) -> Path | None:
    root = Path(log_dir)
    if not root.exists():
        return None
    candidates = sorted(root.rglob(f"{job_id}*.out"))
    return candidates[0] if candidates else None


def extract_val_points(data: dict) -> list[dict]:
    return [{"step": row["step"], "val_loss": row["eval_loss"]} for row in data["val"]]


def plot_val_only(val_points: list[dict], job_id: str, out_path: Path) -> None:
    if not val_points:
        print("No validation loss lines found; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor("#1c1c1e" if dark else "white")

    steps = [row["step"] for row in val_points]
    losses = [row["val_loss"] for row in val_points]
    ax.plot(steps, losses, color=COLORS["val"], lw=2.2, marker="o", markersize=5, label="Eval loss")

    running_min = []
    current = float("inf")
    for value in losses:
        current = min(current, value)
        running_min.append(current)
    ax.plot(steps, running_min, color="#888", lw=1.2, ls="--", label="Best-so-far")

    best_idx = int(np.argmin(losses))
    ax.axvline(steps[best_idx], color=COLORS["val"], lw=1.0, ls="--", alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.6)
    style_ax(ax, "Validation loss", "Step", "Loss")

    fig.suptitle(
        f"TRL eval loss  ·  job {job_id}",
        fontsize=14,
        fontweight="bold",
        color="white" if dark else "#111",
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved -> {out_path}")
    print(f"Best checkpoint: step {steps[best_idx]:,} (eval_loss={losses[best_idx]:.4f})")


def plot_trl(data: dict, job_id: str, out_path: Path) -> None:
    train = data["train"]
    val = data["val"]

    if not train and not val:
        print("No recognisable metric lines found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=200, gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    fig.patch.set_facecolor("#1c1c1e" if dark else "white")

    # Panel 1: train loss + eval loss.
    ax = axes[0, 0]
    if train:
        steps = [row["step"] for row in train]
        losses = [row["loss"] for row in train]
        ax.plot(steps, losses, color=COLORS["train"], alpha=0.2, lw=0.8)
        ax.plot(steps, ema(losses), color=COLORS["train"], lw=2.2, label="Train loss (EMA)")
    if val:
        eval_steps = [row["step"] for row in val]
        eval_losses = [row["eval_loss"] for row in val]
        ax.plot(eval_steps, eval_losses, color=COLORS["val"], lw=2.2, marker="o", markersize=5, label="Eval loss")
        best_idx = int(np.argmin(eval_losses))
        ax.axvline(eval_steps[best_idx], color=COLORS["val"], lw=1.0, ls="--", alpha=0.6)
    if train or val:
        ax.legend(fontsize=9, framealpha=0.6)
    style_ax(ax, "Train/Eval loss", "Step", "Loss")

    # Panel 2: learning rate.
    ax = axes[0, 1]
    lr_points = [(row["step"], row["learning_rate"]) for row in train if row["learning_rate"] is not None]
    if lr_points:
        lr_steps, lrs = zip(*lr_points)
        ax.plot(lr_steps, lrs, color=COLORS["lr"], lw=2.0, label="Learning rate")
        ax.legend(fontsize=9, framealpha=0.6)
    style_ax(ax, "Learning rate", "Step", "LR")

    # Panel 3: entropy and mean token accuracy.
    ax = axes[1, 0]
    ent_points = [(row["step"], row["entropy"]) for row in train if row["entropy"] is not None]
    acc_points = [(row["step"], row["mean_token_accuracy"]) for row in train if row["mean_token_accuracy"] is not None]
    if ent_points:
        e_steps, ent = zip(*ent_points)
        ax.plot(e_steps, ent, color=COLORS["entropy"], lw=2.0, label="Entropy")
    if acc_points:
        ax2 = ax.twinx()
        a_steps, acc = zip(*acc_points)
        ax2.plot(a_steps, acc, color=COLORS["acc"], lw=2.0, ls="--", label="Mean token accuracy")
        ax2.set_ylabel("Mean token accuracy", fontsize=10, color="white" if dark else "#444")
        if dark:
            ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#555")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.6)
    elif ent_points:
        ax.legend(fontsize=9, framealpha=0.6)
    style_ax(ax, "Entropy & token accuracy", "Step", "Entropy")

    # Panel 4: cumulative tokens.
    ax = axes[1, 1]
    token_points = [(row["step"], row["num_tokens"]) for row in train if row["num_tokens"] is not None]
    if token_points:
        t_steps, tokens = zip(*token_points)
        ax.plot(t_steps, tokens, color=COLORS["tokens"], lw=2.2, label="Num tokens")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
        ax.legend(fontsize=9, framealpha=0.6)
    style_ax(ax, "Token progress", "Step", "Tokens")

    all_steps = [row["step"] for row in train] + [row["step"] for row in val]
    total_steps = max(all_steps) if all_steps else 0
    fig.suptitle(
        f"TRL training run  ·  job {job_id}  ·  {total_steps:,} steps",
        fontsize=14,
        fontweight="bold",
        color="white" if dark else "#111",
        y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved -> {out_path}")
    if val:
        eval_losses = [row["eval_loss"] for row in val]
        best_idx = int(np.argmin(eval_losses))
        best = val[best_idx]
        print(f"Best checkpoint: step {best['step']:,} (eval_loss={best['eval_loss']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TRL training curves from .out logs.")
    parser.add_argument("job_id", help="Job ID to match against <job_id>*.out")
    parser.add_argument("--log_path", default=None, help="Explicit .out path (skips search)")
    parser.add_argument("--log_dir", default="logs", help="Directory to search recursively (default: logs)")
    parser.add_argument("--out_dir", default="experiments", help="Output directory (default: experiments)")
    parser.add_argument("--val_only", action="store_true", help="Plot eval loss only")
    args = parser.parse_args()

    if args.log_path:
        log_path = Path(args.log_path)
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            sys.exit(1)
    else:
        log_path = find_log(args.job_id, args.log_dir)
        if log_path is None:
            print(f"No .out file found for job {args.job_id} under {args.log_dir}")
            sys.exit(1)
        print(f"Found log: {log_path}")

    print(f"Parsing {log_path} ...")
    data = parse_log(log_path)
    print(f"  Found {len(data['train'])} train points, {len(data['val'])} eval points")

    out_name = "val_loss.png" if args.val_only else "loss_curve.png"
    out_path = Path(args.out_dir) / args.job_id / out_name

    if args.val_only:
        val_points = extract_val_points(data)
        plot_val_only(val_points, args.job_id, out_path)
        if not val_points:
            sys.exit(1)
        return

    if not data["train"] and not data["val"]:
        print("No recognisable metric lines found. Check --log_dir / --log_path.")
        sys.exit(1)

    plot_trl(data, args.job_id, out_path)


if __name__ == "__main__":
    main()
