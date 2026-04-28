#!/usr/bin/env python
"""Plot training dynamics from run log files only (no summary_stats.parquet).

Reads epoch_log.csv and training_log.csv from each run directory and produces
a multi-page PDF with:
  Page 1 — Training loss curves (total + per-modality) over epochs
  Page 2 — Validation loss curves (total + per-modality) over epochs
  Page 3 — Validation correlation metrics over epochs
  Page 4 — Per-step loss curves (smoothed) from training_log.csv

Usage:
    python src/scripts/plot_run_summary.py \
        --run-dirs results/.../run1 results/.../run2 \
        --output results/.../run_summary.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F", "#BCBD22"]

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_epoch_log(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "epoch_log.csv")
    df["run_name"] = run_dir.name
    return df


def load_training_log(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "training_log.csv")
    df["run_name"] = run_dir.name
    return df


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #

TRAIN_LOSS_COLS = {
    "train_loss": "total",
    "rna_seq_loss": "rna_seq",
    "splice_site_loss": "splice_site",
    "splice_usage_loss": "splice_usage",
    "splice_junctions_loss": "splice_junctions",
}

VAL_LOSS_COLS = {
    "val_loss": "total",
    "val_loss_rna_seq_loss": "rna_seq",
    "val_loss_splice_site_loss": "splice_site",
    "val_loss_splice_usage_loss": "splice_usage",
    "val_loss_splice_junctions_loss": "splice_junctions",
}

CORR_COLS = {
    "rna_seq_1bp_profile_pearson_r_mean": "rna_seq Pearson r",
    "splice_usage_pearson_r": "splice_usage Pearson r",
    "splice_junctions_pearson_r": "splice_junctions Pearson r",
}

STEP_LOSS_COLS = {
    "loss": "total",
    "rna_seq_loss": "rna_seq",
    "splice_site_loss": "splice_site",
    "splice_usage_loss": "splice_usage",
    "splice_junctions_loss": "splice_junctions",
}

# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def _has_negatives(dfs: list[pd.DataFrame], col: str) -> bool:
    return any((df[col].dropna() < 0).any() for df in dfs if col in df.columns)


def _epoch_loss_page(
    epoch_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
    loss_cols: dict[str, str],
    row_label: str,
    title: str,
) -> plt.Figure:
    present = [col for col in loss_cols if any(col in df.columns for df in epoch_dfs)]
    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, y=1.01)

    use_log = True
    for ax, col in zip(axes, present):
        label = loss_cols[col]
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=run, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("epoch")
        if not _has_negatives(epoch_dfs, col):
            ax.set_yscale("log")
        else:
            use_log = False
        ax.grid(True, alpha=0.3)

    y_label = f"{row_label} loss" + (" (log scale)" if use_log else "")
    axes[0].set_ylabel(y_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="run")
    fig.tight_layout()
    return fig


def page_train_loss(epoch_dfs, runs, colors):
    return _epoch_loss_page(
        epoch_dfs, runs, colors, TRAIN_LOSS_COLS, "train",
        "Training loss curves per epoch",
    )


def page_val_loss(epoch_dfs, runs, colors):
    return _epoch_loss_page(
        epoch_dfs, runs, colors, VAL_LOSS_COLS, "val",
        "Validation loss curves per epoch",
    )


def page_val_correlations(epoch_dfs: list[pd.DataFrame], runs: list[str], colors: list) -> plt.Figure:
    present = [col for col in CORR_COLS if any(col in df.columns for df in epoch_dfs)]
    n_cols = len(present) or 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Validation correlation metrics over epochs", fontsize=13, y=1.01)

    for ax, col in zip(axes, present):
        label = CORR_COLS[col]
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=run, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("epoch")
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Pearson r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="run")
    fig.tight_layout()
    return fig


def page_step_loss(
    train_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
    smooth_window: int = 20,
) -> plt.Figure:
    present = [col for col in STEP_LOSS_COLS if any(col in df.columns for df in train_dfs)]
    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Per-step training loss (smoothed, window={smooth_window})", fontsize=13, y=1.01)

    use_log = True
    for ax, col in zip(axes, present):
        label = STEP_LOSS_COLS[col]
        for df, run, color in zip(train_dfs, runs, colors):
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            s = smooth(vals, smooth_window)
            x = np.arange(len(s))
            ax.plot(x, s, label=run, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("step (smoothed)")
        if not _has_negatives(train_dfs, col):
            ax.set_yscale("log")
        else:
            use_log = False
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("loss" + (" (log scale)" if use_log else ""))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="run")
    fig.tight_layout()
    return fig

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True, help="One directory per run")
    p.add_argument("--output", required=True, help="Output PDF path")
    p.add_argument("--smooth", type=int, default=20, help="Smoothing window for step-loss plot (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    runs = [d.name for d in run_dirs]
    colors = PALETTE[: len(runs)]

    epoch_dfs = [load_epoch_log(d) for d in run_dirs]
    train_dfs = [load_training_log(d) for d in run_dirs]

    from matplotlib.backends.backend_pdf import PdfPages
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for fig in [
            page_train_loss(epoch_dfs, runs, colors),
            page_val_loss(epoch_dfs, runs, colors),
            page_val_correlations(epoch_dfs, runs, colors),
            page_step_loss(train_dfs, runs, colors, args.smooth),
        ]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved to", out_path)


if __name__ == "__main__":
    main()
