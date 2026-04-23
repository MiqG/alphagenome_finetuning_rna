#!/usr/bin/env python
"""Plot overfitting training dynamics and final prediction correlations.

Reads epoch_log.csv, training_log.csv, and summary_stats.parquet from
each run directory and produces a multi-page PDF with:
  Page 1 — Training loss curves (total + per-modality) over epochs
  Page 2 — Validation correlation metrics over epochs
  Page 3 — Final prediction correlations per modality (stripplot)
  Page 4 — Junction recall: n_pred vs n_real, coloured by junction_correlation

Usage:
    python src/scripts/plot_overfit_summary.py \
        --run-dirs results/.../run1 results/.../run2 \
        --output results/.../overfit_summary.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #

RUN_ORDER = ["all", "rna_seq_only", "splice_site_only", "splice_usage_only", "splice_junctions_only"]
RUN_LABELS = {
    "all": "all",
    "rna_seq_only": "rna_seq",
    "splice_site_only": "splice_site",
    "splice_usage_only": "splice_usage",
    "splice_junctions_only": "splice_junctions",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_epoch_log(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "epoch_log.csv"
    df = pd.read_csv(path)
    df["run_name"] = run_dir.name
    return df


def load_training_log(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "training_log.csv"
    df = pd.read_csv(path)
    df["run_name"] = run_dir.name
    return df


def aggregate_training_log_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    modality_cols = [c for c in ["rna_seq_loss", "splice_site_loss", "splice_usage_loss", "splice_junctions_loss"] if c in df.columns]
    agg = df.groupby(["epoch", "run_name"])[modality_cols].mean().reset_index()
    return agg


def load_summary_stats(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "visualization" / "summary_stats.parquet"
    df = pd.read_parquet(path)
    df["run_name"] = run_dir.name
    return df


# --------------------------------------------------------------------------- #
# Page 1 — training loss curves (epoch-level)
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


def page_loss_curves(
    epoch_dfs: list[pd.DataFrame],
    train_epoch_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
) -> plt.Figure:
    n_cols = max(len(TRAIN_LOSS_COLS), len(VAL_LOSS_COLS))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), sharey=False)
    fig.suptitle("Training & validation loss curves", fontsize=13, y=1.01)

    for col_idx, (col, label) in enumerate(TRAIN_LOSS_COLS.items()):
        ax = axes[0, col_idx]
        src_dfs = train_epoch_dfs if col != "train_loss" else epoch_dfs
        for df, run, color in zip(src_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=RUN_LABELS.get(run, run), color=color)
        ax.set_title(f"train {label}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for col_idx, (col, label) in enumerate(VAL_LOSS_COLS.items()):
        ax = axes[1, col_idx]
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=RUN_LABELS.get(run, run), color=color)
        ax.set_title(f"val {label}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("loss (log scale)")
    axes[1, 0].set_ylabel("loss (log scale)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="run")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Page 2 — validation correlation metrics over epochs
# --------------------------------------------------------------------------- #

CORR_COLS = {
    "rna_seq_1bp_profile_pearson_r_mean": "rna_seq Pearson r",
    "splice_usage_pearson_r": "splice_usage Pearson r",
    "splice_junctions_pearson_r": "splice_junctions Pearson r",
}


def page_val_correlations(epoch_dfs: list[pd.DataFrame], runs: list[str], colors: list) -> plt.Figure:
    n_cols = len(CORR_COLS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    fig.suptitle("Validation correlation metrics over epochs", fontsize=13, y=1.01)

    for ax, (col, label) in zip(axes, CORR_COLS.items()):
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=RUN_LABELS.get(run, run), color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("epoch")
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Pearson r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="run")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Page 3 — final prediction correlations (summary_stats)
# --------------------------------------------------------------------------- #

SUMMARY_CORR_COLS = {
    "rna_seq_correlation": "rna_seq",
    "donor_correlation": "donor",
    "acceptor_correlation": "acceptor",
    "usage_correlation": "usage",
    "junction_correlation": "junction",
}


def page_final_correlations(summary_df: pd.DataFrame, runs: list[str], colors: list) -> plt.Figure:
    n_cols = len(SUMMARY_CORR_COLS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 5), sharey=True)
    fig.suptitle("Final prediction correlations (per gene × sample)", fontsize=13, y=1.01)

    color_map = {run: c for run, c in zip(runs, colors)}
    run_order = [r for r in RUN_ORDER if r in runs]
    x_positions = {run: i for i, run in enumerate(run_order)}

    for ax, (col, label) in zip(axes, SUMMARY_CORR_COLS.items()):
        for run in run_order:
            sub = summary_df.loc[summary_df["run_name"] == run, col].dropna()
            x = x_positions[run]
            ax.scatter(
                np.full(len(sub), x) + np.random.uniform(-0.15, 0.15, len(sub)),
                sub.values,
                color=color_map[run],
                alpha=0.6,
                s=20,
                zorder=3,
            )
            ax.plot([x - 0.3, x + 0.3], [sub.median(), sub.median()],
                    color=color_map[run], linewidth=2, zorder=4)

        ax.set_title(label, fontsize=9)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(
            [RUN_LABELS.get(r, r) for r in run_order],
            rotation=45, ha="right", fontsize=7,
        )
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("correlation")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Page 4 — junction recall scatter
# --------------------------------------------------------------------------- #

def page_junction_recall(summary_df: pd.DataFrame, runs: list[str], colors: list) -> plt.Figure:
    run_order = [r for r in RUN_ORDER if r in runs]
    n_cols = len(run_order)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4), sharey=True, sharex=True)
    fig.suptitle("Junction recall: predicted vs real count", fontsize=13, y=1.01)

    if n_cols == 1:
        axes = [axes]

    for ax, run, color in zip(axes, run_order, colors):
        sub = summary_df.loc[summary_df["run_name"] == run].dropna(subset=["junction_correlation"])
        sc = ax.scatter(
            sub["n_real_junctions"],
            sub["n_pred_junctions"],
            c=sub["junction_correlation"],
            cmap="RdYlGn",
            vmin=-1, vmax=1,
            alpha=0.8,
            s=30,
            edgecolors="grey",
            linewidths=0.3,
        )
        max_val = max(sub[["n_real_junctions", "n_pred_junctions"]].max().max(), 1)
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)
        ax.set_title(RUN_LABELS.get(run, run), fontsize=9)
        ax.set_xlabel("real junctions")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("predicted junctions")
    fig.colorbar(sc, ax=axes[-1], label="junction_correlation", shrink=0.8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True, help="One directory per run")
    p.add_argument("--output", required=True, help="Output PDF path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    runs = [d.name for d in run_dirs]
    run_order = [r for r in RUN_ORDER if r in runs] + [r for r in runs if r not in RUN_ORDER]
    run_dirs = sorted(run_dirs, key=lambda d: run_order.index(d.name) if d.name in run_order else 999)
    runs = [d.name for d in run_dirs]
    colors = PALETTE[: len(runs)]

    epoch_dfs = [load_epoch_log(d) for d in run_dirs]
    train_epoch_dfs = [aggregate_training_log_per_epoch(load_training_log(d)) for d in run_dirs]
    summary_df = pd.concat([load_summary_stats(d) for d in run_dirs], ignore_index=True)

    from matplotlib.backends.backend_pdf import PdfPages
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for fig in [
            page_loss_curves(epoch_dfs, train_epoch_dfs, runs, colors),
            page_val_correlations(epoch_dfs, runs, colors),
            page_final_correlations(summary_df, runs, colors),
            page_junction_recall(summary_df, runs, colors),
        ]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved to", out_path)


if __name__ == "__main__":
    main()
