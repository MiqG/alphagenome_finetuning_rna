#!/usr/bin/env python
"""Plot training loss curves for overfitting single runs.

One panel per density (high / medium / low), lines coloured by modality.
Designed for original__all runs only.

Usage:
    python src/scripts/plot_overfitting_single_curves.py \
        --run-dirs results/.../high/original__all \
                   results/.../medium/original__all \
                   results/.../low/original__all \
        --output results/.../loss_curves.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figutils import set_figure_style

cm = 1 / 2.54

MODALITY_COLS = {
    "train_loss":             ("total",             "#333333"),
    "rna_seq_loss":           ("rna_seq",           "#4C72B0"),
    "splice_site_loss":       ("splice_site",       "#DD8452"),
    "splice_usage_loss":      ("splice_usage",      "#55A868"),
    "splice_junctions_loss":  ("splice_junctions",  "#C44E52"),
}

# (label, color) — all from the same modality palette
CORR_COLS = {
    "rna_seq_1bp_profile_pearson_r_mean": ("rna_seq profile",   "#4C72B0"),
    "splice_usage_pearson_r":             ("splice_usage",       "#55A868"),
    "splice_junctions_pearson_r":         ("splice_junctions",   "#C44E52"),
}

# (label, color, linestyle)
AUPRC_COLS = {
    "splice_site_auprc_donor_pos":    ("donor pos",    "#DD8452", "-"),
    "splice_site_auprc_acceptor_pos": ("acceptor pos", "#C44E52", "-"),
    "splice_site_auprc_donor_neg":    ("donor neg",    "#DD8452", "--"),
    "splice_site_auprc_acceptor_neg": ("acceptor neg", "#C44E52", "--"),
    "splice_site_auprc_macro":        ("macro",        "#333333", "-"),
}


def load_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_log = pd.read_csv(run_dir / "epoch_log.csv")
    training_log = pd.read_csv(run_dir / "training_log.csv")

    epoch_log["sample"] = epoch_log.groupby("epoch").cumcount() + 1
    training_log["sample"] = training_log.groupby("epoch").cumcount() + 1

    modality_cols = [
        c for c in ["rna_seq_loss", "splice_site_loss", "splice_usage_loss", "splice_junctions_loss"]
        if c in training_log.columns
    ]
    total = epoch_log[["epoch", "sample", "train_loss"]]
    per_step = training_log[["epoch", "sample"] + modality_cols]
    loss_df = total.merge(per_step, on=["epoch", "sample"], how="left")

    return loss_df, epoch_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="One directory per density (e.g. high/original__all)")
    p.add_argument("--output", required=True, help="Output path (.pdf or .svg)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]

    set_figure_style()

    n = len(run_dirs)
    nrows = 3
    fig, axes = plt.subplots(
        nrows, n,
        figsize=(n * 4 * cm, nrows * 4 * cm),
        sharey="row", sharex=True,
    )
    if n == 1:
        axes = axes.reshape(nrows, 1)

    SAMPLE_LS = {1: "-", 2: "--"}
    loss_handles, loss_labels = [], []
    corr_handles, corr_labels = [], []
    auprc_handles, auprc_labels = [], []

    for ax_idx, run_dir in enumerate(run_dirs):
        loss_df, epoch_log = load_data(run_dir)
        density = run_dir.parent.name

        # --- row 0: loss curves ---
        ax = axes[0, ax_idx]
        for sample, ls in SAMPLE_LS.items():
            sub = loss_df[loss_df["sample"] == sample]
            for col, (label, color) in MODALITY_COLS.items():
                if col not in sub.columns:
                    continue
                line, = ax.plot(sub["epoch"], sub[col], color=color, linewidth=0.8, linestyle=ls)
                if ax_idx == 0 and sample == 1:
                    loss_handles.append(line)
                    loss_labels.append(label)
        ax.set_yscale("log")
        ax.set_title(density)
        ax.grid(True, alpha=0.3, linewidth=0.3)

        # --- row 1: Pearson r correlations ---
        ax = axes[1, ax_idx]
        for col, (label, color) in CORR_COLS.items():
            if col not in epoch_log.columns:
                continue
            line, = ax.plot(epoch_log["epoch"], epoch_log[col], color=color, linewidth=0.8)
            if ax_idx == 0:
                corr_handles.append(line)
                corr_labels.append(label)
        ax.grid(True, alpha=0.3, linewidth=0.3)

        # --- row 2: AUPRC splice sites ---
        ax = axes[2, ax_idx]
        for col, (label, color, ls) in AUPRC_COLS.items():
            if col not in epoch_log.columns:
                continue
            line, = ax.plot(epoch_log["epoch"], epoch_log[col], color=color, linewidth=0.8, linestyle=ls)
            if ax_idx == 0:
                auprc_handles.append(line)
                auprc_labels.append(label)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3, linewidth=0.3)

    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("Pearson r")
    axes[2, 0].set_ylabel("AUPRC")

    legend_kw = dict(fontsize=5, frameon=False, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    axes[0, -1].legend(loss_handles, loss_labels, **legend_kw)
    axes[1, -1].legend(corr_handles, corr_labels, **legend_kw)
    axes[2, -1].legend(auprc_handles, auprc_labels, **legend_kw)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved to", out)


if __name__ == "__main__":
    main()
