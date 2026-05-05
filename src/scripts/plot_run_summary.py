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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figutils import cm, set_figure_style

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F", "#BCBD22"]

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_epoch_log(run_dir: Path) -> pd.DataFrame:
    import csv as _csv
    path = run_dir / "epoch_log.csv"
    with open(path, newline="") as f:
        rows = list(_csv.reader(f))
    if not rows:
        return pd.DataFrame()
    ncols = max(len(r) for r in rows)
    header = rows[0] + [f"_extra{i}" for i in range(len(rows[0]), ncols)]
    data = [r + [""] * (ncols - len(r)) for r in rows[1:]]
    df = pd.DataFrame(data, columns=header)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
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

TRAIN_CORR_COLS = {
    "train_rna_seq_1bp_profile_pearson_r_mean": "rna_seq Pearson r",
    "train_splice_usage_pearson_r": "splice_usage Pearson r",
    "train_splice_junctions_pearson_r": "splice_junctions Pearson r",
}

VAL_CORR_COLS = {
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

# modality → (train_col, val_col)
MODALITY_LOSS_COLS = {
    "total":            ("train_loss",             "val_loss"),
    "rna_seq":          ("rna_seq_loss",            "val_loss_rna_seq_loss"),
    "splice_site":      ("splice_site_loss",        "val_loss_splice_site_loss"),
    "splice_usage":     ("splice_usage_loss",       "val_loss_splice_usage_loss"),
    "splice_junctions": ("splice_junctions_loss",   "val_loss_splice_junctions_loss"),
}

MODALITY_CORR_COLS = {
    "rna_seq":          ("train_rna_seq_1bp_profile_pearson_r_mean", "rna_seq_1bp_profile_pearson_r_mean"),
    "splice_usage":     ("train_splice_usage_pearson_r",             "splice_usage_pearson_r"),
    "splice_junctions": ("train_splice_junctions_pearson_r",         "splice_junctions_pearson_r"),
}

# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def _linestyle(run: str) -> str:
    return "--" if "pretrinit" in run else "-"


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
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols * cm, 4 * cm), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(title, y=1.01)

    use_log = True
    for ax, col in zip(axes, present):
        label = loss_cols[col]
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=run, color=color, linestyle=_linestyle(run), marker="o", markersize=2, linewidth=0.6)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_box_aspect(1)
        if not _has_negatives(epoch_dfs, col):
            ax.set_yscale("log")
        else:
            use_log = False
        ax.grid(True, alpha=0.3)

    y_label = f"{row_label} loss" + (" (log scale)" if use_log else "")
    axes[0].set_ylabel(y_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="run")
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


def _corr_page(
    epoch_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
    corr_cols: dict[str, str],
    title: str,
) -> plt.Figure:
    present = [col for col in corr_cols if any(col in df.columns for df in epoch_dfs)]
    n_cols = len(present) or 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols * cm, 4 * cm), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(title, y=1.01)

    for ax, col in zip(axes, present):
        label = corr_cols[col]
        for df, run, color in zip(epoch_dfs, runs, colors):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=run, color=color, linestyle=_linestyle(run), marker="o", markersize=2, linewidth=0.6)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_box_aspect(1)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Pearson r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="run")
    fig.tight_layout()
    return fig


def page_train_correlations(epoch_dfs, runs, colors):
    return _corr_page(epoch_dfs, runs, colors, TRAIN_CORR_COLS, "Training correlation metrics over epochs")


def page_val_correlations(epoch_dfs, runs, colors):
    return _corr_page(epoch_dfs, runs, colors, VAL_CORR_COLS, "Validation correlation metrics over epochs")


def page_step_loss(
    train_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
    smooth_window: int = 20,
) -> plt.Figure:
    present = [col for col in STEP_LOSS_COLS if any(col in df.columns for df in train_dfs)]
    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols * cm, 4 * cm), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Per-step training loss (smoothed, window={smooth_window})", y=1.01)

    use_log = True
    for ax, col in zip(axes, present):
        label = STEP_LOSS_COLS[col]
        for df, run, color in zip(train_dfs, runs, colors):
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            s = smooth(vals, smooth_window)
            x = np.arange(len(s))
            ax.plot(x, s, label=run, color=color, linestyle=_linestyle(run), marker="o", markersize=2, linewidth=0.6)
        ax.set_title(label)
        ax.set_xlabel("step (smoothed)")
        ax.set_box_aspect(1)
        if not _has_negatives(train_dfs, col):
            ax.set_yscale("log")
        else:
            use_log = False
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("loss" + (" (log scale)" if use_log else ""))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="run")
    fig.tight_layout()
    return fig

def _bar_summary_page(
    epoch_dfs: list[pd.DataFrame],
    runs: list[str],
    colors: list,
    modality_cols: dict[str, tuple[str, str]],
    xlabel: str,
    title: str,
    epoch: int,
    sort_ascending: bool = True,
) -> plt.Figure:
    present = [
        (mod, tc, vc) for mod, (tc, vc) in modality_cols.items()
        if any(tc in df.columns or vc in df.columns for df in epoch_dfs)
    ]
    n_cols = len(present)
    # 2 rows: train (top) and val (bottom); no shared y so each subplot sorts independently
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols * cm, 8 * cm))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"{title} (epoch {epoch})", y=1.01)

    run_color = dict(zip(runs, colors))

    for col_idx, (mod, train_col, val_col) in enumerate(present):
        for row_idx, (split_col, split_label) in enumerate([(train_col, "train"), (val_col, "val")]):
            ax = axes[row_idx, col_idx]

            # collect values for all runs
            vals = []
            for df in epoch_dfs:
                row = df[df["epoch"] == epoch]
                if row.empty:
                    row = df.iloc[[-1]]
                vals.append(row[split_col].values[0] if split_col in row.columns else np.nan)

            # sort worst → best (best at bottom)
            order = np.argsort(vals)[::-1] if sort_ascending else np.argsort(vals)
            sorted_runs = [runs[i] for i in order]
            sorted_vals = [vals[i] for i in order]
            sorted_colors = [run_color[r] for r in sorted_runs]

            y = np.arange(len(sorted_runs))
            for i, (val, color) in enumerate(zip(sorted_vals, sorted_colors)):
                ax.barh(i, val, color=color, height=0.6)

            # x-axis: omit 0, start just below minimum value
            valid = [v for v in sorted_vals if not np.isnan(v)]
            if valid:
                xmin, xmax = min(valid), max(valid)
                pad = (xmax - xmin) * 0.05 if xmax != xmin else abs(xmin) * 0.05 or 0.01
                ax.set_xlim(left=xmin - pad)

            ax.set_title(f"{mod}\n{split_label}" if row_idx == 0 else split_label)
            ax.set_yticks([])
            ax.set_xlabel(xlabel)
            ax.set_box_aspect(1)
            ax.grid(True, alpha=0.3, axis="x")

    # legend: run → color
    handles = [plt.Rectangle((0, 0), 1, 1, color=run_color[r]) for r in runs]
    fig.legend(handles, runs, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="run")
    fig.tight_layout()
    return fig


def page_loss_bar(epoch_dfs, runs, colors, epoch=5):
    return _bar_summary_page(
        epoch_dfs, runs, colors, MODALITY_LOSS_COLS,
        "loss", "Loss summary", epoch,
    )


def page_corr_bar(epoch_dfs, runs, colors, epoch=5):
    return _bar_summary_page(
        epoch_dfs, runs, colors, MODALITY_CORR_COLS,
        "Pearson r", "Pearson r summary", epoch, sort_ascending=False,
    )



# --------------------------------------------------------------------------- #
# Diagnostics page
# --------------------------------------------------------------------------- #

_DIAG_COLS = [
    ("splice_junctions_pearson_r",                "full_r"),
    ("splice_junctions_pearson_r_nonzero",         "nonzero_r"),
    ("splice_junctions_pearson_r_donor_marginal",  "donor_marg_r"),
    ("splice_junctions_pearson_r_acceptor_marginal", "accept_marg_r"),
    ("val_loss_splice_junctions_target_nonzero_frac", "target_nz_frac"),
]


def page_eval_diagnostics(epoch_dfs: list[pd.DataFrame], runs: list[str]) -> plt.Figure | None:
    """Table of splice junction diagnostics from the last epoch of each run."""
    present_cols = [
        (col, label) for col, label in _DIAG_COLS
        if any(col in df.columns for df in epoch_dfs)
    ]
    if not present_cols:
        return None

    header = ["run_name"] + [label for _, label in present_cols]
    rows = [header]
    for df, run_name in zip(epoch_dfs, runs):
        last = df.iloc[-1]
        row = [run_name[:20]]
        for col, _ in present_cols:
            val = last[col] if col in last.index else float("nan")
            row.append(f"{val:.4f}" if not np.isnan(float(val)) else "—")
        rows.append(row)

    n_cols = len(header)
    col_widths = [0.25] + [0.15] * (n_cols - 1)
    fig, ax = plt.subplots(figsize=(20 * cm, 12 * cm))
    ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="center", loc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i in range(n_cols):
        table[(0, i)].set_facecolor("#CCCCCC")
        table[(0, i)].set_text_props(weight="bold")
    for i in range(1, len(rows)):
        color = "#F0F0F0" if i % 2 else "#FFFFFF"
        for j in range(n_cols):
            table[(i, j)].set_facecolor(color)

    fig.suptitle("Splice Junction Pearson Diagnostics (last epoch)", fontsize=14, weight="bold", y=0.98)
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True, help="One directory per run")
    p.add_argument("--output", required=True, help="Output PDF path")
    p.add_argument("--smooth", type=int, default=20, help="Smoothing window for step-loss plot (default: 20)")
    p.add_argument("--summary-epoch", type=int, default=5, help="Epoch to use for bar summary plots (default: 5)")
    return p.parse_args()


def main() -> None:
    set_figure_style()
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
        pages = [
            page_train_loss(epoch_dfs, runs, colors),
            page_val_loss(epoch_dfs, runs, colors),
            page_train_correlations(epoch_dfs, runs, colors),
            page_val_correlations(epoch_dfs, runs, colors),
            page_step_loss(train_dfs, runs, colors, args.smooth),
            page_loss_bar(epoch_dfs, runs, colors, args.summary_epoch),
            page_corr_bar(epoch_dfs, runs, colors, args.summary_epoch),
            page_eval_diagnostics(epoch_dfs, runs),
        ]
        for fig in pages:
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print("Saved to", out_path)


if __name__ == "__main__":
    main()
