#!/usr/bin/env python
"""Compare SpliSER SSE against our junction-based SSU approximations.

For each splice site in the benchmark region, joins:
  - SpliSER SSE  = α / (α + β1_bam + β2_bam)  [both counts from BAM]
  - ssu_full     = α / (α + β1_bam + β2_junc)  [β2 from junction counts]
  - ssu_approx   = α / (α + β2_junc)            [no BAM at all]

The SpliSER .SpliSER.tsv file covers the whole chromosome; this script
filters it to the same BED intervals used in the benchmark.

Coordinate note
---------------
SpliSER Site column is 0-based (from pysam cigar parsing):
  donor    site = intron_start(1-based) - 1  → same integer as our position
  acceptor site = intron_end(1-based)        → our position - 1

So the join key is:
  donors:    spliser.Site == ours.position
  acceptors: spliser.Site == ours.position - 1

Usage:
    python src/scripts/compare_spliser_ssu.py \\
        --spliser  results/sanity_checks/ssu_benchmark/SRR17111301/SRR17111301.SpliSER.tsv \\
        --ssu      results/sanity_checks/ssu_benchmark/SRR17111301/ssu_comparison.parquet \\
        --bed      support/overfit.bed \\
        --output-dir results/sanity_checks/ssu_benchmark/SRR17111301
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spliser", required=True, help="SpliSER .SpliSER.tsv output file")
    p.add_argument("--ssu", required=True, help="ssu_comparison.parquet from benchmark script")
    p.add_argument("--bed", required=True, help="BED file defining benchmark intervals (0-based half-open)")
    p.add_argument("--output-dir", default=".", help="Directory to write outputs")
    return p.parse_args()


def load_bed(path: str) -> list[tuple[str, int, int]]:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    return [(r.chrom, int(r.start), int(r.end)) for _, r in df.iterrows()]


def load_spliser(path: str, intervals: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Load and filter SpliSER TSV to the benchmark intervals."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "Region": "chrom",
        "Site": "spliser_site",
        "Strand": "strand",
        "SSE": "sse",
        "alpha_count": "spliser_alpha",
        "beta1_count": "spliser_beta1",
        "beta2_count": "spliser_beta2",
    })

    # Filter to BED intervals (spliser_site is 0-based)
    masks = []
    for chrom, start_0, end_0 in intervals:
        masks.append(
            (df["chrom"] == chrom)
            & (df["spliser_site"] >= start_0)
            & (df["spliser_site"] < end_0)
        )
    if masks:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        df = df[combined].copy()

    return df[["chrom", "spliser_site", "strand", "sse",
               "spliser_alpha", "spliser_beta1", "spliser_beta2"]].reset_index(drop=True)


def join_spliser_to_ssu(spliser: pd.DataFrame, ssu: pd.DataFrame) -> pd.DataFrame:
    """Join SpliSER and SSU tables.

    Donors:    spliser_site == position  (same integer)
    Acceptors: spliser_site == position - 1
    """

    donors = ssu[ssu["role"] == "donor"].copy()
    donors["spliser_site"] = donors["position"]

    acceptors = ssu[ssu["role"] == "acceptor"].copy()
    acceptors["spliser_site"] = acceptors["position"] - 1

    ssu_keyed = pd.concat([donors, acceptors], ignore_index=True)

    merged = ssu_keyed.merge(
        spliser,
        on=["chrom", "spliser_site", "strand"],
        how="inner",
    )
    return merged


def scatter_pair(ax, x, y, color_vals, vmax, xlabel, ylabel, title):
    if len(x) == 0:
        ax.set_title(title)
        return
    sc = ax.scatter(x, y, c=color_vals, vmin=0, vmax=vmax,
                    cmap="viridis", s=12, alpha=0.6, linewidths=0)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, alpha=0.5)
    if len(x) >= 3:
        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
        ax.text(0.05, 0.95,
                f"Pearson r  = {r_p:.3f}\nSpearman r = {r_s:.3f}\nN = {len(x)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return sc


def plot_comparison(df: pd.DataFrame, out_path: Path,
                    timing: dict | None = None) -> None:
    """4×2 grid: rows = donor/acceptor, cols = SSE vs ssu_spliser / SSE vs ssu_full / SSE vs ssu_approx / ssu_spliser vs ssu_full."""
    roles = ["donor", "acceptor"]
    comparisons = [
        ("sse", "ssu_spliser", "SpliSER SSE", "ssu_spliser (BAM-only)"),
        ("sse", "ssu_full",    "SpliSER SSE", "ssu_full (BAM β1, junc β2)"),
        ("sse", "ssu_approx",  "SpliSER SSE", "ssu_approx (junc-only)"),
        ("ssu_spliser", "ssu_full", "ssu_spliser", "ssu_full"),
    ]

    has_spliser_col = "ssu_spliser" in df.columns and df["ssu_spliser"].notna().any()
    if not has_spliser_col:
        comparisons = comparisons[1:]   # drop ssu_spliser columns if missing

    ncols = len(comparisons)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9), sharex=False, sharey=False)
    if ncols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("SpliSER SSE vs SSU approximations", fontsize=12)

    dropna_cols = ["sse", "ssu_full", "ssu_approx"]
    if has_spliser_col:
        dropna_cols.append("ssu_spliser")

    vmax = float(np.log10(df["alpha"].max() + 1)) if not df.empty else 1.0
    sc_ref = None

    for row_i, role in enumerate(roles):
        sub = df[df["role"] == role].dropna(subset=["sse", "ssu_full", "ssu_approx"])
        color_vals = np.log10(sub["alpha"].values + 1) if not sub.empty else []

        for col_i, (xcol, ycol, xlabel, ylabel) in enumerate(comparisons):
            ax = axes[row_i][col_i]
            sub_pair = sub.dropna(subset=[xcol, ycol])
            if not sub_pair.empty:
                cv = np.log10(sub_pair["alpha"].values + 1)
                sc = scatter_pair(
                    ax,
                    sub_pair[xcol].values, sub_pair[ycol].values,
                    cv, vmax,
                    xlabel, ylabel,
                    f"{role}  (N={len(sub_pair)})",
                )
                if sc is not None:
                    sc_ref = sc
            else:
                ax.set_title(f"{role}  (N=0)")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

    if sc_ref is not None:
        fig.colorbar(sc_ref, ax=axes, label="log10(α + 1)", shrink=0.5, pad=0.02)

    if timing:
        short = {
            "ssu_full/approx — alpha+beta2 (junctions)": "α+β2 junctions",
            "ssu_full — beta1 (BAM scan)":               "β1 BAM scan",
            "ssu_spliser — alpha+beta1+beta2 (BAM scan)": "ssu_spliser BAM",
        }
        lines = ["Compute time / peak mem:"]
        for key, entry in timing.items():
            label = short.get(key, key)
            lines.append(f"  {label}: {entry['seconds']:.1f}s / {entry['peak_mb']:.1f} MB")
        fig.text(
            0.01, 0.01, "\n".join(lines),
            fontsize=7, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85),
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = load_bed(args.bed)
    print(f"Benchmark intervals: {intervals}")

    print(f"Loading SpliSER output from {args.spliser!r} …")
    spliser = load_spliser(args.spliser, intervals)
    print(f"  {len(spliser)} SpliSER sites in region")

    print(f"Loading SSU benchmark from {args.ssu!r} …")
    ssu = pd.read_parquet(args.ssu)
    print(f"  {len(ssu)} SSU sites")

    print("Joining …")
    merged = join_spliser_to_ssu(spliser, ssu)
    print(f"  {len(merged)} sites matched")

    if merged.empty:
        print("No matching sites — check coordinate conventions.")
        return

    unmatched_spliser = len(spliser) - len(merged)
    unmatched_ssu = len(ssu) - len(merged)
    print(f"  unmatched SpliSER sites: {unmatched_spliser}")
    print(f"  unmatched SSU sites:     {unmatched_ssu}")

    parquet_path = out_dir / "spliser_comparison.parquet"
    merged.to_parquet(parquet_path, index=False)
    print(f"  wrote {parquet_path}")

    timing_path = Path(args.ssu).parent / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None

    plot_comparison(merged, out_dir / "spliser_vs_ssu_scatterplot.pdf", timing=timing)


if __name__ == "__main__":
    main()
