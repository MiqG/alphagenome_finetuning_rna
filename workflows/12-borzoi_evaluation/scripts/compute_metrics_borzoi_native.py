#!/usr/bin/env python
"""
Summarise the native Borzoi 32 bp tables into the upstream (Abuen) metrics.

Reads, from --predictions-dir, what collect_predictions_borzoi.py wrote:
  borzoi_native_32bp_per_region.parquet
  borzoi_native_32bp_per_track.parquet
  borzoi_native_32bp_gene_expression.parquet
  rna_seq_track_totals_per_interval.parquet   (region totals; resolution independent)

and writes metrics_native.json with, at Borzoi's native 32 bp resolution
(bin sums, log1p for RNA profiles, exon-base-weighted gene means):

  profile Pearson r (per region, track-pooled): mean / median
  accumulated per-track Pearson r: mean
  count Pearson r (raw and log1p), pooled over (region, track) pairs
  JS divergence mean, JS distance mean, MSE
  gene expression: raw across genes, normalized across genes (per track)
  normalized across tracks (per gene) is reported only with >= 3 tracks: with
  two tracks the centred values are (x, -x) and the correlation is always +-1.

Definitions are the upstream ones (src evaluate_borzoi_checkpoint.py /
compute_gene_metrics.py); only their summary step is repeated here. No GPU or
TensorFlow needed.

Usage:
    python workflows/12-borzoi_evaluation/scripts/compute_metrics_borzoi_native.py \\
        --predictions-dir results/evaluation/borzoi/full/test/predictions \\
        --output results/evaluation/borzoi/full/test/metrics_native.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))
from compute_gene_metrics import (  # noqa: E402
    _clean_json,
    _pearson_or_nan,
    summarize_gene_expression,
)

POOL = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = args.predictions_dir
    per_region = pd.read_parquet(os.path.join(d, "borzoi_native_32bp_per_region.parquet"))
    per_track = pd.read_parquet(os.path.join(d, "borzoi_native_32bp_per_track.parquet"))
    gene_rows = pd.read_parquet(os.path.join(d, "borzoi_native_32bp_gene_expression.parquet"))
    totals = pd.read_parquet(os.path.join(d, "rna_seq_track_totals_per_interval.parquet"))

    profile = per_region["profile_pearson_r_log1p"].to_numpy(dtype=float)
    js_div = per_region["js_divergence"].to_numpy(dtype=float)
    js_dist = per_region["js_distance"].to_numpy(dtype=float)
    pred_total = totals["pred_sum_central"].to_numpy(dtype=float)
    obs_total = totals["obs_sum_central"].to_numpy(dtype=float)

    metrics = {
        "profile_transform": "log1p",
        "profile_pearson_r_mean": float(np.nanmean(profile)),
        "profile_pearson_r_median": float(np.nanmedian(profile)),
        "track_pearson_r_accumulated_mean": float(np.nanmean(per_track["pearson_r_accumulated_log1p"])),
        "track_pearson_r_accumulated_all": per_track["pearson_r_accumulated_log1p"].to_numpy(dtype=float),
        "count_pearson_r_raw": _pearson_or_nan(pred_total, obs_total),
        "count_pearson_r_log1p": _pearson_or_nan(np.log1p(pred_total), np.log1p(obs_total)),
        "js_divergence_mean": float(np.nanmean(js_div)),
        "js_distance_mean": float(np.nanmean(js_dist)),
        "mse": float(per_region["mse"].mean()),
        "n_regions": int(len(per_region)),
        "n_tracks": int(len(per_track)),
    }

    gene_metrics, per_track_genes, per_gene = summarize_gene_expression(gene_rows, (POOL,))
    gene_summary = gene_metrics[POOL]
    n_tracks_gene = gene_summary["n_tracks"]
    if n_tracks_gene < 3:
        # Degenerate with fewer than 3 tracks (see module docstring).
        gene_summary.pop("normalized_pearson_r_mean_across_tracks_by_gene", None)
        gene_summary["note"] = (
            "normalized across-tracks-by-gene omitted: {} tracks give +-1 correlations".format(n_tracks_gene)
        )

    summary = {
        "evaluation": {
            "native_resolution_bp": POOL,
            "view": "32 bp bin sums (Borzoi native), ensemble mean over trunks",
            "gene_selection": "at_least_50pct_unique_exon_bp_in_scored_window",
            "gene_resolution_conversion": "exact_exon_bp_weighting_of_32bp_bins",
            "predictions_dir": d,
        },
        "metrics": {POOL: metrics},
        "gene_expression_metrics": {POOL: gene_summary},
    }
    tmp = args.output + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(_clean_json(summary), handle, indent=2)
    os.replace(tmp, args.output)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
