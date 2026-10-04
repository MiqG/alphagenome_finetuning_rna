#!/usr/bin/env python
"""
Compute evaluation metrics from collected Borzoi ensemble predictions.

Step 2 of 2 (the Borzoi counterpart of compute_eval_metrics.py). Reads the
output of collect_predictions_borzoi.py, reads observed coverage from the
bigwigs, and computes, at 1 bp (native bin repeated) and 32 bp resolution:

  - profile Pearson r (log1p for RNA), per region, track-pooled
  - accumulated per-track Pearson r
  - count Pearson r (raw and log1p), Jensen-Shannon divergence, MSE
  - AlphaGenome-paper gene expression metrics (RNA only)

The metric functions are imported from the vendored, unmodified upstream
script (upstream/evaluate_borzoi_checkpoint.py); the code below only
reassembles the tail of that script's main() so that, for identical
predictions, the outputs are identical to a one-step upstream evaluation.
It needs no GPU.

Outputs (in --output-dir):
  summary.json                       — all metrics (completion marker, written last)
  metrics_per_region.csv             — per-region profile Pearson / JS
  gene_expression_per_gene.csv       — per gene, per sample exon-mean values
  gene_expression_metrics_per_track.csv, gene_expression_metrics_per_gene.csv

Usage:
    python workflows/12-borzoi_evaluation/scripts/compute_metrics_borzoi.py \\
        --predictions-dir results/evaluation/borzoi/full/test/predictions \\
        --bigwig s1_fwd.bw s1_rev.bw s2_fwd.bw s2_rev.bw \\
        --samples SRR17111303 SRR17111311 \\
        --gtf data/raw/GENCODE/release_46/gencode.v46.annotation.gtf.parquet \\
        --output-dir results/evaluation/borzoi/full/test
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream"))

import evaluate_borzoi_checkpoint as ebc  # noqa: E402
from compute_gene_metrics import (  # noqa: E402
    summarize_gene_expression,
    validate_gencode_v46_path,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--bigwig", nargs="+", required=True,
                   help="Observed coverage bigwigs in target-track order "
                        "(interleaved forward/reverse per sample)")
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--gtf", required=True,
                   help="GENCODE v46 GTF/GTF.GZ or processed GTF parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pool-width", type=int, default=32)
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.predictions_dir, "run_info.json")) as handle:
        run_info = json.load(handle)
    if run_info["pool_width_bp"] != args.pool_width:
        raise ValueError(
            "--pool-width {} does not match predictions ({})".format(
                args.pool_width, run_info["pool_width_bp"],
            )
        )
    if len(args.bigwig) % 2 or len(args.samples) * 2 != len(args.bigwig):
        raise ValueError("Expected two interleaved strand bigwigs per sample")
    validate_gencode_v46_path(args.gtf)

    predictions_32bp = np.load(os.path.join(args.predictions_dir, "predictions_32bp.npy"))
    regions = pd.read_parquet(os.path.join(args.predictions_dir, "regions.parquet")).to_dict("records")
    if predictions_32bp.shape[0] != len(regions):
        raise ValueError("predictions and regions.parquet are not row-aligned")
    if predictions_32bp.shape[2] != len(args.bigwig):
        raise ValueError(
            "{} prediction tracks vs {} bigwigs".format(
                predictions_32bp.shape[2], len(args.bigwig),
            )
        )

    targets_1bp, targets_32bp = ebc.read_target_views(args.bigwig, regions, args.pool_width)
    if predictions_32bp.shape != targets_32bp.shape:
        raise ValueError(
            "Prediction/target shape mismatch: {} vs {}".format(
                predictions_32bp.shape, targets_32bp.shape,
            )
        )
    # Untransformed Borzoi outputs are sums over each native bin; convert to
    # per-base coverage before repeating so integrated signal is preserved.
    predictions_1bp = np.repeat(
        predictions_32bp / args.pool_width, args.pool_width, axis=1,
    )
    if predictions_1bp.shape != targets_1bp.shape:
        raise ValueError(
            "Upsampled prediction/1-bp target shape mismatch: {} vs {}".format(
                predictions_1bp.shape, targets_1bp.shape,
            )
        )

    profile_transform = "log1p"
    metrics_by_bin = {
        1: ebc.compute_metrics(predictions_1bp, targets_1bp, profile_transform),
        args.pool_width: ebc.compute_metrics(predictions_32bp, targets_32bp, profile_transform),
    }
    positions = [(r["chrom"], r["target_start"], r["target_end"]) for r in regions]

    gene_prediction_views = {1: predictions_1bp, args.pool_width: predictions_32bp / args.pool_width}
    gene_target_views = {1: targets_1bp, args.pool_width: targets_32bp / args.pool_width}
    gene_rows = ebc.compute_gene_rows(
        ebc.load_gtf(args.gtf), positions,
        [r["original_interval_idx"] for r in regions],
        gene_prediction_views, gene_target_views,
        args.samples, args.pool_width,
    )
    if gene_rows.empty:
        raise ValueError(
            "No genes passed the >=50% unique-exon criterion; check the "
            "GENCODE v46 build and chromosome naming"
        )
    gene_metrics, per_track, per_gene = summarize_gene_expression(
        gene_rows, (1, args.pool_width),
    )
    gene_rows.to_csv(os.path.join(args.output_dir, "gene_expression_per_gene.csv"), index=False)
    per_track.to_csv(os.path.join(args.output_dir, "gene_expression_metrics_per_track.csv"), index=False)
    per_gene.to_csv(os.path.join(args.output_dir, "gene_expression_metrics_per_gene.csv"), index=False)

    with open(os.path.join(args.output_dir, "metrics_per_region.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "bin_size_bp", "region_index", "chrom", "start", "end",
            "profile_pearson_r", "js_divergence", "js_distance",
        ])
        for bin_size, metrics in metrics_by_bin.items():
            for index, position in enumerate(positions):
                writer.writerow([
                    bin_size, index, *position,
                    metrics["profile_pearson_r_all"][index],
                    metrics["js_divergence_all"][index],
                    metrics["js_distance_all"][index],
                ])

    summary = {
        "evaluation": {
            "assay": "rna",
            "predictions_dir": args.predictions_dir,
            "collect_run_info": run_info,
            "bigwig": args.bigwig,
            "samples": args.samples,
            "ensemble": "mean_raw_predictions_across_replicate_trunks",
            "score_window_bp": run_info["target_length_bp"],
            "native_resolution_bp": args.pool_width,
            "metric_bin_sizes_bp": [1, args.pool_width],
            "one_bp_prediction_view": "repeat_each_unscaled_native_bin_sum_divided_by_pool_width",
            "native_prediction_view": "unscaled_bin_sums",
            "gene_annotation": "GENCODE_v46_required",
            "gene_selection": "at_least_50pct_unique_exon_bp_in_test_interval",
            "gene_mask_aggregation": "union_all_annotated_exons_then_mean",
            "gene_resolution_conversion": "exact_exon_bp_weighting_of_upsampled_unscaled_borzoi_bins",
        },
        "metrics": metrics_by_bin,
        "gene_expression_metrics": gene_metrics,
    }
    if args.save_predictions:
        np.save(os.path.join(args.output_dir, "targets_32bp.npy"), targets_32bp.astype(np.float16))
    del predictions_1bp, targets_1bp
    gc.collect()
    # summary.json is the completion marker, written last.
    summary_tmp = os.path.join(args.output_dir, "summary.json.tmp")
    with open(summary_tmp, "w") as handle:
        json.dump(ebc.clean_json(summary), handle, indent=2)
    os.replace(summary_tmp, os.path.join(args.output_dir, "summary.json"))
    log.info("Wrote Borzoi metrics to %s", args.output_dir)


if __name__ == "__main__":
    main()
