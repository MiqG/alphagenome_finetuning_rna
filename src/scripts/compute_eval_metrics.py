#!/usr/bin/env python
"""
Compute evaluation metrics from collected AlphaGenome predictions.

Reads the four parquets written by collect_predictions.py and computes:

  Gene Expression Correlation (rna_seq_per_gene.parquet):
    - Raw; Across Genes           — per-track Pearson r between pred/obs log-mean
    - Normalized; Across Genes    — after quantile norm + gene-mean centering, per-track Pearson r
    - Normalized; Across Tracks   — same normalization, per-gene Pearson r across tracks

  Splice Site Classification (splice_site_scores.parquet):
    - auPRC per class (Donor+, Acceptor+, Donor-, Acceptor-) with RNA-seq and GTF labels
    - Macro-average auPRC

  Splice Site Usage (ssu_scores.parquet):
    - Pearson r per sample (pred_ssu vs obs_ssu)

  Splice Junctions (junction_scores.parquet):
    - auPRC for true/false junction classification (per sample, then average)
    - Pearson r on log1p counts for non-zero observed junctions (per sample)

  PSI (psi_scores.parquet):
    - PSI5 Pearson r per sample (chr2)
    - PSI3 Pearson r per sample (chr2)

Outputs:
  metrics.json    — flat dict with all scalar metrics
  metrics.parquet — long-format table (metric_group, metric_name, value)

Usage:
    python src/scripts/compute_eval_metrics.py \\
        --predictions-dir results/evaluation/.../predictions \\
        --output-dir results/evaluation/...
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score  # used by splice site metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--predictions-dir", required=True,
                   help="Directory containing the four prediction parquets")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-gene-tracks", type=int, default=2,
                   help="Minimum number of tracks a gene must appear in to be included in correlation")
    p.add_argument("--min-alpha-juncs", type=int, default=5,
                   help="When >0 and alpha_juncs column is present in ssu_scores.parquet, report SSU "
                        "Pearson both unfiltered and filtered to alpha_juncs >= this value")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def quantile_normalize(mat: np.ndarray) -> np.ndarray:
    """Quantile-normalize columns of mat (genes × tracks) in-place."""
    from scipy.stats import rankdata

    n_genes, n_tracks = mat.shape
    target = np.sort(mat, axis=0).mean(axis=1)  # average sorted row
    result = np.empty_like(mat)
    for t in range(n_tracks):
        col = mat[:, t]
        ranks = rankdata(col, method="average") - 1  # 0-based fractional ranks
        indices = np.round(ranks).astype(int).clip(0, n_genes - 1)
        result[:, t] = target[indices]
    return result


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r; returns None if fewer than 3 finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    r, _ = pearsonr(x[mask], y[mask])
    return float(r)


# ---------------------------------------------------------------------------
# Gene expression correlation
# ---------------------------------------------------------------------------

def compute_gene_expression_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Three correlation approaches from the paper."""
    metrics: dict[str, float] = {}

    if df.empty:
        return metrics

    # Pivot to (genes × tracks) for pred and obs
    tracks = sorted(df["track_name"].unique())
    genes = sorted(df["gene_id"].unique())

    gene_idx = {g: i for i, g in enumerate(genes)}
    track_idx = {t: i for i, t in enumerate(tracks)}
    n_genes, n_tracks = len(genes), len(tracks)

    pred_mat = np.full((n_genes, n_tracks), np.nan)
    obs_mat = np.full((n_genes, n_tracks), np.nan)

    for _, row in df.iterrows():
        gi = gene_idx[row["gene_id"]]
        ti = track_idx[row["track_name"]]
        pred_mat[gi, ti] = row["pred_log_mean"]
        obs_mat[gi, ti] = row["obs_log_mean"]

    # Keep only genes present in all tracks (complete cases)
    complete = np.all(np.isfinite(pred_mat) & np.isfinite(obs_mat), axis=1)
    pred_mat = pred_mat[complete]
    obs_mat = obs_mat[complete]

    if pred_mat.shape[0] < 3:
        return metrics

    # 1. Raw; Across Genes — per track
    per_track_raw = []
    for t in range(n_tracks):
        r = safe_pearson(pred_mat[:, t], obs_mat[:, t])
        if r is not None:
            metrics["gene_expr_raw_track_{}".format(tracks[t])] = r
            per_track_raw.append(r)
    if per_track_raw:
        metrics["gene_expr_raw_mean"] = float(np.mean(per_track_raw))

    # 2 & 3. Normalized
    pred_qn = quantile_normalize(pred_mat.copy())
    obs_qn = quantile_normalize(obs_mat.copy())

    # Subtract per-gene mean across tracks
    pred_centered = pred_qn - pred_qn.mean(axis=1, keepdims=True)
    obs_centered = obs_qn - obs_qn.mean(axis=1, keepdims=True)

    # Normalized; Across Genes — per track
    per_track_norm = []
    for t in range(n_tracks):
        r = safe_pearson(pred_centered[:, t], obs_centered[:, t])
        if r is not None:
            metrics["gene_expr_norm_across_genes_track_{}".format(tracks[t])] = r
            per_track_norm.append(r)
    if per_track_norm:
        metrics["gene_expr_norm_across_genes_mean"] = float(np.mean(per_track_norm))

    # Normalized; Across Tracks — per gene
    per_gene_norm = []
    for g in range(pred_centered.shape[0]):
        r = safe_pearson(pred_centered[g], obs_centered[g])
        if r is not None:
            per_gene_norm.append(r)
    if per_gene_norm:
        arr = np.array(per_gene_norm)
        metrics["gene_expr_norm_across_tracks_median"] = float(np.median(arr))
        metrics["gene_expr_norm_across_tracks_mean"] = float(np.mean(arr))
        metrics["gene_expr_norm_across_tracks_iqr"] = float(
            np.percentile(arr, 75) - np.percentile(arr, 25)
        )

    return metrics


# ---------------------------------------------------------------------------
# Splice site classification
# ---------------------------------------------------------------------------

def compute_splice_site_metrics(df: pd.DataFrame) -> dict[str, float]:
    """auPRC per class following the publication's definition.

    The evaluation set is the union of all annotated splice site positions.
    For each of the four classes the positives are positions of that class;
    the negatives are all other annotated positions (other classes) — NOT random
    genomic background.  This matches "at each relevant genomic position" in the paper.
    """
    metrics: dict[str, float] = {}
    if df.empty:
        return metrics

    # Map each class to (strand, role, predicted_probability_column)
    classes = [
        ("donor_pos",    "+", "donor",    "pred_donor_pos"),
        ("acceptor_pos", "+", "acceptor", "pred_acceptor_pos"),
        ("donor_neg",    "-", "donor",    "pred_donor_neg"),
        ("acceptor_neg", "-", "acceptor", "pred_acceptor_neg"),
    ]

    for label_col, label_prefix in [("label_rnaseq", "rnaseq"), ("label_gtf", "gtf")]:
        if label_col not in df.columns:
            continue

        aps = []
        for cls_name, strand_sign, role, pred_col in classes:
            # Positives: this class, observed under the given label source
            pos_mask = (
                (df["strand"] == strand_sign)
                & (df["role"] == role)
                & (df[label_col] == 1)
            )
            # Negatives: all other annotated positions in the dataframe.
            # The full df already contains only annotated splice sites
            # (no random background), so every non-positive row is a same-type
            # negative (other class), giving the strictest discrimination task.
            y_score = df[pred_col].values
            y_true = pos_mask.astype(int).values
            if y_true.sum() == 0 or len(y_true) < 2:
                continue
            ap = average_precision_score(y_true, y_score)
            metrics["splice_site_auprc_{}_{}".format(cls_name, label_prefix)] = float(ap)
            aps.append(ap)

        if aps:
            metrics["splice_site_auprc_macro_{}".format(label_prefix)] = float(np.mean(aps))

    return metrics


# ---------------------------------------------------------------------------
# SSU Pearson R
# ---------------------------------------------------------------------------

def compute_ssu_metrics(df: pd.DataFrame, suffix: str = "") -> dict[str, float]:
    """Pearson r per sample between pred_ssu and obs_ssu.

    suffix: inserted before the sample id, e.g. "_alpha5" → ssu_pearson_r_alpha5_{sample}.
    Use "" for the unfiltered set and e.g. "_alphaN" for the depth-filtered set.
    """
    metrics: dict[str, float] = {}
    if df.empty:
        return metrics

    per_sample = []
    for sample_id, grp in df.groupby("sample_id"):
        r = safe_pearson(grp["pred_ssu"].values, grp["obs_ssu"].values)
        if r is not None:
            metrics["ssu_pearson_r{}_{}".format(suffix, sample_id)] = r
            per_sample.append(r)

    if per_sample:
        metrics["ssu_pearson_r{}_mean".format(suffix)] = float(np.mean(per_sample))

    return metrics


# ---------------------------------------------------------------------------
# Junction metrics
# ---------------------------------------------------------------------------

def compute_junction_metrics(
    df: pd.DataFrame,
    totals_df: pd.DataFrame,
) -> dict[str, float]:
    """Junction auPRC and count Pearson R.

    auPRC uses the full K×K denominator by reconstructing zero-score negatives
    from junction_totals.parquet (n_valid_pairs per interval/strand/sample).
    Pairs with pred=0 AND obs=0 were dropped at collection time; they are
    added back as synthetic zero-score negatives so the denominator is correct.
    """
    metrics: dict[str, float] = {}

    if df.empty:
        return metrics

    # Total valid pairs per sample (sum n_valid_pairs across all intervals and strands)
    n_total_per_sample: dict[str, int] = {}
    if not totals_df.empty:
        n_total_per_sample = totals_df.groupby("sample_id")["n_valid_pairs"].sum().to_dict()

    # auPRC — reconstruct correct denominator
    auprc_vals = []
    for sample_id, grp in df.groupby("sample_id"):
        y_score = grp["pred_count"].values.astype(np.float32)
        y_true = (grp["obs_count"].values > 0).astype(np.int8)
        n_stored = len(y_score)
        n_total = int(n_total_per_sample.get(sample_id, n_stored))
        n_synthetic = max(0, n_total - n_stored)

        if n_synthetic > 0:
            y_score = np.concatenate([y_score, np.zeros(n_synthetic, dtype=np.float32)])
            y_true = np.concatenate([y_true, np.zeros(n_synthetic, dtype=np.int8)])

        if y_true.sum() == 0 or len(y_true) < 2:
            continue
        ap = average_precision_score(y_true, y_score)
        metrics["junction_auprc_{}".format(sample_id)] = float(ap)
        auprc_vals.append(ap)

    if auprc_vals:
        metrics["junction_auprc_mean"] = float(np.mean(auprc_vals))

    # Count Pearson R (log1p, non-zero observed junctions only)
    pearson_vals = []
    for sample_id, grp in df.groupby("sample_id"):
        sub = grp[grp["obs_count"] > 0]
        if len(sub) < 3:
            continue
        r = safe_pearson(
            np.log1p(sub["pred_count"].values),
            np.log1p(sub["obs_count"].values),
        )
        if r is not None:
            metrics["junction_count_pearson_r_{}".format(sample_id)] = r
            pearson_vals.append(r)
    if pearson_vals:
        metrics["junction_count_pearson_r_mean"] = float(np.mean(pearson_vals))

    return metrics


# ---------------------------------------------------------------------------
# PSI metrics
# ---------------------------------------------------------------------------

def compute_psi_metrics(df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if df.empty:
        return metrics

    for psi_type in ["psi5", "psi3"]:
        vals = []
        for sample_id, grp in df.groupby("sample_id"):
            pred_col = "pred_{}".format(psi_type)
            obs_col = "obs_{}".format(psi_type)
            if pred_col not in grp.columns or obs_col not in grp.columns:
                continue
            r = safe_pearson(grp[pred_col].values, grp[obs_col].values)
            if r is not None:
                metrics["{}_pearson_r_{}".format(psi_type, sample_id)] = r
                vals.append(r)
        if vals:
            metrics["{}_pearson_r_mean".format(psi_type)] = float(np.mean(vals))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_dir = args.predictions_dir

    print("Loading prediction parquets...")
    rna_df = pd.read_parquet(os.path.join(pred_dir, "rna_seq_per_gene.parquet"))
    cls_df = pd.read_parquet(os.path.join(pred_dir, "splice_site_scores.parquet"))
    ssu_df = pd.read_parquet(os.path.join(pred_dir, "ssu_scores.parquet"))
    junc_df = pd.read_parquet(os.path.join(pred_dir, "junction_scores.parquet"))
    totals_path = os.path.join(pred_dir, "junction_totals.parquet")
    totals_df = pd.read_parquet(totals_path) if os.path.exists(totals_path) else pd.DataFrame()
    psi_path = os.path.join(pred_dir, "psi_scores.parquet")
    psi_df = pd.read_parquet(psi_path) if os.path.exists(psi_path) else pd.DataFrame()

    print("  rna_seq rows: {}".format(len(rna_df)))
    print("  splice_site rows: {}".format(len(cls_df)))
    print("  ssu rows: {}".format(len(ssu_df)))
    print("  junction rows: {}".format(len(junc_df)))
    print("  psi rows: {}".format(len(psi_df)))

    # --- Compute metrics ---
    all_metrics: dict[str, float] = {}

    print("Computing gene expression correlations...")
    all_metrics.update(compute_gene_expression_metrics(rna_df))

    print("Computing splice site auPRC...")
    all_metrics.update(compute_splice_site_metrics(cls_df))

    print("Computing SSU Pearson r...")
    all_metrics.update(compute_ssu_metrics(ssu_df))
    if args.min_alpha_juncs > 0 and "alpha_juncs" in ssu_df.columns:
        ssu_df_filtered = ssu_df[ssu_df["alpha_juncs"] >= args.min_alpha_juncs]
        suffix = "_alpha{}".format(args.min_alpha_juncs)
        print("  also computing SSU Pearson r with alpha_juncs >= {} ({:,} / {:,} sites)".format(
            args.min_alpha_juncs, len(ssu_df_filtered), len(ssu_df)))
        all_metrics.update(compute_ssu_metrics(ssu_df_filtered, suffix=suffix))

    print("Computing junction metrics...")
    all_metrics.update(compute_junction_metrics(junc_df, totals_df))

    print("Computing PSI Pearson r...")
    all_metrics.update(compute_psi_metrics(psi_df))

    # --- Write JSON ---
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nWrote {}".format(json_path))

    # --- Write long-format parquet ---
    rows = []
    for key, val in all_metrics.items():
        parts = key.split("_")
        group = "_".join(parts[:2]) if len(parts) >= 2 else key
        rows.append({"metric_group": group, "metric_name": key, "value": val})
    metrics_df = pd.DataFrame(rows)
    parquet_path = os.path.join(args.output_dir, "metrics.parquet")
    metrics_df.to_parquet(parquet_path, index=False, compression="zstd")
    print("Wrote {}".format(parquet_path))

    # Print summary
    summary_keys = [
        "gene_expr_raw_mean",
        "gene_expr_norm_across_genes_mean",
        "gene_expr_norm_across_tracks_median",
        "splice_site_auprc_macro_rnaseq",
        "splice_site_auprc_macro_gtf",
        "ssu_pearson_r_mean",
        "ssu_pearson_r_alpha{}_mean".format(args.min_alpha_juncs),
        "junction_auprc_mean",
        "junction_count_pearson_r_mean",
        "psi5_pearson_r_mean",
        "psi3_pearson_r_mean",
    ]
    print("\n--- Summary ---")
    for k in summary_keys:
        if k in all_metrics:
            print("  {}: {:.4f}".format(k, all_metrics[k]))


if __name__ == "__main__":
    main()
