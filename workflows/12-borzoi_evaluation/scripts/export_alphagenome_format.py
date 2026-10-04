#!/usr/bin/env python
"""
Export collected Borzoi predictions in the AlphaGenome prediction-parquet schema.

Step 2a of the Borzoi evaluation (see ../Snakefile). Reads the output of
collect_predictions_borzoi.py plus the observed bigwigs and writes the RNA-seq
parquets that src/scripts/collect_predictions.py writes for AlphaGenome, so the
unchanged src/scripts/compute_eval_metrics.py (and the paper figure scripts)
work on Borzoi runs too:

  rna_seq_per_gene.parquet / rna_seq_per_gene_32bp.parquet
  rna_seq_profile_corr_{exon,full,central}_{1bp,32bp}.parquet
  rna_seq_profile_corr_per_interval.parquet
  rna_seq_profile_metrics_per_interval.parquet
  rna_seq_track_totals_per_interval.parquet

Every definition is the AlphaGenome one, by construction: the helper functions
(gene/exon means, accumulators, Pearson, JSD) are imported from
src/scripts/collect_predictions.py. Conventions that follow from Borzoi:

  - Borzoi only predicts the central 196,608 bp of each BED interval, so that
    window plays the role of the whole window. Genes are assigned (>=50% exon bp)
    and exon means are taken inside it. "full"-scope outputs are NaN / empty.
  - 1 bp prediction = native bin sum / 32 repeated 32x; 32 bp prediction = mean
    per base in the bin (AlphaGenome's mean-pooled convention).
  - interval_idx and start/end are those of the BED interval (not the target
    window), so tables join with the AlphaGenome ones.

Splice-site, SSU, junction and PSI outputs do not exist for Borzoi
(compute_eval_metrics.py skips missing parquets).

Runs in the alphagenome_pytorch env (no TensorFlow or GPU needed).

Usage:
    python workflows/12-borzoi_evaluation/scripts/export_alphagenome_format.py \\
        --predictions-dir results/evaluation/borzoi/full/test/predictions \\
        --test-bed data/prep/finetuning/alphagenome/FOLD_1/test.bed \\
        --bigwigs s1_fwd.bw s1_rev.bw s2_fwd.bw s2_rev.bw \\
        --samples SRR17111303 SRR17111311 \\
        --gtf-parquet data/raw/GENCODE/release_46/gencode.v46.annotation.gtf.parquet \\
        --output-dir results/evaluation/borzoi/full/test/predictions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyBigWig

# collect_predictions.py imports torch/alphagenome_pytorch at module level
# (present in the alphagenome_pytorch env); we only use its numpy/pandas helpers.
_SRC_SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "scripts",
))
sys.path.insert(0, _SRC_SCRIPTS_DIR)
from collect_predictions import (  # noqa: E402
    ProfileCorrAccumulator,
    accumulator_to_df,
    compute_gene_interval_map,
    fetch_obs_window,
    get_exon_mean_obs,
    get_exon_mean_pred,
    get_exon_mean_pred_binned,
    jsd_per_track,
    merge_gene_exons,
    pearson_r_per_track,
    pearson_r_pooled,
)

POOL = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--predictions-dir", required=True,
                   help="Output dir of collect_predictions_borzoi.py")
    p.add_argument("--test-bed", required=True,
                   help="The BED used for collection (interval_idx / start / end come from it)")
    p.add_argument("--bigwigs", nargs="+", required=True,
                   help="Observed bigwigs in track order: sample0/forward, sample0/reverse, ...")
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--gtf-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.predictions_dir, "run_info.json")) as handle:
        run_info = json.load(handle)
    if run_info["pool_width_bp"] != POOL:
        raise ValueError("Expected 32 bp native resolution, found {}".format(run_info["pool_width_bp"]))
    predictions = np.load(os.path.join(args.predictions_dir, "predictions_32bp.npy"), mmap_mode="r")
    regions = pd.read_parquet(os.path.join(args.predictions_dir, "regions.parquet"))
    n_regions, n_bins, n_tracks = predictions.shape
    if n_regions != len(regions):
        raise ValueError("predictions and regions.parquet are not row-aligned")
    if n_tracks != len(args.bigwigs) or n_tracks != 2 * len(args.samples):
        raise ValueError(
            "{} prediction tracks vs {} bigwigs and {} samples".format(
                n_tracks, len(args.bigwigs), len(args.samples),
            )
        )
    seq_len = n_bins * POOL

    bigwigs = [pyBigWig.open(p) for p in args.bigwigs]
    chrom_sizes = dict(bigwigs[0].chroms())

    # --- GTF exons: same preparation as collect_predictions.py ---
    gtf = pd.read_parquet(
        args.gtf_parquet,
        columns=["Chromosome", "Start", "End", "Strand", "Feature", "gene_id", "gene_name"],
    )
    exons = (
        gtf[gtf["Feature"] == "exon"][["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    exons["Strand"] = exons["Strand"].astype(str)
    exons = merge_gene_exons(exons)
    exons_by_gene = {gid: df for gid, df in exons.groupby("gene_id")}

    # BED intervals (their row index is the AlphaGenome interval_idx).
    bed = pd.read_csv(args.test_bed, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"])

    # Gene assignment inside the scored (central) window, keyed by BED row index.
    windows = regions[["original_interval_idx", "chrom", "target_start", "target_end"]].rename(
        columns={"target_start": "start", "target_end": "end"},
    ).set_index("original_interval_idx")
    gene_interval_map = compute_gene_interval_map(exons, windows)
    genes_per_interval: dict[int, list[str]] = defaultdict(list)
    for gid, (iv_idx, _, _) in gene_interval_map.items():
        genes_per_interval[iv_idx].append(gid)
    print("Regions: {} | genes assigned: {}".format(n_regions, len(gene_interval_map)))

    forward_track_indices = list(range(0, n_tracks, 2))
    reverse_track_indices = list(range(1, n_tracks, 2))

    acc = {name: ProfileCorrAccumulator(n_tracks) for name in (
        "exon_1bp", "full_1bp", "exon_32bp", "full_32bp", "central_1bp", "central_32bp",
    )}  # the two "full" accumulators stay empty (Borzoi has no full-window prediction)
    rna_rows, rna_rows_32bp = [], []
    interval_corr_rows, pooled_rows, totals_rows = [], [], []

    for pos in range(n_regions):
        region = regions.iloc[pos]
        iv_idx = int(region["original_interval_idx"])
        chrom = region["chrom"]
        win_start, win_end = int(region["target_start"]), int(region["target_end"])
        iv_start, iv_end = int(bed.at[iv_idx, "start"]), int(bed.at[iv_idx, "end"])

        pred_32 = np.asarray(predictions[pos], dtype=np.float64) / POOL   # per-base mean in each bin
        pred_1 = np.repeat(pred_32, POOL, axis=0)
        obs_1 = fetch_obs_window(bigwigs, chrom, win_start, win_end, chrom_sizes)
        obs_32 = obs_1.reshape(n_bins, POOL, -1).mean(axis=1)

        log_p1, log_o1 = np.log1p(pred_1), np.log1p(obs_1)
        log_p32, log_o32 = np.log1p(pred_32), np.log1p(obs_32)
        acc["central_1bp"].update(log_p1, log_o1)
        acc["central_32bp"].update(log_p32, log_o32)
        r_1bp = pearson_r_per_track(log_p1, log_o1)
        r_32bp = pearson_r_per_track(log_p32, log_o32)

        profile_r_track = pearson_r_per_track(pred_1, obs_1)
        jsd_track = jsd_per_track(pred_1, obs_1)
        pooled_rows.append({
            "interval_idx": iv_idx, "chrom": chrom, "start": iv_start, "end": iv_end,
            "profile_pearson_r_full": float("nan"),
            "profile_pearson_r_central": float(pearson_r_pooled(pred_1, obs_1)),
            "jsd_full": float("nan"),
            "jsd_central": float(np.nanmean(jsd_track)),
            "n_positions_full": 0,
            "n_positions_central": seq_len,
        })
        pred_sum, obs_sum = pred_1.sum(axis=0), obs_1.sum(axis=0)
        for t_idx in range(n_tracks):
            track_name = args.samples[t_idx // 2]
            strand = "forward" if t_idx % 2 == 0 else "reverse"
            interval_corr_rows.append({
                "interval_idx": iv_idx, "chrom": chrom, "start": iv_start, "end": iv_end,
                "track_idx": t_idx, "track_name": track_name, "strand": strand,
                "pearson_r_1bp": float("nan"), "pearson_r_32bp": float("nan"),
                "n_positions_1bp": 0, "n_positions_32bp": 0,
                "pearson_r_central_1bp": float(r_1bp[t_idx]),
                "pearson_r_central_32bp": float(r_32bp[t_idx]),
                "n_positions_central_1bp": seq_len, "n_positions_central_32bp": n_bins,
                "profile_pearson_r_raw_full": float("nan"),
                "profile_pearson_r_raw_central": float(profile_r_track[t_idx]),
                "jsd_full": float("nan"), "jsd_central": float(jsd_track[t_idx]),
            })
            totals_rows.append({
                "interval_idx": iv_idx, "chrom": chrom, "track_idx": t_idx,
                "track_name": track_name, "strand": strand,
                "pred_sum_full": float("nan"), "obs_sum_full": float("nan"),
                "pred_sum_central": float(pred_sum[t_idx]), "obs_sum_central": float(obs_sum[t_idx]),
            })

        # --- Gene expression (strand-matched exon means inside the window) ---
        for gid in genes_per_interval.get(iv_idx, []):
            gene_exons = exons_by_gene.get(gid)
            if gene_exons is None:
                continue
            _, g_strand, g_name = gene_interval_map[gid]
            track_indices = forward_track_indices if g_strand == "+" else reverse_track_indices
            pred_means = get_exon_mean_pred(pred_1, gene_exons, win_start, seq_len)
            pred_means_32 = get_exon_mean_pred_binned(pred_32, gene_exons, win_start, seq_len)
            obs_means = get_exon_mean_obs(bigwigs, gene_exons, chrom, chrom_sizes, win_start, seq_len)
            if pred_means is None or obs_means is None:
                continue
            for _, ex in gene_exons.iterrows():
                rel_s = max(0, int(ex["Start"]) - win_start)
                rel_e = min(seq_len, int(ex["End"]) - win_start)
                if rel_e <= rel_s:
                    continue
                acc["exon_1bp"].update(log_p1[rel_s:rel_e], log_o1[rel_s:rel_e])
                bin_s = rel_s // POOL
                bin_e = min(((rel_e - 1) // POOL) + 1, n_bins)
                if bin_e > bin_s:
                    acc["exon_32bp"].update(log_p32[bin_s:bin_e], log_o32[bin_s:bin_e])
            for t_idx in track_indices:
                row = {
                    "gene_id": gid, "gene_name": g_name, "chrom": chrom, "strand": g_strand,
                    "interval_idx": iv_idx, "track_idx": t_idx,
                    "track_name": args.samples[t_idx // 2],
                    "obs_log_mean": float(np.log1p(obs_means[t_idx])),
                }
                rna_rows.append({**row, "pred_log_mean": float(np.log1p(pred_means[t_idx]))})
                if pred_means_32 is not None:
                    rna_rows_32bp.append({**row, "pred_log_mean": float(np.log1p(pred_means_32[t_idx]))})
        if (pos + 1) % 500 == 0:
            print("  region {}/{}".format(pos + 1, n_regions))

    for bw in bigwigs:
        bw.close()

    out = args.output_dir
    _write_parquet(pd.DataFrame(rna_rows), os.path.join(out, "rna_seq_per_gene.parquet"))
    _write_parquet(pd.DataFrame(rna_rows_32bp), os.path.join(out, "rna_seq_per_gene_32bp.parquet"))
    for name, accumulator in acc.items():
        _write_parquet(
            accumulator_to_df(accumulator, args.samples),
            os.path.join(out, "rna_seq_profile_corr_{}.parquet".format(name)),
        )
    _write_parquet(pd.DataFrame(interval_corr_rows), os.path.join(out, "rna_seq_profile_corr_per_interval.parquet"))
    _write_parquet(pd.DataFrame(pooled_rows), os.path.join(out, "rna_seq_profile_metrics_per_interval.parquet"))
    _write_parquet(pd.DataFrame(totals_rows), os.path.join(out, "rna_seq_track_totals_per_interval.parquet"))
    print("Wrote AlphaGenome-format RNA-seq parquets to {}".format(out))


if __name__ == "__main__":
    main()
