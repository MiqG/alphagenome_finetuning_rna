#!/usr/bin/env python
"""
Collect fine-tuned Borzoi ensemble predictions on held-out test intervals.

The Borzoi counterpart of src/scripts/collect_predictions.py. Runs GPU inference
for each replicate trunk, undoes the Borzoi target transform (soft clip, sqrt,
scale), averages the raw predictions over trunks, and writes the RNA-seq
prediction parquets that collect_predictions.py writes for AlphaGenome, so the
unchanged src/scripts/compute_eval_metrics.py works on Borzoi runs too:

  rna_seq_per_gene.parquet / rna_seq_per_gene_32bp.parquet
  rna_seq_profile_corr_{exon,full,central}_{1bp,32bp}.parquet
  rna_seq_profile_corr_per_interval.parquet
  rna_seq_profile_metrics_per_interval.parquet
  rna_seq_track_totals_per_interval.parquet

Model and target-transform code is imported from the vendored, unmodified
upstream script (upstream/evaluate_borzoi_checkpoint.py). The parquet helpers
(gene/exon means, accumulators, Pearson, JSD) are verbatim copies of the
AlphaGenome ones (rna_eval_helpers.py), so definitions match by construction.

Conventions that follow from Borzoi:
  - Borzoi only predicts the central 196,608 bp of each BED interval, so that
    window plays the role of the whole window: genes are assigned (>=50% exon bp)
    and exon means are taken inside it. "full"-scope outputs are NaN / empty.
  - 1 bp prediction = native bin sum / 32 repeated 32x; 32 bp prediction = mean
    per base in the bin (AlphaGenome's mean-pooled convention).
  - interval_idx and start/end are those of the BED interval (not the target
    window), so tables join with the AlphaGenome ones.

Splice-site, SSU, junction and PSI outputs do not exist for Borzoi
(compute_eval_metrics.py skips missing parquets).

Native Borzoi view (32 bp, sum-pooled bins, exon-base-weighted gene means), using
the upstream functions on the full 32 bp arrays (small: ~625 MB each), also as
parquets for compute_metrics_borzoi_native.py:
  borzoi_native_32bp_per_region.parquet   — per region: log1p track-pooled profile
                                            Pearson r, JS divergence / distance, MSE
  borzoi_native_32bp_per_track.parquet    — per track: accumulated log1p Pearson r
  borzoi_native_32bp_gene_expression.parquet — per gene and sample: exon-base-weighted
                                            mean coverage (pred / obs)

Also written next to the parquets (not declared workflow outputs):
  regions.parquet, run_info.json — retained regions and run settings
  .progress/trunk_<i>.npy        — per-trunk raw predictions (untransformed, per-bin
                                   coverage sums), used to resume

Usage:
    python workflows/12-borzoi_evaluation/scripts/collect_predictions_borzoi.py \\
        --models results/finetuning/borzoi/trunk_r{0,1,2,3}/model_best.mergeW.h5 \\
        --params results/finetuning/borzoi/trunk_r0/params.json \\
        --targets support/borzoi/targets.txt \\
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \\
        --test-bed data/prep/finetuning/alphagenome/FOLD_1/test.bed \\
        --bigwigs s1_fwd.bw s1_rev.bw s2_fwd.bw s2_rev.bw \\
        --samples SRR17111303 SRR17111311 \\
        --gtf-parquet data/raw/GENCODE/release_46/gencode.v46.annotation.gtf.parquet \\
        --output-dir results/evaluation/borzoi/full/test/predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# evaluation_provenance.py is shared with the AlphaGenome evaluations.
SRC_SCRIPTS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "scripts"))
sys.path.insert(0, SRC_SCRIPTS_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "upstream"))

import evaluation_provenance  # noqa: E402
import evaluate_borzoi_checkpoint as ebc  # noqa: E402
from compute_gene_metrics import collect_gene_expression_rows  # noqa: E402
from rna_eval_helpers import (  # noqa: E402
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

log = logging.getLogger(__name__)

POOL = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--params", nargs="+", required=True,
                   help="One params.json reused by all models, or one per model")
    p.add_argument("--targets", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--test-bed", required=True)
    p.add_argument("--bigwigs", nargs="+", required=True,
                   help="Observed bigwigs in track order: sample0/forward, sample0/reverse, ...")
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--gtf-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--crop-bp", type=int, default=163_840)
    p.add_argument("--pool-width", type=int, default=POOL)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-regions", type=int, default=0)
    p.add_argument("--mixed-precision", action="store_true")
    return p.parse_args()


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def write_alphagenome_parquets(
    predictions: np.ndarray,
    regions: pd.DataFrame,
    test_bed: str,
    bigwig_paths: list[str],
    samples: list[str],
    gtf_parquet: str,
    output_dir: str,
) -> np.ndarray:
    """Write the AlphaGenome-schema RNA-seq parquets (see module docstring).

    Returns the observed 32 bp bin sums (n_regions, n_bins, n_tracks), float32,
    gathered along the way for the native-view tables.
    """
    import pyBigWig

    n_regions, n_bins, n_tracks = predictions.shape
    seq_len = n_bins * POOL
    bigwigs = [pyBigWig.open(p) for p in bigwig_paths]
    chrom_sizes = dict(bigwigs[0].chroms())

    # GTF exons: same preparation as collect_predictions.py
    gtf = pd.read_parquet(
        gtf_parquet,
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
    bed = pd.read_csv(test_bed, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"])

    # Gene assignment inside the scored (central) window, keyed by BED row index.
    windows = regions[["original_interval_idx", "chrom", "target_start", "target_end"]].rename(
        columns={"target_start": "start", "target_end": "end"},
    ).set_index("original_interval_idx")
    gene_interval_map = compute_gene_interval_map(exons, windows)
    genes_per_interval: dict[int, list[str]] = defaultdict(list)
    for gid, (iv_idx, _, _) in gene_interval_map.items():
        genes_per_interval[iv_idx].append(gid)
    log.info("Regions: %d | genes assigned: %d", n_regions, len(gene_interval_map))

    forward_track_indices = list(range(0, n_tracks, 2))
    reverse_track_indices = list(range(1, n_tracks, 2))

    # The two "full" accumulators stay empty: Borzoi has no full-window prediction.
    acc = {name: ProfileCorrAccumulator(n_tracks) for name in (
        "exon_1bp", "full_1bp", "exon_32bp", "full_32bp", "central_1bp", "central_32bp",
    )}
    obs_32_sum = np.zeros((n_regions, n_bins, n_tracks), dtype=np.float32)
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
        obs_32_sum[pos] = obs_32 * POOL

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
            track_name = samples[t_idx // 2]
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

        # Gene expression (strand-matched exon means inside the window)
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
                    "track_name": samples[t_idx // 2],
                    "obs_log_mean": float(np.log1p(obs_means[t_idx])),
                }
                rna_rows.append({**row, "pred_log_mean": float(np.log1p(pred_means[t_idx]))})
                if pred_means_32 is not None:
                    rna_rows_32bp.append({**row, "pred_log_mean": float(np.log1p(pred_means_32[t_idx]))})
        if (pos + 1) % 500 == 0:
            log.info("  region %d/%d", pos + 1, n_regions)

    for bw in bigwigs:
        bw.close()

    _write_parquet(pd.DataFrame(rna_rows), os.path.join(output_dir, "rna_seq_per_gene.parquet"))
    _write_parquet(pd.DataFrame(rna_rows_32bp), os.path.join(output_dir, "rna_seq_per_gene_32bp.parquet"))
    for name, accumulator in acc.items():
        _write_parquet(
            accumulator_to_df(accumulator, samples),
            os.path.join(output_dir, "rna_seq_profile_corr_{}.parquet".format(name)),
        )
    _write_parquet(pd.DataFrame(interval_corr_rows),
                   os.path.join(output_dir, "rna_seq_profile_corr_per_interval.parquet"))
    _write_parquet(pd.DataFrame(pooled_rows),
                   os.path.join(output_dir, "rna_seq_profile_metrics_per_interval.parquet"))
    _write_parquet(pd.DataFrame(totals_rows),
                   os.path.join(output_dir, "rna_seq_track_totals_per_interval.parquet"))
    return obs_32_sum


def write_native_32bp_tables(
    predictions: np.ndarray,
    obs_32_sum: np.ndarray,
    regions: pd.DataFrame,
    test_bed: str,
    samples: list[str],
    gtf_parquet: str,
    output_dir: str,
) -> None:
    """Native Borzoi view: 32 bp bin sums, upstream metric functions (see module docstring)."""
    n_regions, n_bins, n_tracks = predictions.shape
    bed = pd.read_csv(test_bed, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"])

    metrics = ebc.compute_metrics(predictions, obs_32_sum, "log1p")
    mse_per_region = np.array([
        float(np.mean(np.square(predictions[i].astype(np.float64) - obs_32_sum[i])))
        for i in range(n_regions)
    ])
    iv = regions["original_interval_idx"].astype(int).values
    per_region = pd.DataFrame({
        "interval_idx": iv,
        "chrom": regions["chrom"].values,
        "start": bed.loc[iv, "start"].values,
        "end": bed.loc[iv, "end"].values,
        "profile_pearson_r_log1p": metrics["profile_pearson_r_all"],
        "js_divergence": metrics["js_divergence_all"],
        "js_distance": metrics["js_distance_all"],
        "mse": mse_per_region,
    })
    _write_parquet(per_region, os.path.join(output_dir, "borzoi_native_32bp_per_region.parquet"))

    per_track = pd.DataFrame({
        "track_idx": np.arange(n_tracks),
        "track_name": [samples[t // 2] for t in range(n_tracks)],
        "strand": ["forward" if t % 2 == 0 else "reverse" for t in range(n_tracks)],
        "pearson_r_accumulated_log1p": metrics["track_pearson_r_accumulated_all"],
        "n_positions": n_regions * n_bins,
    })
    _write_parquet(per_track, os.path.join(output_dir, "borzoi_native_32bp_per_track.parquet"))

    # Gene expression: upstream selection (>=50% of unique exon bp in the scored
    # window) and exact exon-base weighting of the 32 bp bins.
    positions = [
        (r.chrom, int(r.target_start), int(r.target_end)) for r in regions.itertuples()
    ]
    gene_rows = collect_gene_expression_rows(
        gtf=ebc.load_gtf(gtf_parquet),
        test_intervals=pd.DataFrame({"original_interval_idx": iv}),
        positions=positions,
        prediction_views={POOL: predictions / POOL},
        target_views={POOL: obs_32_sum / POOL},
        bin_sizes=(POOL,),
        sample_names=samples,
    )
    _write_parquet(gene_rows, os.path.join(output_dir, "borzoi_native_32bp_gene_expression.parquet"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if len(args.params) not in {1, len(args.models)}:
        raise ValueError("--params must contain one path or match --models")
    params_paths = args.params * len(args.models) if len(args.params) == 1 else args.params
    if args.pool_width != POOL:
        raise ValueError("Only the native 32 bp pool width is supported")
    if len(args.bigwigs) != 2 * len(args.samples):
        raise ValueError("Expected two interleaved strand bigwigs per sample")

    with open(params_paths[0]) as handle:
        sequence_length = int(json.load(handle)["model"]["seq_length"])
    target_length = sequence_length - 2 * args.crop_bp
    if target_length != 196_608:
        raise ValueError("Expected centered 196,608-bp target, found {}".format(target_length))

    # The fingerprint guards the resumable per-trunk predictions, so it covers
    # only what determines them (models, targets, regions, inference code), not
    # the bigwigs / GTF / export code, which can change without invalidating them.
    fingerprint = evaluation_provenance.build_fingerprint(
        framework="borzoi",
        settings={
            "crop_bp": args.crop_bp,
            "pool_width": args.pool_width,
            "max_regions": args.max_regions,
            "mixed_precision": args.mixed_precision,
        },
        paths={
            "models": args.models,
            "params": params_paths,
            "targets": args.targets,
            "genome": args.genome,
            "test_bed": args.test_bed,
        },
        code_paths=[ebc.__file__],
    )
    # Raises if the directory holds state from a different run.
    evaluation_provenance.ensure_provenance(args.output_dir, fingerprint)
    progress_dir = os.path.join(args.output_dir, ".progress")
    os.makedirs(progress_dir, exist_ok=True)

    import pysam
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info("TensorFlow GPUs: %s", tf.config.list_physical_devices("GPU"))
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    targets_metadata = ebc.load_targets(args.targets)
    if len(targets_metadata) != len(args.bigwigs):
        raise ValueError("{} target rows vs {} bigwigs".format(len(targets_metadata), len(args.bigwigs)))
    fasta = pysam.Fastafile(args.genome)
    chrom_sizes = dict(zip(fasta.references, fasta.lengths))
    fasta.close()
    regions = ebc.prepare_regions(
        args.test_bed, chrom_sizes, sequence_length, target_length, args.max_regions,
    )
    log.info("Retained %d regions from %s", len(regions), args.test_bed)

    ensemble_sum = None
    for index, (model_path, params_path) in enumerate(zip(args.models, params_paths)):
        trunk_file = os.path.join(progress_dir, "trunk_{}.npy".format(index))
        if os.path.exists(trunk_file):
            log.info("Resuming: loading finished trunk %d from %s", index, trunk_file)
            raw = np.load(trunk_file)
        else:
            start = time.time()
            transformed = ebc.predict_one_model(
                model_path, params_path, args.genome, regions, args.batch_size,
            )
            raw = ebc.untransform_predictions(transformed, targets_metadata)
            del transformed
            _atomic_save_npy(trunk_file, raw)
            log.info("Trunk %d done in %.1f min", index, (time.time() - start) / 60)
        ensemble_sum = raw.astype(np.float64) if ensemble_sum is None else ensemble_sum + raw
        del raw

    predictions_32bp = (ensemble_sum / len(args.models)).astype(np.float32)
    n_bins = target_length // POOL
    expected = (len(regions), n_bins, len(targets_metadata))
    if predictions_32bp.shape != expected:
        raise ValueError("Unexpected prediction shape {} (expected {})".format(predictions_32bp.shape, expected))

    os.makedirs(args.output_dir, exist_ok=True)
    regions_df = pd.DataFrame(regions)
    regions_df.to_parquet(os.path.join(args.output_dir, "regions.parquet"), index=False)
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as handle:
        json.dump({
            "models": args.models, "params": params_paths, "targets": args.targets,
            "test_bed": args.test_bed, "genome": args.genome,
            "n_regions": len(regions), "n_tracks": int(len(targets_metadata)),
            "n_ensemble_members": len(args.models),
            "sequence_length_bp": sequence_length, "target_length_bp": target_length,
            "pool_width_bp": POOL,
            "prediction_view": "ensemble_mean_of_untransformed_per_bin_sums",
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
        }, handle, indent=2)

    # The parquets are the declared workflow outputs, written last and atomically.
    obs_32_sum = write_alphagenome_parquets(
        predictions_32bp, regions_df, args.test_bed, args.bigwigs, args.samples,
        args.gtf_parquet, args.output_dir,
    )
    write_native_32bp_tables(
        predictions_32bp, obs_32_sum, regions_df, args.test_bed, args.samples,
        args.gtf_parquet, args.output_dir,
    )
    log.info("Wrote AlphaGenome-format and native 32 bp parquets for %d regions to %s", len(regions), args.output_dir)


if __name__ == "__main__":
    main()
