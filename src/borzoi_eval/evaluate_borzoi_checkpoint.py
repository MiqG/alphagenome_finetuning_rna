#!/usr/bin/env python
"""Evaluate an ensemble of fine-tuned Borzoi checkpoints on fold test regions.

The four replicate-trunk predictions are untransformed to raw coverage and
then averaged before metrics are computed over the centered 196,608-bp target
window. Metrics are reported at native 32-bp resolution and at 1-bp resolution
by repeating each unscaled per-base bin value 32 times.

Metric code is a NumPy port of ``evaluate_checkpoint.py``. RNA gene metrics
reuse the AlphaGenome paper protocol from ``compute_gene_metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyBigWig
import pysam
from scipy import stats
import tensorflow as tf
from tqdm import tqdm

from baskerville import dna, seqnn

from compute_gene_metrics import (
    collect_gene_expression_rows,
    summarize_gene_expression,
    validate_gencode_v46_path,
)

try:
    from baskerville.dataset import untransform_preds as _untransform_preds
except Exception:  # pragma: no cover - installation dependent
    _untransform_preds = None


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a four-replicate Borzoi checkpoint ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--params", nargs="+", required=True,
        help="One params.json reused by all models, or one per model",
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--bigwig", nargs="+", required=True)
    parser.add_argument("--test-bed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assay", choices=["atac", "rna"], required=True)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument(
        "--gtf", default=None,
        help="GTF/GTF.GZ or processed GTF parquet; required for RNA gene metrics",
    )
    parser.add_argument("--crop-bp", type=int, default=163_840)
    parser.add_argument("--pool-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_targets(path: str) -> pd.DataFrame:
    targets = pd.read_csv(path, sep="\t", index_col=0)
    for column in ("clip", "clip_soft", "scale"):
        if column in targets:
            targets[column] = targets[column].astype(float)
    return targets


def manual_untransform(predictions: np.ndarray, targets: pd.DataFrame) -> np.ndarray:
    """Invert hound_data_read's default (non-transform_old) target transform.

    Forward: scale per bp -> sum pool -> sqrt(1 + s) - 1 ->
    soft clip cs - 1 + sqrt(z - cs + 1). Hard clips are not invertible.
    """
    output = predictions.astype(np.float64)
    for track_index, (_, target) in enumerate(targets.iterrows()):
        values = output[..., track_index]
        if pd.notna(target.get("clip_soft")):
            clip_soft = float(target["clip_soft"])
            values = np.where(
                values > clip_soft,
                clip_soft - 1.0 + np.square(values - clip_soft + 1.0),
                values,
            )
        if "_sqrt" in str(target.get("sum_stat", "")):
            values = np.square(values + 1.0) - 1.0
        values /= float(target.get("scale", 1.0))
        output[..., track_index] = values
    return np.maximum(output, 0).astype(np.float32)


def untransform_predictions(
    predictions: np.ndarray, targets: pd.DataFrame,
) -> np.ndarray:
    shape = predictions.shape
    flat = predictions.reshape(-1, shape[-1])
    if _untransform_preds is not None:
        try:
            output = _untransform_preds(flat.copy(), targets, unscale=True)
            return np.maximum(output, 0).reshape(shape).astype(np.float32)
        except Exception as error:  # pragma: no cover - version dependent
            log.warning("Baskerville untransform failed (%s); using port", error)
    return manual_untransform(predictions, targets)


def read_bed(path: str) -> list[tuple[int, str, int, int]]:
    regions = []
    with open(path) as handle:
        for original_index, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            chromosome, start, end, *_ = line.split()
            regions.append((original_index, chromosome, int(start), int(end)))
    return regions


def centered_window(start: int, end: int, width: int) -> tuple[int, int]:
    center = (start + end) // 2
    window_start = center - width // 2
    return window_start, window_start + width


def prepare_regions(
    bed_path: str,
    chrom_sizes: dict[str, int],
    sequence_length: int,
    target_length: int,
    max_regions: int,
) -> list[dict]:
    retained = []
    for original_index, chromosome, start, end in read_bed(bed_path):
        sequence_start, sequence_end = centered_window(start, end, sequence_length)
        target_start, target_end = centered_window(start, end, target_length)
        if (
            chromosome not in chrom_sizes
            or sequence_start < 0
            or sequence_end > chrom_sizes[chromosome]
        ):
            continue
        retained.append({
            "original_interval_idx": original_index,
            "chrom": chromosome,
            "sequence_start": sequence_start,
            "sequence_end": sequence_end,
            "target_start": target_start,
            "target_end": target_end,
        })
    if max_regions and len(retained) > max_regions:
        indices = np.random.default_rng(42).choice(
            len(retained), max_regions, replace=False,
        )
        retained = [retained[index] for index in indices]
    return retained


def predict_one_model(
    model_path: str,
    params_path: str,
    genome: str,
    regions: list[dict],
    batch_size: int,
) -> np.ndarray:
    with open(params_path) as handle:
        params_model = json.load(handle)["model"]
    model = seqnn.SeqNN(params_model)
    model.restore(model_path)
    fasta = pysam.Fastafile(genome)
    outputs = []
    batch = []

    def flush() -> None:
        if not batch:
            return
        values = model(np.stack(batch).astype(np.float32), dtype="float32")
        outputs.extend(np.asarray(values, dtype=np.float32))
        batch.clear()

    for region in tqdm(regions, desc=Path(model_path).parent.name):
        sequence = fasta.fetch(
            region["chrom"], region["sequence_start"], region["sequence_end"],
        )
        batch.append(dna.dna_1hot(sequence, seq_len=len(sequence)))
        if len(batch) >= batch_size:
            flush()
    flush()
    fasta.close()
    del model
    tf.keras.backend.clear_session()
    return np.stack(outputs)


def read_target_views(
    bigwig_paths: list[str], regions: list[dict], pool_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read 1-bp targets and construct sum-pooled native-resolution targets."""
    bigwigs = [pyBigWig.open(path) for path in bigwig_paths]
    one_bp_targets = []
    pooled_targets = []
    try:
        for region in tqdm(regions, desc="BigWig targets"):
            tracks = []
            for bigwig in bigwigs:
                values = np.asarray(bigwig.values(
                    region["chrom"], region["target_start"],
                    region["target_end"], numpy=True,
                ), dtype=np.float32)
                tracks.append(np.nan_to_num(
                    values, nan=0.0, posinf=0.0, neginf=0.0,
                ))
            one_bp = np.stack(tracks, axis=-1)
            one_bp_targets.append(one_bp)
            pooled_targets.append(one_bp.reshape(
                one_bp.shape[0] // pool_width, pool_width, one_bp.shape[1],
            ).sum(axis=1, dtype=np.float32))
    finally:
        for bigwig in bigwigs:
            bigwig.close()
    return np.stack(one_bp_targets), np.stack(pooled_targets)


@dataclass
class PearsonState:
    xy_sum: np.ndarray
    x_sum: np.ndarray
    xx_sum: np.ndarray
    y_sum: np.ndarray
    yy_sum: np.ndarray
    count: np.ndarray


def accumulated_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy port of AlphaGenome's accumulated Pearson metric.

    Sums are accumulated in float64: at 1 bp there are ~1e9 values per track,
    and float32 sums cancel catastrophically (negative variance -> NaN).
    """
    axis = (-2, -3)
    state = PearsonState(
        np.sum(x * y, axis=axis, dtype=np.float64),
        np.sum(x, axis=axis, dtype=np.float64),
        np.sum(np.square(x), axis=axis, dtype=np.float64),
        np.sum(y, axis=axis, dtype=np.float64),
        np.sum(np.square(y), axis=axis, dtype=np.float64),
        np.full(x.shape[-1], x.shape[0] * x.shape[1], dtype=np.float64),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = state.x_sum / state.count
        y_mean = state.y_sum / state.count
        covariance = state.xy_sum - state.count * x_mean * y_mean
        x_var = state.xx_sum - state.count * np.square(x_mean)
        y_var = state.yy_sum - state.count * np.square(y_mean)
        denominator = np.sqrt(x_var) * np.sqrt(y_var)
        return covariance / (denominator + np.finfo(denominator.dtype).eps)


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def compute_metrics(
    predictions: np.ndarray, targets: np.ndarray, profile_transform: str,
) -> dict:
    profile_predictions = (
        np.log1p(predictions) if profile_transform == "log1p" else predictions
    )
    profile_targets = np.log1p(targets) if profile_transform == "log1p" else targets
    profile_r = np.asarray([
        pearson_or_nan(prediction, target)
        for prediction, target in zip(profile_predictions, profile_targets)
    ])
    eps = 1e-8
    p = targets / (targets.sum(axis=1, keepdims=True) + eps)
    q = predictions / (predictions.sum(axis=1, keepdims=True) + eps)
    m = 0.5 * (p + q)
    js_by_track = 0.5 * (
        np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
        + np.sum(q * np.log((q + eps) / (m + eps)), axis=1)
    )
    js_divergence = js_by_track.mean(axis=1)
    track_r = accumulated_pearson(profile_targets, profile_predictions)
    return {
        "profile_transform": profile_transform,
        "profile_pearson_r_all": profile_r,
        "profile_pearson_r_mean": float(np.nanmean(profile_r)),
        "profile_pearson_r_median": float(np.nanmedian(profile_r)),
        "track_pearson_r_accumulated_all": track_r,
        "track_pearson_r_accumulated_mean": float(np.nanmean(track_r)),
        "count_pearson_r_raw": pearson_or_nan(
            predictions.sum(axis=1), targets.sum(axis=1),
        ),
        "count_pearson_r_log1p": pearson_or_nan(
            np.log1p(predictions.sum(axis=1)), np.log1p(targets.sum(axis=1)),
        ),
        "js_divergence_all": js_divergence,
        "js_divergence_mean": float(np.nanmean(js_divergence)),
        "js_distance_all": np.sqrt(np.maximum(js_divergence, 0)),
        "js_distance_mean": float(np.nanmean(np.sqrt(np.maximum(js_divergence, 0)))),
        "mse": float(np.mean(np.square(predictions - targets))),
        "n_regions": int(predictions.shape[0]),
        "n_positions_per_region": int(predictions.shape[1]),
        "n_tracks": int(predictions.shape[2]),
    }


def load_gtf(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    attribute_pattern = re.compile(r'(\S+) "([^"]*)"')
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] not in {"gene", "transcript", "exon"}:
                continue
            attributes = dict(attribute_pattern.findall(fields[8]))
            rows.append({
                "Chromosome": fields[0], "Start": int(fields[3]) - 1,
                "End": int(fields[4]), "Strand": fields[6],
                "Feature": fields[2], "gene_id": attributes.get("gene_id"),
                "gene_name": attributes.get("gene_name", attributes.get("gene_id")),
                "gene_type": attributes.get("gene_type", attributes.get("gene_biotype")),
                "transcript_id": attributes.get("transcript_id"),
            })
    return pd.DataFrame(rows)


def compute_gene_rows(
    gtf: pd.DataFrame,
    positions: list[tuple[str, int, int]],
    original_indices: list[int],
    prediction_views: dict[int, np.ndarray],
    target_views: dict[int, np.ndarray],
    samples: list[str],
    resolution: int,
) -> pd.DataFrame:
    test_intervals = pd.DataFrame({
        "original_interval_idx": original_indices,
    })
    return collect_gene_expression_rows(
        gtf=gtf,
        test_intervals=test_intervals,
        positions=positions,
        prediction_views=prediction_views,
        target_views=target_views,
        bin_sizes=(1, resolution),
        sample_names=samples,
    )


def clean_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    if (output_dir / "summary.json").exists() and not args.force:
        log.info("Skipping completed evaluation: %s", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.params) not in {1, len(args.models)}:
        raise ValueError("--params must contain one path or match --models")
    params_paths = args.params * len(args.models) if len(args.params) == 1 else args.params
    targets_metadata = load_targets(args.targets)
    if len(targets_metadata) != len(args.bigwig):
        raise ValueError("--bigwig count must match targets.txt rows")
    if args.assay == "rna":
        if len(args.bigwig) % 2 or not args.samples or not args.gtf:
            raise ValueError("RNA requires paired --bigwig tracks, --samples, and --gtf")
        if len(args.samples) * 2 != len(args.bigwig):
            raise ValueError("RNA expects two interleaved strand tracks per sample")
        validate_gencode_v46_path(args.gtf)

    with open(params_paths[0]) as handle:
        sequence_length = int(json.load(handle)["model"]["seq_length"])
    target_length = sequence_length - 2 * args.crop_bp
    if target_length != 196_608:
        raise ValueError(f"Expected centered 196,608-bp target, found {target_length}")
    if target_length % args.pool_width:
        raise ValueError("Target length must be divisible by pool width")

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    fasta = pysam.Fastafile(args.genome)
    chrom_sizes = dict(zip(fasta.references, fasta.lengths))
    fasta.close()
    regions = prepare_regions(
        args.test_bed, chrom_sizes, sequence_length, target_length, args.max_regions,
    )

    ensemble_sum = None
    for model_path, params_path in zip(args.models, params_paths):
        transformed = predict_one_model(
            model_path, params_path, args.genome, regions, args.batch_size,
        )
        raw = untransform_predictions(transformed, targets_metadata)
        ensemble_sum = raw.astype(np.float64) if ensemble_sum is None else ensemble_sum + raw
    predictions_32bp = (ensemble_sum / len(args.models)).astype(np.float32)
    targets_1bp, targets_32bp = read_target_views(
        args.bigwig, regions, args.pool_width,
    )
    if predictions_32bp.shape != targets_32bp.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{predictions_32bp.shape} vs {targets_32bp.shape}"
        )
    # Untransformed Borzoi outputs are sums over each native bin. Convert to
    # per-base coverage before repeating so integrated signal is preserved.
    predictions_1bp = np.repeat(
        predictions_32bp / args.pool_width,
        args.pool_width,
        axis=1,
    )
    if predictions_1bp.shape != targets_1bp.shape:
        raise ValueError(
            "Upsampled prediction/1-bp target shape mismatch: "
            f"{predictions_1bp.shape} vs {targets_1bp.shape}"
        )

    profile_transform = "log1p" if args.assay == "rna" else "none"
    metrics_by_bin = {
        1: compute_metrics(predictions_1bp, targets_1bp, profile_transform),
        args.pool_width: compute_metrics(
            predictions_32bp, targets_32bp, profile_transform,
        ),
    }
    positions = [
        (region["chrom"], region["target_start"], region["target_end"])
        for region in regions
    ]
    gene_metrics = None
    if args.assay == "rna":
        gene_prediction_views = {
            1: predictions_1bp,
            args.pool_width: predictions_32bp / args.pool_width,
        }
        gene_target_views = {
            1: targets_1bp,
            args.pool_width: targets_32bp / args.pool_width,
        }
        gene_rows = compute_gene_rows(
            load_gtf(args.gtf), positions,
            [region["original_interval_idx"] for region in regions],
            gene_prediction_views, gene_target_views,
            args.samples, args.pool_width,
        )
        if gene_rows.empty:
            raise ValueError(
                "No genes passed the >=50% unique-exon criterion; check the "
                "GENCODE v46 build and chromosome naming"
            )
        gene_metrics_by_bin, per_track, per_gene = summarize_gene_expression(
            gene_rows, (1, args.pool_width),
        )
        gene_metrics = gene_metrics_by_bin
        gene_rows.to_csv(output_dir / "gene_expression_per_gene.csv", index=False)
        per_track.to_csv(output_dir / "gene_expression_metrics_per_track.csv", index=False)
        per_gene.to_csv(output_dir / "gene_expression_metrics_per_gene.csv", index=False)

    with open(output_dir / "metrics_per_region.csv", "w", newline="") as handle:
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
            "assay": args.assay, "models": args.models,
            "ensemble": "mean_raw_predictions_across_replicate_trunks",
            "n_ensemble_members": len(args.models), "test_bed": args.test_bed,
            "sequence_length_bp": sequence_length,
            "score_window_bp": target_length,
            "native_resolution_bp": args.pool_width,
            "metric_bin_sizes_bp": [1, args.pool_width],
            "one_bp_prediction_view": (
                "repeat_each_unscaled_native_bin_sum_divided_by_pool_width"
            ),
            "native_prediction_view": "unscaled_bin_sums",
            "targets": args.targets, "bigwig": args.bigwig,
            "gene_annotation": "GENCODE_v46_required" if args.assay == "rna" else None,
            "gene_selection": (
                "at_least_50pct_unique_exon_bp_in_test_interval"
                if args.assay == "rna" else None
            ),
            "gene_mask_aggregation": (
                "union_all_annotated_exons_then_mean"
                if args.assay == "rna" else None
            ),
            "gene_resolution_conversion": (
                "exact_exon_bp_weighting_of_upsampled_unscaled_borzoi_bins"
                if args.assay == "rna" else None
            ),
        },
        "metrics": metrics_by_bin,
        "gene_expression_metrics": gene_metrics,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(clean_json(summary), handle, indent=2)
    if args.save_predictions:
        np.save(
            output_dir / "ensemble_predictions_1bp.npy",
            predictions_1bp.astype(np.float16),
        )
        np.save(output_dir / "targets_1bp.npy", targets_1bp.astype(np.float16))
        np.save(
            output_dir / "ensemble_predictions_32bp.npy",
            predictions_32bp.astype(np.float16),
        )
        np.save(output_dir / "targets_32bp.npy", targets_32bp.astype(np.float16))
    log.info("Wrote %s ensemble evaluation to %s", args.assay, output_dir)


if __name__ == "__main__":
    main()
