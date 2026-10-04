#!/usr/bin/env python
"""
Collect fine-tuned Borzoi ensemble predictions on held-out test intervals.

Step 1 of 2 (the Borzoi counterpart of collect_predictions.py). Runs GPU
inference for each replicate trunk, undoes the Borzoi target transform
(soft clip, sqrt, scale) and averages the raw predictions over trunks.
Metrics are computed separately by compute_metrics_borzoi.py, so different
metrics can be tried without repeating inference.

All model/target logic is imported from the vendored, unmodified upstream
script (src/borzoi_eval/evaluate_borzoi_checkpoint.py), so predictions are
identical to what that script's own evaluation would produce.

Outputs (in --output-dir):
  predictions_32bp.npy  — ensemble mean raw predictions, float32,
                          (n_regions, target_length / pool_width, n_tracks).
                          Each value is the sum of per-base coverage in a bin.
  regions.parquet       — retained regions (chrom, sequence/target windows,
                          original BED index), row-aligned with the array
  run_info.json         — models, targets, settings, provenance fingerprint
  .progress/trunk_<i>.npy — per-trunk raw predictions, kept for resume and for
                          per-trunk analyses (not declared workflow outputs)

Usage:
    python src/scripts/collect_predictions_borzoi.py \\
        --models results/finetuning/borzoi/trunk_r{0,1,2,3}/model_best.mergeW.h5 \\
        --params results/finetuning/borzoi/trunk_r0/params.json \\
        --targets support/borzoi/targets.txt \\
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \\
        --test-bed data/prep/finetuning/alphagenome/FOLD_1/test.bed \\
        --output-dir results/evaluation/borzoi/full/test/predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "borzoi_eval"))

import evaluation_provenance  # noqa: E402
import evaluate_borzoi_checkpoint as ebc  # noqa: E402

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--params", nargs="+", required=True,
                   help="One params.json reused by all models, or one per model")
    p.add_argument("--targets", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--test-bed", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--crop-bp", type=int, default=163_840)
    p.add_argument("--pool-width", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-regions", type=int, default=0)
    p.add_argument("--mixed-precision", action="store_true")
    return p.parse_args()


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if len(args.params) not in {1, len(args.models)}:
        raise ValueError("--params must contain one path or match --models")
    params_paths = args.params * len(args.models) if len(args.params) == 1 else args.params

    with open(params_paths[0]) as handle:
        sequence_length = int(json.load(handle)["model"]["seq_length"])
    target_length = sequence_length - 2 * args.crop_bp
    if target_length != 196_608:
        raise ValueError("Expected centered 196,608-bp target, found {}".format(target_length))
    if target_length % args.pool_width:
        raise ValueError("Target length must be divisible by pool width")

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
        code_paths=[
            __file__,
            os.path.join(HERE, "evaluation_provenance.py"),
            ebc.__file__,
        ],
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
    n_bins = target_length // args.pool_width
    if predictions_32bp.shape != (len(regions), n_bins, len(targets_metadata)):
        raise ValueError(
            "Unexpected prediction shape {} (expected {})".format(
                predictions_32bp.shape, (len(regions), n_bins, len(targets_metadata)),
            )
        )

    # Final artifacts are written last and atomically; run_info.json is the
    # completion marker.
    _atomic_save_npy(os.path.join(args.output_dir, "predictions_32bp.npy"), predictions_32bp)
    regions_df = pd.DataFrame(regions)
    regions_tmp = os.path.join(args.output_dir, "regions.parquet.tmp")
    regions_df.to_parquet(regions_tmp, index=False)
    os.replace(regions_tmp, os.path.join(args.output_dir, "regions.parquet"))
    run_info = {
        "models": args.models,
        "params": params_paths,
        "targets": args.targets,
        "test_bed": args.test_bed,
        "genome": args.genome,
        "n_regions": len(regions),
        "n_tracks": int(len(targets_metadata)),
        "n_ensemble_members": len(args.models),
        "sequence_length_bp": sequence_length,
        "target_length_bp": target_length,
        "pool_width_bp": args.pool_width,
        "prediction_view": "ensemble_mean_of_untransformed_per_bin_sums",
        "fingerprint_sha256": fingerprint["fingerprint_sha256"],
    }
    run_info_tmp = os.path.join(args.output_dir, "run_info.json.tmp")
    with open(run_info_tmp, "w") as handle:
        json.dump(run_info, handle, indent=2)
    os.replace(run_info_tmp, os.path.join(args.output_dir, "run_info.json"))
    log.info("Wrote predictions for %d regions to %s", len(regions), args.output_dir)


if __name__ == "__main__":
    main()
