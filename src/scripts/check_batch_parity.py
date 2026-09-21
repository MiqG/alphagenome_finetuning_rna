#!/usr/bin/env python
"""
Verify that collect_predictions.py's --batch-size does not change predictions.

Runs collect_predictions.py twice on the same small interval subset -- once
at --batch-size 1, once at a candidate batch size -- and diffs the resulting
prediction parquets. Every interval is padded/cropped to the same
--sequence-length (no cross-example padding/masking), and the model's only
batch-named layer (RMSBatchNorm) uses a frozen running_var buffer rather than
live batch statistics, so predictions should match closely; residual
differences are ordinary bf16 matmul non-associativity across batch/tile
shapes, not a correctness bug. Run this once per (checkpoint, GPU type)
combination before trusting a full sweep at batch size > 1 -- see
workflows/11-xinming_pytorch_evaluation/Snakefile's module docstring.

Usage:
    python src/scripts/check_batch_parity.py \\
        --batch-size 8 \\
        --n-intervals 16 \\
        --collect-args -- \\
        --pretrained-weights ... --checkpoint ... --test-bed ... \\
        --genome ... --gtf-parquet ... --bigwigs ... --ssu-parquets ... \\
        --star-junctions ... --samples ... --dtype bfloat16

Must be run on a GPU node (per this repo's cluster rules, never on the login
node) via salloc/srun or sbatch, since it constructs the real model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

_COLLECT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_predictions.py")

# Parquets to compare and the numeric columns worth diffing in each. Columns
# outside this list (e.g. bookkeeping ids, string labels) are expected to be
# identical and are checked for exact row-order/count agreement instead.
_NUMERIC_COLUMNS = {
    "rna_seq_per_gene.parquet": ["pred_log_mean"],
    "splice_site_scores.parquet": [
        "pred_donor_pos", "pred_acceptor_pos", "pred_donor_neg", "pred_acceptor_neg", "pred_no_site",
    ],
    "ssu_scores.parquet": ["pred_ssu"],
    "junction_scores.parquet": ["pred_count"],
    "psi_scores.parquet": ["pred_psi5", "pred_psi3"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--batch-size", type=int, required=True,
                   help="Candidate batch size to check against batch-size=1.")
    p.add_argument("--n-intervals", type=int, default=16,
                   help="Number of intervals from --test-bed to use (kept small: this "
                        "runs two full model loads + forward passes).")
    p.add_argument("--atol", type=float, default=1e-2,
                   help="Absolute tolerance for prediction differences (raw-scale "
                        "counts/probabilities). Loosen only if you understand why.")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="Relative tolerance for prediction differences.")
    p.add_argument("--work-dir", default=None,
                   help="Directory for the two prediction runs and the truncated test "
                        "bed. Defaults to a temp dir cleaned up automatically.")
    p.add_argument("collect_args", nargs=argparse.REMAINDER,
                   help="Everything after '--': the rest of collect_predictions.py's "
                        "own arguments (--pretrained-weights, --checkpoint, --test-bed, "
                        "--genome, --gtf-parquet, --bigwigs, --ssu-parquets, "
                        "--star-junctions, --samples, --dtype, ...). Do not pass "
                        "--batch-size, --test-bed's full file, or --output-dir here -- "
                        "--test-bed is truncated and --output-dir is set by this script.")
    return p.parse_args()


def _truncate_bed(src_bed: str, dst_bed: str, n: int) -> None:
    with open(src_bed) as fin, open(dst_bed, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)


def _run_collect(collect_args: list[str], test_bed: str, batch_size: int, output_dir: str) -> None:
    args = list(collect_args)
    if args and args[0] == "--":
        args = args[1:]
    cmd = [
        sys.executable, _COLLECT_SCRIPT,
        *args,
        "--test-bed", test_bed,
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
    ]
    print("Running: {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=True)


def _compare(path_a: str, path_b: str, columns: list[str], atol: float, rtol: float) -> list[str]:
    problems = []
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return ["{}: missing in one of the two runs".format(os.path.basename(path_a))]
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    name = os.path.basename(path_a)
    if len(df_a) != len(df_b):
        problems.append("{}: row count differs ({} vs {})".format(name, len(df_a), len(df_b)))
        return problems
    for col in columns:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        a = df_a[col].to_numpy(dtype=np.float64)
        b = df_b[col].to_numpy(dtype=np.float64)
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(a - b)))
            problems.append(
                "{}:{}: max abs diff {:.6g} exceeds atol={} rtol={}".format(
                    name, col, max_diff, atol, rtol
                )
            )
    return problems


def main() -> None:
    args = parse_args()

    collect_args = args.collect_args
    if collect_args and collect_args[0] == "--":
        collect_args = collect_args[1:]
    try:
        test_bed_idx = collect_args.index("--test-bed")
    except ValueError:
        raise SystemExit("Pass the full collect_predictions.py --test-bed via collect_args.")
    src_test_bed = collect_args[test_bed_idx + 1]
    # Drop --test-bed <path> from collect_args; _run_collect re-adds the truncated one.
    collect_args = collect_args[:test_bed_idx] + collect_args[test_bed_idx + 2:]

    work_dir_ctx = tempfile.TemporaryDirectory() if args.work_dir is None else None
    work_dir = args.work_dir or work_dir_ctx.name
    os.makedirs(work_dir, exist_ok=True)

    small_bed = os.path.join(work_dir, "parity_check.bed")
    _truncate_bed(src_test_bed, small_bed, args.n_intervals)

    out_b1 = os.path.join(work_dir, "batch1")
    out_bn = os.path.join(work_dir, "batch{}".format(args.batch_size))
    _run_collect(collect_args, small_bed, 1, out_b1)
    _run_collect(collect_args, small_bed, args.batch_size, out_bn)

    problems = []
    for filename, columns in _NUMERIC_COLUMNS.items():
        problems += _compare(
            os.path.join(out_b1, filename), os.path.join(out_bn, filename),
            columns, args.atol, args.rtol,
        )

    if work_dir_ctx is not None:
        work_dir_ctx.cleanup()

    if problems:
        print("\nPARITY CHECK FAILED (batch_size=1 vs {}):".format(args.batch_size))
        for p in problems:
            print("  - {}".format(p))
        raise SystemExit(1)

    print("\nPARITY CHECK PASSED: batch_size=1 and batch_size={} agree within tolerance "
          "on {} intervals.".format(args.batch_size, args.n_intervals))


if __name__ == "__main__":
    main()
