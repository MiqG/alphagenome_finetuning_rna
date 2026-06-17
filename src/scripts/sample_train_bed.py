"""
Sample a random subset of train.bed matching the size of test.bed.

Usage:
    python src/scripts/sample_train_bed.py \
        --folds-dir data/prep/finetuning/alphagenome \
        --fold FOLD_1 \
        --seed 42
"""

import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds-dir", required=True, help="Base folds directory")
    p.add_argument("--fold", required=True, help="Fold name (e.g. FOLD_1)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    fold_dir = os.path.join(args.folds_dir, args.fold)

    train_bed = os.path.join(fold_dir, "train.bed")
    test_bed  = os.path.join(fold_dir, "test.bed")
    out_bed   = os.path.join(fold_dir, "train_sample.bed")

    train = pd.read_csv(train_bed, sep="\t", header=None)
    test  = pd.read_csv(test_bed,  sep="\t", header=None)

    n = len(test)
    sample = train.sample(n=n, random_state=args.seed)
    sample.to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"Wrote {n} intervals to {out_bed}")


if __name__ == "__main__":
    main()
