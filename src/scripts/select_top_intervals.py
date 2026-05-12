"""Select top N genomic intervals ranked by splice junction activity.

Reads a BED3 file and one or more STAR SJ.out.tab files, scores each interval
by the total uniquely-mapped junction reads from junctions fully contained
within it (summed across all SJ files), and writes the top N intervals.

Optionally mixes in --n-random randomly sampled intervals (disjoint from the
top-N selection, seeded by --seed for reproducibility).
"""

import argparse

import numpy as np
import pandas as pd
import pyranges as pr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bed", required=True, help="Input BED3 file (intervals to score)")
    p.add_argument("--star-junctions", nargs="+", required=True,
                   help="STAR SJ.out.tab files (one or more samples)")
    p.add_argument("--n", type=int, required=True,
                   help="Number of top intervals to select")
    p.add_argument("--n-random", type=int, default=0,
                   help="Additional randomly sampled intervals (disjoint from top-N)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --n-random sampling")
    p.add_argument("--output", required=True, help="Output BED3 file")
    return p.parse_args()


def main():
    args = parse_args()

    intervals = pd.read_csv(
        args.bed, sep="\t", header=None, names=["chrom", "start", "end"],
    )

    gr_intervals = pr.PyRanges(
        intervals.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End"})
    )

    per_sample_rows = []
    for i, sj_path in enumerate(args.star_junctions):
        df = pd.read_csv(
            sj_path, sep="\t", header=None,
            names=["chrom", "intron_start", "intron_end", "strand", "motif",
                   "annotated", "n_unique", "n_multi", "max_overhang"],
        )
        df = df[df["n_unique"] > 0]

        gr_junctions = pr.PyRanges(pd.DataFrame({
            "Chromosome": df["chrom"].values,
            "Start":      (df["intron_start"] - 1).values,
            "End":        df["intron_end"].values,
            "n_unique":   df["n_unique"].values,
        }))

        # Overlap join then filter to strict containment (junction fully inside interval)
        joined_df = gr_intervals.join(gr_junctions, how=None).df
        contained = joined_df[
            (joined_df["Start_b"] >= joined_df["Start"]) &
            (joined_df["End_b"]   <= joined_df["End"])
        ]

        sample_agg = (
            contained
            .groupby(["Chromosome", "Start", "End"])
            .agg(total_reads=("n_unique", "sum"), n_junctions=("n_unique", "count"))
            .reset_index()
        )
        sample_agg["sample"] = i
        per_sample_rows.append(sample_agg)

    per_sample = pd.concat(per_sample_rows, ignore_index=True)

    agg = (
        per_sample
        .groupby(["Chromosome", "Start", "End"])
        .agg(total_reads=("total_reads", "mean"), n_junctions=("n_junctions", "mean"))
        .reset_index()
        .rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end"})
    )

    scored = intervals.merge(agg, on=["chrom", "start", "end"], how="left").fillna(0)

    top = (
        scored
        .sort_values(["total_reads", "n_junctions"], ascending=False)
        .head(args.n)
    )

    if args.n_random > 0:
        top_idx = set(top.index)
        remaining = scored[~scored.index.isin(top_idx)]
        n_sample = min(args.n_random, len(remaining))
        rng = np.random.default_rng(args.seed)
        random_rows = remaining.iloc[rng.choice(len(remaining), size=n_sample, replace=False)]
        top = pd.concat([top, random_rows], ignore_index=True)

    top[["chrom", "start", "end"]].to_csv(
        args.output, sep="\t", header=False, index=False,
    )


if __name__ == "__main__":
    main()
