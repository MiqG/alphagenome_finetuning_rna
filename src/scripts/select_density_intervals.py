"""Select one genomic interval at a given splice-junction density tier.

Reads a BED3 file and one or more STAR SJ.out.tab files. Each file is treated
as a separate sample. Per-interval metrics (total_reads, avg_reads, n_junctions)
are computed per sample then averaged across samples. Intervals are filtered to
those with n_junctions > 10 and avg_reads > 1, then ranked by avg_reads. The
density tier selects by percentile within the filtered set:

  high   — 5th percentile (most junctions / highest avg coverage)
  medium — 50th percentile (median)
  low    — 95th percentile (sparse but still junction-rich)

Optionally writes the full scored table (all intervals, pre-filter) to
--output-ranking as a tab-separated gzipped file.
"""

import argparse

import pandas as pd
import pyranges as pr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bed", required=True, help="Input BED3 file")
    p.add_argument("--star-junctions", nargs="+", required=True,
                   help="STAR SJ.out.tab files (one or more samples)")
    p.add_argument("--density", required=True, choices=["high", "medium", "low"],
                   help="Density tier to select")
    p.add_argument("--output", default=None, help="Output BED3 file (single interval)")
    p.add_argument("--output-ranking", default=None,
                   help="Optional: write full scored interval table (tsv.gz)")
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

        joined_df = gr_intervals.join(gr_junctions, how=None).df
        contained = joined_df[
            (joined_df["Start_b"] >= joined_df["Start"]) &
            (joined_df["End_b"]   <= joined_df["End"])
        ]

        sample_agg = (
            contained
            .groupby(["Chromosome", "Start", "End"])
            .agg(
                total_reads=("n_unique", "sum"),
                avg_reads=("n_unique", "mean"),
                n_junctions=("n_unique", "count"),
            )
            .reset_index()
        )
        sample_agg["sample"] = i
        per_sample_rows.append(sample_agg)

    per_sample = pd.concat(per_sample_rows, ignore_index=True)

    agg = (
        per_sample
        .groupby(["Chromosome", "Start", "End"])
        .agg(
            total_reads=("total_reads", "mean"),
            avg_reads=("avg_reads", "mean"),
            n_junctions=("n_junctions", "mean"),
        )
        .reset_index()
        .rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end"})
    )

    scored = (
        intervals.merge(agg, on=["chrom", "start", "end"], how="left")
        .fillna(0)
    )

    if args.output_ranking is not None:
        scored.to_csv(args.output_ranking, sep="\t", index=False, compression="gzip")

    if args.output is not None:
        sele = (
            scored
            .query("n_junctions > 10 & avg_reads > 1")
            .sort_values("avg_reads", ascending=False)
            .reset_index(drop=True)
        )

        if sele.empty:
            raise ValueError("No intervals pass the junction filter (n_junctions > 10, avg_reads > 1)")

        percentiles = {"high": 0.05, "medium": 0.5, "low": 0.95}
        idx = int(percentiles[args.density] * (len(sele) - 1))
        selected = sele.iloc[[idx]]

        selected[["chrom", "start", "end"]].to_csv(
            args.output, sep="\t", header=False, index=False,
        )


if __name__ == "__main__":
    main()
