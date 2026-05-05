"""Select one genomic interval at a given splice-junction density tier.

Reads a BED3 file and one or more STAR SJ.out.tab files, scores each interval
by total uniquely-mapped junction reads (summed across all SJ files), then
picks one interval according to --density:

  high   — highest ranked interval (most junctions)
  medium — interval at the median rank
  low    — lowest-ranked interval that still has at least one junction

Optionally writes the full ranked table (all intervals with scores) to
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
                   help="Optional: write full ranked interval table (tsv.gz)")
    return p.parse_args()


def main():
    args = parse_args()

    intervals = pd.read_csv(
        args.bed, sep="\t", header=None, names=["chrom", "start", "end"],
    )

    sj_frames = []
    for sj_path in args.star_junctions:
        df = pd.read_csv(
            sj_path, sep="\t", header=None,
            names=["chrom", "intron_start", "intron_end", "strand", "motif",
                   "annotated", "n_unique", "n_multi", "max_overhang"],
        )
        sj_frames.append(df)
    junctions = pd.concat(sj_frames, ignore_index=True)

    junctions["j_start"] = junctions["intron_start"] - 1
    junctions["j_end"]   = junctions["intron_end"]

    gr_intervals = pr.PyRanges(
        intervals.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End"})
    )
    gr_junctions = pr.PyRanges(pd.DataFrame({
        "Chromosome": junctions["chrom"].values,
        "Start":      junctions["j_start"].values,
        "End":        junctions["j_end"].values,
        "n_unique":   junctions["n_unique"].values,
    }))

    joined_df = gr_intervals.join(gr_junctions, how=None).df
    contained = joined_df[
        (joined_df["Start_b"] >= joined_df["Start"]) &
        (joined_df["End_b"]   <= joined_df["End"])
    ]

    agg = (
        contained
        .groupby(["Chromosome", "Start", "End"])
        .agg(total_reads=("n_unique", "sum"), n_junctions=("n_unique", "count"))
        .reset_index()
        .rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end"})
    )

    scored = (
        intervals.merge(agg, on=["chrom", "start", "end"], how="left")
        .fillna(0)
        .sort_values(["total_reads", "n_junctions"], ascending=False)
        .reset_index(drop=True)
    )
    scored.insert(0, "rank", range(1, len(scored) + 1))

    if args.output_ranking is not None:
        scored.to_csv(args.output_ranking, sep="\t", index=False, compression="gzip")

    if args.output is not None:
        if args.density == "high":
            selected = scored.iloc[[0]]
        elif args.density == "medium":
            selected = scored.iloc[[len(scored) // 2]]
        else:  # low: least-active interval that still has at least one junction
            with_junctions = scored[scored["n_junctions"] > 0]
            if with_junctions.empty:
                raise ValueError("No intervals with junctions found for 'low' tier")
            selected = with_junctions.iloc[[-1]]

        selected[["chrom", "start", "end"]].to_csv(
            args.output, sep="\t", header=False, index=False,
        )


if __name__ == "__main__":
    main()
