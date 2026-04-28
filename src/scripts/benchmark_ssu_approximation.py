#!/usr/bin/env python
"""Benchmark junction-only SSU approximation against BAM-derived ground truth.

Computes, for each splice site in a genomic interval:
  - SSU full  = α / (α + β1 + β2)   [SpliSER definition, requires BAM]
  - SSU approx = α / (α + β2)        [junction-only, no BAM needed]

where:
  α  = split reads using this site (from SJ.out.tab)
  β1 = reads spanning the site continuously without splicing (from BAM)
  β2 = reads using a competing site for the same partner (from junction data)

Outputs:
  ssu_comparison.parquet  — one row per splice site
  ssu_scatterplot.pdf     — 2×2 scatter: full vs approx, by strand and role

Usage:
    python src/scripts/benchmark_ssu_approximation.py \
        --bam data/raw/ENA/sf3b1mut/STAR/SRR17111301/second_pass.Aligned.sortedByCoord.out.filtered.bam \
        --junctions data/raw/ENA/sf3b1mut/STAR/SRR17111301/second_pass.SJ.out.tab \
        --interval chr7:117480025-117668665 \
        --output-dir results/sanity_checks/ssu_benchmark/SRR17111301
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Allow importing from the alphagenome-pytorch package in the repo
sys.path.insert(
    0,
    str(Path(__file__).parents[2] / "src" / "alphagenome-pytorch" / "src"),
)
from alphagenome_pytorch.extensions.finetuning.star_junctions import read_star_junctions


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark junction-only SSU approximation vs BAM ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bam", required=True,
                   help="Coordinate-sorted, indexed BAM file (STAR-aligned)")
    p.add_argument("--junctions", required=True,
                   help="STAR SJ.out.tab file")
    p.add_argument("--interval", required=True,
                   help='"chr1:1000000-2000000" (1-based inclusive) or path to BED file '
                        "(0-based half-open, chrom/start/end columns)")
    p.add_argument("--output-dir", default=".",
                   help="Directory to write outputs")
    p.add_argument("--min-unique-reads", type=int, default=1,
                   help="Minimum n_uniquely_mapped_reads to retain a junction")
    p.add_argument("--mapq", type=int, default=30,
                   help="Minimum MAPQ for β1 reads")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Interval parsing
# ------------------------------------------------------------------ #

def parse_interval(s: str) -> list[tuple[str, int, int]]:
    """Return list of (chrom, start_0based, end_0based_exclusive).

    If s is a path to a .bed file, read all rows (chrom, start, end — 0-based half-open).
    Otherwise parse "chrom:start-end" as 1-based inclusive UCSC coords.
    """
    if os.path.isfile(s):
        df = pd.read_csv(s, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"])
        return [(row.chrom, int(row.start), int(row.end)) for _, row in df.iterrows()]

    # UCSC-style: chr1:1000000-2000000
    chrom, coords = s.split(":")
    start_str, end_str = coords.split("-")
    start_0 = int(start_str) - 1   # 1-based inclusive → 0-based
    end_0   = int(end_str)          # 1-based inclusive → 0-based exclusive
    return [(chrom, start_0, end_0)]


# ------------------------------------------------------------------ #
# Step 1: load and filter junctions
# ------------------------------------------------------------------ #

def load_all_junctions(path: str, min_unique_reads: int) -> pd.DataFrame:
    """Read and quality-filter the full SJ.out.tab once.

    Called once per script run; interval subsetting is done separately
    so the file is not re-read for each interval in a multi-interval BED.
    """
    junctions = read_star_junctions(path)
    junctions = junctions.loc[
        (junctions["n_uniquely_mapped_reads"] >= min_unique_reads)
        & (junctions["strand"].isin(["+", "-"]))
    ].copy()
    # Derive 1-based exon coordinates (matching datasets.py convention)
    junctions["exon_start"] = junctions["intron_start"] - 1   # donor: last exon base
    junctions["exon_end"]   = junctions["intron_end"]   + 1   # acceptor: first exon base
    junctions["count"]      = junctions["n_uniquely_mapped_reads"]
    return junctions


def filter_junctions_to_interval(
    junctions: pd.DataFrame,
    chrom: str,
    start_0: int,
    end_0: int,
) -> pd.DataFrame:
    """Subset pre-loaded junctions to a single interval."""
    mask = (
        (junctions["chrom"] == chrom)
        & (junctions["intron_start"] > start_0)   # 1-based > 0-based start ≡ ≥ start+1
        & (junctions["intron_end"] <= end_0)
    )
    return junctions.loc[mask].copy()


# ------------------------------------------------------------------ #
# Steps 2–3: α and β2 from junction data
# ------------------------------------------------------------------ #

def compute_alpha_beta2(
    junctions: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    """Return (donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2).

    All dicts keyed by (chrom, 1-based position, strand).

    β2(D) = Σ_{A: D→A} acceptor_total(A) − α(D)
    β2(A) = Σ_{D: D→A} donor_total(D) − α(A)

    Fully vectorized: no iterrows().
    """
    # Site-level totals (α)
    donor_alpha_s = (
        junctions.groupby(["chrom", "exon_start", "strand"])["count"].sum()
        .rename("donor_total")
    )
    acceptor_alpha_s = (
        junctions.groupby(["chrom", "exon_end", "strand"])["count"].sum()
        .rename("acceptor_total")
    )

    # Join site totals back onto each junction row (vectorized lookup)
    j = junctions.join(acceptor_alpha_s, on=["chrom", "exon_end", "strand"])
    j = j.join(donor_alpha_s,   on=["chrom", "exon_start", "strand"])

    # β2 per donor: Σ_A acceptor_total(A) for all A reachable from D, minus α(D)
    donor_denom    = j.groupby(["chrom", "exon_start", "strand"])["acceptor_total"].sum()
    donor_beta2_s  = (donor_denom - donor_alpha_s).rename("donor_beta2")

    # β2 per acceptor: Σ_D donor_total(D) for all D pointing to A, minus α(A)
    acceptor_denom   = j.groupby(["chrom", "exon_end", "strand"])["donor_total"].sum()
    acceptor_beta2_s = (acceptor_denom - acceptor_alpha_s).rename("acceptor_beta2")

    return (
        donor_alpha_s.to_dict(),
        acceptor_alpha_s.to_dict(),
        donor_beta2_s.to_dict(),
        acceptor_beta2_s.to_dict(),
    )


# ------------------------------------------------------------------ #
# Step 4: β1 from BAM
# ------------------------------------------------------------------ #

def build_beta1_counts(
    bam_path: str,
    chrom: str,
    start_0: int,
    end_0: int,
    site_positions_0based: set[int],
    site_strands: dict[int, set[str]],
    mapq_min: int = 30,
) -> dict[int, int]:
    """Return {0-based position → β1 count}.

    Positions passed in must be intron-side positions (first intron base for
    donors, last intron base for acceptors).  A read contributes β1 when it:
      - overlaps the position
      - has MAPQ >= mapq_min
      - is not a PCR/optical duplicate
      - has no N CIGAR operation covering the position (continuous across it,
        i.e. the read spans into the intron without splicing)
      - has the same strand as the splice site (XS tag must match)

    Reads are fetched once for the whole interval; per-read site lookup uses
    binary search to avoid an O(reads × sites) inner loop.

    Args:
        site_strands: {0-based position → set of strands} for the sites at that position.
    """
    try:
        import pysam
    except ImportError as e:
        raise ImportError("pysam is required for β1 computation") from e

    beta1: dict[int, int] = {pos: 0 for pos in site_positions_0based}
    sites_sorted = sorted(site_positions_0based)

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        for read in bam.fetch(chrom, start_0, end_0):
            if read.is_unmapped or read.is_duplicate:
                continue
            if read.mapping_quality < mapq_min:
                continue
            if not read.cigartuples:
                continue

            # Get read strand from XS tag (set by tagXSstrandedData.awk)
            try:
                read_strand = read.get_tag("XS")
            except KeyError:
                read_strand = None

            # Collect intron intervals (N CIGAR ops)
            introns: list[tuple[int, int]] = []
            ref_pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:                     # N = intron / splice gap
                    introns.append((ref_pos, ref_pos + length))
                    ref_pos += length
                elif op in (0, 2, 7, 8):        # M, D, =, X consume reference
                    ref_pos += length
                # I, S, H, P do not consume reference

            read_start = read.reference_start
            read_end   = read.reference_end     # 0-based exclusive

            # Binary-search to find only site positions overlapped by this read
            lo = bisect.bisect_left(sites_sorted, read_start)
            hi = bisect.bisect_right(sites_sorted, read_end - 1)

            for site_pos in sites_sorted[lo:hi]:
                # Only count reads matching the site's strand
                if read_strand is not None:
                    site_str = site_strands.get(site_pos, set())
                    if read_strand not in site_str:
                        continue

                # β1: read is continuous (not spliced) across site_pos
                if not any(iv_s <= site_pos < iv_e for iv_s, iv_e in introns):
                    beta1[site_pos] += 1
    finally:
        bam.close()

    return beta1


# ------------------------------------------------------------------ #
# Step 5: assemble site table
# ------------------------------------------------------------------ #

def assemble_site_table(
    donor_alpha: dict,
    acceptor_alpha: dict,
    donor_beta2: dict,
    acceptor_beta2: dict,
    beta1_counts: dict[int, int],
) -> pd.DataFrame:
    """Build one row per splice site with α, β1, β2, ssu_full, ssu_approx."""
    rows: list[dict] = []

    for (chrom_j, exon_start, strand), alpha in donor_alpha.items():
        pos_0 = exon_start       # first intron base (0-based) = intron_start - 1
        b1 = beta1_counts.get(pos_0, 0)
        b2 = donor_beta2.get((chrom_j, exon_start, strand), 0)
        d_full   = alpha + b1 + b2
        d_approx = alpha + b2
        rows.append({
            "chrom":      chrom_j,
            "position":   exon_start,
            "strand":     strand,
            "role":       "donor",
            "alpha":      int(alpha),
            "beta1":      int(b1),
            "beta2":      int(b2),
            "ssu_full":   alpha / d_full   if d_full   > 0 else float("nan"),
            "ssu_approx": alpha / d_approx if d_approx > 0 else float("nan"),
        })

    for (chrom_j, exon_end, strand), alpha in acceptor_alpha.items():
        pos_0 = exon_end - 2     # last intron base (0-based) = intron_end - 1
        b1 = beta1_counts.get(pos_0, 0)
        b2 = acceptor_beta2.get((chrom_j, exon_end, strand), 0)
        d_full   = alpha + b1 + b2
        d_approx = alpha + b2
        rows.append({
            "chrom":      chrom_j,
            "position":   exon_end,
            "strand":     strand,
            "role":       "acceptor",
            "alpha":      int(alpha),
            "beta1":      int(b1),
            "beta2":      int(b2),
            "ssu_full":   alpha / d_full   if d_full   > 0 else float("nan"),
            "ssu_approx": alpha / d_approx if d_approx > 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["chrom", "position", "strand", "role"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Step 7: scatterplot
# ------------------------------------------------------------------ #

def plot_scatterplot(df: pd.DataFrame, out_path: Path) -> None:
    """2×2 scatter of ssu_full vs ssu_approx, faceted by strand and role."""
    strands = ["+", "-"]
    roles   = ["donor", "acceptor"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    fig.suptitle("SSU: junction-only approximation vs BAM ground truth", fontsize=13)

    vmax = float(np.log10(df["alpha"].max() + 1)) if not df.empty else 1.0
    sc_ref = None

    for row_i, strand in enumerate(strands):
        for col_i, role in enumerate(roles):
            ax = axes[row_i][col_i]
            sub = df[
                (df["strand"] == strand) & (df["role"] == role)
            ].dropna(subset=["ssu_full", "ssu_approx"])

            if not sub.empty:
                color_vals = np.log10(sub["alpha"].values + 1)
                sc = ax.scatter(
                    sub["ssu_full"],
                    sub["ssu_approx"],
                    c=color_vals,
                    vmin=0,
                    vmax=vmax,
                    cmap="viridis",
                    s=12,
                    alpha=0.6,
                    linewidths=0,
                )
                sc_ref = sc

            # Diagonal reference line
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, alpha=0.5)

            n = len(sub)
            if n >= 3:
                r_p, _ = pearsonr(sub["ssu_full"], sub["ssu_approx"])
                r_s, _ = spearmanr(sub["ssu_full"], sub["ssu_approx"])
                # Flag how many sites have β2=0 (approximation collapses to 1.0)
                n_uncontested = int((sub["beta2"] == 0).sum())
                ax.text(
                    0.05, 0.95,
                    (
                        f"Pearson r  = {r_p:.3f}\n"
                        f"Spearman r = {r_s:.3f}\n"
                        f"N = {n}  (β2=0: {n_uncontested})"
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
                )
            elif n > 0:
                ax.text(0.05, 0.95, f"N = {n} (too few)",
                        transform=ax.transAxes, fontsize=8, va="top")

            ax.set_title(f"strand={strand}  role={role}", fontsize=10)
            ax.set_xlabel("SSU full (BAM)")
            ax.set_ylabel("SSU approx (junction-only)")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

    if sc_ref is not None:
        fig.colorbar(sc_ref, ax=axes, label="log10(α + 1)", shrink=0.55, pad=0.02)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = parse_interval(args.interval)
    print(f"Processing {len(intervals)} interval(s) from {args.interval!r}")

    # Read junctions once for all intervals
    print(f"Loading junctions from {args.junctions!r} …")
    all_junctions = load_all_junctions(args.junctions, args.min_unique_reads)
    print(f"  {len(all_junctions)} junctions after quality filtering")

    all_frames: list[pd.DataFrame] = []

    for chrom, start_0, end_0 in intervals:
        region = f"{chrom}:{start_0 + 1}-{end_0}"
        print(f"\n[{region}]")

        # Step 1: subset to interval
        junctions = filter_junctions_to_interval(all_junctions, chrom, start_0, end_0)
        print(f"  {len(junctions)} junctions in interval")
        if junctions.empty:
            print("  no junctions — skipping interval")
            continue

        # Steps 2–3: α and β2
        donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2 = compute_alpha_beta2(junctions)
        n_donors    = len(donor_alpha)
        n_acceptors = len(acceptor_alpha)
        print(f"  {n_donors} donor sites, {n_acceptors} acceptor sites")

        # Step 4: β1 from BAM
        sites_0based: set[int] = set()
        site_strands: dict[int, set[str]] = {}

        for (chrom_j, pos, strand) in donor_alpha:
            if chrom_j == chrom:
                p0 = pos          # first intron base (0-based) = intron_start - 1
                sites_0based.add(p0)
                site_strands.setdefault(p0, set()).add(strand)
        for (chrom_j, pos, strand) in acceptor_alpha:
            if chrom_j == chrom:
                p0 = pos - 2      # last intron base (0-based) = intron_end - 1
                sites_0based.add(p0)
                site_strands.setdefault(p0, set()).add(strand)

        print(f"  computing β1 for {len(sites_0based)} positions from BAM …")
        beta1_counts = build_beta1_counts(
            args.bam, chrom, start_0, end_0, sites_0based, site_strands, args.mapq
        )
        total_b1 = sum(beta1_counts.values())
        print(f"  total β1 reads counted: {total_b1}")

        # Step 5: assemble
        df_interval = assemble_site_table(
            donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2, beta1_counts
        )
        all_frames.append(df_interval)

    if not all_frames:
        print("\nNo data produced. Check that the interval overlaps junctions in the SJ file.")
        return

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(subset=["chrom", "position", "strand", "role"]).reset_index(drop=True)
    print(f"\nTotal splice sites: {len(df)}")

    # β2=0 summary
    n_b2_zero = int((df["beta2"] == 0).sum())
    print(f"  sites with β2=0 (uncontested, ssu_approx=1.0): {n_b2_zero} ({100*n_b2_zero/len(df):.1f}%)")

    # Step 6: write parquet
    parquet_path = out_dir / "ssu_comparison.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  wrote {parquet_path}")

    # Step 7: scatterplot
    plot_scatterplot(df, out_dir / "ssu_scatterplot.pdf")


if __name__ == "__main__":
    main()
