"""
Sanity-check pipeline for finetuning data quality.

Contains:
  benchmark_ssu_approximation — compare junction-only SSU approximation against
      BAM-derived ground-truth SSU (SpliSER definition) for one or more samples.
      Outputs: ssu_comparison.parquet, ssu_scatterplot.pdf

  spliser_process — run SpliSER process on the benchmark chromosome to get
      BAM-derived SSE (α, β1, β2 all from BAM) for direct comparison.
      Outputs: {sample}.SpliSER.tsv

  compare_spliser_ssu — join SpliSER SSE against our ssu_full and ssu_approx.
      Outputs: spliser_comparison.parquet, spliser_vs_ssu_scatterplot.pdf

Run with:
    snakemake -s workflows/sanity_checks.smk --use-conda

To run a specific target:
    snakemake -s workflows/sanity_checks.smk --use-conda \\
        results/sanity_checks/ssu_benchmark/SRR17111301/spliser_vs_ssu_scatterplot.pdf
"""

import os
import pandas as pd

configfile: "config/config.yaml"

DATA_DIR = config["rnaseq"]["sf3b1mut"]["path"]

# ------------------------------------------------------------------ #
# Samples to benchmark
# Two samples that have completed two-pass STAR alignment and BAM prep.
# ------------------------------------------------------------------ #
SSU_BENCHMARK_SAMPLES = ["SRR17111301", "SRR17111311"]

# BED file defining the interval(s) to benchmark.
# Reuses support/overfit.bed so the benchmark runs on the same region as the
# overfitting and oracle workflows.
SSU_BED = os.path.join("support", "overfit.bed")

OUTPUT_DIR = "results/sanity_checks/ssu_benchmark"

# Chromosome(s) to pass to SpliSER process (derived from the BED file).
# SpliSER runs per-chromosome; the comparison script filters to the exact intervals.
_bed_df = pd.read_csv(SSU_BED, sep="\t", header=None, usecols=[0], names=["chrom"])
SSU_CHROMS = sorted(_bed_df["chrom"].unique().tolist())

# ------------------------------------------------------------------ #
# Paths derived from sf3b1mut pipeline outputs
# ------------------------------------------------------------------ #
def bam_path(sample):
    return os.path.join(
        DATA_DIR, "STAR", sample,
        "second_pass.Aligned.sortedByCoord.out.filtered.bam",
    )

def junctions_path(sample):
    return os.path.join(
        DATA_DIR, "STAR", sample,
        "second_pass.SJ.out.tab",
    )


# ------------------------------------------------------------------ #
# Rule: all
# ------------------------------------------------------------------ #
rule all:
    input:
        expand(
            os.path.join(OUTPUT_DIR, "{sample}", "ssu_scatterplot.pdf"),
            sample=SSU_BENCHMARK_SAMPLES,
        ),
        expand(
            os.path.join(OUTPUT_DIR, "{sample}", "ssu_comparison.parquet"),
            sample=SSU_BENCHMARK_SAMPLES,
        ),
        expand(
            os.path.join(OUTPUT_DIR, "{sample}", "spliser_vs_ssu_scatterplot.pdf"),
            sample=SSU_BENCHMARK_SAMPLES,
        ),


# ------------------------------------------------------------------ #
# Rule: benchmark_ssu_approximation
# ------------------------------------------------------------------ #
rule benchmark_ssu_approximation:
    """Compare junction-only SSU approximation to BAM-derived ground truth.

    Runs benchmark_ssu_approximation.py for a single sample over the intervals
    defined in SSU_BED. Outputs a parquet table and a 2x2 scatterplot PDF.
    """
    input:
        bam       = lambda wc: bam_path(wc.sample),
        bam_bai   = lambda wc: bam_path(wc.sample) + ".bai",
        junctions = lambda wc: junctions_path(wc.sample),
        bed       = SSU_BED,
    output:
        parquet = os.path.join(OUTPUT_DIR, "{sample}", "ssu_comparison.parquet"),
        pdf     = os.path.join(OUTPUT_DIR, "{sample}", "ssu_scatterplot.pdf"),
    params:
        script           = "src/scripts/benchmark_ssu_approximation.py",
        output_dir       = lambda wc: os.path.join(OUTPUT_DIR, wc.sample),
        min_unique_reads = 1,
        mapq             = 30,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        mkdir -p {params.output_dir}
        python {params.script} \
            --bam {input.bam} \
            --junctions {input.junctions} \
            --interval {input.bed} \
            --output-dir {params.output_dir} \
            --min-unique-reads {params.min_unique_reads} \
            --mapq {params.mapq}
        """


# ------------------------------------------------------------------ #
# Rule: spliser_process
# ------------------------------------------------------------------ #
rule spliser_process:
    """Run SpliSER process on each benchmark chromosome to get BAM-derived SSE.

    SpliSER computes α, β1, β2 all directly from the BAM (no junction file).
    Restricted to the chromosome(s) in SSU_BED for speed; the comparison script
    further filters to the exact BED intervals.
    """
    input:
        bam     = lambda wc: bam_path(wc.sample),
        bam_bai = lambda wc: bam_path(wc.sample) + ".bai",
    output:
        tsv = os.path.join(OUTPUT_DIR, "{sample}", "{sample}.SpliSER.tsv"),
    params:
        output_prefix = lambda wc: os.path.join(OUTPUT_DIR, wc.sample, wc.sample),
        chroms        = " ".join(SSU_CHROMS),
        output_dir    = lambda wc: os.path.join(OUTPUT_DIR, wc.sample),
    conda:
        "spliser"
    shell:
        """
        mkdir -p {params.output_dir}
        spliser process \
            -B {input.bam} \
            -o {params.output_prefix} \
            -c {params.chroms} \
            --isStranded \
            -s rf
        """


# ------------------------------------------------------------------ #
# Rule: compare_spliser_ssu
# ------------------------------------------------------------------ #
rule compare_spliser_ssu:
    """Join SpliSER SSE against our junction-based SSU estimates.

    Produces a merged parquet and a 2×3 scatter PDF comparing:
      SpliSER SSE vs ssu_full, SpliSER SSE vs ssu_approx, ssu_full vs ssu_approx
    faceted by donor/acceptor role.
    """
    input:
        spliser = os.path.join(OUTPUT_DIR, "{sample}", "{sample}.SpliSER.tsv"),
        ssu     = os.path.join(OUTPUT_DIR, "{sample}", "ssu_comparison.parquet"),
        bed     = SSU_BED,
    output:
        parquet = os.path.join(OUTPUT_DIR, "{sample}", "spliser_comparison.parquet"),
        pdf     = os.path.join(OUTPUT_DIR, "{sample}", "spliser_vs_ssu_scatterplot.pdf"),
    params:
        script     = "src/scripts/compare_spliser_ssu.py",
        output_dir = lambda wc: os.path.join(OUTPUT_DIR, wc.sample),
    conda:
        "spliser"
    shell:
        """
        python {params.script} \
            --spliser   {input.spliser} \
            --ssu       {input.ssu} \
            --bed       {input.bed} \
            --output-dir {params.output_dir}
        """
