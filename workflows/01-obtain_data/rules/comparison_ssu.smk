import pandas as pd

# Two samples with completed two-pass STAR alignment and BAM prep
SSU_BENCHMARK_SAMPLES = ["SRR17111301", "SRR17111311"]

# Reuses support/overfit.bed so the benchmark runs on the same intervals as overfitting
SSU_BED = os.path.join("support", "overfit.bed")

SSU_OUTPUT_DIR = "results/sanity_checks/ssu_benchmark"

# SpliSER runs per-chromosome; derived from BED file
_bed_df = pd.read_csv(SSU_BED, sep="\t", header=None, usecols=[0], names=["chrom"])
SSU_CHROMS = sorted(_bed_df["chrom"].unique().tolist())


def _bam_path(sample):
    return os.path.join(
        DATA_DIR, "STAR", sample,
        "paper_pass.Aligned.sortedByCoord.out.filtered.bam",
    )

def _junctions_path(sample):
    return os.path.join(
        DATA_DIR, "STAR", sample,
        "paper_pass.SJ.out.tab",
    )


rule benchmark_ssu_approximation:
    """Compare junction-only SSU approximation to BAM-derived ground truth."""
    input:
        bam       = lambda wc: _bam_path(wc.sample),
        bam_bai   = lambda wc: _bam_path(wc.sample) + ".bai",
        junctions = lambda wc: _junctions_path(wc.sample),
        bed       = SSU_BED,
    output:
        parquet = os.path.join(SSU_OUTPUT_DIR, "{sample}", "ssu_comparison.parquet"),
        pdf     = os.path.join(SSU_OUTPUT_DIR, "{sample}", "ssu_scatterplot.pdf"),
    params:
        script           = "src/scripts/benchmark_ssu_approximation.py",
        output_dir       = lambda wc: os.path.join(SSU_OUTPUT_DIR, wc.sample),
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


rule spliser_process:
    """Run SpliSER process on benchmark chromosomes to get BAM-derived SSE."""
    input:
        bam     = lambda wc: _bam_path(wc.sample),
        bam_bai = lambda wc: _bam_path(wc.sample) + ".bai",
    output:
        tsv = os.path.join(SSU_OUTPUT_DIR, "{sample}", "{sample}.SpliSER.tsv"),
    params:
        output_prefix = lambda wc: os.path.join(SSU_OUTPUT_DIR, wc.sample, wc.sample),
        chroms        = " ".join(SSU_CHROMS),
        output_dir    = lambda wc: os.path.join(SSU_OUTPUT_DIR, wc.sample),
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


rule compare_spliser_ssu:
    """Join SpliSER SSE against junction-based SSU estimates."""
    input:
        spliser = os.path.join(SSU_OUTPUT_DIR, "{sample}", "{sample}.SpliSER.tsv"),
        ssu     = os.path.join(SSU_OUTPUT_DIR, "{sample}", "ssu_comparison.parquet"),
        bed     = SSU_BED,
    output:
        parquet = os.path.join(SSU_OUTPUT_DIR, "{sample}", "spliser_comparison.parquet"),
        pdf     = os.path.join(SSU_OUTPUT_DIR, "{sample}", "spliser_vs_ssu_scatterplot.pdf"),
    params:
        script     = "src/scripts/compare_spliser_ssu.py",
        output_dir = lambda wc: os.path.join(SSU_OUTPUT_DIR, wc.sample),
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
