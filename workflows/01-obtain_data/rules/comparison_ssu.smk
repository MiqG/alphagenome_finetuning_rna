import pandas as pd

# Two samples with completed two-pass STAR alignment and BAM prep
SSU_BENCHMARK_SAMPLES = ["SRR17111303","SRR17111311"]

SSU_OUTPUT_DIR = "results/sanity_checks/comparison_ssu"

# chromosome for benchmark
SSU_CHROMS = ["chr1"]

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

rule get_junctions:
    """Extract splice junctions from BAM in STAR SJ.out.tab format via regtools."""
    input:
        bam       = os.path.join(DATA_DIR,"STAR","{sample}","paper_pass.Aligned.sortedByCoord.out.bam"),
        bam_bai   = os.path.join(DATA_DIR,"STAR","{sample}","paper_pass.Aligned.sortedByCoord.out.bam.bai"),
    output:
        sj = os.path.join(SSU_OUTPUT_DIR, "junctions", "{sample}.starlike.SJ.out.tab"),
    benchmark:
        os.path.join(SSU_OUTPUT_DIR, "benchmarks", "{sample}", "get_junctions.tsv")
    params:
        script = os.path.join(SCRIPTS_DIR, "get_star_junctions.py"),
        chroms = " ".join(SSU_CHROMS),
    threads: 2  # unique + total regtools passes run in parallel
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 2*60,
        memory    = 8
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        python {params.script} \
            --bam {input.bam} \
            --output {output.sj} \
            --chroms {params.chroms}

        echo "Done!"
        """

rule compute_ssu_benchmark:
    input:
        junctions = os.path.join(DATA_DIR,"STAR","{sample}","paper_pass.SJ.out.tab"),
        bam       = os.path.join(DATA_DIR,"STAR","{sample}","paper_pass.Aligned.sortedByCoord.out.filtered.bam"),
        bam_bai   = os.path.join(DATA_DIR,"STAR","{sample}","paper_pass.Aligned.sortedByCoord.out.filtered.bam.bai"),
    output:
        ssu = os.path.join(SSU_OUTPUT_DIR,"custom","{sample}","paper_pass.ssu.parquet")
    benchmark:
        os.path.join(SSU_OUTPUT_DIR, "benchmarks", "{sample}", "compute_ssu.tsv")
    params:
        script = "src/alphagenome-pytorch/scripts/compute_ssu.py",
        chroms = " ".join(SSU_CHROMS)
    threads: 1
    resources:
        gres = "none",
        partition = "genoa64",
        runtime = 2*60,  # h in minutes
        memory = 8  # G
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {params.script} \
            --junctions {input.junctions} \
            --bam {input.bam} \
            --chroms {params.chroms} \
            --output {output.ssu}

        echo "Done!"
        """

rule spliser_process:
    """Run SpliSER process on benchmark chromosomes to get BAM-derived SSE."""
    input:
        bam     = lambda wc: _bam_path(wc.sample),
        bam_bai = lambda wc: _bam_path(wc.sample) + ".bai",
    output:
        tsv = os.path.join(SSU_OUTPUT_DIR, "spliser", "{sample}", "{sample}.SpliSER.tsv"),
    benchmark:
        os.path.join(SSU_OUTPUT_DIR, "benchmarks", "{sample}", "spliser_process.tsv")
    params:
        output_prefix = lambda wc: os.path.join(SSU_OUTPUT_DIR, "spliser", wc.sample, wc.sample),
        chroms        = " ".join(SSU_CHROMS),
        output_dir    = lambda wc: os.path.join(SSU_OUTPUT_DIR, "spliser", wc.sample),
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

rule merge_benchmarks:
    """Concatenate per-sample benchmark TSVs into one parquet for comparison."""
    input:
        expand(
            os.path.join(SSU_OUTPUT_DIR, "benchmarks", "{sample}", "{tool}.tsv"),
            sample=SSU_BENCHMARK_SAMPLES,
            tool=["compute_ssu", "spliser_process"],
        ),
    output:
        parquet = os.path.join(SSU_OUTPUT_DIR, "benchmarks", "benchmarks.parquet"),
    run:
        dfs = []
        for f in input:
            parts = f.split(os.sep)
            sample = parts[-2]
            tool = os.path.splitext(parts[-1])[0]
            df = pd.read_csv(f, sep="\t")
            df["sample"] = sample
            df["tool"] = tool
            dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_parquet(output.parquet, index=False)