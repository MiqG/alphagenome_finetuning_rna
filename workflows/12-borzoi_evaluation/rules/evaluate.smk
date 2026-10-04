"""
Borzoi evaluation rules (see ../Snakefile for the overview).

Resumability follows the repo convention (see CLAUDE.md, "never make a
resumable checkpoint the declared rule output"): collect_predictions_borzoi
keeps per-trunk intermediates in <output>/.progress/, which Snakemake never
tracks, and declares only its final artifacts as outputs.
"""

import os

DATA_DIR       = config["rnaseq"]["sf3b1mut"]["path"]
FOLDS_DIR      = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD           = config["preprocessing"]["overfitting"]["fold"]
SAMPLES        = config["preprocessing"]["overfitting"]["samples"]
BIGWIG_STRANDS = ["forward", "reverse"]

BORZOI_EVAL    = config["finetuning"]["borzoi"]["evaluation"]
TRUNKS_DIR     = BORZOI_EVAL["trunks_dir"]
NUM_REPS       = BORZOI_EVAL["num_reps"]
TARGETS_FILE   = BORZOI_EVAL["targets"]
EVAL_OUTPUT_DIR = BORZOI_EVAL["output_dir"]

# Repo-root-relative (see workflows/10-jax_finetuning/rules/finetune.smk for why).
SCRIPTS_DIR    = "workflows/12-borzoi_evaluation/scripts"
COLLECT_SCRIPT = os.path.join(SCRIPTS_DIR, "collect_predictions_borzoi.py")
EXPORT_SCRIPT  = os.path.join(SCRIPTS_DIR, "export_alphagenome_format.py")
METRICS_SCRIPT = os.path.join(SCRIPTS_DIR, "compute_metrics_borzoi.py")
# Shared with the AlphaGenome evaluations (workflows 06 and 10), unchanged.
EVAL_METRICS_SCRIPT = "src/scripts/compute_eval_metrics.py"

# Subsets to evaluate: test (held-out) + train_sample (overfitting check)
EVAL_SUBSETS = ["test", "train_sample"]

SUBSET_BED = {
    "test":         os.path.join(FOLDS_DIR, FOLD, "test.bed"),
    "train_sample": os.path.join(FOLDS_DIR, FOLD, "train_sample.bed"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bigwigs(wildcards):
    # Order matters: interleaved forward/reverse per sample, matching the
    # rows of targets.txt and the trunk output units.
    return [
        os.path.join(DATA_DIR, "STAR", sample,
                     "paper_pass.Aligned.sortedByCoord.out.filtered.{}.bw".format(strand))
        for sample in SAMPLES
        for strand in BIGWIG_STRANDS
    ]


def _interval_bed(wildcards):
    return SUBSET_BED[wildcards.subset]


TRUNK_MODELS = [
    os.path.join(TRUNKS_DIR, "trunk_r{}".format(rep), "model_best.mergeW.h5")
    for rep in range(NUM_REPS)
]
TRUNK_PARAMS = os.path.join(TRUNKS_DIR, "trunk_r0", "params.json")

PRED_DIR = os.path.join(EVAL_OUTPUT_DIR, "{subset}", "predictions")

# RNA-seq parquets in the AlphaGenome prediction schema (see export_alphagenome_format.py).
EXPORT_FILES = [
    "rna_seq_per_gene.parquet",
    "rna_seq_per_gene_32bp.parquet",
    "rna_seq_profile_corr_exon_1bp.parquet",
    "rna_seq_profile_corr_exon_32bp.parquet",
    "rna_seq_profile_corr_full_1bp.parquet",
    "rna_seq_profile_corr_full_32bp.parquet",
    "rna_seq_profile_corr_central_1bp.parquet",
    "rna_seq_profile_corr_central_32bp.parquet",
    "rna_seq_profile_corr_per_interval.parquet",
    "rna_seq_profile_metrics_per_interval.parquet",
    "rna_seq_track_totals_per_interval.parquet",
]

# Default targets: metrics.parquet, computed by the same script as the
# AlphaGenome runs. The upstream summary.json (compute_metrics_borzoi) is
# optional and not part of `rule all`: request it explicitly,
#   results/evaluation/borzoi/full/{subset}/summary.json
EVAL_TARGETS = expand(
    os.path.join(EVAL_OUTPUT_DIR, "{subset}", "metrics.parquet"),
    subset=EVAL_SUBSETS,
)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule collect_predictions_borzoi:
    """GPU inference for every replicate trunk; writes the mean raw prediction."""
    wildcard_constraints:
        subset = "|".join(EVAL_SUBSETS),
    input:
        models       = TRUNK_MODELS,
        params       = TRUNK_PARAMS,
        targets      = TARGETS_FILE,
        interval_bed = _interval_bed,
        genome       = config["gencode"]["paths"]["fasta"],
    output:
        predictions = os.path.join(PRED_DIR, "predictions_32bp.npy"),
        regions     = os.path.join(PRED_DIR, "regions.parquet"),
        run_info    = os.path.join(PRED_DIR, "run_info.json"),
    params:
        output_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset, "predictions"),
        crop_bp    = BORZOI_EVAL["crop_bp"],
        pool_width = BORZOI_EVAL["pool_width"],
        batch_size = BORZOI_EVAL["batch_size"],
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "collect_predictions.tsv")
    threads: 8
    resources:
        runtime   = int(24 * 60),
        partition = "gpu_diasfrazer",
        gres      = "gpu:h100:1",
        memory    = 64,
    conda:
        "borzoi"
    # Per-trunk intermediates live in <output_dir>/.progress/ (not declared
    # outputs), so a retry resumes after the last finished trunk.
    retries: 3
    shell:
        """
        set -eo pipefail

        python -u {COLLECT_SCRIPT} \
            --models {input.models} \
            --params {input.params} \
            --targets {input.targets} \
            --genome {input.genome} \
            --test-bed {input.interval_bed} \
            --crop-bp {params.crop_bp} \
            --pool-width {params.pool_width} \
            --batch-size {params.batch_size} \
            --output-dir {params.output_dir}

        echo "Done collecting Borzoi predictions for subset {wildcards.subset}"
        """


rule export_alphagenome_format:
    """Write the AlphaGenome-schema RNA-seq parquets from the raw predictions."""
    wildcard_constraints:
        subset = "|".join(EVAL_SUBSETS),
    input:
        predictions  = os.path.join(PRED_DIR, "predictions_32bp.npy"),
        regions      = os.path.join(PRED_DIR, "regions.parquet"),
        run_info     = os.path.join(PRED_DIR, "run_info.json"),
        interval_bed = _interval_bed,
        bigwigs      = _bigwigs,
        gtf_parquet  = config["gencode"]["paths"]["gtf_parquet"],
    output:
        files = expand(os.path.join(EVAL_OUTPUT_DIR, "{{subset}}", "predictions", "{name}"), name=EXPORT_FILES),
    params:
        predictions_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset, "predictions"),
        samples         = " ".join(SAMPLES),
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "export_alphagenome_format.tsv")
    threads: 4
    resources:
        runtime   = int(6 * 60),
        gres      = "none",
        partition = "genoa64",
        qos       = "normal",
        # Regions are processed one at a time (no 1 bp arrays held in memory).
        memory    = 32,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python -u {EXPORT_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --test-bed {input.interval_bed} \
            --bigwigs {input.bigwigs} \
            --samples {params.samples} \
            --gtf-parquet {input.gtf_parquet} \
            --output-dir {params.predictions_dir}

        echo "Done exporting AlphaGenome-format parquets for subset {wildcards.subset}"
        """


rule compute_eval_metrics:
    """Metrics from the exported parquets, with the same script as the AlphaGenome runs."""
    wildcard_constraints:
        subset = "|".join(EVAL_SUBSETS),
    input:
        files = expand(os.path.join(EVAL_OUTPUT_DIR, "{{subset}}", "predictions", "{name}"), name=EXPORT_FILES),
    output:
        metrics_json    = os.path.join(EVAL_OUTPUT_DIR, "{subset}", "metrics.json"),
        metrics_parquet = os.path.join(EVAL_OUTPUT_DIR, "{subset}", "metrics.parquet"),
    params:
        predictions_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset, "predictions"),
        output_dir      = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset),
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "compute_eval_metrics.tsv")
    threads: 4
    resources:
        runtime   = int(0.5 * 60),
        gres      = "none",
        partition = "genoa64",
        qos       = "normal",
        memory    = 16,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {EVAL_METRICS_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --output-dir {params.output_dir} \
            --min-junction-counts 5

        echo "Done computing metrics for subset {wildcards.subset}"
        """


rule compute_metrics_borzoi:
    """Optional: upstream Borzoi metrics (summary.json), computed directly from the raw predictions."""
    wildcard_constraints:
        subset = "|".join(EVAL_SUBSETS),
    input:
        predictions = os.path.join(PRED_DIR, "predictions_32bp.npy"),
        regions     = os.path.join(PRED_DIR, "regions.parquet"),
        run_info    = os.path.join(PRED_DIR, "run_info.json"),
        bigwigs     = _bigwigs,
        gtf_parquet = config["gencode"]["paths"]["gtf_parquet"],
    output:
        summary = os.path.join(EVAL_OUTPUT_DIR, "{subset}", "summary.json"),
    params:
        predictions_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset, "predictions"),
        output_dir      = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset),
        samples         = " ".join(SAMPLES),
        pool_width      = BORZOI_EVAL["pool_width"],
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "compute_metrics.tsv")
    threads: 8
    resources:
        runtime   = int(4 * 60),
        gres      = "none",
        partition = "genoa64",
        qos       = "normal",
        # The 1 bp predictions and targets are 20 GB each for the full test set
        # (6359 regions x 196,608 bp x 4 tracks, float32), plus copies and metric
        # temporaries. Not measured at full size; genoa64 nodes have ~700 GB.
        memory    = 512,
    conda:
        "borzoi"
    shell:
        """
        set -eo pipefail

        python -u {METRICS_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --bigwig {input.bigwigs} \
            --samples {params.samples} \
            --gtf {input.gtf_parquet} \
            --pool-width {params.pool_width} \
            --output-dir {params.output_dir}

        echo "Done computing Borzoi metrics for subset {wildcards.subset}"
        """
