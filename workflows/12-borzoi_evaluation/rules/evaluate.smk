"""
Borzoi evaluation rules (see ../Snakefile for the overview).

Resumability follows the repo convention (see CLAUDE.md, "never make a
resumable checkpoint the declared rule output"): collect_predictions keeps
per-trunk intermediates in <output>/.progress/, which Snakemake never tracks,
and declares only its final artifacts (the prediction parquets) as outputs.
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
# Shared with the AlphaGenome evaluations (workflows 06 and 10), unchanged.
METRICS_SCRIPT = "src/scripts/compute_eval_metrics.py"

# Subsets to evaluate: test (held-out) + train_sample (overfitting check)
EVAL_SUBSETS = ["test"] #, "train_sample"]

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

# RNA-seq parquets in the AlphaGenome prediction schema (see collect_predictions_borzoi.py).
EXPORT_FILES = [
    "rna_seq_per_gene.parquet",
    "rna_seq_per_gene_32bp.parquet",
    "rna_seq_profile_corr_exon_1bp.parquet",
    "rna_seq_profile_corr_exon_32bp.parquet",
    "rna_seq_profile_corr_central_1bp.parquet",
    "rna_seq_profile_corr_central_32bp.parquet",
    "rna_seq_profile_corr_per_interval.parquet",
    "rna_seq_profile_metrics_per_interval.parquet",
    "rna_seq_track_totals_per_interval.parquet",
    # 32 bp view with native bin-sum pooling (the *_32bp files above are mean-pooled).
    "rna_seq_profile_corr_exon_32bp_sum.parquet",
    "rna_seq_profile_corr_central_32bp_sum.parquet",
    "rna_seq_profile_corr_per_interval_32bp_sum.parquet",
]

EVAL_TARGETS = expand(
    os.path.join(EVAL_OUTPUT_DIR, "{subset}", "metrics.parquet"),
    subset=EVAL_SUBSETS,
)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule collect_predictions:
    """GPU inference for every replicate trunk; writes the AlphaGenome-schema parquets."""
    wildcard_constraints:
        subset = "|".join(EVAL_SUBSETS),
    input:
        models       = TRUNK_MODELS,
        params       = TRUNK_PARAMS,
        targets      = TARGETS_FILE,
        interval_bed = _interval_bed,
        genome       = config["gencode"]["paths"]["fasta"],
        bigwigs      = _bigwigs,
        gtf_parquet  = config["gencode"]["paths"]["gtf_parquet"],
    output:
        files = expand(os.path.join(EVAL_OUTPUT_DIR, "{{subset}}", "predictions", "{name}"), name=EXPORT_FILES),
    params:
        output_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.subset, "predictions"),
        samples    = " ".join(SAMPLES),
        crop_bp    = BORZOI_EVAL["crop_bp"],
        pool_width = BORZOI_EVAL["pool_width"],
        batch_size = BORZOI_EVAL["batch_size"],
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "collect_predictions.tsv")
    threads: 2
    resources:
        runtime   = int(24 * 60),
        partition = "gpu_diasfrazer",
        gres      = "gpu:1g.12gb:1",
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
            --bigwigs {input.bigwigs} \
            --samples {params.samples} \
            --gtf-parquet {input.gtf_parquet} \
            --crop-bp {params.crop_bp} \
            --pool-width {params.pool_width} \
            --batch-size {params.batch_size} \
            --output-dir {params.output_dir}

        echo "Done collecting Borzoi predictions for subset {wildcards.subset}"
        """


rule compute_metrics:
    """Metrics from the prediction parquets, with the same script as the AlphaGenome runs."""
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
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{subset}", "compute_metrics.tsv")
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

        python {METRICS_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --output-dir {params.output_dir} \
            --min-junction-counts 5

        echo "Done computing metrics for subset {wildcards.subset}"
        """
