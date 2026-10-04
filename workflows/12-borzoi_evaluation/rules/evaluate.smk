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

COLLECT_SCRIPT = "src/scripts/collect_predictions_borzoi.py"
METRICS_SCRIPT = "src/scripts/compute_metrics_borzoi.py"

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

EVAL_TARGETS = expand(
    os.path.join(EVAL_OUTPUT_DIR, "{subset}", "summary.json"),
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


rule compute_metrics_borzoi:
    """CPU metrics from the collected predictions (no GPU needed)."""
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
        # 1 bp predictions and targets are ~5 GB each for the full test set,
        # plus metric temporaries.
        memory    = 256,
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
