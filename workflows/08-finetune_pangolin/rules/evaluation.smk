"""
evaluation.smk — Collect Pangolin predictions and compute metrics.

Mirrors workflow 06 (AlphaGenome evaluation).  For each (run_name, epoch, subset):
  1. collect_predictions  — single-GPU inference, writes ssu_scores.parquet and
                            splice_site_scores.parquet
  2. compute_metrics      — Pearson r (SSU) and auPRC (splice sites) via the shared
                            src/scripts/compute_eval_metrics.py

Note: Pangolin's classification head produces a unified splice-probability (not separate
donor / acceptor probabilities), so splice-site auPRC is computed as a 2-class task.
compute_eval_metrics.py handles this transparently because the four pred_donor_pos /
pred_acceptor_pos / pred_donor_neg / pred_acceptor_neg columns are all set to the same
per-strand splice probability.
"""

import os

DATA_DIR         = config["rnaseq"]["sf3b1mut"]["path"]
COLLECT_SCRIPT   = "src/custom-pangolin/scripts/collect_predictions.py"
METRICS_SCRIPT   = "src/scripts/compute_eval_metrics.py"
FOLDS_DIR        = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD             = config["preprocessing"]["overfitting"]["fold"]
SAMPLES          = config["preprocessing"]["overfitting"]["samples"]

FINETUNE_DIR     = "results/finetuning/pangolin/full"
EVAL_OUTPUT_DIR  = "results/evaluation/pangolin/full"

EVAL_SUBSETS = ["test", "train_sample"]

SUBSET_BED = {
    "test":         os.path.join(FOLDS_DIR, FOLD, "test.bed"),
    "train_sample": os.path.join(FOLDS_DIR, FOLD, "train_sample.bed"),
}

_FULL_EPOCHS = 10   # must match full.smk

# Map run_name -> list of epochs to evaluate
# finetune.py saves a checkpoint after every epoch, so any epoch ≤ _FULL_EPOCHS is valid
# once the training rule has completed.
EVAL_RUNS = {
    "annotated__frozen__1gpu":        [_FULL_EPOCHS],
    "annotated__frozen__1gpu__alpha0": [_FULL_EPOCHS],
}

_RUN_NAMES = [r for r, epochs in EVAL_RUNS.items() for e in epochs for s in EVAL_SUBSETS]
_EPOCHS    = [e for r, epochs in EVAL_RUNS.items() for e in epochs for s in EVAL_SUBSETS]
_SUBSETS   = [s for r, epochs in EVAL_RUNS.items() for e in epochs for s in EVAL_SUBSETS]


def _ssu_parquets(wildcards):
    return [
        os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
        for sample in SAMPLES
    ]


def _interval_bed(wildcards):
    return SUBSET_BED[wildcards.subset]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule all_eval:
    input:
        expand(
            os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "metrics.parquet"),
            zip,
            run_name=_RUN_NAMES,
            epoch=_EPOCHS,
            subset=_SUBSETS,
        ),


rule pangolin_collect_predictions:
    """Single-GPU Pangolin inference on an interval BED."""
    wildcard_constraints:
        epoch  = r"\d+",
        subset = r"[a-z_]+",
    input:
        checkpoint     = os.path.join(FINETUNE_DIR, "{run_name}", "checkpoint_epoch{epoch}.pth"),
        interval_bed   = _interval_bed,
        genome         = config["gencode"]["paths"]["fasta"],
        ssu_parquets   = _ssu_parquets,
    output:
        ssu            = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "ssu_scores.parquet"),
        splice_site    = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "splice_site_scores.parquet"),
    params:
        output_dir     = lambda wildcards: os.path.join(
            EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), wildcards.subset, "predictions"
        ),
        samples        = " ".join(SAMPLES),
        min_alpha      = 5,
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "epoch{epoch}", "{subset}", "collect_predictions.tsv")
    threads: 8
    resources:
        runtime   = int(4 * 60),
        gres      = "gpu:1",
        partition = "acc_ehpc",
        qos       = "acc_ehpc",
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {COLLECT_SCRIPT} \
            --checkpoint {input.checkpoint} \
            --test-bed {input.interval_bed} \
            --genome {input.genome} \
            --ssu-parquets {input.ssu_parquets} \
            --samples {params.samples} \
            --min-alpha-juncs {params.min_alpha} \
            --output-dir {params.output_dir}

        echo "Done collecting predictions for {wildcards.run_name} epoch {wildcards.epoch} {wildcards.subset}"
        """


rule pangolin_compute_metrics:
    """Compute SSU Pearson r and splice-site auPRC from prediction parquets."""
    wildcard_constraints:
        epoch  = r"\d+",
        subset = r"[a-z_]+",
    input:
        ssu         = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "ssu_scores.parquet"),
        splice_site = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "splice_site_scores.parquet"),
    output:
        metrics_json    = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "metrics.json"),
        metrics_parquet = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "metrics.parquet"),
    params:
        predictions_dir = lambda wildcards: os.path.join(
            EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), wildcards.subset, "predictions"
        ),
        output_dir = lambda wildcards: os.path.join(
            EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), wildcards.subset
        ),
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "epoch{epoch}", "{subset}", "compute_metrics.tsv")
    threads: 8
    resources:
        runtime   = int(30),
        gres      = "none",
        partition = "gpp",
        qos       = "gp_ehpc",
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {METRICS_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --output-dir {params.output_dir} \
            --min-junction-counts 5

        echo "Done computing metrics for {wildcards.run_name} epoch {wildcards.epoch} {wildcards.subset}"
        """
