"""
dev.smk — short Pangolin training runs for parameter validation.

Uses the dev BED (same as workflow 04) and a single GPU.
5 epochs is enough to check that gradients flow, loss decreases, and checkpoints save.

Also includes collect_predictions + compute_metrics on the val BED so correlation
can be assessed without running the full evaluation workflow.
"""

import os

DATA_DIR        = config["rnaseq"]["sf3b1mut"]["path"]
FINETUNE_SCRIPT = "src/custom-pangolin/scripts/finetune.py"
COLLECT_SCRIPT  = config["pangolin"]["collect_script"]
METRICS_SCRIPT  = "src/scripts/compute_eval_metrics.py"
FOLDS_DIR       = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD            = config["preprocessing"]["overfitting"]["fold"]
SAMPLES         = config["preprocessing"]["overfitting"]["samples"]
PANGOLIN_CFG    = config["pangolin"]

DEV_OUTPUT_DIR      = "results/finetuning/pangolin/dev"
DEV_EVAL_OUTPUT_DIR = "results/evaluation/pangolin/dev"

_DEV_VAL_BED = os.path.join(config["preprocessing"]["overfitting"]["dev"]["output_dir"], "valid.bed")

# All dev runs share the same epoch count so the output path stays a literal
_DEV_EPOCHS = 5

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

DEV_RUNS = {
    "annotated__frozen__1gpu": {
        "mode":            "linear-probe",
        "epochs":          _DEV_EPOCHS,
        "lr":              1e-3,
        "warmup_steps":    0,
        "batch_size":      128,
        "min_alpha_juncs": 0,
        "num_gpus":        1,
    },
    # "annotated__full__1gpu": {
    #     "mode":            "full",
    #     "epochs":          _DEV_EPOCHS,
    #     "lr":              5e-5,
    #     "warmup_steps":    50,
    #     "batch_size":      12,
    #     "min_alpha_juncs": 5,
    #     "num_gpus":        1,
    # },
}


def _dev_run(key):
    return lambda wildcards: DEV_RUNS[wildcards.run_name][key]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule all_dev:
    input:
        expand(
            os.path.join(DEV_OUTPUT_DIR, "{run_name}", "finetune.done"),
            run_name=list(DEV_RUNS.keys()),
        ),
        expand(
            os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "metrics.parquet"),
            run_name=list(DEV_RUNS.keys()),
            epoch=list(range(1, _DEV_EPOCHS + 1)),
        ),


rule pangolin_dev_finetune:
    """Short Pangolin training on the dev BED — verify everything runs."""
    wildcard_constraints:
        run_name="|".join(DEV_RUNS.keys()),
    input:
        weights        = PANGOLIN_CFG["pretrained_weights"],
        genome         = config["gencode"]["paths"]["fasta"],
        train_bed      = os.path.join(config["preprocessing"]["overfitting"]["dev"]["output_dir"], "train.bed"),
        val_bed        = os.path.join(config["preprocessing"]["overfitting"]["dev"]["output_dir"], "valid.bed"),
        ssu_parquets   = [
            os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
            for sample in SAMPLES
        ],
    output:
        done = touch(os.path.join(DEV_OUTPUT_DIR, "{run_name}", "finetune.done")),
    params:
        num_gpus        = _dev_run("num_gpus"),
        mode            = _dev_run("mode"),
        epochs          = _dev_run("epochs"),
        lr              = _dev_run("lr"),
        warmup_steps    = _dev_run("warmup_steps"),
        batch_size      = _dev_run("batch_size"),
        min_alpha_juncs = _dev_run("min_alpha_juncs"),
        output_dir      = DEV_OUTPUT_DIR,
        samples         = " ".join(SAMPLES),
    benchmark:
        os.path.join(DEV_OUTPUT_DIR, "benchmarks", "{run_name}", "finetune.tsv")
    threads: 8
    resources:
        runtime   = int(2 * 60),
        gres      = "gpu:1",
        partition = "acc_ehpc",
        qos       = "acc_ehpc",
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        FINETUNE_SCRIPT=$(mktemp /tmp/pangolin_finetune_XXXXXX.py)
        cp {FINETUNE_SCRIPT} "$FINETUNE_SCRIPT"

        MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

        torchrun --nproc_per_node={params.num_gpus} --master_port "$MASTER_PORT" "$FINETUNE_SCRIPT" \
            --genome {input.genome} \
            --ssu-parquets {input.ssu_parquets} \
            --samples {params.samples} \
            --train-bed {input.train_bed} \
            --val-bed {input.val_bed} \
            --pretrained-weights {input.weights} \
            --mode {params.mode} \
            --epochs {params.epochs} \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --batch-size {params.batch_size} \
            --min-alpha-juncs {params.min_alpha_juncs} \
            --output-dir {params.output_dir} \
            --run-name {wildcards.run_name} \
            --seed 42

        rm -f "$FINETUNE_SCRIPT"
        echo "Done!"
        """


rule pangolin_dev_collect_predictions:
    """Single-GPU inference on the dev val BED — produces ssu_scores + splice_site_scores."""
    wildcard_constraints:
        run_name = "|".join(DEV_RUNS.keys()),
        epoch    = r"\d+",
    input:
        done         = os.path.join(DEV_OUTPUT_DIR, "{run_name}", "finetune.done"),
        interval_bed = _DEV_VAL_BED,
        genome       = config["gencode"]["paths"]["fasta"],
        ssu_parquets = [
            os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
            for sample in SAMPLES
        ],
    output:
        ssu         = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "predictions", "ssu_scores.parquet"),
        splice_site = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "predictions", "splice_site_scores.parquet"),
    params:
        checkpoint = lambda wildcards: os.path.join(
            DEV_OUTPUT_DIR, wildcards.run_name,
            "checkpoint_epoch{}.pth".format(wildcards.epoch)
        ),
        output_dir = lambda wildcards: os.path.join(
            DEV_EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), "val", "predictions"
        ),
        samples    = " ".join(SAMPLES),
        min_alpha  = 5,
    benchmark:
        os.path.join(DEV_EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "epoch{epoch}", "val", "collect_predictions.tsv")
    threads: 8
    resources:
        runtime   = int(2 * 60),
        gres      = "gpu:1",
        partition = "acc_ehpc",
        qos       = "acc_ehpc",
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {COLLECT_SCRIPT} \
            --checkpoint {params.checkpoint} \
            --test-bed {input.interval_bed} \
            --genome {input.genome} \
            --ssu-parquets {input.ssu_parquets} \
            --samples {params.samples} \
            --min-alpha-juncs {params.min_alpha} \
            --output-dir {params.output_dir}

        echo "Done collecting dev predictions for {wildcards.run_name} epoch {wildcards.epoch}"
        """


rule pangolin_dev_compute_metrics:
    """Compute SSU Pearson r and splice-site auPRC from dev predictions."""
    wildcard_constraints:
        run_name = "|".join(DEV_RUNS.keys()),
        epoch    = r"\d+",
    input:
        ssu         = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "predictions", "ssu_scores.parquet"),
        splice_site = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "predictions", "splice_site_scores.parquet"),
    output:
        metrics_json    = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "metrics.json"),
        metrics_parquet = os.path.join(DEV_EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "val", "metrics.parquet"),
    params:
        predictions_dir = lambda wildcards: os.path.join(
            DEV_EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), "val", "predictions"
        ),
        output_dir = lambda wildcards: os.path.join(
            DEV_EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), "val"
        ),
    benchmark:
        os.path.join(DEV_EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "epoch{epoch}", "val", "compute_metrics.tsv")
    threads: 4
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

        echo "Done computing dev metrics for {wildcards.run_name} epoch {wildcards.epoch}"
        """
