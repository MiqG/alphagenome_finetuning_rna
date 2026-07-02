"""
full.smk — Full Pangolin finetuning on FOLD_1 train/val split, plus evaluation.

Mirrors workflow 05 (AlphaGenome full finetuning) as closely as possible.
Primary run: linear-probe (frozen backbone), 1 GPU.

Evaluation rules (collect_predictions + compute_metrics) are included here so
that _EPOCHS is defined in one place only — no separate evaluation.smk needed.
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

FULL_OUTPUT_DIR = "results/finetuning/pangolin/full"
EVAL_OUTPUT_DIR = "results/evaluation/pangolin/full"

# ── Change _EPOCHS here only — evaluation rules pick it up automatically ──────
_EPOCHS = 5

EVAL_SUBSETS = ["test", "train_sample"]

SUBSET_BED = {
    "test":         os.path.join(FOLDS_DIR, FOLD, "test.bed"),
    "train_sample": os.path.join(FOLDS_DIR, FOLD, "train_sample.bed"),
}

# ---------------------------------------------------------------------------
# Run matrix — mirrors workflow 05 naming conventions
# ---------------------------------------------------------------------------

FULL_RUNS = {
    # "annotated__frozen__1gpu": {
    #     "mode":            "linear-probe",
    #     "epochs":          _EPOCHS,
    #     "lr":              1e-3,
    #     "warmup_steps":    0,
    #     "weight_decay":    0,
    #     "batch_size":      128,
    #     "min_alpha_juncs": 0,
    #     "num_gpus":        1,
    # },
    "annotated__full__1gpu": {
        "mode":            "full",
        "epochs":          _EPOCHS,
        "lr":              1e-4,
        "warmup_steps":    0,
        "weight_decay":    0,
        "batch_size":      128,
        "min_alpha_juncs": 0,
        "num_gpus":        1,
    },
}


def _full_run(key):
    return lambda wildcards: FULL_RUNS[wildcards.run_name][key]


def _gres(wildcards):
    return "gpu:{}".format(FULL_RUNS[wildcards.run_name]["num_gpus"])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_EVAL_RUN_NAMES = [r for r in FULL_RUNS for s in EVAL_SUBSETS]
_EVAL_SUBSETS   = [s for r in FULL_RUNS for s in EVAL_SUBSETS]


def _ssu_parquets(wildcards):
    return [
        os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
        for sample in SAMPLES
    ]


def _interval_bed(wildcards):
    return SUBSET_BED[wildcards.subset]


rule all_full:
    input:
        expand(
            os.path.join(FULL_OUTPUT_DIR, "{run_name}", "finetune.done"),
            run_name=list(FULL_RUNS.keys()),
        ),
        os.path.join(FULL_OUTPUT_DIR, "summary", "epoch_logs.parquet"),
        expand(
            os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "metrics.parquet"),
            run_name=list(FULL_RUNS.keys()),
            epoch=list(range(1, _EPOCHS + 1)),
            subset=EVAL_SUBSETS,
        ),


rule pangolin_full_finetune:
    """Finetune Pangolin on the full FOLD_1 train split."""
    wildcard_constraints:
        run_name="|".join(FULL_RUNS.keys()),
    input:
        weights      = PANGOLIN_CFG["pretrained_weights"],
        genome       = config["gencode"]["paths"]["fasta"],
        train_bed    = os.path.join(FOLDS_DIR, FOLD, "train.bed"),
        val_bed      = os.path.join(FOLDS_DIR, FOLD, "valid.bed"),
        ssu_parquets = [
            os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
            for sample in SAMPLES
        ],
    output:
        done = touch(os.path.join(FULL_OUTPUT_DIR, "{run_name}", "finetune.done")),
    benchmark:
        os.path.join(FULL_OUTPUT_DIR, "benchmarks", "{run_name}", "finetune.tsv")
    params:
        num_gpus        = _full_run("num_gpus"),
        mode            = _full_run("mode"),
        epochs          = _full_run("epochs"),
        lr              = _full_run("lr"),
        warmup_steps    = _full_run("warmup_steps"),
        weight_decay    = _full_run("weight_decay"),
        batch_size      = _full_run("batch_size"),
        min_alpha_juncs = _full_run("min_alpha_juncs"),
        output_dir      = FULL_OUTPUT_DIR,
        samples         = " ".join(SAMPLES),
    threads: lambda wildcards: FULL_RUNS[wildcards.run_name]["num_gpus"] * 10
    resources:
        runtime   = int(7 * 24 * 60),
        # gres      = _gres,
        # partition = "acc_ehpc",
        # qos       = "acc_ehpc",
        memory = 42,
        gres      = "gpu:mig_24gb:1",
        partition = "gpu_diasfrazer",
    retries: 1
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
            --weight-decay {params.weight_decay} \
            --batch-size {params.batch_size} \
            --min-alpha-juncs {params.min_alpha_juncs} \
            --output-dir {params.output_dir} \
            --run-name {wildcards.run_name} \
            --resume auto \
            --seed 42

        rm -f "$FINETUNE_SCRIPT"
        echo "Done!"
        """


rule pangolin_combine_epoch_logs:
    params:
        logs = lambda wildcards: expand(
            os.path.join(FULL_OUTPUT_DIR, "{run_name}", "epoch_log.csv"),
            run_name=list(FULL_RUNS.keys()),
        )
    input:
        models = lambda wildcards: expand(
            os.path.join(FULL_OUTPUT_DIR, "{run_name}", "finetune.done"),
            run_name=list(FULL_RUNS.keys()),
        )
    output:
        combined = os.path.join(FULL_OUTPUT_DIR, "summary", "epoch_logs.parquet")
    threads: 1
    resources:
        runtime   = int(0.1 * 60),
        gres      = "none",
        # partition = "gpp",
        # qos       = "gp_ehpc",
        memory = 8,
        partition = "gpu_diasfrazer",
    run:
        import pandas as pd
        dfs = []
        for f in params.logs:
            if os.path.exists(f):
                dfs.append(
                    pd.read_csv(f).assign(run_name=os.path.basename(os.path.dirname(f)))
                )
        if dfs:
            pd.concat(dfs).to_parquet(output.combined, index=False, compression="zstd")
        else:
            pd.DataFrame().to_parquet(output.combined, index=False)
        print("Done!")


rule pangolin_collect_predictions:
    """Single-GPU Pangolin inference on an interval BED."""
    wildcard_constraints:
        epoch  = r"\d+",
        subset = r"[a-z_]+",
    input:
        done         = os.path.join(FULL_OUTPUT_DIR, "{run_name}", "finetune.done"),
        interval_bed = _interval_bed,
        genome       = config["gencode"]["paths"]["fasta"],
        ssu_parquets = _ssu_parquets,
    output:
        ssu         = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "ssu_scores.parquet"),
        splice_site = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "epoch{epoch}", "{subset}", "predictions", "splice_site_scores.parquet"),
    params:
        checkpoint = lambda wildcards: os.path.join(
            FULL_OUTPUT_DIR, wildcards.run_name,
            "checkpoint_epoch{}.pth".format(wildcards.epoch)
        ),
        output_dir = lambda wildcards: os.path.join(
            EVAL_OUTPUT_DIR, wildcards.run_name,
            "epoch{}".format(wildcards.epoch), wildcards.subset, "predictions"
        ),
        samples   = " ".join(SAMPLES),
        min_alpha = 5,
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "epoch{epoch}", "{subset}", "collect_predictions.tsv")
    threads: 8
    resources:
        runtime   = int(4 * 60),
        # gres      = _gres,
        # partition = "acc_ehpc",
        # qos       = "acc_ehpc",
        memory = 42,
        gres      = "gpu:3g.47gb:1",
        partition = "gpu_diasfrazer",
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
        # partition = "gpp",
        # qos       = "gp_ehpc",
        memory = 42,
        partition = "gpu_diasfrazer",
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
