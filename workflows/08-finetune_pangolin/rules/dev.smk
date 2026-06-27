"""
dev.smk — short Pangolin training runs for parameter validation.

Uses the dev BED (same as workflow 04) and a single GPU.
5 epochs is enough to check that gradients flow, loss decreases, and checkpoints save.
"""

import os

DATA_DIR        = config["rnaseq"]["sf3b1mut"]["path"]
FINETUNE_SCRIPT = "src/custom-pangolin/scripts/finetune.py"
FOLDS_DIR       = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD            = config["preprocessing"]["overfitting"]["fold"]
SAMPLES         = config["preprocessing"]["overfitting"]["samples"]
PANGOLIN_CFG    = config["pangolin"]

DEV_OUTPUT_DIR  = "results/finetuning/pangolin/dev"

# All dev runs share the same epoch count so the output path stays a literal
_DEV_EPOCHS = 5

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

DEV_RUNS = {
    "annotated__frozen__1gpu": {
        "mode":            "linear-probe",
        "epochs":          _DEV_EPOCHS,
        "lr":              1e-4,
        "warmup_steps":    50,
        "batch_size":      12,
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
            os.path.join(DEV_OUTPUT_DIR, "{run_name}", "checkpoint_epoch{epochs}.pth"),
            run_name=list(DEV_RUNS.keys()),
            epochs=_DEV_EPOCHS,
        ),
        expand(
            os.path.join(DEV_OUTPUT_DIR, "{run_name}", "epoch_log.csv"),
            run_name=list(DEV_RUNS.keys()),
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
        checkpoint = os.path.join(DEV_OUTPUT_DIR, "{run_name}", "checkpoint_epoch{}.pth".format(_DEV_EPOCHS)),
        log        = os.path.join(DEV_OUTPUT_DIR, "{run_name}", "epoch_log.csv"),
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
