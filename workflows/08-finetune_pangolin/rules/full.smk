"""
full.smk — Full Pangolin finetuning on FOLD_1 train/val split.

Mirrors workflow 05 (AlphaGenome full finetuning) as closely as possible.
Primary run: linear-probe (frozen backbone), 10 epochs, 1 GPU.
"""

import os

DATA_DIR        = config["rnaseq"]["sf3b1mut"]["path"]
FINETUNE_SCRIPT = "src/custom-pangolin/scripts/finetune.py"
FOLDS_DIR       = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD            = config["preprocessing"]["overfitting"]["fold"]
SAMPLES         = config["preprocessing"]["overfitting"]["samples"]
PANGOLIN_CFG    = config["pangolin"]

FULL_OUTPUT_DIR = "results/finetuning/pangolin/full"

_EPOCHS = 10

# ---------------------------------------------------------------------------
# Run matrix — mirrors workflow 05 naming conventions
# ---------------------------------------------------------------------------

FULL_RUNS = {
    "annotated__frozen__1gpu": {
        "mode":            "linear-probe",
        "epochs":          _EPOCHS,
        "lr":              1e-4,
        "warmup_steps":    200,
        "weight_decay":    0.1,
        "batch_size":      12,
        "min_alpha_juncs": 5,
        "num_gpus":        1,
    },
    "annotated__frozen__1gpu__alpha0": {
        "mode":            "linear-probe",
        "epochs":          _EPOCHS,
        "lr":              1e-4,
        "warmup_steps":    200,
        "weight_decay":    0.1,
        "batch_size":      12,
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

rule all_full:
    input:
        expand(
            os.path.join(FULL_OUTPUT_DIR, "{run_name}", "checkpoint_epoch{epochs}.pth"),
            run_name=list(FULL_RUNS.keys()),
            epochs=_EPOCHS,
        ),
        os.path.join(FULL_OUTPUT_DIR, "summary", "epoch_logs.parquet"),


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
    benchmark:
        os.path.join(FULL_OUTPUT_DIR, "benchmarks", "{run_name}", "finetune.tsv")
    output:
        checkpoint = os.path.join(
            FULL_OUTPUT_DIR, "{run_name}",
            "checkpoint_epoch{}.pth".format(_EPOCHS)
        ),
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
    threads: lambda wildcards: FULL_RUNS[wildcards.run_name]["num_gpus"] * 8
    resources:
        runtime   = int(48 * 60),
        gres      = _gres,
        partition = "acc_ehpc",
        qos       = "acc_ehpc",
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
            os.path.join(FULL_OUTPUT_DIR, "{run_name}", "checkpoint_epoch{epochs}.pth"),
            run_name=list(FULL_RUNS.keys()),
            epochs=_EPOCHS,
        )
    output:
        combined = os.path.join(FULL_OUTPUT_DIR, "summary", "epoch_logs.parquet")
    threads: 1
    resources:
        runtime   = int(0.1 * 60),
        gres      = "none",
        partition = "gpp",
        qos       = "gp_ehpc",
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
