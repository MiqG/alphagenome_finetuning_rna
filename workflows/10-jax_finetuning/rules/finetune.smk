"""
JAX finetuning workflow (alphagenome_ft): probing reproduction of the
PyTorch paper run "randinit__newloss__annotated__frozen__multigpu_ddp".

Cross-checks the alphagenome-pytorch splicing fine-tuning results against a
JAX/alphagenome_research-based finetuning path, using the sibling
alphagenome_ft package (splice-finetuning branch,
/users/diasfrazer/manglada/repositories/alphagenome_ft).

Trains all 4 modalities jointly (rna_seq + splice_site + splice_usage +
splice_junctions), matching workflows/05-full_finetuning/Snakefile's
"randinit__newloss__annotated__frozen__multigpu_ddp" run as closely as the
JAX/alphagenome_ft stack allows. See
plans/10-jax-alphagenome_ft-reproduction.md ("Redo" section) for the
remaining known differences (Kaggle-hosted weights instead of HuggingFace,
no LoRA-parity, no pretrained-head-init for splice_site) that a straight
comparison against the PyTorch run still needs to account for.

Run with:
    snakemake -s workflows/10-jax_finetuning/Snakefile --use-conda [-n]

As with other GPU workflows in this repo, submit the actual (non-dry-run)
run via SLURM rather than on the login node:
    ./src/scripts/submit_snakemake_slurm.sh \
        "snakemake -s workflows/10-jax_finetuning/Snakefile --use-conda -j 4"
"""

import os

configfile: "config/config.yaml"

DATA_DIR   = config["rnaseq"]["sf3b1mut"]["path"]
FOLDS_DIR  = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD       = config["preprocessing"]["overfitting"]["fold"]
SAMPLES    = config["preprocessing"]["overfitting"]["samples"]
BIGWIG_STRANDS = ["forward", "reverse"]  # matches workflows/05-full_finetuning/Snakefile

FT_CFG          = config["finetuning"]["alphagenome_ft"]
SF3B1_FT_CFG    = FT_CFG["sf3b1mut"]
OUTPUT_DIR      = SF3B1_FT_CFG["output_dir"]
CHECKPOINT_CACHE_DIR = FT_CFG["checkpoint_cache_dir"]

# Repo-root-relative (not workflow.snakefile-derived): workflow.snakefile
# resolves to whichever file Snakemake is currently parsing, which is this
# included rules/finetune.smk itself, not the top-level workflows/
# 10-jax_finetuning/Snakefile that includes it.
SCRIPTS_DIR = "workflows/10-jax_finetuning/scripts"

# One entry per parallel run (config["finetuning"]["alphagenome_ft"]["sf3b1mut"]["runs"]),
# mirroring workflows/05-full_finetuning/Snakefile's ALL_RUNS dict + {run_name}
# wildcard pattern, keyed by run_name (not the config's own short keys) so the
# wildcard matches 1:1 against each run's actual --run-name/output directory.
ALL_RUNS = {
    run_cfg["run_name"]: run_cfg
    for run_cfg in SF3B1_FT_CFG["runs"].values()
}


def _mode_args(wildcards):
    cfg = ALL_RUNS[wildcards.run_name]
    if cfg["mode"] == "lora":
        return "--mode lora --lora-rank {} --lora-alpha {} --lora-targets {}".format(
            cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_targets"],
        )
    return "--mode linear-probe"

# GPU target for jax_full_finetune, overridable via --config gpu_target=mig.
# "h100" (default): gpu_diasfrazer's single non-MIG H100 — no partition time
#   limit, preferred whenever free.
# "mig": genoa64-09a's gpu:7g.80gb MIG slice on the shared "gpu" partition —
#   memory-/compute-equivalent fallback when the H100 is occupied by another
#   user's job, but that partition rejects --time beyond 12h outright
#   (confirmed empirically: --qos=marathon still gets "Requested time limit
#   is invalid" above 12h, despite marathon being nominally available to our
#   account there per sacctmgr — this looks like a separate, lower cap
#   enforced specifically on this shared/billed partition). --resume auto
#   (see finetune_alphagenome_jax.py) makes cycling through repeated 12h
#   allocations, or bouncing between this and "h100", safe.
GPU_PRESETS = {
    "h100": {"partition": "gpu_diasfrazer", "gres_type": "h100", "runtime": 7 * 24 * 60, "qos": "marathon"},
    "mig":  {"partition": "gpu", "gres_type": "7g.80gb", "runtime": 12 * 60, "qos": "normal"},
}
GPU_CFG = GPU_PRESETS[config.get("gpu_target", "h100")]


# Target list for the top-level Snakefile's rule all (see ../Snakefile).
FINETUNE_TARGETS = expand(
    os.path.join(OUTPUT_DIR, "{run_name}", "training_complete.marker"),
    run_name=list(ALL_RUNS.keys()),
)


rule download_alphagenome_jax_weights:
    """One-time Kaggle checkpoint cache — run on the login node (needs Kaggle
    credentials + internet), not inside the SLURM GPU job. See the script
    docstring for why."""
    output:
        checkpoint_path_file = os.path.join(CHECKPOINT_CACHE_DIR, "checkpoint_path.txt"),
    params:
        model_version = FT_CFG["kaggle_model_version"],
        script        = os.path.join(SCRIPTS_DIR, "download_alphagenome_jax_weights.py"),
    threads: 1
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 60,
        memory    = 4,
    conda:
        "alphagenome"
    shell:
        """
        # This login node has no /etc/ssl/certs/ca-certificates.crt, which
        # TensorFlow's GCS client (used by kagglehub's dependencies) hardcodes
        # by default — point it at the conda env's own cert bundle instead.
        export CURL_CA_BUNDLE="${{CONDA_PREFIX}}/ssl/cacert.pem"
        export SSL_CERT_FILE="${{CONDA_PREFIX}}/ssl/cacert.pem"

        python {params.script} \
            --model-version {params.model_version} \
            --output-path-file {output.checkpoint_path_file}
        """


rule jax_full_finetune:
    """4-modality (rna_seq + 3 splice heads) finetune on the full FOLD_1 train/val split.

    {run_name} wildcard selects among ALL_RUNS (probing vs lora, see
    _mode_args) — matches workflows/05-full_finetuning/Snakefile's pattern of
    one rule driving multiple parallel runs via a run_name-keyed config dict.
    """
    wildcard_constraints:
        run_name = "|".join(ALL_RUNS.keys()),
    input:
        checkpoint_path_file = os.path.join(CHECKPOINT_CACHE_DIR, "checkpoint_path.txt"),
        genome         = config["gencode"]["paths"]["fasta"],
        train_bed      = os.path.join(FOLDS_DIR, FOLD, "train.bed"),
        val_bed        = os.path.join(FOLDS_DIR, FOLD, "valid.bed"),
        bigwigs        = [
            os.path.join(DATA_DIR, "STAR", sample,
                         "paper_pass.Aligned.sortedByCoord.out.filtered." + strand + ".bw")
            for sample in SAMPLES
            for strand in BIGWIG_STRANDS
        ],
        star_junctions = [
            os.path.join(DATA_DIR, "STAR", sample, "paper_pass.SJ.out.tab")
            for sample in SAMPLES
        ],
        ssu_parquets = [
            os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
            for sample in SAMPLES
        ],
    output:
        # Deliberately NOT checkpoint_dir/{last,best} (which --resume auto
        # reads/writes across invocations) — a distinct marker, matching
        # workflows/05-full_finetuning/Snakefile's pattern of declaring only
        # the final-epoch artifact as output. See the plan doc's "Redo"
        # section: declaring the resumable directory itself as the output
        # is what let --rerun-incomplete silently wipe a real run's
        # progress.
        marker = os.path.join(OUTPUT_DIR, "{run_name}", "training_complete.marker"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "{run_name}", "jax_full_finetune.tsv")
    params:
        script          = os.path.join(SCRIPTS_DIR, "finetune_alphagenome_jax.py"),
        mode_args       = _mode_args,
        sequence_length = SF3B1_FT_CFG["sequence_length"],
        junction_position_source = SF3B1_FT_CFG["junction_position_source"],
        max_splice_sites = SF3B1_FT_CFG["max_splice_sites"],
        lr              = SF3B1_FT_CFG["lr"],
        weight_decay    = SF3B1_FT_CFG["weight_decay"],
        epochs          = SF3B1_FT_CFG["epochs"],
        batch_size      = SF3B1_FT_CFG["batch_size"],
        gradient_accumulation_steps = SF3B1_FT_CFG["gradient_accumulation_steps"],
        num_devices     = SF3B1_FT_CFG["num_devices"],
        save_every_steps = SF3B1_FT_CFG["save_every_steps"],
        track_means_samples = SF3B1_FT_CFG["track_means_samples"],
        dtype           = SF3B1_FT_CFG["dtype"],
        output_dir      = OUTPUT_DIR,
    threads: SF3B1_FT_CFG["num_devices"] * 8
    resources:
        runtime   = GPU_CFG["runtime"],
        memory    = 128,
        gres      = "gpu:{}:{}".format(GPU_CFG["gres_type"], SF3B1_FT_CFG["num_devices"]),
        partition = GPU_CFG["partition"],
        qos       = GPU_CFG["qos"],
    conda:
        "alphagenome"
    # Generous: on the "mig" GPU target (12h QOS cap), completing all
    # SF3B1_FT_CFG["epochs"] epochs needs multiple SLURM allocations in a
    # row, each resuming via --resume auto from the last one's checkpoint
    # (safe now that the checkpoint dir is no longer the declared output —
    # see above). Harmless on "h100" (no time limit): finishes in one
    # attempt there, retries just go unused.
    retries: 20
    shell:
        """
        set -eo pipefail

        # See download_alphagenome_jax_weights rule above for why this is needed
        # (real AlphaGenomeModel construction also fetches calibration scores
        # from GCS, hitting the same missing-system-CA-bundle issue).
        export CURL_CA_BUNDLE="${{CONDA_PREFIX}}/ssl/cacert.pem"
        export SSL_CERT_FILE="${{CONDA_PREFIX}}/ssl/cacert.pem"

        CHECKPOINT_PATH=$(cat {input.checkpoint_path_file})

        python {params.script} \
            --checkpoint-path "$CHECKPOINT_PATH" \
            --genome {input.genome} \
            --train-bed {input.train_bed} \
            --val-bed {input.val_bed} \
            --bigwig {input.bigwigs} \
            --track-means-samples {params.track_means_samples} \
            --star-junctions {input.star_junctions} \
            --ssu {input.ssu_parquets} \
            --junction-position-source {params.junction_position_source} \
            --sequence-length {params.sequence_length} \
            --max-splice-sites {params.max_splice_sites} \
            --lr {params.lr} \
            --weight-decay {params.weight_decay} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --num-devices {params.num_devices} \
            --save-every-steps {params.save_every_steps} \
            --dtype {params.dtype} \
            --max-grad-norm 1.0 \
            --gradient-checkpointing \
            {params.mode_args} \
            --resume auto \
            --seed 1234 \
            --output-dir {params.output_dir} \
            --run-name {wildcards.run_name}

        echo "Done!"
        """
