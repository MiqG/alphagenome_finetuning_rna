"""
Benchmark pipeline: pretrained-oracle self-consistency test.

Tests whether a new output head can recover the pretrained AlphaGenome model's
own predictions for HepG2 total RNA-seq (tracks 1518, 1789, 5154, 5521, 5888).

Two target representations are compared ({target_type} wildcard):
  - data_like    : targets derived via BigWig + STAR junction files (same pipeline as
                   real finetuning — quantised, transformed, sparse)
  - distillation : targets are raw model output tensors loaded directly from .npz files
                   via DistillationDataset (continuous, lossless)

Steps:
  1. generate_oracle_targets  — run the pretrained model on overfit.bed and write both
                                BigWig/STAR files AND raw .npz tensors + manifest.
  2. overfit_oracle           — train a new head in linear-probe mode; {target_type}
                                selects which target representation to use.
  3. visualize_oracle_overfit — compare trained head predictions vs oracle signal.
  4. plot_oracle_summary      — summary PDF across all run × target_type combinations.

Run with:
    snakemake -s workflows/benchmark_pretrained_oracle.smk --use-conda
"""

import os
import pandas as pd

configfile: "config/config.yaml"

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
FINETUNE_SCRIPT    = config["finetuning"]["alphagenome"]["finetune_script"]
FOLDS_DIR          = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD_TRAIN_BED     = os.path.join(FOLDS_DIR, "FOLD_1", "train.bed")
OVERFIT_BED        = os.path.join("support", "overfit.bed")

OUTPUT_DIR = os.path.join(
    config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"].replace("sf3b1mut", "oracle_benchmark")
)

ORACLE_DIR              = os.path.join(OUTPUT_DIR, "targets", "hepg2")
ORACLE_RNA_FWD          = os.path.join(ORACLE_DIR, "oracle_rna_fwd.bw")
ORACLE_RNA_REV          = os.path.join(ORACLE_DIR, "oracle_rna_rev.bw")
ORACLE_JUNCTIONS        = os.path.join(ORACLE_DIR, "oracle_junctions.SJ.out.tab")
ORACLE_JUNCTIONS_PLUS1  = os.path.join(ORACLE_DIR, "oracle_junctions_start_plus1.SJ.out.tab")
ORACLE_MANIFEST         = os.path.join(ORACLE_DIR, "distillation_manifest.parquet")

# ------------------------------------------------------------------ #
# Run configurations
# ------------------------------------------------------------------ #
_MODALITIES = ["rna_seq", "splice_site", "splice_usage", "splice_junctions"]

ORACLE_RUNS = {
    "all": ",".join(f"{m}:1.0" for m in _MODALITIES),
}

_TARGET_TYPES = ["data_like", "data_like_plus1", "distillation"]

# ------------------------------------------------------------------ #
# Rule: all
# ------------------------------------------------------------------ #
rule all:
    input:
        expand(
            os.path.join(OUTPUT_DIR, "{target_type}", "{run_name}", "visualization", "summary_stats.parquet"),
            target_type=_TARGET_TYPES,
            run_name=list(ORACLE_RUNS.keys()),
        ),
        expand(
            os.path.join(OUTPUT_DIR, "{target_type}", "oracle_summary.pdf"),
            target_type=_TARGET_TYPES,
        ),


# ------------------------------------------------------------------ #
# Rule 0: create overfit.bed (single interval from FOLD_1/train.bed)
# ------------------------------------------------------------------ #
rule create_overfit_bed:
    """Extract first interval from FOLD_1/train.bed."""
    input:
        fold_train_bed = FOLD_TRAIN_BED,
    output:
        overfit_bed = OVERFIT_BED,
    shell:
        "head -1 {input.fold_train_bed} > {output.overfit_bed}"


# ------------------------------------------------------------------ #
# Rule 1: generate oracle targets (BigWig + STAR + .npz distillation)
# ------------------------------------------------------------------ #
rule generate_oracle_targets:
    """Run pretrained AlphaGenome on overfit.bed; write BigWig, STAR, and .npz targets."""
    input:
        weights        = config["alphagenome_pytorch"]["paths"]["weights"],
        track_metadata = config["examples"]["alphagenome"]["track_metadata"],
        bed            = OVERFIT_BED,
        genome         = config["gencode"]["paths"]["fasta"],
    output:
        rna_fwd          = ORACLE_RNA_FWD,
        rna_rev          = ORACLE_RNA_REV,
        junctions        = ORACLE_JUNCTIONS,
        junctions_plus1  = ORACLE_JUNCTIONS_PLUS1,
        manifest         = ORACLE_MANIFEST,
    params:
        script          = "src/scripts/generate_oracle_targets.py",
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        output_dir      = ORACLE_DIR,
    threads: 4
    resources:
        gres      = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime   = 2 * 60,
        memory    = 80,
        nodelist  = "genoa64-09",
    conda:
        "alphagenome_pytorch"
    shell:
        """
        mkdir -p {params.output_dir}
        python {params.script} \
            --weights {input.weights} \
            --track-metadata {input.track_metadata} \
            --bed {input.bed} \
            --genome {input.genome} \
            --sequence-length {params.sequence_length} \
            --output-dir {params.output_dir}
        """


# ------------------------------------------------------------------ #
# Rule 2: overfit new head on oracle targets
# ------------------------------------------------------------------ #
def _junctions_path(wildcards):
    if wildcards.target_type == "data_like_plus1":
        return ORACLE_JUNCTIONS_PLUS1
    return ORACLE_JUNCTIONS


def _distillation_arg(wildcards, input):
    if wildcards.target_type == "distillation":
        return f"--distillation-targets {input.manifest}"
    return ""


rule overfit_oracle:
    """Fine-tune a new head in linear-probe mode against oracle targets."""
    input:
        weights     = config["alphagenome_pytorch"]["paths"]["weights"],
        genome      = config["gencode"]["paths"]["fasta"],
        overfit_bed = OVERFIT_BED,
        rna_fwd     = ORACLE_RNA_FWD,
        rna_rev     = ORACLE_RNA_REV,
        junctions   = _junctions_path,
        manifest    = ORACLE_MANIFEST,
    output:
        done       = touch(os.path.join(OUTPUT_DIR, "{target_type}", "{run_name}", ".done")),
        checkpoint = os.path.join(OUTPUT_DIR, "{target_type}", "{run_name}", "best_model.pth"),
    params:
        num_gpus                  = 1,
        sequence_length           = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        lr                        = 1e-2,
        weight_decay              = 1e-2,
        epochs                    = 500,
        batch_size                = 1,
        gradient_accumulation_steps = 1,
        track_means_samples       = config["finetuning"]["alphagenome"]["sf3b1mut"]["track_means_samples"],
        save_every_steps          = 50,
        loss_partitions           = "rna_seq:8,splice_site:8,splice_usage:8",
        modality_weights          = lambda wildcards: ORACLE_RUNS[wildcards.run_name],
        distillation_arg          = _distillation_arg,
        output_base               = lambda wildcards: os.path.join(OUTPUT_DIR, wildcards.target_type),
    threads: 6
    resources:
        gres      = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime   = 12 * 60,
        memory    = 42,
        nodelist  = "genoa64-09",
    conda:
        "alphagenome_pytorch"
    retries: 0
    shell:
        """
        set -eo pipefail
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        TMP_SCRIPT=$(mktemp /tmp/finetune_XXXXXX.py)
        cp {FINETUNE_SCRIPT} "$TMP_SCRIPT"

        torchrun --nproc_per_node={params.num_gpus} "$TMP_SCRIPT" \
            --num-workers {threads} \
            --mode linear-probe \
            --modality-weights "{params.modality_weights}" \
            --loss-partitions {params.loss_partitions} \
            --genome {input.genome} \
            --modality rna_seq --bigwig {input.rna_fwd} {input.rna_rev} \
            --modality splice_site,splice_usage,splice_junctions \
            --star-junctions {input.junctions} \
            --train-bed {input.overfit_bed} \
            --val-bed {input.overfit_bed} \
            --pretrained-weights {input.weights} \
            --pretrained-head-samples rna_seq:NA,splice_usage:NA,splice_junctions:NA,splice_site:NA \
            --organism human \
            --gradient-checkpointing \
            --resume auto \
            --lr {params.lr} \
            --weight-decay {params.weight_decay} \
            --warmup-steps 20 \
            --lr-schedule cosine \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir {params.output_base} \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --run-name {wildcards.run_name} \
            --max-grad-norm inf \
            --seed 1234 \
            {params.distillation_arg}

        rm -f "$TMP_SCRIPT"

        find "{params.output_base}/{wildcards.run_name}" -name "*.pth" ! -name "best_model.pth" -delete

        echo "Oracle overfit complete: target_type={wildcards.target_type} run={wildcards.run_name}"
        """


# ------------------------------------------------------------------ #
# Rule 3: visualize oracle overfit
# ------------------------------------------------------------------ #
rule visualize_oracle_overfit:
    """Plot trained head predictions vs oracle (pretrained model) signal."""
    input:
        checkpoint  = os.path.join(OUTPUT_DIR, "{target_type}", "{run_name}", "best_model.pth"),
        overfit_bed = OVERFIT_BED,
        genome      = config["gencode"]["paths"]["fasta"],
        gtf         = config["gencode"]["paths"]["gtf"],
        rna_fwd     = ORACLE_RNA_FWD,
        rna_rev     = ORACLE_RNA_REV,
        junctions   = _junctions_path,
    output:
        summary = os.path.join(OUTPUT_DIR, "{target_type}", "{run_name}", "visualization", "summary_stats.parquet"),
    params:
        script          = "src/scripts/visualize_overfit.py",
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        viz_dir         = lambda wildcards: os.path.join(
            OUTPUT_DIR, wildcards.target_type, wildcards.run_name, "visualization"
        ),
    conda:
        "alphagenome_pytorch"
    shell:
        """
        mkdir -p {params.viz_dir}
        python {params.script} \
            --checkpoint {input.checkpoint} \
            --bed {input.overfit_bed} \
            --genome {input.genome} \
            --gtf {input.gtf} \
            --bigwig {input.rna_fwd} {input.rna_rev} \
            --star-junctions {input.junctions} \
            --sequence-length {params.sequence_length} \
            --output-dir {params.viz_dir}
        """


# ------------------------------------------------------------------ #
# Rule 4: summary plot per target_type
# ------------------------------------------------------------------ #
rule plot_oracle_summary:
    """Summary PDF of training dynamics and correlations for one target_type."""
    input:
        summaries = expand(
            os.path.join(OUTPUT_DIR, "{{target_type}}", "{run_name}", "visualization", "summary_stats.parquet"),
            run_name=list(ORACLE_RUNS.keys()),
        ),
    output:
        pdf = os.path.join(OUTPUT_DIR, "{target_type}", "oracle_summary.pdf"),
    params:
        script   = "src/scripts/plot_overfit_summary.py",
        run_dirs = lambda wildcards: [
            os.path.join(OUTPUT_DIR, wildcards.target_type, rn) for rn in ORACLE_RUNS.keys()
        ],
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --run-dirs {params.run_dirs} \
            --output {output.pdf}
        """
