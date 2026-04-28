"""
Standalone Snakefile for AlphaGenome overfitting + visualization debugging.

This workflow is independent of the main pipeline. It:
1. Creates a minimal 8-interval training set from FOLD_1
2. Overfits on those 8 intervals (50 epochs, constant LR, no warmup)
   - All modalities with equal weight (1.0)
   - Each modality individually (weight 1.0, others 0.0)
   For each run above, two variants:
   - normal: initialize output heads with random weights
   - pretrained: initialize output heads from pretrained model weights
3. Visualizes predictions vs real signals at gene level
4. Generates summary plots per initialization strategy

Output structure:
  {overfit_output_dir}/
    normal/
      all/best_model.pth, visualization/summary_stats.parquet, ...
      {modality}_only/...
      overfit_summary.pdf
    pretrained/
      all/best_model.pth, visualization/summary_stats.parquet, ...
      {modality}_only/...
      overfit_summary.pdf

Run with:
    snakemake -s workflows/overfit_alphagenome.smk --use-conda
"""

import os
import pandas as pd

# Load config
configfile: "config/config.yaml"

# Extract dataset paths
DATA_DIR = config["rnaseq"]["sf3b1mut"]["path"]
BIGWIG_STRANDS = ["forward", "reverse"]
JUNCTION_STRANDS = ["fwd", "rev"]

# Read SAMPLES from metadata TSV (same as sf3b1mut.smk)
metadata = pd.read_table(config["rnaseq"]["sf3b1mut"]["metadata"])
metadata = metadata.loc[metadata["library_source"] == "TRANSCRIPTOMIC"]
URLS = metadata["fastq_ftp"].str.split(";").str[0].apply(os.path.dirname).to_list()
URLS = {os.path.basename(url): url for url in URLS}
SAMPLES = list(URLS.keys())

# Use samples that have strand-split junction files (.fwd.tab, .rev.tab)
# Currently: SRR17111301 and SRR17111311
OVERFIT_SAMPLES = ["SRR17111301", "SRR17111311"]

# Paths
FINETUNE_SCRIPT = config["finetuning"]["alphagenome"]["finetune_script"]
ALPHAGENOME_FOLDS_DIR = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD_TRAIN_BED = os.path.join(ALPHAGENOME_FOLDS_DIR, "FOLD_1", "train.bed")
OVERFIT_BED = os.path.join("support", "overfit.bed")
OVERFIT_OUTPUT_DIR = os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"].replace("sf3b1mut", "overfit"))

# Modality weight configurations: run_name -> weights string
_MODALITIES = ["splice_site"]#["rna_seq", "splice_site", "splice_usage", "splice_junctions"]
OVERFIT_RUNS = {
    "all": ",".join("{}:1.0".format(m) for m in _MODALITIES),
    **{
        "{}_only".format(m): ",".join(
            "{}:{}".format(mod, "1.0" if mod == m else "0.0") for mod in _MODALITIES
        )
        for m in _MODALITIES
    },
}

# Head initialization strategies
_INIT_STRATEGIES = ["normal", "pretrained"]

# GTF annotation variants
_GTF_VARIANTS = ["no_gtf", "with_gtf"]

rule all:
    input:
        expand(
            os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "{run_name}", "visualization", "summary_stats.parquet"),
            gtf_variant=_GTF_VARIANTS,
            init_strategy=_INIT_STRATEGIES,
            run_name=list(OVERFIT_RUNS.keys()),
        ),
        expand(
            os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "overfit_summary.pdf"),
            gtf_variant=_GTF_VARIANTS,
            init_strategy=_INIT_STRATEGIES,
        ),

rule create_overfit_bed:
    """Extract first 8 intervals from FOLD_1/train.bed for overfitting."""
    input:
        fold_train_bed = FOLD_TRAIN_BED,
    output:
        overfit_bed = OVERFIT_BED,
    shell:
        """
        head -1 {input.fold_train_bed} > {output.overfit_bed}
        """

rule overfit_sf3b1mut:
    """Fine-tune AlphaGenome on 8 intervals to verify training loop."""
    input:
        weights = config["alphagenome_pytorch"]["paths"]["weights"],
        genome = config["gencode"]["paths"]["fasta"],
        overfit_bed = OVERFIT_BED,
        bigwigs = [
            os.path.join(
                DATA_DIR, "STAR", sample,
                "second_pass.Aligned.sortedByCoord.out.filtered." + strand + ".bw"
            )
            for sample in OVERFIT_SAMPLES
            for strand in BIGWIG_STRANDS
        ],
        star_junctions = [
            os.path.join(
                DATA_DIR, "STAR", sample,
                "second_pass.SJ.out.tab"
            )
            for sample in OVERFIT_SAMPLES
        ],
        gtf_parquet = config["gencode"]["paths"]["gtf_parquet"],
    output:
        done = touch(os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "{run_name}", ".done")),
        checkpoint = os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "{run_name}", "best_model.pth"),
    params:
        num_gpus = 1,
        modality_bigwig = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_bigwig"],
        modality_splicing = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_splicing"],
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        overlap_highres = 1024,
        lr = 1e-3,
        epochs = 100, 
        batch_size = 1,
        gradient_accumulation_steps = 1,
        track_means_samples = config["finetuning"]["alphagenome"]["sf3b1mut"]["track_means_samples"],
        save_every_steps = 50,
        output_base = OVERFIT_OUTPUT_DIR,
        pretrained_weights = config["alphagenome_pytorch"]["paths"]["weights"],
        modality_weights = lambda wildcards: OVERFIT_RUNS[wildcards.run_name],
        loss_partitions = "rna_seq:8,splice_site:8,splice_usage:8",
        pretrained_head_arg = lambda wildcards: (
            "--pretrained-head-samples rna_seq:119,splice_usage:139,splice_junctions:139,splice_site:0 --organism human"
            if wildcards.init_strategy == "pretrained"
            else "--pretrained-head-samples rna_seq:NA,splice_usage:NA,splice_junctions:NA,splice_site:NA --organism human"
        ),
        gtf_arg = lambda wildcards, input: (
            "--gtf {}".format(input.gtf_parquet)
            if wildcards.gtf_variant == "with_gtf"
            else ""
        ),
    threads: 6
    resources:
        gres = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime = 12*60,  # minutes
        memory = 80,  # G
        nodelist = "genoa64-09"
    conda:
        "alphagenome_pytorch"
    retries: 0
    shell:
        """
        set -eo pipefail
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # Copy finetune script to tmp to avoid NFS issues under torchrun
        FINETUNE_SCRIPT=$(mktemp /tmp/finetune_XXXXXX.py)
        cp {FINETUNE_SCRIPT} "$FINETUNE_SCRIPT"

        OUTPUT_DIR="{params.output_base}/{wildcards.gtf_variant}/{wildcards.init_strategy}"

        torchrun --nproc_per_node={params.num_gpus} "$FINETUNE_SCRIPT" \
            --num-workers {threads} \
            --mode linear-probe \
            --modality-weights "{params.modality_weights}" \
            --loss-partitions {params.loss_partitions} \
            --genome {input.genome} \
            --modality {params.modality_bigwig} --bigwig {input.bigwigs} \
            --modality {params.modality_splicing} --star-junctions {input.star_junctions} \
            --train-bed {input.overfit_bed} \
            --val-bed {input.overfit_bed} \
            --pretrained-weights {params.pretrained_weights} \
            --gradient-checkpointing \
            --resume auto \
            --lr {params.lr} \
            --warmup-steps 0 \
            --lr-schedule constant \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir "$OUTPUT_DIR" \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --run-name {wildcards.run_name} \
            --max-grad-norm inf \
            --seed 1234 \
            {params.pretrained_head_arg} \
            {params.gtf_arg}

        rm -f "$FINETUNE_SCRIPT"

        # Remove intermediate checkpoints, keep only best_model.pth
        find "$OUTPUT_DIR/{wildcards.run_name}" -name "*.pth" ! -name "best_model.pth" -delete

        echo "Overfitting complete!"
        """

rule visualize_overfit:
    """Visualize predictions vs real tracks for genes in overfitting intervals."""
    input:
        checkpoint = os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "{run_name}", "best_model.pth"),
        overfit_bed = OVERFIT_BED,
        genome = config["gencode"]["paths"]["fasta"],
        gtf = config["gencode"]["paths"]["gtf"],
        gtf_parquet = config["gencode"]["paths"]["gtf_parquet"],
        bigwigs = [
            os.path.join(
                DATA_DIR, "STAR", sample,
                "second_pass.Aligned.sortedByCoord.out.filtered." + strand + ".bw"
            )
            for sample in OVERFIT_SAMPLES
            for strand in BIGWIG_STRANDS
        ],
        star_junctions = [
            os.path.join(
                DATA_DIR, "STAR", sample,
                "second_pass.SJ.out.tab"
            )
            for sample in OVERFIT_SAMPLES
        ],
    output:
        summary = os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "{run_name}", "visualization", "summary_stats.parquet"),
    params:
        script = "src/scripts/visualize_overfit.py",
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        viz_dir = lambda wildcards: os.path.join(OVERFIT_OUTPUT_DIR, wildcards.gtf_variant, wildcards.init_strategy, wildcards.run_name, "visualization"),
        gtf_arg = lambda wildcards, input: (
            "--gtf-splice-sites {}".format(input.gtf_parquet)
            if wildcards.gtf_variant == "with_gtf"
            else ""
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
            --bigwig {input.bigwigs} \
            --star-junctions {input.star_junctions} \
            --sequence-length {params.sequence_length} \
            --output-dir {params.viz_dir} \
            {params.gtf_arg}
        """

rule plot_overfit_summary:
    """Plot training dynamics and final prediction correlations across all overfit runs."""
    input:
        summaries = expand(
            os.path.join(OVERFIT_OUTPUT_DIR, "{{gtf_variant}}", "{{init_strategy}}", "{run_name}", "visualization", "summary_stats.parquet"),
            run_name=list(OVERFIT_RUNS.keys()),
        ),
    output:
        pdf = os.path.join(OVERFIT_OUTPUT_DIR, "{gtf_variant}", "{init_strategy}", "overfit_summary.pdf"),
    params:
        script = "src/scripts/plot_overfit_summary.py",
        run_dirs = lambda wildcards: [os.path.join(OVERFIT_OUTPUT_DIR, wildcards.gtf_variant, wildcards.init_strategy, run_name) for run_name in OVERFIT_RUNS.keys()],
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --run-dirs {params.run_dirs} \
            --output {output.pdf}
        """
