"""
Standalone Snakefile for AlphaGenome overfitting + visualization debugging.

This workflow is independent of the main pipeline. It:
1. Creates a minimal 16-interval training set from FOLD_0
2. Overfits on those 16 intervals (100 epochs, constant LR, no warmup)
3. Visualizes predictions vs real signals on a multi-page PDF

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
FOLD_TRAIN_BED = os.path.join(ALPHAGENOME_FOLDS_DIR, "FOLD_0", "train.bed")
OVERFIT_BED = os.path.join("support", "overfit.bed")
OVERFIT_OUTPUT_DIR = os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"].replace("sf3b1mut", "overfit"))
VIZ_OUTPUT_DIR = os.path.join(OVERFIT_OUTPUT_DIR, "visualization")

rule all:
    input:
        viz_pdf = os.path.join(VIZ_OUTPUT_DIR, "tracks.pdf"),

rule create_overfit_bed:
    """Extract first 16 intervals from FOLD_0/train.bed for overfitting."""
    input:
        fold_train_bed = FOLD_TRAIN_BED,
    output:
        overfit_bed = OVERFIT_BED,
    shell:
        """
        head -16 {input.fold_train_bed} > {output.overfit_bed}
        """

rule overfit_sf3b1mut:
    """Fine-tune AlphaGenome on 16 intervals to verify training loop."""
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
                "second_pass.SJ." + strand + ".tab"
            )
            for sample in OVERFIT_SAMPLES
            for strand in JUNCTION_STRANDS
        ],
    output:
        done = touch(os.path.join(OVERFIT_OUTPUT_DIR, "overfit", ".done")),
        checkpoint = os.path.join(OVERFIT_OUTPUT_DIR, "overfit", "best_model.pth"),
    params:
        num_gpus = 1,
        modality_bigwig = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_bigwig"],
        modality_splicing = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_splicing"],
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        overlap_highres = config["finetuning"]["alphagenome"]["sf3b1mut"]["overlap_highres"],
        lr = config["finetuning"]["alphagenome"]["sf3b1mut"]["lr"],
        epochs = 100,
        batch_size = 1,
        gradient_accumulation_steps = 1,
        track_means_samples = config["finetuning"]["alphagenome"]["sf3b1mut"]["track_means_samples"],
        save_every_steps = 50,
        output_dir = OVERFIT_OUTPUT_DIR,
        pretrained_weights = os.path.join(
            config["alphagenome_pytorch"]["paths"]["weights"], "model_all_folds.safetensors"
        ),
    threads: 6
    resources:
        gres = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime = 12*60,  # minutes
        memory = 80  # G
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

        torchrun --nproc_per_node={params.num_gpus} "$FINETUNE_SCRIPT" \
            --num-workers {threads} \
            --mode lora \
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
            --output-dir {params.output_dir} \
            --overlap-highres {params.overlap_highres} \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --run-name overfit

        rm -f "$FINETUNE_SCRIPT"
        echo "Overfitting complete!"
        """

rule visualize_overfit:
    """Visualize predictions vs real tracks from overfitting."""
    input:
        checkpoint = os.path.join(OVERFIT_OUTPUT_DIR, "overfit", "best_model.pth"),
        overfit_bed = OVERFIT_BED,
        genome = config["gencode"]["paths"]["fasta"],
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
                "second_pass.SJ." + strand + ".tab"
            )
            for sample in OVERFIT_SAMPLES
            for strand in JUNCTION_STRANDS
        ],
    output:
        pdf = os.path.join(VIZ_OUTPUT_DIR, "tracks.pdf"),
    params:
        script = "src/scripts/visualize_overfit.py",
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
    conda:
        "alphagenome_pytorch"
    shell:
        """
        mkdir -p {VIZ_OUTPUT_DIR}
        python {params.script} \
            --checkpoint {input.checkpoint} \
            --bed {input.overfit_bed} \
            --genome {input.genome} \
            --bigwig {input.bigwigs} \
            --star-junctions {input.star_junctions} \
            --sequence-length {params.sequence_length} \
            --output {output.pdf}
        """
