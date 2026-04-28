"""
Development finetuning workflow for AlphaGenome on SF3B1-mutant RNA-seq data.

Subsets 2000 intervals from FOLD_1 train/valid beds and fine-tunes on two
representative samples (SRR17111301, SRR17111311) using precomputed SSU
parquets and GTF annotation for splice site classification.

Three runs are compared:
  - annotated_lp_nolora: junction positions from STAR SJ.out.tab files (linear-probe)
  - predicted_lp_nolora: junction positions from top-k=512 splice_site head predictions (linear-probe)
  - annotated_lp_lora:   same as annotated_lp_nolora but with LoRA (rank=8) adapters

Run with:
    snakemake -s workflows/finetune_dev.smk --use-conda [-n]
"""

import os

configfile: "config/config.yaml"

DATA_DIR        = config["rnaseq"]["sf3b1mut"]["path"]
FINETUNE_SCRIPT = config["finetuning"]["alphagenome"]["finetune_script"]
FOLDS_DIR       = config["finetuning"]["alphagenome"]["folds_dir"]
DEV_OUTPUT_DIR  = config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"].replace("sf3b1mut", "dev")

# DEV_SAMPLES    = ["SRR17111301", "SRR17111311"]
DEV_SAMPLES = ["SRR17111303","SRR17111311"]
BIGWIG_STRANDS = ["forward", "reverse"]
N_INTERVALS    = 2000
JUNCTION_TOP_K = 512

RUN_NAMES = ["annotated_lp_nolora", "predicted_lp_nolora", "annotated_lp_lora"]

# Map run name → --junction-position-source flag value
JUNCTION_POSITION_SOURCE = {
    "annotated_lp_nolora": "annotated",
    "predicted_lp_nolora": "predicted",
    "annotated_lp_lora":   "annotated",
}

# LoRA / LoCon settings for the annotated_lora run.
# Targets use substring matching against full dotted module paths.
# Backbone submodules are: encoder, tower, decoder, embedder_1bp.
# Heads live under "heads." — excluded by using backbone-prefixed substrings so
# that freshly-initialised head parameters are not double-wrapped with adapters.
#   LoRA  (Linear): attention projections and FFN linears in tower + decoder linears
#   LoCon (Conv1d): conv blocks in encoder down-path, decoder up-path, embedder skip
# Rank=8/alpha=8 (scale=1) for LoRA; rank=4/alpha=1 for LoCon — conservative scaling
# to avoid adapter magnitude dominating the small dev training set.
LORA_RANK     = 8
LORA_ALPHA    = 8
LORA_TARGETS  = "encoder.linear,encoder.proj,tower.linear,tower.proj,tower.fc,decoder.linear,decoder.proj,embedder_1bp.linear,embedder_1bp.proj"
LOCON_RANK    = None
LOCON_ALPHA   = None
LOCON_TARGETS = None # "encoder.conv,encoder.dna_embedder,decoder.conv,embedder_1bp.conv,embedder_1bp.project_skip"


rule all:
    input:
        os.path.join(DEV_OUTPUT_DIR, "summary.pdf"),


rule subset_bed:
    """Subset first N_INTERVALS lines from a FOLD_1 BED file."""
    input:
        bed = os.path.join(FOLDS_DIR, "FOLD_1", "{split}.bed"),
    output:
        bed = os.path.join(DEV_OUTPUT_DIR, "beds", "{split}.bed"),
    shell:
        "head -n {N_INTERVALS} {input.bed} > {output.bed}"


rule finetune_dev:
    """Fine-tune AlphaGenome on the dev subset using precomputed SSU parquets."""
    wildcard_constraints:
        run_name = "|".join(RUN_NAMES),
    input:
        weights      = config["alphagenome_pytorch"]["paths"]["weights"],
        genome       = config["gencode"]["paths"]["fasta"],
        gtf_parquet  = config["gencode"]["paths"]["gtf_parquet"],
        train_bed    = os.path.join(DEV_OUTPUT_DIR, "beds", "train.bed"),
        val_bed      = os.path.join(DEV_OUTPUT_DIR, "beds", "valid.bed"),
        bigwigs      = [
            os.path.join(DATA_DIR, "STAR", sample,
                         "second_pass.Aligned.sortedByCoord.out.filtered." + strand + ".bw")
            for sample in DEV_SAMPLES
            for strand in BIGWIG_STRANDS
        ],
        star_junctions = [
            os.path.join(DATA_DIR, "STAR", sample, "second_pass.SJ.out.tab")
            for sample in DEV_SAMPLES
        ],
        ssu_parquets = [
            os.path.join(DATA_DIR, "STAR", sample, "second_pass.ssu.parquet")
            for sample in DEV_SAMPLES
        ],
    output:
        checkpoint = os.path.join(DEV_OUTPUT_DIR, "run", "{run_name}", "best_model.pth"),
    params:
        num_gpus                    = 1,
        modality_bigwig             = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_bigwig"],
        modality_splicing           = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_splicing"],
        sequence_length             = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        overlap_highres             = config["finetuning"]["alphagenome"]["sf3b1mut"]["overlap_highres"],
        lr                          = 3e-4,
        warmup_steps                = 200,
        epochs                      = config["finetuning"]["alphagenome"]["sf3b1mut"]["epochs"],
        batch_size                  = config["finetuning"]["alphagenome"]["sf3b1mut"]["batch_size"],
        gradient_accumulation_steps = config["finetuning"]["alphagenome"]["sf3b1mut"]["gradient_accumulation_steps"],
        track_means_samples         = N_INTERVALS,
        save_every_steps            = config["finetuning"]["alphagenome"]["sf3b1mut"]["save_every_steps"],
        output_dir                  = os.path.join(DEV_OUTPUT_DIR, "run"),
        junction_position_source    = lambda wc: JUNCTION_POSITION_SOURCE[wc.run_name],
        junction_top_k_flag         = lambda wc: f"--junction-top-k {JUNCTION_TOP_K}" if wc.run_name == "predicted_lp_nolora" else "",
        mode_flag                   = lambda wc: "lora" if wc.run_name == "annotated_lp_lora" else "linear-probe",
        lora_flags                  = lambda wc: (
            f"--lora-rank {LORA_RANK} --lora-alpha {LORA_ALPHA} --lora-targets {LORA_TARGETS}"
        ) if wc.run_name == "annotated_lp_lora" else "",
        run_name                    = "{run_name}",
    threads: 6
    resources:
        gres      = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime   = 12 * 60,
        memory    = 80,
        nodelist  = "genoa64-09",
    conda:
        "alphagenome_pytorch"
    retries: 0
    shell:
        """
        set -eo pipefail
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        FINETUNE_SCRIPT=$(mktemp /tmp/finetune_XXXXXX.py)
        cp {FINETUNE_SCRIPT} "$FINETUNE_SCRIPT"

        torchrun --nproc_per_node={params.num_gpus} "$FINETUNE_SCRIPT" \
            --num-workers {threads} \
            --mode {params.mode_flag} \
            --genome {input.genome} \
            --modality {params.modality_bigwig} --bigwig {input.bigwigs} \
            --modality {params.modality_splicing} \
                --star-junctions {input.star_junctions} \
                --ssu {input.ssu_parquets} \
            --gtf {input.gtf_parquet} \
            --train-bed {input.train_bed} \
            --val-bed {input.val_bed} \
            --pretrained-weights {input.weights} \
            --gradient-checkpointing \
            --resume auto \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --lr-schedule cosine \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir {params.output_dir} \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --max-grad-norm inf \
            --junction-position-source {params.junction_position_source} \
            {params.junction_top_k_flag} \
            {params.lora_flags} \
            --pretrained-head-samples splice_site:0 \
            --run-name {params.run_name} \
            --max-grad-norm 1.0 \
            --organism human

        rm -f "$FINETUNE_SCRIPT"

        #find {params.output_dir} -name "*.pth" ! -name "best_model.pth" -delete
        """


rule plot_dev_summary:
    """Plot training dynamics and prediction correlations for both dev runs."""
    input:
        checkpoints = expand(
            os.path.join(DEV_OUTPUT_DIR, "run", "{run_name}", "best_model.pth"),
            run_name = RUN_NAMES,
        ),
    output:
        pdf = os.path.join(DEV_OUTPUT_DIR, "summary.pdf"),
    params:
        script   = "src/scripts/plot_run_summary.py",
        run_dirs = " ".join(
            os.path.join(DEV_OUTPUT_DIR, "run", r) for r in RUN_NAMES
        ),
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --run-dirs {params.run_dirs} \
            --output {output.pdf}
        """
