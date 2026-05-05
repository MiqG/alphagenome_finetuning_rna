"""
Development finetuning workflow for AlphaGenome on SF3B1-mutant RNA-seq data.

Selects train intervals (120 top + 50 random) and val intervals (20 top + 10
random) from FOLD_1 beds ranked by splice junction activity (total
uniquely-mapped junction reads across both DEV_SAMPLES, with number of
distinct junctions as tiebreaker). Fine-tunes on two representative samples
(SRR17111303, SRR17111311) using precomputed SSU parquets and GTF annotation
for splice site classification.

Run names follow a four-axis combinatorial scheme:
  {annotated|predicted}_{randinit|pretrinit}_{frozen|lora}_{seed0|seed1}

  annotated / predicted  — splice junction position source
                           (annotated: STAR SJ.out.tab; predicted: top-k=512 splice_site head)
  randinit  / pretrinit  — head weight initialisation
                           (randinit: default truncated-normal; pretrinit: all heads seeded from
                            organism track index 0 of the pretrained model)
  frozen    / lora       — backbone training mode
                           (frozen: linear-probe on frozen backbone; lora: LoRA rank=8 adapters)
  seed0     / seed1      — random seed for weight init and data shuffling

Run with:
    snakemake -s workflows/finetune_dev.smk --use-conda [-n]
"""

import itertools
import os

configfile: "config/config.yaml"

DATA_DIR        = config["rnaseq"]["sf3b1mut"]["path"]
FINETUNE_SCRIPT = config["finetuning"]["alphagenome"]["finetune_script"]
DEV_OUTPUT_DIR  = config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"].replace("sf3b1mut", "dev")
DEV_BED_DIR     = config["preprocessing"]["overfitting"]["dev"]["output_dir"]

DEV_SAMPLES    = config["preprocessing"]["overfitting"]["samples"]
BIGWIG_STRANDS = ["forward", "reverse"]
N_TRAIN_INTERVALS = (
    config["preprocessing"]["overfitting"]["dev"]["n_train_top"]
    + config["preprocessing"]["overfitting"]["dev"]["n_train_random"]
)
JUNCTION_TOP_K = 512
SEEDS = [0]#[0, 1]
EPOCHS = 10

JUNCTION_SOURCES = ["annotated", "predicted"]
HEAD_INITS       = ["randinit", "pretrinit"]
BACKBONE_MODES   = ["frozen", "lora"]

RUN_NAMES = [
    "{}_{}_{}_{}" .format(js, hi, bm, "seed{}".format(s))
    for js, hi, bm, s in itertools.product(JUNCTION_SOURCES, HEAD_INITS, BACKBONE_MODES, SEEDS)
]

# LoRA settings for backbone="lora" runs.
# Targets use substring matching against full dotted module paths.
# Backbone submodules are: encoder, tower, decoder, embedder_1bp.
# Heads live under "heads." — excluded by using backbone-prefixed substrings so
# that freshly-initialised head parameters are not double-wrapped with adapters.
# Rank=8/alpha=8 (scale=1) — conservative scaling for the small dev training set.
LORA_RANK    = 8
LORA_ALPHA   = 8
LORA_TARGETS = "encoder.linear,encoder.proj,tower.linear,tower.proj,tower.fc,decoder.linear,decoder.proj,embedder_1bp.linear,embedder_1bp.proj"

# Pretrained-head samples flag: initialise all modality heads from organism track index 0.
# pretrained track selected with: 
# track_metadata.query("output_type=='rna_seq'").reset_index(drop=True).query("track_name=='EFO:0002067 total RNA-seq'")
# df.query("output_type=='splice_sites_usage'").reset_index(drop=True).query("track_name=='usage_EFO:0002067 total RNA-seq'")
_PRETRINIT_FLAG = (
    "--pretrained-head-samples 'rna_seq:120|391|120|391'" # forward|reverse|forward|reverse
    " --pretrained-head-samples splice_site:0" # just one head
    " --pretrained-head-samples 'splice_usage:140|507|140|507'" # forward|reverse|forward|reverse
    " --pretrained-head-samples splice_junctions:140" # forward and reverse
)


def _junction_source(wc):
    return wc.run_name.split("_")[0]          # "annotated" | "predicted"

def _head_init(wc):
    return wc.run_name.split("_")[1]          # "randinit" | "pretrinit"

def _backbone_mode(wc):
    return wc.run_name.split("_")[2]          # "frozen" | "lora"

def _seed(wc):
    return int(wc.run_name.split("seed")[1])  # 0 | 1


rule all:
    input:
        expand(
            os.path.join(DEV_OUTPUT_DIR, "run", "{run_name}", "best_model.pth"),
            run_name = RUN_NAMES,
        ),
        os.path.join(DEV_OUTPUT_DIR, "run", "summary.pdf"),


rule finetune_dev:
    """Fine-tune AlphaGenome on the dev subset using precomputed SSU parquets."""
    wildcard_constraints:
        run_name = "|".join(RUN_NAMES),
    input:
        weights      = config["alphagenome_pytorch"]["paths"]["weights"],
        genome       = config["gencode"]["paths"]["fasta"],
        gtf_parquet  = config["gencode"]["paths"]["gtf_parquet"],
        train_bed    = os.path.join(DEV_BED_DIR, "train.bed"),
        val_bed      = os.path.join(DEV_BED_DIR, "valid.bed"),
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
        lr                          = 1e-3, #lambda wc: 2e-4 if _head_init(wc) == "pretrinit" else 1e-3,
        warmup_steps                = lambda wc: 0 if _head_init(wc) == "pretrinit" else 2,
        lr_schedule                 = "cosine", #lambda wc: "constant" if _head_init(wc) == "pretrinit" else "cosine",
        epochs                      = EPOCHS,
        batch_size                  = config["finetuning"]["alphagenome"]["sf3b1mut"]["batch_size"],
        gradient_accumulation_steps = 8,
        track_means_samples         = N_TRAIN_INTERVALS,
        save_every_steps            = config["finetuning"]["alphagenome"]["sf3b1mut"]["save_every_steps"],
        output_dir                  = os.path.join(DEV_OUTPUT_DIR, "run"),
        junction_position_source    = lambda wc: _junction_source(wc),
        junction_top_k_flag         = lambda wc: "--junction-top-k {}".format(JUNCTION_TOP_K) if _junction_source(wc) == "predicted" else "",
        pretrained_head_flag        = lambda wc: _PRETRINIT_FLAG if _head_init(wc) == "pretrinit" else "",
        mode_flag                   = lambda wc: "lora" if _backbone_mode(wc) == "lora" else "linear-probe",
        lora_flags                  = lambda wc: (
            "--lora-rank {} --lora-alpha {} --lora-targets {}".format(LORA_RANK, LORA_ALPHA, LORA_TARGETS)
        ) if _backbone_mode(wc) == "lora" else "",
        seed                        = lambda wc: _seed(wc),
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
            --lr-schedule {params.lr_schedule} \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir {params.output_dir} \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --junction-position-source {params.junction_position_source} \
            {params.junction_top_k_flag} \
            {params.pretrained_head_flag} \
            {params.lora_flags} \
            --max-grad-norm 1.0 \
            --log-every 20 \
            --seed {params.seed} \
            --run-name {params.run_name} \
            --organism human \
            --eval-train-pearson

        rm -f "$FINETUNE_SCRIPT"

        RUN_DIR="{params.output_dir}/{params.run_name}"
        find "$RUN_DIR" -name "checkpoint_epoch*.pth" | sort -V | head -n -1 | xargs -r rm -f
        rm -f "$RUN_DIR/checkpoint_preempt.pth"
        """


rule plot_dev_summary:
    """Plot training dynamics and prediction correlations for all dev runs."""
    input:
        checkpoints = expand(
            os.path.join(DEV_OUTPUT_DIR, "run", "{run_name}", "best_model.pth"),
            run_name = RUN_NAMES,
        ),
    output:
        pdf = os.path.join(DEV_OUTPUT_DIR, "run", "summary.pdf"),
    params:
        script   = "src/scripts/plot_run_summary.py",
        run_dirs = " ".join(
            os.path.join(DEV_OUTPUT_DIR, "run", r) for r in RUN_NAMES
        ),
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        python {params.script} \
            --run-dirs {params.run_dirs} \
            --output {output.pdf}
        """
