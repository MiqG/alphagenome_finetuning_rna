FINETUNE_SCRIPT = config["finetuning"]["alphagenome"]["finetune_script"]
ALPHAGENOME_FOLDS_DIR = config["finetuning"]["alphagenome"]["folds_dir"]

# Finetuning supports multiple modalities (rna_seq, atac, etc.) plus splice modalities
# (splice_site, splice_usage, splice_junctions) trained jointly via comma-separated --modality arg

rule finetune_sf3b1mut:
    wildcard_constraints:
        fold = "|".join(ALPHAGENOME_FOLDS)
    input:
        weights = config["alphagenome_pytorch"]["paths"]["weights"],
        genome = config["gencode"]["paths"]["fasta"],
        train_bed = os.path.join(ALPHAGENOME_FOLDS_DIR, "{fold}", "train.bed"),
        val_bed = os.path.join(ALPHAGENOME_FOLDS_DIR, "{fold}", "valid.bed"),
        bigwigs = [
            os.path.join(
                config["rnaseq"]["sf3b1mut"]["path"], "STAR", sample,
                "second_pass.Aligned.sortedByCoord.out.filtered.{strand}.bw".format(strand=strand)
            )
            for sample in [SAMPLES[1], SAMPLES[4]] # DEV
            for strand in STRANDS
        ],
        star_junctions = [
            os.path.join(
                config["rnaseq"]["sf3b1mut"]["path"], "STAR", sample,
                "second_pass.SJ.{strand}.tab".format(strand=strand)
            )
            for sample in [SAMPLES[1], SAMPLES[4]] # DEV
            for strand in ["fwd", "rev"]
        ],
        bigwig_mapping = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", "bigwig_mapping-with_mapped_reads.tsv.gz"
        ),
    output:
        done = touch(os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"], "{fold}", ".done"))
    params:
        num_gpus = 1, #config["finetuning"]["alphagenome"]["sf3b1mut"]["num_gpus"],
        modality_bigwig = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_bigwig"],
        modality_splicing = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality_splicing"],
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        overlap_highres = config["finetuning"]["alphagenome"]["sf3b1mut"]["overlap_highres"],
        lr = config["finetuning"]["alphagenome"]["sf3b1mut"]["lr"],
        warmup_steps = config["finetuning"]["alphagenome"]["sf3b1mut"]["warmup_steps"],
        epochs = config["finetuning"]["alphagenome"]["sf3b1mut"]["epochs"],
        batch_size = config["finetuning"]["alphagenome"]["sf3b1mut"]["batch_size"],
        gradient_accumulation_steps = config["finetuning"]["alphagenome"]["sf3b1mut"]["gradient_accumulation_steps"],
        track_means_samples = config["finetuning"]["alphagenome"]["sf3b1mut"]["track_means_samples"],
        save_every_steps = config["finetuning"]["alphagenome"]["sf3b1mut"]["save_every_steps"],
        output_dir = os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"], "{fold}"),
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
    retries: 0 #DEV 4 # ~ 2 epochs
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
            --train-bed {input.train_bed} \
            --val-bed {input.val_bed} \
            --pretrained-weights {params.pretrained_weights} \
            --gradient-checkpointing \
            --resume auto \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --batch-size {params.batch_size} \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir {params.output_dir} \
            --sequence-parallel \
            --overlap-highres {params.overlap_highres} \
            --sequence-length {params.sequence_length} \
            --track-means-samples {params.track_means_samples} \
            --save-every-steps {params.save_every_steps} \
            --run-name run

        rm -f "$FINETUNE_SCRIPT"
        echo "Done!"
        """
