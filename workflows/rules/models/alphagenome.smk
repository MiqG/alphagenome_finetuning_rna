FINETUNE_SCRIPT = config["finetuning"]["alphagenome"]["finetune_script"]
ALPHAGENOME_FOLDS_DIR = config["finetuning"]["alphagenome"]["folds_dir"]



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
            for sample in SAMPLES[:2]
            for strand in STRANDS
        ],
        bigwig_mapping = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", "bigwig_mapping-with_mapped_reads.tsv.gz"
        ),
    output:
        done = touch(os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"], "{fold}", ".done"))
    params:
        num_gpus = config["finetuning"]["alphagenome"]["sf3b1mut"]["num_gpus"],
        modality = config["finetuning"]["alphagenome"]["sf3b1mut"]["modality"],
        sequence_length = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"],
        overlap_highres = config["finetuning"]["alphagenome"]["sf3b1mut"]["overlap_highres"],
        lr = config["finetuning"]["alphagenome"]["sf3b1mut"]["lr"],
        epochs = config["finetuning"]["alphagenome"]["sf3b1mut"]["epochs"],
        gradient_accumulation_steps = config["finetuning"]["alphagenome"]["sf3b1mut"]["gradient_accumulation_steps"],
        output_dir = os.path.join(config["finetuning"]["alphagenome"]["sf3b1mut"]["output_dir"], "{fold}"),
        pretrained_weights = os.path.join(
            config["alphagenome_pytorch"]["paths"]["weights"], "model_all_folds.safetensors"
        ),
    threads: 4  # number of GPUs
    resources:
        gres = "gpu:4",
        partition = "gpu_diasfrazer",
        runtime = 24*60,  # 24h
        memory = 80  # G
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        # Copy finetune script to tmp to avoid NFS issues under torchrun
        FINETUNE_SCRIPT=$(mktemp /tmp/finetune_XXXXXX.py)
        cp {FINETUNE_SCRIPT} "$FINETUNE_SCRIPT"

        torchrun --nproc_per_node={params.num_gpus} "$FINETUNE_SCRIPT" \
            --mode lora \
            --genome {input.genome} \
            --modality {params.modality} \
            --bigwig {input.bigwigs} \
            --train-bed {input.train_bed} \
            --val-bed {input.val_bed} \
            --pretrained-weights {params.pretrained_weights} \
            --gradient-checkpointing \
            --no-save-checkpoints \
            --lr {params.lr} \
            --warmup-steps 0 \
            --gradient-accumulation-steps {params.gradient_accumulation_steps} \
            --epochs {params.epochs} \
            --output-dir {params.output_dir} \
            --sequence-parallel \
            --overlap-highres {params.overlap_highres} \
            --sequence-length {params.sequence_length}

        rm -f "$FINETUNE_SCRIPT"
        echo "Done!"
        """
