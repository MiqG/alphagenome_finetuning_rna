rule alphagenome_download_sequences_bed:
    """Download Borzoi sequences_human.bed.gz (fold assignments for hg38)."""
    output:
        sequences_bed_gz = config["finetuning"]["alphagenome"]["sequences_bed_gz"],
    params:
        url = config["borzoi"]["support"]["sequences_bed_url"],
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 15,
        memory = 2
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        wget -q -O {output.sequences_bed_gz} {params.url}
        echo "Done!"
        """


rule alphagenome_make_folds:
    """Convert Borzoi sequences.bed to per-fold AlphaGenome 1 Mb BED files."""
    input:
        sequences_bed = config["finetuning"]["alphagenome"]["sequences_bed_gz"],
    output:
        beds = expand(
            os.path.join(config["finetuning"]["alphagenome"]["folds_dir"], "{fold}", "{split}.bed"),
            fold=ALPHAGENOME_FOLDS,
            split=["train", "valid", "test"],
        ),
    params:
        folds_dir = config["finetuning"]["alphagenome"]["folds_dir"],
        script = config["finetuning"]["alphagenome"]["convert_folds_script"],
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 30,
        memory = 4
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --input {input.sequences_bed} \
            --output-dir {params.folds_dir} \
            --organism human

        echo "Done!"
        """


rule download_weights:
    params:
        repository = config["alphagenome_pytorch"]["urls"]["repository"],
        weights = config["alphagenome_pytorch"]["urls"]["weights"]
    output:
        weights = config["alphagenome_pytorch"]["paths"]["weights"]
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 2*60,
        memory = 4
    conda:
        "alphagenome_pytorch"
    shell:
        """
        # Create parent directory
        mkdir -p "$(dirname {output.weights})"

        # Download to parent directory to get file directly (not nested)
        hf download {params.repository} {params.weights} --local-dir "$(dirname {output.weights})"

        echo "Done!"
        """
