rule download_weights:
    params:
        weights = config["alphagenome_pytorch"]["urls"]["weights"]
    output:
        weights = directory(config["alphagenome_pytorch"]["paths"]["weights"])
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
        hf download {params.weights} model_all_folds.safetensors --local-dir {output.weights}

        echo "Done!"
        """
