rule download_weights:
    params:
        weights = config["alphagenome_pytorch"]["urls"]["weights"]
    output:
        weights = directory(config["alphagenome_pytorch"]["paths"]["weights"])
    conda:
        "publication_likelihood"
    shell:
        """
        hf download {params.weights} model_all_folds.safetensors --local-dir {output.weights}
        
        echo "Done!"
        """