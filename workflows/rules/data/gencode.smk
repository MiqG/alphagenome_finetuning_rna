rule download_genome_sequence:
    params:
        url = config["gencode"]["urls"]["fasta"],
    output:
        fasta = config["gencode"]["paths"]["fasta"],
        index = config["gencode"]["paths"]["fasta"] + ".fai"
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        wget --user-agent="Chrome" --no-check-certificate {params.url} -O - | gunzip | bgzip -c > {output.fasta}

        samtools faidx {output.fasta}

        echo "Done!"
        """

rule download_genome_annotation:
    params:
        url = config["gencode"]["urls"]["gtf"],
    output:
        gtf = config["gencode"]["paths"]["gtf"]
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        wget --user-agent="Chrome" --no-check-certificate {params.url} -O {output.gtf}

        echo "Done!"
        """

rule gtf_to_parquet:
    input:
        gtf = config["gencode"]["paths"]["gtf"]
    output:
        parquet = config["gencode"]["paths"]["gtf_parquet"]
    run:
        import pyranges as pr
        gtf = pr.read_gtf(input.gtf)
        gtf.df.to_parquet(output.parquet, compression="zstd", index=False)

        print("Done!")

rule build_star_index:
    input:
        gtf = config["gencode"]["paths"]["gtf"],
        fasta = config["gencode"]["paths"]["fasta"]
    output:
        directory(config["gencode"]["paths"]["star_index"])
    conda:
        "alphagenome_finetuning_rna"
    threads: 20
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60, # h
        memory = 40 # G
    shell:
        """
        set -euo pipefail

        STAR \
            --runThreadN {threads} \
            --runMode genomeGenerate \
            --genomeDir {output} \
            --genomeFastaFiles {input.fasta} \
            --sjdbGTFfile {input.gtf}

        echo "Done!"
        """