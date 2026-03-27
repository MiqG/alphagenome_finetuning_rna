rule download_genome_sequence:
    params:
        url = config["gencode"]["urls"]["fasta"],
    output:
        fasta = config["gencode"]["paths"]["fasta"],
        index = config["gencode"]["paths"]["fasta"] + ".fai"
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 2*60,
        memory = 4
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
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60,
        memory = 4
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
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = int(0.5*60),
        memory = 12
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
    threads: 20
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60,
        memory = 40
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -euo pipefail

        mkdir -p {output}
        bgzip -cd -@ {threads} {input.fasta} > {output}/genome.fa
        zcat {input.gtf} > {output}/annotation.gtf

        STAR \
            --runThreadN {threads} \
            --runMode genomeGenerate \
            --genomeDir {output} \
            --genomeFastaFiles {output}/genome.fa \
            --sjdbGTFfile {output}/annotation.gtf

        rm {output}/genome.fa {output}/annotation.gtf

        echo "Done!"
        """
