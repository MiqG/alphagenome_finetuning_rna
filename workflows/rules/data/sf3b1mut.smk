import os
import numpy as np
import pandas as pd

DATA_DIR = config["rnaseq"]["sf3b1mut"]["path"]

# paired end reads
ENDS = ["1","2"]
STRANDS = ["forward","reverse"]

# get urls and fastq sizes
metadata = pd.read_table(config["rnaseq"]["sf3b1mut"]["metadata"])
metadata = metadata.loc[metadata["library_source"]=="TRANSCRIPTOMIC"]

## URLS to download
URLS = metadata["fastq_ftp"].str.split(";").str[0].apply(os.path.dirname).to_list()
URLS = {os.path.basename(url): url for url in URLS}
SAMPLES = list(URLS.keys())
N_SAMPLES = len(SAMPLES)

## fastq sizes
SIZES = metadata.set_index("run_accession")["fastq_bytes"].astype("str").str.split(";").apply(lambda x: max(np.array(x, dtype=int))).to_dict()
SIZE_THRESH = 5e9

rule download_fastq:
    params:
        sample = "{sample}",
        end = "{end}",
        url = lambda wildcards: URLS[wildcards.sample],
        fastqs_dir = os.path.join(DATA_DIR,"fastqs")
    output:
        download_done = os.path.join(DATA_DIR,"fastqs",".done","{sample}_{end}")
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 3600*2, # 2h
        memory = 2
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        # download
        echo "Downloading {params.sample}..."
        
        wget --user-agent="Chrome" \
             --no-check-certificate \
             {params.url}/{params.sample}_{params.end}.fastq.gz \
             -O {params.fastqs_dir}/{params.sample}_{params.end}.fastq.gz
        
        touch {output.download_done}
        echo "Finished downloading {params.sample}."
        echo $(date)
        
        echo "Done!"
        """
        
rule star_first_pass:
    input:
        download_done = [os.path.join(DATA_DIR,"fastqs",".done","{sample}_{end}").format(end=end, sample="{sample}") for end in ENDS],
        genome_dir = config["gencode"]["paths"]["star_index"]
    params:
        sample = "{sample}",
        fastqs_dir = os.path.join(DATA_DIR,"fastqs"),        
        output_dir = os.path.join(DATA_DIR,"STAR","{sample}"),
        tmp_dir = os.path.join(TMP_ROOT,"{sample}"),
    output:
        align_done = touch(os.path.join(DATA_DIR,"STAR",".done_align_first","{sample}"))
    threads: 6
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 6*60, # h
        memory = 40 # G
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail
        
        echo $(ulimit -n)
        ulimit -n 2048 # otherwise error
        echo $(ulimit -n)
        
        if [ -d {params.tmp_dir} ]; then
          rm -r {params.tmp_dir}
        fi

        nice STAR \
            --genomeDir {input.genome_dir} \
            --genomeLoad NoSharedMemory \
            --readFilesIn {params.fastqs_dir}/{params.sample}_1.fastq.gz {params.fastqs_dir}/{params.sample}_2.fastq.gz \
            --readFilesCommand "pigz -cd -p {threads}" \
            --outSAMtype BAM Unsorted \
            --outFileNamePrefix {params.output_dir}/first_pass. \
            --outTmpDir {params.tmp_dir} \
            --runThreadN {threads}
        
        if [ -d {params.tmp_dir} ]; then
          rm -r {params.tmp_dir}
        fi
        
        echo "Done!"
        """
        
rule merge_first_pass_splice_junctions:
    input:
        align_done = os.path.join(DATA_DIR,"STAR",".done_align_first","{sample}")
    params:
        output_dir = os.path.join(DATA_DIR,"STAR","{sample}")
    output:
        merge_junctions_done = touch(os.path.join(DATA_DIR,"STAR",".done_merge_junc","{sample}"))
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60, # h
        memory = 2 # GB
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail
        
        cat {params.output_dir}/first_pass.SJ.out.tab \
        | awk '$7 >= 3' \
        | cut -f1-4 \
        | sort -u \
        > {params.output_dir}/first_pass.SJ.out.merged.tab
        
        echo "Done!"
        """
        
rule star_second_pass:
    input:
        merge_junctions_done = os.path.join(DATA_DIR,"STAR",".done_merge_junc","{sample}"),
        genome_dir = config["gencode"]["paths"]["star_index"]
    params:
        sample = "{sample}",
        fastqs_dir = os.path.join(DATA_DIR,"fastqs"),        
        output_dir = os.path.join(DATA_DIR,"STAR","{sample}"),
        tmp_dir = os.path.join(TMP_ROOT,"{sample}"),
        memory_limit = 20000000000
    output:
        align_done = touch(os.path.join(DATA_DIR,"STAR",".done_align_second","{sample}"))
    threads: 6
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 6*60, # h
        memory = 40 # G
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail
        
        if [ -d {params.tmp_dir} ]; then
          rm -r {params.tmp_dir}
        fi

        STAR \
            --genomeDir {input.genome_dir} \
            --genomeLoad NoSharedMemory \
            --readFilesIn {params.fastqs_dir}/{params.sample}_1.fastq.gz {params.fastqs_dir}/{params.sample}_2.fastq.gz \
            --readFilesCommand "pigz -cd -p {threads}" \
            --outSAMtype BAM Unsorted \
            --outFileNamePrefix {params.output_dir}/second_pass. \
            --outTmpDir {params.tmp_dir} \
            --runThreadN {threads} \
            --sjdbFileChrStartEnd {params.output_dir}/first_pass.SJ.out.merged.tab \
            --outFilterType BySJout \
            --outFilterMultimapNmax 20 \
            --alignSJoverhangMin 8 \
            --alignSJDBoverhangMin 1 \
            --outFilterMismatchNmax 999 \
            --outFilterMismatchNoverReadLmax 0.04 \
            --alignIntronMin 20 \
            --alignIntronMax 1000000 \
            --alignMatesGapMax 1000000 \
            --quantMode GeneCounts TranscriptomeSAM
                    
        # sort BAM
        sambamba sort \
            --nthreads {threads} \
            --show-progress \
            --tmpdir {params.tmp_dir} \
            --memory-limit {params.memory_limit} \
            --out {params.output_dir}/second_pass.Aligned.sortedByCoord.out.bam \
            {params.output_dir}/second_pass.Aligned.out.bam

        if [ -d {params.tmp_dir} ]; then
          rm -r {params.tmp_dir}
        fi

        echo "Done!"
        """   
        
rule star_combine_genexpr:
    input:
        [os.path.join(DATA_DIR,"STAR",".done_align_second","{sample}").format(sample=sample) for sample in SAMPLES]
    output:
        counts = os.path.join(DATA_DIR,"STAR","merged.second_pass.ReadsPerGene.out.tab.gz")
    params:
        counts_col = 2,
        star_out = os.path.join(DATA_DIR,"STAR"),
        done_dir = os.path.join(DATA_DIR,"STAR",".done_align_second")
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 6*60, # h
        memory = 2 # G
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -euo pipefail
        
        echo "Combining gene expression..."
        
        SAMPLES=$(ls {params.done_dir})
        COUNTER=0
        for SAMPLE in $SAMPLES; do
            
            echo $SAMPLE
            
            if [ "$COUNTER" -eq "0" ]; then
                # for first iteration, add gene column
                cat {params.star_out}/$SAMPLE/second_pass.ReadsPerGene.out.tab | sed 1,4d | cut -f1,{params.counts_col} | sed "1i ENSEMBL\t$SAMPLE" > {params.star_out}/tmp
                
            else
                # take only mRNA counts for the rest
                paste {params.star_out}/tmprev <(cat {params.star_out}/$SAMPLE/second_pass.ReadsPerGene.out.tab | sed 1,4d | cut -f{params.counts_col} | sed "1i $SAMPLE") > {params.star_out}/tmp
            fi
            
            mv {params.star_out}/tmp {params.star_out}/tmprev
            
            COUNTER=$[$COUNTER +1]
        done
        
        mv {params.star_out}/tmprev {output.counts}
        gzip -f {output.counts}
        mv {output.counts}.gz {output.counts}
        
        echo "Done!"
        """

rule prep_bam:
    input:
        align_done = os.path.join(DATA_DIR,"STAR",".done_align_second","{sample}")
    params:
        star_scripts_dir = "~/repositories/STAR-2.7.11a/extras/scripts",
        output_dir = os.path.join(DATA_DIR,"STAR","{sample}"),
        chromosomes_oi = "' or ref_name=='".join(
            pd.read_table(os.path.join(SUPPORT_DIR,"chromosomes_oi.txt"), header=None)[0].to_list()
        )
    output:
        filt_done = touch(os.path.join(DATA_DIR,"STAR",".done_prep_bam","{sample}"))
    threads: 6
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60, # h
        memory = 2 # G
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail
        
        echo "Indexing BAM..."
        sambamba index \
            --show-progress \
            --nthreads {threads} \
            {params.output_dir}/second_pass.Aligned.sortedByCoord.out.bam
        
        echo "Filtering chromosomes and low quality mappings, and adding strand in BAM..."
        sambamba view \
            --nthreads {threads} \
            --show-progress \
            --format bam \
            --filter "ref_name=='{params.chromosomes_oi}'" \
            {params.output_dir}/second_pass.Aligned.sortedByCoord.out.bam \
        | samtools view \
            --threads {threads} \
            -q 255 \
            -h \
        | awk \
            -v strType=2 \
            -f {params.star_scripts_dir}/tagXSstrandedData.awk \
        | samtools view \
            --threads {threads} \
            -bS \
            - \
        > {params.output_dir}/second_pass.Aligned.sortedByCoord.out.filtered.bam
        
        echo "Done!"
        """
        
rule make_bigwig:
    input:
        filt_done = os.path.join(DATA_DIR,"STAR",".done_prep_bam","{sample}")
    output:
        bw_done = touch(os.path.join(DATA_DIR,"STAR",".done_make_bw","{sample}-{strand}"))
    params:
        strand = "{strand}",
        output_dir = os.path.join(DATA_DIR,"STAR","{sample}")        
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 12*60, # h in minutes
        memory = 10 # G
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        echo "Indexing BAM..."
        sambamba index \
            --show-progress \
            --nthreads {threads} \
            {params.output_dir}/second_pass.Aligned.sortedByCoord.out.filtered.bam
            
        echo "Making coverage..."
        bamCoverage \
            --bam {params.output_dir}/second_pass.Aligned.sortedByCoord.out.filtered.bam \
            --filterRNAstrand {params.strand} \
            --outFileFormat "bigwig" \
            --binSize 1 \
            --outFileName {params.output_dir}/second_pass.Aligned.sortedByCoord.out.filtered.{params.strand}.bw
        
        echo "Done!"
        """
        
rule make_bigwig_mapping:
    input:
        bws_done = [os.path.join(DATA_DIR,"STAR",".done_make_bw","{sample}-{strand}").format(sample=o, strand=s) for o in SAMPLES for s in STRANDS]
    output:
        bigwig_mapping = os.path.join(DATA_DIR,"STAR","bigwig_mapping.tsv.gz")
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = int(0.5*60), # h in minutes
        memory = 2 # G
    run:
        import pandas as pd
        
        bigwig_mapping = []
        for f in input.bws_done:
            filename = os.path.basename(f)
            sample, strand = filename.split("-")
            output_dir = os.path.join(os.path.dirname(os.path.dirname(f)), sample)
            bigwig_file = os.path.join(output_dir, "second_pass.Aligned.sortedByCoord.out.filtered.{strand}.bw").format(strand=strand)
            
            assert os.path.isfile(bigwig_file)
            
            bigwig_mapping.append({
                "bigwig_file": bigwig_file,
                "sampleID": sample,
                "strand": strand,
                "clip_soft": 384 # bulk
            })
            
        bigwig_mapping = pd.DataFrame(bigwig_mapping)
        
        bigwig_mapping.to_csv(output.bigwig_mapping, **SAVE_PARAMS)
        
        print("Done!")
        
        
rule get_mapped_reads:
    input:
        filt_done = [os.path.join(DATA_DIR,"STAR",".done_make_bw","{sample}-{strand}").format(sample=o, strand=s) for o in SAMPLES for s in STRANDS],
        mapping = os.path.join(DATA_DIR,"STAR","bigwig_mapping.tsv.gz")
    output:
        mapped_reads = os.path.join(DATA_DIR,"STAR","bigwig_mapping-with_mapped_reads.tsv.gz")
    threads: 12
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60, # h in minutes
        memory = 10 # G
    run:
        import pandas as pd
        import pysam
        from tqdm import tqdm
        from joblib import Parallel, delayed
        
        # load
        mapping = pd.read_table(input.mapping)
        n_jobs = threads
        
        # get mapped reads in bam file
        def forward_strand_read(read):
            return (
                not read.is_unmapped
                and not read.is_secondary
                and not read.is_duplicate
                and not read.is_qcfail
                and not read.is_reverse  # forward strand only
            )
        
        def get_mapped_reads(row):
            # get mapped reads in bam
            bigwig_file = row["bigwig_file"]
            in_bam = bigwig_file.replace(".forward.bw",".bam").replace(".reverse.bw",".bam")
            samfile = pysam.AlignmentFile(in_bam)
            mapped_reads = float(samfile.mapped)
            
            # prepare output
            row["mapped_reads_total"] = mapped_reads
            
            # add reverse
            forward_mapped = samfile.count(read_callback=forward_strand_read)
            row["mapped_reads_forward"] = forward_mapped
            row["mapped_reads_reverse"] = mapped_reads - forward_mapped
            
            # fill overall with corresponding strand
            if bigwig_file.endswith(".forward.bw"):
                row["mapped_reads"] = row["mapped_reads_forward"]
            elif bigwig_file.endswith(".reverse.bw"):
                row["mapped_reads"] = row["mapped_reads_reverse"]
            
            return row
        
        mapped_reads = Parallel(n_jobs)(
            delayed(get_mapped_reads)(row)
            for idx, row in tqdm(mapping.iterrows(), total=len(mapping))
        )
        mapped_reads = pd.DataFrame(mapped_reads)
        
        # save
        mapped_reads.to_csv(output.mapped_reads, **SAVE_PARAMS)
        
        print("Done!")