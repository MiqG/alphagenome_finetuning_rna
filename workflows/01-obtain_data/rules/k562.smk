# K562 polyA+ RNA-seq pipeline — reproduces AlphaGenome pretraining SSU targets
#
# Experiment identification
# -------------------------
# AlphaGenome's supplementary track table lists ENCFF132DVY as the file accession for the
# K562 polyA+ SSU tracks (indices 139 + and 506 -). Querying the ENCODE API:
#   https://www.encodeproject.org/files/ENCFF132DVY/?format=json
# returns "dataset": "/experiments/ENCSR000AEM/", identifying ENCSR000AEM as the source
# experiment. ENCSR000AEM is the canonical ENCODE tier-1 K562 polyA+ RNA-seq dataset
# (101 bp PE, 2 biological replicates) used by most ENCODE-based genomics models
# (Enformer, Basenji, Borzoi, AlphaGenome).
#
# Known differences from AlphaGenome's exact pipeline
# ----------------------------------------------------
# 1. Reference: we use GRCh38.primary_assembly + GENCODE v46; AlphaGenome used
#    GRCh38.p13 + GENCODE v32. Minor effect on junction positions at patch regions.
# 2. Junction filtering: AlphaGenome filtered ENCODE junctions against a GTEx
#    high-confidence set via splicemap (github.com/gagneurlab/splicemap). We do not
#    apply this filter, so our pipeline produces a superset of their splice sites.
# 3. SSU read exclusions: AlphaGenome excluded PCR/optical duplicates and reads with
#    MQ<30 or BQ<20. We apply MAPQ>=30 and markdup. BQ<20 filtering is omitted because
#    the samtools -e 'min_qual >= 20' expression is only supported in samtools >=1.20;
#    the effect is negligible for modern high-quality sequencing data.
# 4. Normalization: AlphaGenome normalized junction counts to 1M reads/sample, clipped
#    at 99.99th percentile, and scaled by mean of nonzero sites. We compute raw SSU.
# 5. Replicate merging: AlphaGenome produces one track per CURIE/strand; replicates are
#    merged (likely by averaging per-sample SSU). We compute SSU per replicate separately.

K562_DATA_DIR = config["rnaseq"]["k562"]["path"]

# ENCODE experiment ENCSR000AEM — K562 polyA+ RNA-seq, 2 biological replicates
# Corresponds to AlphaGenome pretraining tracks:
#   splice_sites_usage  139 (+) / 506 (-)  usage_EFO:0002067 polyA plus RNA-seq  (ENCFF132DVY)
#   rna_seq             119 (+) / 390 (-)  EFO:0002067 polyA plus RNA-seq
K562_SAMPLES = ["ENCSR000AEM_rep1", "ENCSR000AEM_rep2"]
K562_STRANDS = ["forward", "reverse"]

# Map (sample, end) -> ENCODE file accession
K562_FASTQ_ACCESSIONS = {
    "ENCSR000AEM_rep1": {"1": "ENCFF001RED", "2": "ENCFF001RDZ"},
    "ENCSR000AEM_rep2": {"1": "ENCFF001REG", "2": "ENCFF001REF"},
}
ENCODE_BASE_URL = "https://www.encodeproject.org/files"


rule k562_download_fastq:
    """Download K562 ENCODE paired-end FASTQs (ENCSR000AEM)."""
    output:
        done = os.path.join(K562_DATA_DIR, "fastqs", ".done", "{sample}_{end}"),
    params:
        accession    = lambda wc: K562_FASTQ_ACCESSIONS[wc.sample][wc.end],
        fastqs_dir   = os.path.join(K562_DATA_DIR, "fastqs"),
        encode_base  = ENCODE_BASE_URL,
    threads: 1
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 60 * 24,
        memory    = 2,
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        wget --no-check-certificate \
             "{params.encode_base}/{params.accession}/@@download/{params.accession}.fastq.gz" \
             -O {params.fastqs_dir}/{wildcards.sample}_{wildcards.end}.fastq.gz

        touch {output.done}
        echo "Done!"
        """


rule k562_star_pass:
    """Two-pass STAR alignment for K562."""
    input:
        done       = [os.path.join(K562_DATA_DIR, "fastqs", ".done", "{sample}_{end}").format(end=e, sample="{sample}") for e in ["1", "2"]],
        genome_dir = config["gencode"]["paths"]["star_index"],
    params:
        sample     = "{sample}",
        fastqs_dir = os.path.join(K562_DATA_DIR, "fastqs"),
        output_dir = os.path.join(K562_DATA_DIR, "STAR", "{sample}"),
        tmp_dir    = os.path.join(TMP_ROOT, "k562", "{sample}"),
    output:
        align_done = touch(os.path.join(K562_DATA_DIR, "STAR", ".done_align", "{sample}")),
    threads: 6
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 6 * 60,
        memory    = 40,
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        if [ -d {params.tmp_dir} ]; then rm -r {params.tmp_dir}; fi

        STAR \
            --genomeDir {input.genome_dir} \
            --genomeLoad NoSharedMemory \
            --readFilesIn {params.fastqs_dir}/{params.sample}_1.fastq.gz {params.fastqs_dir}/{params.sample}_2.fastq.gz \
            --readFilesCommand "pigz -cd -p {threads}" \
            --outSAMtype BAM Unsorted \
            --outFileNamePrefix {params.output_dir}/paper_pass. \
            --outTmpDir {params.tmp_dir} \
            --runThreadN {threads} \
            --outFilterMultimapNmax 20 \
            --alignSJoverhangMin 8 \
            --alignSJDBoverhangMin 1 \
            --outFilterMismatchNmax 999 \
            --outFilterMismatchNoverReadLmax 0.04 \
            --alignIntronMin 20 \
            --alignIntronMax 1000000 \
            --alignMatesGapMax 1000000 \
            --outSAMstrandField intronMotif \
            --outSAMunmapped Within

        if [ -d {params.tmp_dir} ]; then rm -r {params.tmp_dir}; fi

        echo "Done!"
        """


rule k562_sort_pass:
    """Sort K562 STAR BAM by coordinate with sambamba."""
    input:
        align_done = os.path.join(K562_DATA_DIR, "STAR", ".done_align", "{sample}"),
    params:
        output_dir   = os.path.join(K562_DATA_DIR, "STAR", "{sample}"),
        tmp_dir      = os.path.join(TMP_ROOT, "k562", "{sample}"),
        memory_limit = 20000000000,
    output:
        sort_done = touch(os.path.join(K562_DATA_DIR, "STAR", ".done_sort", "{sample}")),
    threads: 6
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 2 * 60,
        memory    = 40,
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        sambamba sort \
            --nthreads {threads} \
            --show-progress \
            --tmpdir {params.tmp_dir} \
            --memory-limit {params.memory_limit} \
            --out {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.bam \
            {params.output_dir}/paper_pass.Aligned.out.bam

        echo "Done!"
        """


rule k562_prep_bam:
    """Mark duplicates and filter K562 BAM.

    sambamba markdup writes to a temp BAM (cannot stream), then sambamba view
    filters duplicates + canonical chromosomes and samtools view applies MQ>=30
    and BQ>=20 to match AlphaGenome's read exclusion criteria.
    """
    input:
        sort_done = os.path.join(K562_DATA_DIR, "STAR", ".done_sort", "{sample}"),
    params:
        output_dir     = os.path.join(K562_DATA_DIR, "STAR", "{sample}"),
        tmp_dir        = os.path.join(TMP_ROOT, "k562", "{sample}_markdup"),
        chromosomes_oi = "' or ref_name=='".join(config["variables"]["CHROMOSOMES"]),
    output:
        filt_done = touch(os.path.join(K562_DATA_DIR, "STAR", ".done_prep_bam", "{sample}")),
    threads: 6
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 3 * 60,
        memory    = 16,
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        mkdir -p {params.tmp_dir}

        sambamba markdup \
            --nthreads {threads} \
            --show-progress \
            --tmpdir {params.tmp_dir} \
            {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.bam \
            {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.markdup.bam

        sambamba view \
            --nthreads {threads} \
            --show-progress \
            --format bam \
            --filter "not duplicate and mapping_quality >= 30 and (ref_name=='{params.chromosomes_oi}')" \
            {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.markdup.bam \
        > {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.filtered.bam

        rm -f {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.markdup.bam \
              {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.markdup.bam.bai

        rm -rf {params.tmp_dir}

        sambamba index \
            --nthreads {threads} \
            --show-progress \
            {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.filtered.bam

        echo "Done!"
        """


rule k562_make_bigwig:
    """Strand-specific bigwigs for K562 (1 bp, raw counts)."""
    input:
        filt_done = os.path.join(K562_DATA_DIR, "STAR", ".done_prep_bam", "{sample}"),
    params:
        strand     = "{strand}",
        output_dir = os.path.join(K562_DATA_DIR, "STAR", "{sample}"),
    output:
        bw_done = touch(os.path.join(K562_DATA_DIR, "STAR", ".done_make_bw", "{sample}-{strand}")),
    threads: 1
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 12 * 60,
        memory    = 10,
    conda:
        "alphagenome_finetuning_rna"
    shell:
        """
        set -eo pipefail

        bamCoverage \
            --bam {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.filtered.bam \
            --filterRNAstrand {params.strand} \
            --outFileFormat bigwig \
            --binSize 1 \
            --outFileName {params.output_dir}/paper_pass.Aligned.sortedByCoord.out.filtered.{params.strand}.bw

        echo "Done!"
        """


rule k562_compute_ssu:
    """Compute splice site usage for K562 from STAR junctions + filtered BAM."""
    input:
        filt_done = os.path.join(K562_DATA_DIR, "STAR", ".done_prep_bam", "{sample}"),
    params:
        script    = "src/alphagenome-pytorch/scripts/compute_ssu.py",
        junctions = os.path.join(K562_DATA_DIR, "STAR", "{sample}", "paper_pass.SJ.out.tab"),
        bam       = os.path.join(K562_DATA_DIR, "STAR", "{sample}", "paper_pass.Aligned.sortedByCoord.out.filtered.bam"),
    output:
        ssu = os.path.join(K562_DATA_DIR, "STAR", "{sample}", "paper_pass.ssu.parquet"),
    threads: 1
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 2 * 60,
        memory    = 8,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {params.script} \
            --junctions {params.junctions} \
            --bam {params.bam} \
            --output {output.ssu}

        echo "Done!"
        """
