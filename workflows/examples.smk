"""
Standalone Snakefile for AlphaGenome inference examples.

This workflow is independent of the main pipeline. It:
1. Extracts track metadata from the JAX AlphaGenome model (optional; needs alphagenome JAX env)
2. Visualizes AlphaGenome pretrained predictions vs real K562 data for LDLR and TSPYL2,
   using K562 output head track indices (rna_seq:119, splice_usage:139, splice_junction:139)

Run with:
    snakemake -s workflows/examples.smk --use-conda
"""

import os

configfile: "config/config.yaml"

OUTPUT_DIR     = config["examples"]["alphagenome"]["output_dir"]
TRACK_METADATA = config["examples"]["alphagenome"]["track_metadata"]

VIZ_SCRIPT             = "src/scripts/visualize_gene_pretrained.py"
EXTRACT_METADATA_SCRIPT = "src/alphagenome-pytorch/scripts/extract_track_metadata.py"

# K562 output head track indices (polyA plus RNA-seq, EFO:0002067)
K562_TRACK_RNA      = 119   # rna_seq, + strand
K562_TRACK_USAGE    = 139   # splice_sites_usage, + strand
K562_TRACK_JUNCTION = 139   # splice_sites_junction

# Gene regions of interest (0-based, hg38)
# ROI gives ~30 kb upstream + gene body + ~30 kb downstream
GENES = {
    "LDLR":   {"chrom": "chr19", "roi_start": 11_066_619, "roi_end": 11_136_619, "strand": "+"},
    "TSPYL2": {"chrom": "chrX",  "roi_start": 53_052_000, "roi_end": 53_119_000, "strand": "+"},
}

K562_SAMPLE = "SRR2103591"


rule all:
    input:
        expand(
            os.path.join(OUTPUT_DIR, "{gene}_k562_pretrained.pdf"),
            gene=list(GENES.keys()),
        ),


rule alphagenome_extract_track_metadata:
    """Extract track metadata from JAX AlphaGenome model.

    Requires the JAX alphagenome and alphagenome_research packages.
    Run this once to generate track_metadata.parquet for track selection.
    """
    output:
        parquet = TRACK_METADATA,
    params:
        script = EXTRACT_METADATA_SCRIPT,
    conda:
        "publication_likelihood"
    shell:
        """
        python {params.script} --output-file {output.parquet}
        """


rule alphagenome_visualize_gene:
    """Visualize AlphaGenome pretrained predictions vs real K562 data for a gene.

    Uses K562 polyA RNA-seq output head tracks as initialized in the finetuning workflow.
    Per-position parquets include a 'nucleotide' column with the reference base at each position.
    """
    input:
        weights        = config["alphagenome_pytorch"]["paths"]["weights"],
        genome         = config["gencode"]["paths"]["fasta"],
        bigwig_fwd     = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", K562_SAMPLE,
            "second_pass.Aligned.sortedByCoord.out.filtered.forward.bw",
        ),
        bigwig_rev     = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", K562_SAMPLE,
            "second_pass.Aligned.sortedByCoord.out.filtered.reverse.bw",
        ),
        star_junctions = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", K562_SAMPLE,
            "second_pass.SJ.out.tab",
        ),
    output:
        pdf = os.path.join(OUTPUT_DIR, "{gene}_k562_pretrained.pdf"),
    params:
        script      = VIZ_SCRIPT,
        chrom       = lambda wildcards: GENES[wildcards.gene]["chrom"],
        roi_start   = lambda wildcards: GENES[wildcards.gene]["roi_start"],
        roi_end     = lambda wildcards: GENES[wildcards.gene]["roi_end"],
        strand      = lambda wildcards: GENES[wildcards.gene]["strand"],
        track_rna   = K562_TRACK_RNA,
        track_usage = K562_TRACK_USAGE,
        track_junc  = K562_TRACK_JUNCTION,
    resources:
        gres      = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime   = 60,
        memory    = 40,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --gene-name    {wildcards.gene} \
            --chrom        {params.chrom} \
            --roi-start    {params.roi_start} \
            --roi-end      {params.roi_end} \
            --strand       {params.strand} \
            --weights      {input.weights} \
            --genome       {input.genome} \
            --bigwig-fwd   {input.bigwig_fwd} \
            --bigwig-rev   {input.bigwig_rev} \
            --star-junctions {input.star_junctions} \
            --track-index-rna             {params.track_rna} \
            --track-index-splice-usage    {params.track_usage} \
            --track-index-splice-junction {params.track_junc} \
            --output       {output.pdf}
        """
