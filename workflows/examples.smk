"""
Standalone Snakefile for AlphaGenome inference examples.

This workflow is independent of the main pipeline. It:
1. Extracts track metadata from the JAX AlphaGenome model (optional; needs alphagenome JAX env)
2. Visualizes AlphaGenome pretrained predictions for the LDLR gene in HepG2

Run with:
    snakemake -s workflows/examples.smk --use-conda
"""

import os

configfile: "config/config.yaml"

OUTPUT_DIR     = config["examples"]["alphagenome"]["output_dir"]
TRACK_METADATA = config["examples"]["alphagenome"]["track_metadata"]

VIZ_SCRIPT = "src/scripts/visualize_ldlr_pretrained.py"
EXTRACT_METADATA_SCRIPT = "src/alphagenome-pytorch/scripts/extract_track_metadata.py"


rule all:
    input:
        os.path.join(OUTPUT_DIR, "ldlr_hepg2_pretrained.pdf"),


rule alphagenome_extract_track_metadata:
    """Extract track metadata from JAX AlphaGenome model.

    Requires the JAX alphagenome and alphagenome_research packages.
    Run this once to generate track_metadata.parquet for HepG2 track selection.
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


rule alphagenome_visualize_ldlr:
    """Visualize AlphaGenome pretrained predictions vs real K562 data for LDLR gene.

    Plots predicted and real RNA-seq, splice donor/acceptor classification,
    splice site usage, and splice junctions for chr19:11066619-11136619
    (positive strand).
    """
    input:
        weights     = config["alphagenome_pytorch"]["paths"]["weights"],
        genome      = config["gencode"]["paths"]["fasta"],
        metadata    = TRACK_METADATA,
        bigwig_fwd  = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", "SRR2103591",
            "second_pass.Aligned.sortedByCoord.out.filtered.forward.bw",
        ),
        bigwig_rev  = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", "SRR2103591",
            "second_pass.Aligned.sortedByCoord.out.filtered.reverse.bw",
        ),
        star_junctions = os.path.join(
            config["rnaseq"]["sf3b1mut"]["path"], "STAR", "SRR2103591",
            "second_pass.SJ.out.tab",
        ),
    output:
        pdf = os.path.join(OUTPUT_DIR, "ldlr_hepg2_pretrained.pdf"),
    params:
        script = VIZ_SCRIPT,
    resources:
        gres      = "gpu:7g.80gb:1",
        partition = "gpu",
        runtime   = 60,   # minutes
        memory    = 40,   # G
    conda:
        "alphagenome_pytorch"
    shell:
        """
        python {params.script} \
            --weights        {input.weights} \
            --genome         {input.genome} \
            --track-metadata {input.metadata} \
            --bigwig-fwd     {input.bigwig_fwd} \
            --bigwig-rev     {input.bigwig_rev} \
            --star-junctions {input.star_junctions} \
            --output         {output.pdf}
        """
