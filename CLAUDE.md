# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Snakemake-based genomics pipeline to prepare RNA-seq data (SF3B1 mutation study, MEC1 cell line) for fine-tuning the AlphaGenome RNA prediction model. The pipeline handles reference genome download, STAR alignment, and bigwig coverage track generation as inputs to model fine-tuning.

## Running the Pipeline

### Local execution
```bash
snakemake --use-conda -j <cores>
```

### SLURM cluster submission
```bash
./src/scripts/submit_snakemake_slurm.sh "snakemake --use-conda [OPTIONS]"
# Monitor job status:
./src/scripts/status-sacct.sh <SLURM_JOB_ID>
```

### Dry run
```bash
snakemake -n
```

## Configuration

All paths, URLs, and parameters are centralized in `config/config.yaml`. This includes:
- GENCODE v46 reference genome/annotation URLs
- Hugging Face model repo for AlphaGenome weights
- Paths to raw/preprocessed data directories
- RNA-seq metadata file location (`support/ENA_filereport-compendium-sf3b1_mut.tsv`)

## Architecture

**Entry point:** `workflows/Snakefile` — includes three rule modules and defines the `all` target.

**Rule modules:**
- `workflows/rules/data/gencode.smk` — Downloads and indexes GRCh38 FASTA; downloads GENCODE v46 GTF; converts GTF to parquet via pyranges
- `workflows/rules/data/alphagenome.smk` — Downloads AlphaGenome PyTorch model weights from Hugging Face via `hf` CLI
- `workflows/rules/data/rnaseq.smk` — Main RNA-seq processing pipeline (see below)

**RNA-seq pipeline stages (`rnaseq.smk`):**
1. Download paired-end FASTQs from ENA FTP
2. STAR two-pass alignment (first pass → merge splice junctions → second pass)
3. BAM filtering (quality, chromosomes, strand assignment via sambamba)
4. Strand-specific bigwig generation via bamCoverage
5. Mapped read counting via pysam
6. Gene expression matrix merging across samples

**Conda environments:** Defined in `workflows/envs/` (currently empty — need to be populated before execution).

**Key external tools required:** STAR, sambamba, samtools, bamCoverage (deeptools), pigz, pysam, pyranges, pandas.

## Data

- `support/ENA_filereport-compendium-sf3b1_mut.tsv` — 46 RNA-seq runs (SF3B1 WT and K700E mutant MEC1 cells, ±H3B-8800 treatment)
- Raw data → `data/raw/`, preprocessed → `data/prep/`
- Results → `results/`
