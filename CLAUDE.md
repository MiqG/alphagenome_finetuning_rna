# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Snakemake-based genomics pipeline organized as three numbered workflows that download and align SF3B1-mutant RNA-seq data (MEC1 cell line) and develop/validate fine-tuning of AlphaGenome-PyTorch on splicing modalities.

## Running the Workflows

Each workflow is independent and has its own Snakefile:

```bash
snakemake -s workflows/01-obtain_data/Snakefile --use-conda -j <cores>
snakemake -s workflows/02-preprocess_data/Snakefile --use-conda -j <cores>
snakemake -s workflows/03-overfitting_single/Snakefile --use-conda -j <cores>
```

### SLURM cluster submission
```bash
./src/scripts/submit_snakemake_slurm.sh "snakemake -s workflows/<N>-<name>/Snakefile --use-conda [OPTIONS]"
# Monitor job status:
./src/scripts/status-sacct.sh <SLURM_JOB_ID>
```

## Configuration

All paths, URLs, and parameters are centralized in `config/config.yaml`, organized as:
- `gencode` — GENCODE v46 genome/annotation URLs and paths
- `rnaseq.sf3b1mut` — metadata TSV and raw data path for SF3B1 RNA-seq
- `alphagenome_pytorch` — Hugging Face weights repo and local path
- `finetuning.alphagenome` — finetune script path, fold BED dirs, sequences_bed_url, and `sf3b1mut` hyperparameters
- `preprocessing.overfitting` — sample list, fold, and output dirs for single interval selection

## Architecture

There is no top-level `workflows/Snakefile`. Each numbered workflow is self-contained.

### Workflow structure

```
workflows/
  01-obtain_data/
    Snakefile          — defines TMP_ROOT, SUPPORT_DIR, SAVE_PARAMS globals; includes all data rules
    rules/
      gencode.smk      — downloads GRCh38 FASTA and GENCODE v46 GTF; builds STAR index; converts GTF to parquet
      alphagenome.smk  — downloads AlphaGenome-PyTorch weights from Hugging Face via hf CLI; downloads sequences_human.bed.gz and converts to per-fold BEDs
      sf3b1mut.smk     — full RNA-seq processing pipeline (see below); defines SAMPLES and STRANDS
      comparison_ssu.smk — validates compute_ssu.py and get_star_junctions.py against SpliSER/STAR on chr1
  02-preprocess_data/
    Snakefile          — selects single genomic intervals for overfitting experiments
  03-overfitting_single/
    Snakefile          — single-interval overfitting (500 epochs, 1 GPU); three experiment groups
  rules/
    models/
      alphagenome.smk  — AlphaGenome LoRA finetuning rule (finetune_sf3b1mut) for monolithic use
```

### Global variables in `01-obtain_data/Snakefile`
- `TMP_ROOT` — scratch directory for STAR temp files (`~/scratch`)
- `SUPPORT_DIR` — `support/`
- `SAVE_PARAMS` — pandas `to_csv` kwargs (tab-separated, gzipped)

### Variables defined in `01-obtain_data/rules/sf3b1mut.smk`
- `SAMPLES` — hardcoded list of ENA run accessions (SRR17111303, SRR17111311, + 3 more)
- `STRANDS` — `["forward", "reverse"]`

### RNA-seq pipeline (`sf3b1mut.smk`)
1. Download paired-end FASTQs from ENA FTP
2. STAR two-pass alignment (first pass → merge splice junctions → second pass)
3. BAM filtering: chromosomes, MAPQ 255, strand tag via `tagXSstrandedData.awk`
4. Strand-specific bigwig generation via `bamCoverage --binSize 1` (raw counts, no normalization)
5. Splice site usage via `compute_ssu.py` → zstd-compressed parquet
6. Mapped read counting via pysam
7. Gene expression matrix merging across samples

### AlphaGenome finetuning (workflow 03)
- Workflow 03 defines its own run matrix and calls `torchrun` directly
- Script path set via `FINETUNE_SCRIPT` from `config["finetuning"]["alphagenome"]["finetune_script"]`, pointing to `src/alphagenome-pytorch/scripts/finetune.py`
- Script is copied to a tmp file before launch to avoid NFS issues under torchrun
- Key flags: `--mode` (linear-probe or lora), `--rope-init` (zeros or truncated_normal), `--junction-loss` (original, normalized, sparse), `--junction-position-source` (annotated or predicted), `--pretrained-head-samples`

## Conda environments

Defined in `envs/`. All `conda:` directives in rule files use bare env names (e.g. `"alphagenome_pytorch"`), not yaml file paths.

| File | Env name | Used for |
|------|----------|----------|
| `general.yaml` | `alphagenome_finetuning_rna` | RNA-seq processing (STAR, sambamba, samtools, deeptools, pysam, pyranges) |
| `alphagenome_pytorch.yaml` | `alphagenome_pytorch` | AlphaGenome weight download and finetuning (PyTorch, alphagenome-pytorch[finetuning], hf CLI) |
| `spliser.yaml` | `spliser` | SpliSER-based SSU validation |

## External dependencies

| Tool | Used by |
|------|---------|
| STAR | sf3b1mut alignment |
| sambamba, samtools | BAM filtering and indexing |
| bamCoverage (deeptools) | bigwig generation |
| pigz | parallel FASTQ decompression |
| pysam | mapped read counting |
| pyranges | GTF → parquet |
| torchrun | AlphaGenome multi-GPU training |
| hf CLI | AlphaGenome weight download |

## Data

- `support/ENA_filereport-compendium-sf3b1mut.tsv` — RNA-seq runs (SF3B1 WT and K700E mutant MEC1 cells, ±H3B-8800 treatment)
- Raw data → `data/raw/`, preprocessed → `data/prep/`
- Results → `results/`
