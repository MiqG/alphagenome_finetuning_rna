# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Snakemake-based genomics pipeline that downloads and aligns SF3B1-mutant RNA-seq data (MEC1 cell line) and fine-tunes two sequence models on it:
- **AlphaGenome-PyTorch** — fine-tuned via LoRA using `scripts/finetune.py`
- **Borzoi** — fine-tuned via LoRA using the baskerville framework (`hound_transfer.py`)

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

All paths, URLs, and parameters are centralized in `config/config.yaml`, organized as:
- `gencode` — GENCODE v46 genome/annotation URLs and paths
- `rnaseq.sf3b1mut` — metadata TSV and raw data path for SF3B1 RNA-seq
- `alphagenome_pytorch` — Hugging Face weights repo and local path
- `borzoi` — baskerville path, pretrained trunk paths (4 replicates), support files (blacklist, sequences.bed)
- `finetuning.alphagenome.sf3b1mut` — AlphaGenome finetuning hyperparameters and BED files
- `finetuning.borzoi.sf3b1mut` — Borzoi finetuning hyperparameters; `params_lora.json` is **generated dynamically** by `borzoi_make_targets` (not a static file)

## Architecture

**Entry point:** `workflows/Snakefile` — defines globals (`TMP_ROOT`, `SUPPORT_DIR`, `SAVE_PARAMS`), includes all rule modules, and defines the `all` target.

**Global variables defined in Snakefile** (available to all included rule files):
- `TMP_ROOT` — scratch directory for STAR temp files (`~/scratch`)
- `SUPPORT_DIR` — `support/`
- `SAVE_PARAMS` — pandas `to_csv` kwargs (tab-separated, gzipped)
- `SAMPLES`, `STRANDS` — derived from the sf3b1mut metadata at parse time

### Data rule modules (`workflows/rules/data/`)
- `gencode.smk` — downloads GRCh38 FASTA and GENCODE v46 GTF; builds STAR index; converts GTF to parquet
- `alphagenome.smk` — downloads AlphaGenome-PyTorch weights from Hugging Face via `hf` CLI; runs AlphaGenome LoRA finetuning via `torchrun`
- `borzoi.smk` — downloads Borzoi pretrained trunks, blacklist, and sequences.bed from GCS
- `sf3b1mut.smk` — full RNA-seq processing pipeline (see below)

### Model rule modules (`workflows/rules/models/`)
- `borzoi.smk` — Borzoi transfer learning pipeline (see below)

### RNA-seq pipeline (`sf3b1mut.smk`)
1. Download paired-end FASTQs from ENA FTP
2. STAR two-pass alignment (first pass → merge splice junctions → second pass)
3. BAM filtering: chromosomes, MAPQ 255, strand tag via `tagXSstrandedData.awk`
4. Strand-specific bigwig generation via `bamCoverage --binSize 1` (raw counts, no normalization)
5. Mapped read counting via pysam
6. Gene expression matrix merging across samples

### AlphaGenome finetuning (`alphagenome.smk` — `finetune_sf3b1mut`)
- Uses `torchrun --nproc_per_node=4` with sequence parallelism
- Script sourced from `src/alphagenome-pytorch/scripts/finetune.py` (configured via `finetuning.alphagenome.finetune_script`)
- Script is copied to a tmp file before launch to avoid NFS issues under torchrun
- `--bigwig` accepts multiple files (all samples × strands passed as a list)
- Requires `support/finetuning/train.bed` and `support/finetuning/val.bed`

### Borzoi finetuning pipeline (`models/borzoi.smk`)
Multi-step pipeline following the baskerville transfer learning tutorial:
1. `borzoi_bw_to_w5` — converts each bigwig to compressed HDF5 (`.w5`) via `bw_w5.py`
2. `borzoi_make_targets` — builds `targets.txt` (strand pairs, clip/scale params) and generates `params_lora.json` with `head_human.units` set to `N_SAMPLES × 2`
3. `borzoi_make_tfrecords` — runs `hound_data.py` to create TFRecords; uses GENCODE genome (shared with alignment), no umap filtering
4. `borzoi_setup_folds` — runs `setup_folds.py` to arrange train/val/test fold structure
5. `borzoi_transfer` — runs `hound_transfer.py` once per replicate trunk (`{rep}` wildcard, 0–3)

**Key Borzoi design decisions:**
- Genome: reuses `config["gencode"]["paths"]["fasta"]` (not a separate baskerville hg38 download)
- Umap mappability filtering is omitted
- `scale: 0.005` in config assumes ~200bp fragment length for raw coverage bigwigs (1/fragment_length)
- `params_lora.json` is auto-generated from `src/baskerville/tests/data/transfer/json/borzoi_lora.json` — do not create it manually

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
| hound_data.py, hound_transfer.py | Borzoi TFRecord creation and training |
| bw_w5.py, setup_folds.py | Borzoi bigwig conversion and fold setup |
| gsutil | Borzoi support file download from GCS |
| hf CLI | AlphaGenome weight download |

## Data

- `support/ENA_filereport-compendium-sf3b1mut.tsv` — 46 RNA-seq runs (SF3B1 WT and K700E mutant MEC1 cells, ±H3B-8800 treatment)
- Raw data → `data/raw/`, preprocessed → `data/prep/`
- Results → `results/`
- `support/finetuning/` — train/val BED files for AlphaGenome finetuning
- `support/borzoi/` — auto-generated `params_lora.json` (do not edit manually)
