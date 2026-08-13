# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Snakemake-based genomics pipeline organized as five numbered workflows that download and align SF3B1-mutant RNA-seq data (MEC1 cell line) and develop/validate fine-tuning of AlphaGenome-PyTorch on splicing modalities. A Borzoi fine-tuning pipeline is also included but is secondary.

## Running the Workflows

Each workflow is independent and has its own Snakefile:

```bash
snakemake -s workflows/01-obtain_data/Snakefile --use-conda -j <cores>
snakemake -s workflows/02-preprocess_data/Snakefile --use-conda -j <cores>
snakemake -s workflows/03-overfitting_single/Snakefile --use-conda -j <cores>
snakemake -s workflows/04-overfitting_dev/Snakefile --use-conda -j <cores>
snakemake -s workflows/05-full_finetuning/Snakefile --use-conda -j <cores>
```

### SLURM cluster submission
```bash
./src/scripts/submit_snakemake_slurm.sh "snakemake -s workflows/<N>-<name>/Snakefile --use-conda [OPTIONS]"
# Monitor job status:
./src/scripts/status-sacct.sh <SLURM_JOB_ID>
```

## Cluster operational notes (lessons learned the hard way)

- **Never run heavy Python directly on the login node.** Anything that constructs a real model (`dna_model.create(...)`, `create_model_with_heads(...)`, any model init/forward pass) — even "just for inspection" — is real CPU/memory work and does not belong on the login node. Use SLURM instead. For a handful of quick checks, don't write a fresh `sbatch` script each time: grab one interactive allocation with `salloc -p genoa64 --cpus-per-task=4 --mem=16G --time=00:30:00 --no-shell` and reuse it via `srun --jobid=<id> <command>` for each check, then `scancel <id>` when done — much faster than round-tripping through `sbatch` + log polling per check. Reserve real `sbatch` scripts for the actual long-running jobs.
- **`/tmp` is node-local, not shared** — including this assistant's own `/tmp/claude-*` scratchpad. A SLURM job on one node cannot see files written to `/tmp` on the login node or another node. Any script a SLURM job needs to read, or any output you want to inspect afterward, must live under shared project storage (e.g. `results/.../debug_data/`), never under `/tmp`.
- **GPU resources** (as of 2026-08): `gpu_diasfrazer` partition has exactly one non-MIG H100 (`gres=gpu:h100:1`, node `najin`) — no wall-time limit, prefer it when free (`squeue -p gpu_diasfrazer` / `scontrol show node najin` to check). The shared `gpu` partition's `genoa64-09a` node has a `gpu:7g.80gb` MIG slice (full-size profile — memory/compute-equivalent to a whole 80GB GPU) as a fallback when the H100 is occupied, but that partition caps `--time` at ~12h in practice: longer requests are rejected outright ("Requested time limit is invalid") even with `--qos=marathon` passed explicitly, despite `marathon` being nominally available to this account there per `sacctmgr show assoc`.
- A `sbatch: HTTP/1.1 400 Bad Request: ... "Unknown GPU type: 'h100'"` message on submission is benign — the job still submits and runs correctly (verify with `scontrol show job <id>`, which will show the correct `gres/gpu:h100=N`). Some monitoring/accounting hook on this cluster doesn't recognize the short GPU-type alias; it doesn't block scheduling.
- For JAX GPU memory tuning, `TF_GPU_ALLOCATOR` is a **TensorFlow-only** env var — it has no effect on JAX's own PJRT/XLA GPU client (confirmed: the allocator stayed `GPU_0_bfc` with it set, and the same OOM recurred). If JAX-side memory tuning is ever genuinely needed, `XLA_PYTHON_CLIENT_MEM_FRACTION`/`XLA_PYTHON_CLIENT_PREALLOCATE` are the real levers — but check for an actual root cause first (e.g. a frozen-backbone run silently backpropagating through the whole model because gradients were never detached) rather than reaching for allocator env vars.

### Snakemake gotcha: never make a resumable checkpoint the declared rule output

Every submission command in this repo's README uses `--rerun-incomplete`, which deletes a rule's declared `output:` if Snakemake considers the previous attempt incomplete. If that output *is* the directory/file a script also reads/writes across invocations for its own resume logic (e.g. a "latest checkpoint" directory), this silently destroys real training progress on the next retry. Follow `workflows/05-full_finetuning/Snakefile`'s pattern instead: declare only a final, one-time artifact (e.g. `checkpoint_epoch{N}.pth`, or a dedicated completion-marker file) as `output:`, and let the script manage its own intermediate/resumable state entirely in a location Snakemake never tracks or can delete.

## Configuration

All paths, URLs, and parameters are centralized in `config/config.yaml`, organized as:
- `gencode` — GENCODE v46 genome/annotation URLs and paths
- `rnaseq.sf3b1mut` — metadata TSV and raw data path for SF3B1 RNA-seq
- `alphagenome_pytorch` — Hugging Face weights repo and local path
- `borzoi` — baskerville path, pretrained trunk paths (4 replicates), support files (blacklist, sequences.bed)
- `finetuning.alphagenome` — finetune script path, fold BED dirs, and `sf3b1mut` hyperparameters
- `finetuning.borzoi.sf3b1mut` — Borzoi finetuning hyperparameters; `params_lora.json` is **generated dynamically** by `borzoi_make_targets` (not a static file)
- `preprocessing.overfitting` — sample list, fold, and output dirs for single/dev interval selection

## Architecture

There is no top-level `workflows/Snakefile`. Each numbered workflow is self-contained.

### Workflow structure

```
workflows/
  01-obtain_data/
    Snakefile          — defines TMP_ROOT, SUPPORT_DIR, SAVE_PARAMS globals; includes all data rules
    rules/
      gencode.smk      — downloads GRCh38 FASTA and GENCODE v46 GTF; builds STAR index; converts GTF to parquet
      alphagenome.smk  — downloads AlphaGenome-PyTorch weights from Hugging Face via hf CLI; converts Borzoi fold BEDs
      sf3b1mut.smk     — full RNA-seq processing pipeline (see below); defines SAMPLES and STRANDS
      comparison_ssu.smk — validates compute_ssu.py and get_star_junctions.py against SpliSER/STAR on chr1
  02-preprocess_data/
    Snakefile          — selects single and dev genomic intervals for overfitting experiments
  03-overfitting_single/
    Snakefile          — single-interval overfitting (500 epochs, 1 GPU); three experiment groups
  04-overfitting_dev/
    Snakefile          — dev dataset overfitting (50 epochs); ablates GPU parallelism and junction source
  05-full_finetuning/
    Snakefile          — full FOLD_1 fine-tuning (4-GPU DDP, linear-probe)
  rules/
    models/
      alphagenome.smk  — AlphaGenome LoRA finetuning rule (finetune_sf3b1mut) for monolithic use
      borzoi.smk       — Borzoi transfer learning pipeline
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

### AlphaGenome finetuning (workflows 03–05)
- Each workflow defines its own run matrix and calls `torchrun` directly
- Script path set via `FINETUNE_SCRIPT` from `config["finetuning"]["alphagenome"]["finetune_script"]`, pointing to `src/alphagenome-pytorch/scripts/finetune.py`
- Script is copied to a tmp file before launch to avoid NFS issues under torchrun
- Key flags: `--mode` (linear-probe or lora), `--rope-init` (zeros or truncated_normal), `--junction-loss` (original, normalized, sparse), `--junction-position-source` (annotated or predicted), `--pretrained-head-samples`
- Workflows 03 and 04 use the `overfit_single` / `overfit_dev` rules respectively; workflow 05 uses `workflows/rules/models/alphagenome.smk`

### Borzoi finetuning pipeline (`workflows/rules/models/borzoi.smk`)
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

## Conda environments

Defined in `envs/`. All `conda:` directives in rule files use bare env names (e.g. `"alphagenome_pytorch"`), not yaml file paths.

| File | Env name | Used for |
|------|----------|----------|
| `general.yaml` | `alphagenome_finetuning_rna` | RNA-seq processing (STAR, sambamba, samtools, deeptools, pysam, pyranges) |
| `alphagenome_pytorch.yaml` | `alphagenome_pytorch` | AlphaGenome weight download and finetuning (PyTorch, alphagenome-pytorch[finetuning], hf CLI) |
| `spliser.yaml` | `spliser` | SpliSER-based SSU validation |
| `borzoi.yaml` | `borzoi` | Borzoi download and training (baskerville, tensorflow 2.15, gsutil) |

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

- `support/ENA_filereport-compendium-sf3b1mut.tsv` — RNA-seq runs (SF3B1 WT and K700E mutant MEC1 cells, ±H3B-8800 treatment)
- Raw data → `data/raw/`, preprocessed → `data/prep/`
- Results → `results/`
- `support/borzoi/` — auto-generated `params_lora.json` (do not edit manually)
