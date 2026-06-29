# Fine-tuning splicing heads for AlphaGenome

[AlphaGenome](https://deepmind.google/discover/blog/alphagenome-a-foundation-model-for-genome-biology/) is a sequence-to-function foundation model for genome biology. While community implementations such as [alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) support fine-tuning on RNA-seq coverage tracks, extending the model to splicing modalities (splice site classification, splice site usage, and splice junction counts) requires additional preprocessing, data loading, and validation work.

This repository contains the full Snakemake pipeline used to develop and validate that extension, using two RNA-seq samples from SF3B1-mutant MEC1 cells ([López-Oreja 2023](https://doi.org/10.26508/lsa.202301955)) as a case study. It covers everything from raw FASTQ download and alignment to single-interval overfitting experiments and full fine-tuning. The development process, including the bugs we found and fixed along the way, is described in our [blog post]().

## Requirements

- [Snakemake](https://snakemake.readthedocs.io) with conda integration (`--use-conda`)
- SLURM (optional, for cluster execution)
- 4 GPUs (for multi-GPU fine-tuning steps): all linear probing runs have been shown to run in GPUs with 46GB of memory, however, lora runs required at least 80GB.

All software dependencies are managed per-workflow via conda environments defined in `envs/`.

## Workflows

### 1. Obtain data (`workflows/01-obtain_data/Snakefile`)

```bash
snakemake -s workflows/01-obtain_data/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --cpus-per-task={threads} --time={resources.runtime} --mem={resources.memory}G --partition={resources.partition} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 30 --use-conda -s workflows/01-obtain_data/Snakefile --latency-wait 60 --keep-going --rerun-incomplete --rerun-triggers mtime'
```

Downloads and preprocesses all inputs:

- **Genome**: GRCh38 sequence and GENCODE v46 annotation; builds STAR index
- **RNA-seq** (SF3B1-mutant MEC1, ENA): downloads paired-end FASTQs, aligns with STAR two-pass, and derives all four training tracks per sample:
  - per-base coverage bigwigs (stranded)
  - splice site usage (parquet, via `compute_ssu.py`)
  - splice junction counts (TSV, via STAR or `get_star_junctions.py`)
  - splice site classes (derived on the fly from splice site usage files during training)
- **Model weights**: AlphaGenome-PyTorch from Hugging Face; Borzoi pretrained trunks and genome interval folds from GCS
- **Validation** (`rules/comparison_ssu.smk`): confirms that `compute_ssu.py` and `get_star_junctions.py` match SpliSER and STAR output (Pearson r = 1 on chr1)

### 2. Preprocess data (`workflows/02-preprocess_data/Snakefile`)

```bash
snakemake -s workflows/02-preprocess_data/Snakefile --use-conda -j <cores>
```

Selects genomic intervals for overfitting experiments from the Borzoi FOLD_1 splits:

- **Single intervals**: one interval per splice junction density tier (high / medium / low), ranked by total uniquely-mapped junction reads across samples. Output: `data/prep/overfitting/single/{high,medium,low}.bed`
- **Dev dataset**: top-N + random intervals for a small train/val split (120+50 train, 20+10 val). Output: `data/prep/overfitting/dev/{train,valid}.bed`

### 3. Single-interval overfitting (`workflows/03-overfitting_single/Snakefile`)

```bash
snakemake -s workflows/03-overfitting_single/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --cpus-per-task={threads} --mem={resources.memory}G --time={resources.runtime} --partition={resources.partition} --qos=normal --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 30 --use-conda -s workflows/03-overfitting_single/Snakefile --latency-wait 60 --keep-going --rerun-incomplete --rerun-triggers mtime'
```

Overfits AlphaGenome for 500 epochs (constant LR, no warmup, frozen trunk, 1 GPU) on the medium splice junction density interval from step 2. Three experiment groups, all running with all 4 modalities jointly:

- **original**: baseline with randomly initialized heads, no GTF augmentation, original RoPE initialization (zeros).
- **debug_splice_sites**: ablation of head initialization (random vs. pretrained), loss segmentation (none vs. 8 segments), and GTF augmentation for the splice site classification head. Six configurations.
- **debug_splice_junctions**: 4-way factorial ablation of RoPE initialization (zeros vs. truncated normal), junction loss formulation (original, normalized, sparse), junction position source (annotated vs. predicted), and junction head initialization (random vs. pretrained). 24 configurations.

Output: `results/finetuning/alphagenome_pytorch/overfitting/single/`

### 4. Dev dataset overfitting (`workflows/04-overfitting_dev/Snakefile`)

```bash
snakemake -s workflows/04-overfitting_dev/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --account=ehpc708 --cpus-per-task={threads} --time={resources.runtime} --partition={resources.partition} --qos={resources.qos} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 30 --use-conda -s workflows/04-overfitting_dev/Snakefile --latency-wait 60 --keep-going --rerun-incomplete --rerun-triggers mtime'
```

Overfits AlphaGenome for 50 epochs on the dev train/val split from step 2. One experiment group:

- **debug_splice_junctions**: fixed to truncated RoPE initialization and ratio-normalized junction loss; ablates junction position source (annotated vs. predicted) and GPU parallelism strategy (single GPU, 4-GPU sequence parallel, 4-GPU DDP) with frozen trunk and randomly initialized heads. Effective batch size is 8 across all GPU configurations.

Output: `results/finetuning/alphagenome_pytorch/overfitting/dev/`

### 5. Full fine-tuning (`workflows/05-full_finetuning/Snakefile`)

```bash
snakemake -s workflows/05-full_finetuning/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --account=ehpc708 --cpus-per-task={threads} --time={resources.runtime} --partition={resources.partition} --qos={resources.qos} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 365 --use-conda -s workflows/05-full_finetuning/Snakefile --latency-wait 60 --rerun-incomplete --keep-going --rerun-triggers mtime'
```

### 6. Evaluation (`workflows/06-evaluation/Snakefile`)

```bash
snakemake -s workflows/06-evaluation/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --account=ehpc708 --cpus-per-task={threads} --time={resources.runtime} --partition={resources.partition} --qos={resources.qos} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 365 --use-conda -s workflows/06-evaluation/Snakefile --latency-wait 60 --rerun-incomplete --keep-going --rerun-triggers mtime'
```

Fine-tunes AlphaGenome on the full FOLD_1 train/val split (41,699 train + 6,323 val intervals) using the best configuration from the overfitting experiments: randomly initialized heads, ratio-normalized junction loss, predicted junction positions, frozen trunk, 4-GPU DDP, constant LR (1e-4), 5 epochs, effective batch size 64. Uses `--resume auto` for fault tolerance across SLURM preemptions.

Output: `results/finetuning/alphagenome_pytorch/full/`

### 7. Overfitting SSU (`workflows/07-overfit_ssu/Snakefile`)

```bash
snakemake -s workflows/07-overfit_ssu/Snakefile --use-conda -j <cores>
```

```bash
# SLURM
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --account=ehpc708 --cpus-per-task={threads} --time={resources.runtime} --partition={resources.partition} --qos={resources.qos} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 365 --use-conda -s workflows/07-overfit_ssu/Snakefile --latency-wait 60 --rerun-incomplete --keep-going --rerun-triggers mtime'
```

### 8. Finetuning Pangolin

```shell
# BSC
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --account=ehpc708 --cpus-per-task={threads} --time={resources.runtime} --partition={resources.partition} --qos={resources.qos} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 365 --use-conda -s workflows/08-finetune_pangolin/Snakefile --latency-wait 60 --rerun-incomplete --keep-going --rerun-triggers mtime'

# CRG
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --cpus-per-task={threads} --mem={resources.memory}G --time={resources.runtime} --partition={resources.partition} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 30 --use-conda -s workflows/08-finetune_pangolin/Snakefile --latency-wait 60 --rerun-incomplete --keep-going --rerun-triggers mtime'
```

## Citation

If you use this repository, please cite our blog post (link to be added upon publication) and the original AlphaGenome paper:

> Avsec, Ž. et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature*, 649 (2026). https://doi.org/10.1038/s41586-025-10014-0

## License

MIT — see [LICENSE](LICENSE).