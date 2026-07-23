# Fine-tuning splicing heads for AlphaGenome

[AlphaGenome](https://deepmind.google/discover/blog/alphagenome-a-foundation-model-for-genome-biology/) is a sequence-to-function foundation model for genome biology. While community implementations such as [alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) support fine-tuning on RNA-seq coverage tracks, extending the model to splicing modalities (splice site classification, splice site usage, and splice junction counts) requires additional preprocessing, data loading, and validation work.

This repository contains the full Snakemake pipeline used to develop and validate that extension, using two RNA-seq samples from SF3B1-mutant MEC1 cells ([López-Oreja 2023](https://doi.org/10.26508/lsa.202301955)) as a case study. It covers everything from raw FASTQ download and alignment to single-interval overfitting debugging and full FOLD_1 fine-tuning with held-out evaluation. The development process, including the bugs we found and fixed along the way, is described in our [blog post]().

## Requirements

- [Snakemake](https://snakemake.readthedocs.io) with conda integration (`--use-conda`)
- SLURM (optional, for cluster execution)
- 1 GPU (for single-interval overfitting runs): linear probing runs have been shown to run in GPUs with 46GB of memory.
- 2-4 GPUs (for full fine-tuning, step 5 below): 4 GPUs for the frozen-trunk linear-probe run (multi-GPU DDP), 2 GPUs for the LoRA run.

All software dependencies are managed per-workflow via conda environments defined in `envs/`.

## Workflows

### 1. Obtain data (`workflows/01-obtain_data/Snakefile`)

```bash
snakemake -s workflows/01-obtain_data/Snakefile --use-conda -j <cores>
```

Downloads and preprocesses all inputs:

- **Genome**: GRCh38 sequence and GENCODE v46 annotation; builds STAR index
- **RNA-seq** (SF3B1-mutant MEC1, ENA): downloads paired-end FASTQs, aligns with STAR two-pass, and derives all four training tracks per sample:
  - per-base coverage bigwigs (stranded)
  - splice site usage (parquet, via `compute_ssu.py`)
  - splice junction counts (TSV, via STAR or `get_star_junctions.py`)
  - splice site classes (derived on the fly from splice site usage files during training)
- **Model weights**: AlphaGenome-PyTorch from Hugging Face; genome interval fold assignments from the calico/borzoi repository
- **Validation** (`rules/comparison_ssu.smk`): confirms that `compute_ssu.py` and `get_star_junctions.py` match SpliSER and STAR output (Pearson r = 1 on chr1)

### 2. Preprocess data (`workflows/02-preprocess_data/Snakefile`)

```bash
snakemake -s workflows/02-preprocess_data/Snakefile --use-conda -j <cores>
```

Selects genomic intervals for overfitting experiments from the FOLD_1 splits:

- **Single intervals**: one interval per splice junction density tier (high / medium / low), ranked by total uniquely-mapped junction reads across samples. Output: `data/prep/overfitting/single/{high,medium,low}.bed`
- **Train sample**: a seeded random sample of FOLD_1's `train.bed`, matching `test.bed` in size, used by step 6 as an overfitting sanity check (compares held-out test performance against a same-sized sample of training data). Output: `data/prep/finetuning/alphagenome/FOLD_1/train_sample.bed`

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

### 5. Full fine-tuning (`workflows/05-full_finetuning/Snakefile`)

```bash
snakemake -s workflows/05-full_finetuning/Snakefile --use-conda -j <cores>
```

Fine-tunes AlphaGenome on the full FOLD_1 train/validation split (10 epochs), all four modalities jointly, randomly initialized heads, normalized junction loss, and annotated junction positions. Two runs:

- **Frozen-trunk linear-probe**: 4-GPU DDP.
- **LoRA** (rank 8, `q_proj`/`v_proj`): 2 GPU, constant learning rate with no warmup, matching the linear-probe schedule so the two are comparable.

GPU/cluster resource requests in the Snakefile are placeholders — adjust `gres`/`partition` to your own setup. Output: `results/finetuning/alphagenome_pytorch/full/{run_name}/checkpoint_epoch10.pth`

### 6. Evaluation (`workflows/06-evaluation/Snakefile`)

```bash
snakemake -s workflows/06-evaluation/Snakefile --use-conda -j <cores>
```

Evaluates both fine-tuned models (linear-probe and LoRA) at each saved epoch on FOLD_1's held-out `test.bed` intervals, plus a same-sized `train_sample.bed` subset as an overfitting check. For each (run, epoch, subset), runs single-GPU inference (`collect_predictions.py`) and computes gene expression, splice site usage, and splice junction count metrics (`compute_eval_metrics.py`).

Output: `results/evaluation/alphagenome_pytorch/full/{run_name}/epoch{epoch}/{subset}/metrics.parquet`

## Citation

If you use this repository, please cite our blog post (link to be added upon publication) and the original AlphaGenome paper:

> Avsec, Ž. et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature*, 649 (2026). https://doi.org/10.1038/s41586-025-10014-0

## License

MIT — see [LICENSE](LICENSE).
