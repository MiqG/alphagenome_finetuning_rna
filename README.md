# Fine-tuning splicing heads for AlphaGenome

[AlphaGenome](https://deepmind.google/discover/blog/alphagenome-a-foundation-model-for-genome-biology/) is a sequence-to-function foundation model for genome biology. While community implementations such as [alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) support fine-tuning on RNA-seq coverage tracks, extending the model to splicing modalities (splice site classification, splice site usage, and splice junction counts) requires additional preprocessing, data loading, and validation work.

This repository contains the full Snakemake pipeline used to develop and validate that extension, using two RNA-seq samples from SF3B1-mutant MEC1 cells ([López-Oreja 2023](https://doi.org/10.26508/lsa.202301955)) as a case study. It covers everything from raw FASTQ download and alignment to single-interval overfitting experiments. The development process, including the bugs we found and fixed along the way, is described in our [blog post]().

## Requirements

- [Snakemake](https://snakemake.readthedocs.io) with conda integration (`--use-conda`)
- SLURM (optional, for cluster execution)
- 1 GPU (for single-interval overfitting runs): linear probing runs have been shown to run in GPUs with 46GB of memory.

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

## Citation

If you use this repository, please cite our blog post (link to be added upon publication) and the original AlphaGenome paper:

> Avsec, Ž. et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature*, 649 (2026). https://doi.org/10.1038/s41586-025-10014-0

## License

MIT — see [LICENSE](LICENSE).
