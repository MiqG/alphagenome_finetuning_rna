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

## Citation

If you use this repository, please cite our blog post (link to be added upon publication) and the original AlphaGenome paper:

> Avsec, Ž. et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature*, 649 (2026). https://doi.org/10.1038/s41586-025-10014-0

## License

MIT — see [LICENSE](LICENSE).

Gene Expression Correlation
Pearson correlation (r) was also computed between observed and predicted gene expression values on
held-out test intervals. Gene expression values for both observed and predicted tracks were calculated as
the log-transformed mean read coverage across all annotated exons for a given gene, using GENCODE
version 46 annotations and considering strand matching. To avoid duplicate genes across test intervals,
we only considered genes with at least 50% of their exons falling within a test interval. Three specific
correlation approaches were used:
• Raw; Across Genes (Fig. 2d, left): For each cell type/track, correlation was computed across all
genes between their predicted and observed log-transformed expression values.
• Normalized; Across Genes (Fig. 2d, middle): For each track, log-transformed expression values
were quantile normalized across genes. Then, for each gene, its mean expression across all tracks
was subtracted. Correlation was then computed across genes between these normalized predicted
and observed values within each track.
• Normalized; Across Tracks (Fig. 2d, right): Using the same quantile-normalized, gene-meancentered data as above, correlation was computed across tracks/cell types for each gene separately,
assessing per-gene prediction consistency over different cellular contexts.

Splicing Prediction Performance
AlphaGenome’s performance on various splicing-related track prediction tasks was assessed using
specific test sets, metrics, and comparisons depending on the specific output type.
Splice Site Classification The model’s ability to correctly identify and classify splice sites was primarily
evaluated using the auPRC metric. For this task, we evaluated separately with true labels derived from
RNA-seq observed splice sites and those annotated in GENCODE GTF files. At each relevant genomic
position, the model predicted probabilities for four positive splice site classes: Donor site on the plus
strand (Donor+), Acceptor site on the plus strand (Acceptor+), Donor site on the minus strand (Donor-),
and Acceptor site on the minus strand (Acceptor-). The test sequences were split into batches with
input sequence length 1M base pair. A separate auPRC was computed for each of these four classes,
comparing predicted probabilities against true labels, and the overall reported performance is the average
auPRC across them.
For specific comparisons against models like SpliceAI and DeltaSplice, AlphaGenome’s performance
(using the ensemble of four fold-1 models, chosen for maximal test set overlap with these peer models)
was assessed on test intervals derived from human chromosomes 1, 3, 5, 7, and 9 for consistency.
Pangolin was not evaluated for this task as it does not predict separately donors or accepters, only splice
sites in general.

Splice Site Usage Prediction The evaluation of splice site usage (SSU) was conducted using Pearson
correlation coefficients (r). These were computed between the vector of AlphaGenome’s predicted SSU
values and the vector of observed SSU values (derived from RNA-seq) for each tissue across held-out
test intervals, treating each tissue’s SSU profile as a distinct track.
For comparative benchmarking (e.g., against DeltaSplice), the same held-out test intervals and SSU
predictions FROM AlphaGenome fold-1 were used. Annotated SSU values were derived from RNA-seq.
Comparisons with Pangolin for SSU were not performed due to differing SSU definitions and due to
Pangolin not using its SSU prediction head for variant effect prediction.

Splice Junction Prediction The model’s performance in predicting splice junctions was evaluated
through three distinct approaches on the held out test intervals of fold-1 with an ensemble model trained
with fold-1 training data.
1. Classification of true vs. false junctions: This task assessed the model’s ability to distinguish
true junctions (defined as donor-acceptor pairs with supporting RNA-seq read counts after the
filtering steps described in the data section) from the vast majority of false junctions (defined as
donor-acceptor pairs with no RNA-seq supporting). This task is challenging since only a small set
of donor/acceptor pairs are true within 1M base range. The model’s predicted junction counts were
the classification scores. For Splam, since it does not directly predict a junction score, we used the
min of the donor and acceptor splice site probabilities as a prediction for the junction score for each
splice junction. An auPRC was computed for each tissue independently by comparing flattened
matrices of these predicted scores against the corresponding flattened binary ground truth labels.
The final reported performance is the average auPRC across all tissues.
2. Quantitative prediction of junction counts: The accuracy of predicting the strength of junction usage
was assessed using the Pearson correlation between log(1 + 𝑥) transformed predicted junction
counts and log(1 + 𝑥) transformed measured junction counts (interpreted as junction strength).
This correlation was only computed over junctions that had non-zero read counts in the ground
truth test set data.

Prediction of PSI5 and PSI3 levels: The model’s ability to predict local splicing choices was further
evaluated by comparing measured and predicted PSI5 and PSI3 levels using Pearson correlation. These
metrics are defined as: PSI5(D,A)=n(D,A) /A’n(D,A’), PSI3(D,A)=n(D,A) /D’n(D’,A), where n(D,A) is the
number of split reads supporting the specific junction from donor D to acceptor A, A’n(D,A’) is the total
number of split reads supporting all splice junctions originating from donor D, and D’n(D’,A) is the total for
all splice junctions ending at acceptor A. The PSI correlation reported in (Extended Data Fig. 2) are
from the test intervals in chr2 only.
Together, these three approaches provide a multifaceted assessment of AlphaGenome’s ability to
accurately predict both the presence and quantitative strength of splice junctions, as well as local splice
isoform choices.


can you plan a script and a rule in @workflows/05-full_finetuning/Snakefile 
  to run predictions for test regions with a full finetuned model? I have 
  created the file at @src/scripts/predict_interval.py and consider 
  @src/scripts/run_pretrained_forward_pass.py for this. The desired outputs are
  though, predicted gene expression for the intervals taking into account gene
  annotations, for splice sites, classification and usage, splice junctions. 
  take