# Fine-tuning splicing heads for AlphaGenome

## workflows
1. obtain data (`workflows/01-obtain_data/Snakefile`)

   ```bash
   snakemake -s workflows/01-obtain_data/Snakefile --use-conda -j <cores>
   ```
   - genome sequence and annotations from GENCODE
      - download genome sequence
      - download genome annotation

   - RNA-seq (SF3B1-mutant MEC1, ENA)
      - download fastq
      - align (STAR two-pass)
         - splice junctions
         - bam files
      - process bam files
         - splice site usage (contain strand information)
         - coverage bigwig (one if unstranded two if stranded)
      - comparison SSU (`workflows/01-obtain_data/rules/comparison_ssu.smk`)
         - compare that preprocessing is equivalent with SpliSER
         - not only output values, but also time to process every single sample

   - models
      - download model weights (AlphaGenome-PyTorch)
      - download and prepare genome interval folds (Borzoi hg38 sequences.bed)

2. preprocess data (`workflows/02-preprocess_data/Snakefile`)

   ```bash
   snakemake -s workflows/02-preprocess_data/Snakefile --use-conda -j <cores>
   ```
   - single intervals to overfit: select one interval per density tier (high / medium / low) from FOLD_1 train.bed, ranked by total uniquely-mapped junction reads across samples
      - output: `data/prep/overfitting/single/{high,medium,low}.bed`
   - dev dataset intervals: select top-N + random intervals from FOLD_1 train/valid.bed
      - train: 120 top + 50 random
      - valid: 20 top + 10 random
      - output: `data/prep/overfitting/dev/{train,valid}.bed`

3. overfitting single (`workflows/03-overfitting_single/Snakefile`)

   ```bash
   snakemake -s workflows/03-overfitting_single/Snakefile --use-conda -j <cores>
   ```

   Overfits AlphaGenome for 50 epochs (constant LR, no warmup, linear-probe mode) on each of
   the 3 single intervals (high / medium / low splice junction density) from `02-preprocess_data`.
   All runs use `paper_pass` bigwigs and junction files from the 2 preprocessing samples
   (SRR17111303, SRR17111311).

   Three experiment groups:

   - **original** — baseline: randinit heads, no segmented loss, no GTF, original rope
      - `all`: all 4 modalities contribute equally (weight 1.0)
      - `rna_seq_only`, `splice_site_only`, `splice_usage_only`, `splice_junctions_only`:
        one modality at weight 1.0, all others at 0.0
      - store: train/val losses (total + per modality), correlations per epoch, time and memory;
        for splice_site: auPRC against ground-truth categorical labels
      - plot: loss curves (total + per modality), correlation curves — palette by density tier

   - **debug_splice_sites** — ablate head init × segmented loss × GTF for splice-site training.
     Six configurations (head_init × segmented_loss × gtf_variant):
      1. randinit, no segmented loss, with GTF
      2. randinit, segmented loss, with GTF
      3. randinit, segmented loss, no GTF
      4. pretrinit, segmented loss, no GTF
      5. pretrinit, segmented loss, with GTF
      6. pretrinit, no segmented loss, with GTF
     Each config runs twice: `splice_site_only` (weight 1.0) and `all` (all modalities 1.0).
      - store: same as original
      - plot: loss curves, correlations, auPRC across configs

   - **debug_splice_junctions** — ablate rope initialization × GTF for junction training.
     Four configurations (rope_variant × gtf_variant):
      1. original rope, with GTF
      2. original rope, no GTF
      3. truncated rope, with GTF
      4. truncated rope, no GTF
     Each config runs twice: `junction_only` (weight 1.0) and `all` (all modalities 1.0).
      - store: same as original
      - plot: loss curves, junction correlations across configs

   Output: `results/finetuning/alphagenome_pytorch/overfitting/single/`

      

