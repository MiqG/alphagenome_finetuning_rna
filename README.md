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

3. overfitting single
   - experiments
      - overfit 50 epochs the 3 single intervals constant learning rate:
         - original
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, no segmented loss for splice site and splice site usage, all losses contribute, not using GTF
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, no segmented loss for splice site and splice site usage, only one loss contributes (weight 1, rest 0) for every modality, not using GTF
            - store: correlations, losses, of train and validation, as well as time and memory stats, for splice site probability predictions compute auPRC with ground truth categorical data
            - plot with loss overall and for each modality, plot with correlation across epochs for each modality, palette according to high, medium, low density.
         - debug_splice_sites
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, no segmented loss for splice site and splice site usage, only splice site loss contributes vs all, using GTF
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, do segmented loss for splice site and splice site usage, only splice site loss contributes vs all, using GTF
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, do segmented loss for splice site and splice site usage, only splice site loss contributes vs all, not using GTF
            - linear probing output heads with pretrained weights, original rope initialization of splice junctions, and original junction loss, do segmented loss for splice site and splice site usage, only splice site loss contributes vs all, not using GTF
            - linear probing output heads with pretrained weights, original rope initialization of splice junctions, and original junction loss, do segmented loss for splice site and splice site usage, only splice site loss contributes vs all, using GTF
            - linear probing output heads with pretrained weights, original rope initialization of splice junctions, and original junction loss, no segmented loss for splice site and splice site usage, only splice site loss contributes vs all, using GTF
            - store: correlations, losses, of train and validation, as well as time and memory stats, for splice site probability predictions compute auPRC with ground truth categorical data
            - plot overall losses, correlations and corresponding auPRC
         - debug_splice_junctions
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, no segmented loss, only junction loss contributes vs all, using GTF
            - linear probing output heads with random weights, original rope initialization of splice junctions, and original junction loss, no segmented loss, only junction loss contributes vs all, not using GTF
            - linear probing output heads with random weights, truncated rope initialization of splice junctions, and original junction loss, no segmented loss, only junction loss contributes vs all, using GTF
            - linear probing output heads with random weights, truncated rope initialization of splice junctions, and original junction loss, no segmented loss, only junction loss contributes vs all, not using GTF

      

