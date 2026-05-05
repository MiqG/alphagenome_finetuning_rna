# Fine-tuning splicing heads for AlphaGenome

## workflows
1. obtain data
   - genome sequence and annotations from GENCODE
      - download genome sequence
      - download genome annotation

   - RNA-seq
      - download fastq
      - align
         - splice junctions
         - bam files
      - process bam files
         - splice site usage (contain strand information)
         - coverage bigwig (one if unstranded two if stranded)
      - comparison SSU
         - compare that preprocessing is equivalent with SpliSER
         - not only output values, but also time to process every single sample

   - models
      - download model weights
      - download and prepare genome interval folds

2. preprocess data
   - single intervals to overfit: select intervals with many, medium and few splice junctions
   - dev dataset intervals with top intervals to overfit: 120 top for training 50 random, and 20 top and 10 random for validating

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

      

