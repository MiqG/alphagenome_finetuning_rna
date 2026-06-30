Preparation of RNA-seq Data for Splicing Analyses Training data for AlphaGenome’s splicing-related
predictions (splice junctions, splice site usage (SSU), and splice site classification) were derived from the
same ENCODE and GTEx RNA-seq datasets used for gene expression analyses, and samples were
grouped by ontology CURIEs consistently with the RNA-seq processing pipeline.
To quantify splice junction count, reads from each RNA-seq sample were realigned using STAR
(version 2.7.11b)77 from the BAM files downloaded from GTEx and ENCODE. For human samples,
alignment was performed against the GRCh38.p13 reference genome, with GENCODE v32 gene
annotations guiding splice junction discovery. For mouse samples, the GRCm38.p6 reference genome
and GENCODE vM23 annotations were used.
Key STAR alignment parameters were configured to optimize for junction detection, including setting a
minimum splice junction overhang of 8 base pairs (--alignSJoverhangMin 8), allowing a maximum
of 20 alignments for multi-mapping reads (--outFilterMultimapNmax 20), defining standard intron
size limits (e.g., --alignIntronMin 20, --alignIntronMax 1000000), and outputting strand
information derived splice dinucleotide motifs (--outSAMstrandField intronMotif).
The precise STAR command used was:
STAR --runMode alignReads \
--genomeDir "gcs/${GENOME}" \
--readFilesType ${read_file_type} \
--readFilesCommand ${read_file_command} \
--readFilesIn ${input_path} \
--outSAMtype BAM Unsorted \
--outFilterMultimapNmax 20 \
--alignSJoverhangMin 8 \
--alignSJDBoverhangMin 1 \
--outFilterMismatchNmax 999 \
--outFilterMismatchNoverReadLmax 0.04 \
--alignIntronMin 20 \
--alignIntronMax 1000000 \
--alignMatesGapMax 1000000 \
--outSAMstrandField intronMotif --outSAMunmapped Within \
--outFileNamePrefix "${output_prefix}" \
--outTmpDir data/tmp
The primary output files containing splice junction information (sj.out.tab) from STAR served
as the raw data for all subsequent splicing data curation, specifically to define training targets for the
three types of splicing-related predictions. Samtools (version 1.21)78 was used for standard intermediate
processing of BAM files generated during alignment.
6
1. Splice Junction Quantification, Filtering, and Normalization Individual splice junction output files
for each sample were first combined into a single comprehensive table, indexed by chromosome, junction
start coordinate, junction end coordinate, and strand, separately for human and mouse species.
A stringent quality filtering pipeline was then applied to these compiled junction lists using the
splicemap package (available at https://github.com/gagneurlab/splicemap) to ensure high
data fidelity:
• Human GTEx Samples: A junction was retained if its 90th percentile read count across all GTEx
samples was greater than 1, AND the median of total read counts supporting any alternative
splicing event sharing either its donor or acceptor site was at least 1.
• Human ENCODE Samples: Junctions from human ENCODE RNA-seq samples were filtered
against the set of high-confidence junctions derived from GTEx; any ENCODE human RNA-seq
splice junction not present in the filtered GTEx junction set was discarded.
• Mouse ENCODE Samples: For mouse ENCODE RNA-seq samples, a junction was retained if its
median read count across all mouse RNA-seq samples within the same ontology CURIE group
was greater than 3.
After these filtering steps, the retained junction counts for each sample were normalized to 1 million
total filtered junction reads per sample.
For use in model training (loss calculation) and evaluation, these normalized junction counts underwent further tissue-specific preprocessing. Within each tissue:
1. Raw counts were first clipped at the 99.99th percentile for that tissue to mitigate the influence of
extreme outliers.
2. Subsequently, these clipped counts were scaled by dividing by the mean count value, where this
mean was calculated only across actively expressed junctions (defined as those with a clipped
count > 0) within that specific tissue.
During training on splice junctions, the donor-acceptor pairing of a maximum of 512 splice sites on
each strand are considered per input interval. If the sampled interval has more than 512 splice sites, we
narrow the interval size to consider splice junctions until a maximum of 512 splice sites are in the interval
on either strands. (Note: This only affects the interval used for inference; the input sequence length of
the model, for example 1 Mb, remains unchanged). For each strand and each CURIE condition, splice
junction counts are represented as a square matrix of shape [512, 512] corresponding to donor/acceptor
pairs. Each element of the matrix represents the observed normalized read counts for the donor/acceptor
pair. If less than 512 splice sites are observed, the matrix is padded to 512 with 0.
2. Splice Site Usage Splice Site Usage (SSU) was calculated for each potential splice site using the
formula:
SSU =
# reads using the splice site
# reads using the splice site + # reads supporting skipping of the splice site
We adapted the basic splice site strength definition from Dent et al79. SSU quantification was
performed using a custom script. For this calculation, we considered all reads spanning the splice sites
regardless of the strand. SSU counting was done with a custom script. Reads flagged as PCR/optical
duplicates, those with a mapping quality (MQ) below 30, or reads containing base calls with a base quality
(BQ) below 20 were excluded from the counts. The counting of each RNA-seq sample was performed
independently, and the SSU for each RNA-seq sample was calculated independently. Only splice sites
7
that were detected in the corresponding STAR splice junction output (sj.out.tab files) and passed the
above splice junction filtering step were considered in this quantification process.
The splice site usage has two tracks per CURIE condition, corresponding to the two strands. Unlike
splice sites, SSU does not distinguish between donor usage and acceptor usage.

3. Splice Site Definition for Classification The set of splice sites used for training the splice site
classification task was defined as the union of all unique donor and acceptor sites present in the filtered
splice junction data (from step 2 above) for each ontology CURIE. Notably, unlike splice junction counts
or SSU values which can be tissue/sample-specific, the defined splice site training examples were not
treated as tissue-specific.
The splice site classification task was formulated as a 5-class classification problem, where each
relevant position could be classified as:
• Donor site on the positive strand (Donor+)
• Acceptor site on the positive strand (Acceptor+)
• Donor site on the negative strand (Donor-)
• Acceptor site on the negative strand (Acceptor-)
• Not a splice site on either strand

# evaluation

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
• Normalized; Across Tracks (Fig. 2d, right): Using the same quantile-normalized, gene-mean-
centered data as above, correlation was computed across tracks/cell types for each gene separately,
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


Benchmarking Against Borzoi Track Predictions
For a direct comparison with the Borzoi fold-1 model2
, an AlphaGenome model, which was initially
trained on the identical Borzoi fold-1 data split, underwent fine-tuning. This fine-tuning process involved
augmenting the AlphaGenome architecture with two additional heads to mirror Borzoi’s outputs.
The first head aggregates AlphaGenome’s 1 bp embeddings into 32 bp embeddings and makes
predictions matching the Borzoi tracks (7,611 human and 2,608 mouse). This head is trained on Borzoi’s
TFRecords dataset (original resolution and scaling) to allow for a direct comparison with the published
Borzoi model. This head is used for metrics reported in Fig. 1d where we compare against Borzoi at 32
bp resolution.
The second RNA-seq head is trained on the same RNA-seq tracks as Borzoi but reprocessed at 1
bp resolution and without any Borzoi specific scaling. When comparing this head against Borzoi, we
unscale and repeat Borzoi’s predictions 32 times (to equal 1 bp unscaled predictions). This head is used
for metrics reported in Fig. 1d where we compare against Borzoi at 1 bp resolution.
We validated the approach of upsampling and scaling the additional RNA-seq head to match Borzoi by
applying the same procedure to the training data. AlphaGenome’s base-resolution data was aggregated
to a 32 bp resolution and Borzoi’s original scaling methodology was applied. Compared to Borzoi’s
provided TFRecord files, we achieved high concordance (0.988 average Pearson r correlation) . This
comparison excluded unmappable regions, as flagged by the ‘umap’ entry in the Borzoi data examples.
Finally, in the Borzoi’s and AlphaGenome’s RNA-seq comparison at 1 bp resolution, we only include
the tracks for which the Pearson r correlation is larger or equal than 0.99 and exclude the unmappable
regions.

Correlation for Continuous Tracks
Concordance between predicted and observed continuous track signals (such as those for ChIP-seq,
DNase-seq, ATAC-seq, CAGE, and PRO-cap) was primarily measured using the Pearson correlation
coefficient (r). For a given track and a held-out test interval, Pearson r was calculated between the vector
of predicted values and the vector of observed values across all corresponding genomic bins within that
interval. The distributions shown in Fig. 2c represents the Pearson r values calculated for all tracks within
a specific assay group (e.g., all TF ChIP-seq tracks) and organism (e.g., human and mouse) across all
held-out test intervals. The average Pearson r for each group (shown as text and circle) is the mean of
these individual track correlations.

# figures

Extended Data Fig. 2: Splicing track performance.

(a) Schematic overview of splice site (SS) classification, splice site usage (SSU) prediction, and splice junction (SJ) read count prediction tasks. (b) (left) Performance comparison (AUPRC) of SS classification and SJ classification against reference methods. ‘Baseline’ means the fraction of positive splice junctions in the evaluated data. Splice site classification is evaluated with both GTF (GENCODE v46) annotated splice sites only and also splice sites derived from GTEx RNA-seq data (Methods). Splice junction classification discriminates between true splice junctions observed from RNA-seq data versus false junctions not observed from RNA-seq (but where the splice sites are observed). Splice junction classification was evaluated per tissue and then the mean AUPRC across tissues were reported. (right) Performance comparison (Pearson r) of predicted vs. measured SSU and SJ counts (log(1+x) transformed). (c) Scatter plot between predicted and measured donor SSU across seven example human tissues (from GTEx). Pearson r in each tissue is displayed as text. (d) Scatter plot between predicted and measured splice junction counts across seven human tissues (from GTEx). Pearson r in each tissue is displayed as text. (e) Distribution of Pearson correlation coefficients between predicted and measured PSI3 per tissue (left), PSI5 per tissue (middle), and junction counts across tissues (measuring tissue specificity of the splice junction predictions).

Extended Data Fig. 3: Track-level performance benchmarking.

Performance comparison of AlphaGenome with Enformer and Borzoi on held-out genomic track prediction. (a, b) Comparison of AlphaGenome test set performance on Enformer human tracks (each dot is one track) against Enformer models either (a) not fine-tuned or (b) fine-tuned on human data (the main released Enformer version). AlphaGenome model was re-trained for direct comparability using matched training intervals and an additional Enformer prediction head (Methods). (c) Evaluation of RNA-seq prediction performance at base and gene resolution using the same source of RNA-seq data as Borzoi, but processed at base-resolution and not scaled (Methods). Borzoi’s 32 bp RNA-seq predictions were upsampled and unscaled to the original scale for comparison. The larger performance difference observed on the normal scale (first column) likely reflects resolution differences at exon-intron boundaries. This difference decreased when using log(1+x) transformed values (second column), suggesting better agreement on overall gene expression levels. A similar trend was observed when aggregating expression per gene (average exon coverage, third column). Cell-type specificity was evaluated by correlating quantile-normalized, mean-subtracted expression profiles across genes (fourth column) and across tracks (fifth column). (d) Test set performance comparison of AlphaGenome against Borzoi (fold 1) on Borzoi track data at 32 bp resolution (each dot is one track). AlphaGenome was fine-tuned with an additional Borzoi head at matched resolution (Methods). (e) Stratification of cell-type specific prediction accuracy. The per-gene log-fold change correlation performance (from panel c, fourth column) was stratified by gene characteristics: median expression level across tissues (Median TPM; quintile breakpoints: 5.5×10−9, 4.1×10−4, 8.1×10−4, 0.17, 4.1, 3.6×104 TPM), number of tissues with the gene expressed (TPM ≥ 0.001; quintile breakpoints: 9.4×10−8, 9.4×10−4, 8.0, 52, 54, 54 tissues), and housekeeping gene status. Sample sizes in brackets are the number of genes in each category. Box plots display the median (center line), the 25th and 75th percentiles (box bounds), and the whiskers extend to 1.5 times the interquartile range from the box bounds; points beyond whiskers indicate outliers. (f, g) Performance comparison of AlphaGenome against (f) ProCapNet (on PRO-Cap data) and (g) ChromBPNet (on ATAC and DNase). Evaluation was performed on ProCapNet fold 5 and ChromBPNet fold 0 test peak regions, respectively, where regions overlapping with AlphaGenome fold 0 training intervals were excluded. Performance is quantified by track Pearson r, Pearson r on the log total count, and Jensen-Shannon distance (JSD; lower indicates better performance). AlphaGenome outperforms the baselines across all metrics, modalities and cell-lines. For (g), only tracks with matching experiment accessions between AlphaGenome and ChromBPNet training sets were considered.