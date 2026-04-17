## Notes from paper

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

Final data representations Processed genomic tracks, serving as model prediction targets, were
converted to brain floating point (bfloat16) format for numerical efficiency and stored in z-standard
compressed sharded matrices. Most data tracks were maintained at base-pair resolution, except for ChIPseq (TF and Histone) tracks, which represent fold-change values, and were stored as base-resolution
cumulative sums. This strategy allows for efficient querying of their average signal at 128 bp resolution.
The model directly uses the track values resulting from the upstream processing steps, which typically
represent normalized read or insertion counts (often scaled to a total of 100 million signals per track for
sequencing assays) or fold-change enrichments (for ChIP-seq). No additional scaling transformations
(e.g., log-scaling or z-score normalization across tracks) are applied before model input. Predicted splice
junction counts are an exception; while their values are additionally scaled (as described in the training
methods), this does not affect their primary use in calculating ratios for relative splice site assessment.

Splice Sites Classification Output Head This head predicts the probability of each base belonging to
one of five classes: Donor+, Acceptor+, Donor-, Acceptor-, or not a splice site (see Splice Site Definition
for Classification section in Training Data). A linear layer is applied to the 1 bp resolution embeddings
that maps to 5 logits per base, followed by a Softmax activation to produce class probabilities. Training
uses a standard per-base cross-entropy loss function against the true splice site labels.

Splice Site Usage Output Head This head predicts the proportion of splicing events utilizing each
potential splice site (SSU), separately for each strand and tissue/cell type. It takes the 1 bp resolution
embeddings as input. A linear layer maps the embedding dimension to the number of tissues/cell types
per strand at each potential splice site location, followed by a Sigmoid activation to constrain outputs
between 0 and 1. The model is trained using a binary cross-entropy loss against the observed SSU
values.

Splice Junctions Output Head This head predicts counts for potential splice junctions between donor
and acceptor sites. It operates on the 1 bp resolution embeddings. Because each sequence in the batch
might have a different number of donor and acceptor splice sites, in practice we perform the calculation
using batch padding. In the pseudocode below, we show the calculation for a single sequence:
def tissue_scaled_rope(x: Array[S, 768], indices: Array[P]) -> Array[P, num_tissues, 768]:
x = x[indices, :]
scale = GetParameter('scale', (num_tissues, 768))
offset = GetParameter('offset', (num_tissues, 768))
x = scale[None, :, :] * x[:, None, :] + offset[None, :, :]
return apply_rope(x, max_position=2 ** 20, positions=indices)
def splice_junctions(
x: Array[S, C], donor_indices: Array[D], acceptor_indices: Array[A]
) -> Array[D, A, N_tissues]:
x = Linear(768)(x)
donor_embedding = tissue_scaled_rope(x, donor_indices)
acceptor_embedding = tissue_scaled_rope(x, acceptor_indices)
return Softplus(
Einsum('dtk,atk->dat', donor_embedding, acceptor_embedding))
First, a linear layer projects the embeddings to an intermediate dimension. For sequence-level
pre-training and evaluation (as described in this section), the tissue-specific genomic coordinates of
expressed donor (donor_indices) and acceptor (acceptor_indices) sites, derived from the data
processing pipeline, are provided as input. During distillation, these sites are obtained by selecting the
top-k (with 𝑘 = 512) highest probability sites from the teacher’s splice site classification head. The
procedure for identifying splice sites differs during variant effect prediction, as described in the Variant
Scoring section. The projected embeddings corresponding to these ground truth site coordinates are
then processed by the tissue_scaled_rope function. This function first extracts the embeddings at
the provided site locations and then applies the modified RoPE implementation described previously.
Because splice sites are not contiguous, the function uses relative genomic coordinates as positions
to incorporate distance information accurately. The modification to RoPE’s frequency calculation is
particularly relevant here, reducing density at short ranges to better focus on the longer distances
typical for splice junctions. Finally, the function applies a learnable, tissue-specific affine transformation
(with distinct parameters per site type, strand, and organism). The main splice_junctions function
computes the predicted count for each potential junction via an inner product (Einsum) between the
corresponding processed donor and acceptor embeddings, followed by a Softplus activation to ensure
positive values.
The training loss for splice junction predictions (junctions_loss) combines multiple terms. Two
cross-entropy terms compare predicted and target count ratio distributions. These ratios represent
conditional splicing probabilities. The first term evaluates the distribution of acceptor site usage for each
donor, akin to a Percent Spliced In from the 5’ site (PSI5) perspective. This is achieved by normalizing
counts over all acceptors for each donor (i.e., 𝑎𝑥𝑖𝑠 = 1, effectively 𝑃(Acceptor | Donor, Tissue)).
The second term evaluates the distribution of donor site usage for each acceptor, akin to a PSI3
perspective. This normalizes counts over all donors for each acceptor (e.g., axis=0, effectively 𝑃(Donor |
Acceptor, Tissue)). Additionally, two Poisson loss terms compare the marginal sums: one compares the
total predicted and target counts summed across all acceptors for each donor (e.g., 𝑎𝑥𝑖𝑠 = 1), and the
other sums across all donors for each acceptor (e.g., 𝑎𝑥𝑖𝑠 = 0). For numerical stability of the Poisson
loss term calculation, the sums of the target counts undergo an additional soft_clip operation. The
final loss is a weighted combination of the cross-entropy and Poisson components. This entire loss
calculation is performed independently for positive and negative strand predictions using their respective
splice sites and targets. It is specified by the pseudocode:
def soft_clip(x: Array) -> Array:
return Where(x > 10.0, 2 * Sqrt(x * 10.0) - 10.0, x)
def multinomial_cross_entropy(
x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues], axis: int
) -> Array[]:
pred_ratios = (x + 1e-7) / (x + 1e-7).sum(axis=axis, keepdims=True)
target_ratios = (targets + 1e-7) / (targets + 1e-7).sum(
axis=axis, keepdims=True)
return - (targets * Log(pred_ratios)).sum()
def poisson_loss(
x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues], axis: int
) -> Array[]:
sum_pred = x.sum(axis=axis)
sum_targets = soft_clip(targets.sum(axis=axis))
return (sum_pred - sum_targets * Log(sum_pred + 1e-7)).sum()
def junctions_loss(
x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues]
) -> Array[]:
ratios_loss = (multinomial_cross_entropy(x, targets, axis=0) +
multinomial_cross_entropy(x, targets, axis=1))
counts_loss = (
poisson_loss(x, targets, axis=0) + poisson_loss(x, targets, axis=1))
return 0.2 * ratios_loss + 0.04 * counts_loss

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


---
Three things differ between the paper pseudocode and the actual JAX implementation:  
                                                        
1. Loss weights: paper says 0.2 × ratios + 0.04 × counts, JAX actually does 1.0 ×    
ratios + 0.2 × counts.

2. CE formula: paper uses raw targets as weights (-(targets *                        
log(pred_ratios)).sum()), JAX normalizes targets to p_true = target /                
target.sum(axis) first — so it's a true cross-entropy between the target ratio
distribution and the predicted ratio distribution, not a weighted log-likelihood.    

3. Normalization: paper pseudocode uses .sum() throughout, JAX uses _safe_masked_mean
(divide by number of valid elements) for both CE and Poisson. This is what keeps the
loss scale invariant to the number of junctions in a window and makes the 1.0 weight
reasonable. The Poisson also subtracts a min_value baseline (y_true - y_true *      
log(y_true + eps)) so loss is zero at the optimum.        

The masking via pairs_mask (outer product of valid donors × valid acceptors) is      
implied but not shown in the pseudocode.       