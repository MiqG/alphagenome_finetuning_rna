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