"""
intervals.smk — Derive Pangolin-stratified eval BEDs from AlphaGenome FOLD_1 test.bed.

Pangolin's backbone was pretrained with a chromosome-based split:
  seen   (train): even autosomes + chrX + chrY
  unseen (test) : chr1, chr3, chr5, chr7, chr9

Two filtered BEDs are generated so that notebook analysis can compare SSU
correlations on (a) intervals where Pangolin's backbone had pretraining signal
and (b) intervals that were fully held out during Pangolin pretraining.
"""

_FOLDS_DIR = config["finetuning"]["alphagenome"]["folds_dir"]
_FOLD      = config["preprocessing"]["overfitting"]["fold"]

PANGOLIN_UNSEEN_CHROMS = {"chr1", "chr3", "chr5", "chr7", "chr9"}
PANGOLIN_SEEN_CHROMS   = {
    "chr2", "chr4", "chr6", "chr8", "chr10", "chr11", "chr12", "chr13",
    "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21",
    "chr22", "chrX", "chrY",
}


rule pangolin_make_eval_beds:
    """Filter FOLD_1 test.bed into seen/unseen Pangolin pretraining chromosome subsets."""
    input:
        test_bed = os.path.join(_FOLDS_DIR, _FOLD, "test.bed"),
    output:
        seen   = os.path.join(_FOLDS_DIR, _FOLD, "test_pangolin_seen_chroms.bed"),
        unseen = os.path.join(_FOLDS_DIR, _FOLD, "test_pangolin_unseen_chroms.bed"),
    threads: 1
    resources:
        runtime   = 5,
        gres      = "none",
        partition = "gpp",
        qos       = "gp_ehpc",
    run:
        import pandas as pd
        bed = pd.read_csv(input.test_bed, sep="\t", header=None, names=["chrom", "start", "end"])
        bed[bed["chrom"].isin(PANGOLIN_SEEN_CHROMS)].to_csv(
            output.seen, sep="\t", header=False, index=False
        )
        bed[bed["chrom"].isin(PANGOLIN_UNSEEN_CHROMS)].to_csv(
            output.unseen, sep="\t", header=False, index=False
        )
