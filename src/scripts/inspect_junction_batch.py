import sys
sys.path.insert(0, "src/alphagenome-pytorch/src")
import yaml
import numpy as np

cfg = yaml.safe_load(open("config/config.yaml"))
DATA_DIR = cfg["rnaseq"]["sf3b1mut"]["path"]
SAMPLES  = cfg["preprocessing"]["overfitting"]["samples"]

star_files = [f"{DATA_DIR}/STAR/{s}/paper_pass.SJ.out.tab" for s in SAMPLES]
ssu_files  = [f"{DATA_DIR}/STAR/{s}/paper_pass.ssu.parquet" for s in SAMPLES]

chrom, win_start, win_end = "chr6", 89090380, 90138956
seq_len = win_end - win_start

from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    read_star_junctions, normalize_junctions_per_sample,
    junctions_to_junction_matrix, splice_sites_to_classification_array,
    read_ssu_parquet,
)

# Load junctions
all_juncs = []
for path in star_files:
    junc = read_star_junctions(path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc = junc.loc[junc["chrom"].str.contains("chr", na=False) & junc["strand"].isin(["+", "-"])].drop_duplicates()
    junc["exon_start"] = junc["intron_start"] - 1
    junc["exon_end"]   = junc["intron_end"] + 1
    junc["count"]      = junc["n_uniquely_mapped_reads"]
    junc = normalize_junctions_per_sample(junc)
    all_juncs.append(junc)

# Junctions in window
for i, (junc, s) in enumerate(zip(all_juncs, SAMPLES)):
    in_win = junc[(junc["chrom"] == chrom) & (junc["exon_start"] > win_start) & (junc["exon_end"] <= win_end)]
    print(f"Sample {s}: {len(in_win)} junctions, count [{in_win['count'].min():.3f}, {in_win['count'].max():.3f}]")

# Build cls_arr from SSU
ssu_dfs = [read_ssu_parquet(p, chrom, win_start, win_end) for p in ssu_files]
cls_arr = splice_sites_to_classification_array(ssu_dfs, chrom, win_start, seq_len)
print(f"\ncls sites — D+:{(cls_arr[:,0]>0).sum()} A+:{(cls_arr[:,1]>0).sum()} D-:{(cls_arr[:,2]>0).sum()} A-:{(cls_arr[:,3]>0).sum()}")

# Build junction matrix
jpos, jmat = junctions_to_junction_matrix(all_juncs, cls_arr, chrom, win_start, seq_len, max_splice_sites=256)
nonzero = (jmat > 0).sum()
print(f"Matrix: {nonzero}/{jmat.size} non-zero ({100*nonzero/jmat.size:.4f}%), sum/sample: {jmat.sum(axis=(0,1))}")

# Sanity: non-zero entries should land on cls sites
print("\nFirst 5 non-zero junctions (sample 0, + strand):")
for d, a in zip(*np.where(jmat[:,:,0] > 0)):
    d_rel, a_rel = int(jpos[0,d]), int(jpos[1,a])
    print(f"  d_rel={d_rel} a_rel={a_rel} count={jmat[d,a,0]:.4f} cls[d,D+]={cls_arr[d_rel,0]:.2f} cls[a,A+]={cls_arr[a_rel,1]:.2f}")
    if d >= 4:
        break

# How many STAR junctions are dropped (no cls site found)?
in_win0 = all_juncs[0][(all_juncs[0]["chrom"]==chrom) & (all_juncs[0]["exon_start"]>win_start) & (all_juncs[0]["exon_end"]<=win_end)]
missed = 0
for _, row in in_win0.iterrows():
    d_rel = int(row["exon_start"]) - 1 - win_start
    a_rel = int(row["exon_end"])   - 1 - win_start
    ch = 0 if row["strand"] == "+" else 2
    has_d = 0 <= d_rel < seq_len and cls_arr[d_rel, ch] > 0
    has_a = 0 <= a_rel < seq_len and cls_arr[a_rel, ch+1] > 0
    if not (has_d and has_a):
        missed += 1
print(f"\nSTAR junctions in window: {len(in_win0)}, dropped (no cls site): {missed} ({100*missed/max(len(in_win0),1):.1f}%)")
