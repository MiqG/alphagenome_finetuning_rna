#!/usr/bin/env python
"""Reproduce training Pearson r from saved splice_junctions_annotated.npz.

Runs from the project root. Usage:
    conda run -n alphagenome_pytorch python src/scripts/debug_junction_pearson.py
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT = "/users/diasfrazer/manglada/projects/alphagenome_finetuning_rna"
RUN     = "debug_splice_junctions__truncrope__origloss__annotated__pretrinit__bfloat16"
DENSITY = "medium"

NPZ_PATH  = f"{PROJECT}/results/finetuning/alphagenome_pytorch/overfitting/single/{DENSITY}/{RUN}/splice_junctions_annotated.npz"
META_PATH = f"{PROJECT}/results/finetuning/alphagenome_pytorch/overfitting/single/{DENSITY}/{RUN}/metadata.json"

SJ_PATHS = [
    f"{PROJECT}/data/raw/ENA/sf3b1mut/STAR/SRR17111303/paper_pass.SJ.out.tab",
    f"{PROJECT}/data/raw/ENA/sf3b1mut/STAR/SRR17111311/paper_pass.SJ.out.tab",
]

sys.path.insert(0, f"{PROJECT}/src/alphagenome-pytorch/src")
from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    read_star_junctions,
    junctions_to_junction_matrix,
    normalize_junctions_per_sample,
)

# ── 1. Load npz ──────────────────────────────────────────────────────────────
npz = np.load(NPZ_PATH)
positions   = npz["junction_positions"]   # (4, 512) int32
pred_counts = npz["junction_counts"]       # (512, 512, 4) float32
print(f"positions shape : {positions.shape}")
print(f"pred_counts shape: {pred_counts.shape}")
n_s = pred_counts.shape[2] // 2           # n_tissues = 2

valid_per_role = [(positions[r] >= 0).sum() for r in range(4)]
print(f"Valid positions per role [D+, A+, D-, A-]: {valid_per_role}")

# ── 2. Load metadata ──────────────────────────────────────────────────────────
with open(META_PATH) as f:
    meta = json.load(f)
chrom        = meta["chrom"]
padded_start = meta["padded_start"]
padded_end   = meta["padded_end"]
seq_len      = padded_end - padded_start
print(f"Window: {chrom}:{padded_start}-{padded_end}  seq_len={seq_len}")

# ── 3. Load + normalise STAR junctions (exactly as SplicingDataset does) ────
all_juncs = []
for path in SJ_PATHS:
    junc = read_star_junctions(path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc = junc.loc[
        junc["chrom"].str.contains("chr", na=False)
        & junc["strand"].isin(["+", "-"])
    ].drop_duplicates()
    junc["exon_start"] = junc["intron_start"] - 1
    junc["exon_end"]   = junc["intron_end"]   + 1
    junc["count"]      = junc["n_uniquely_mapped_reads"]
    junc = normalize_junctions_per_sample(junc)
    all_juncs.append(junc)
    print(f"  {path.split('/')[-2]}: {len(junc)} junctions after normalisation")

# ── 4. Pre-filter per-sample DataFrames and add d_rel / a_rel ────────────────
end = padded_start + seq_len
all_juncs_local = []
for junc_df in all_juncs:
    mask = (
        (junc_df["chrom"] == chrom)
        & (junc_df["exon_start"] > padded_start)
        & (junc_df["exon_start"] <= end)
        & (junc_df["exon_end"]   > padded_start)
        & (junc_df["exon_end"]   <= end)
    )
    local = junc_df.loc[mask].copy()
    if not local.empty:
        local["d_rel"] = local["exon_start"].astype(int) - 1 - padded_start
        local["a_rel"] = local["exon_end"].astype(int)   - 1 - padded_start
    all_juncs_local.append(local)
    print(f"  window junctions: {len(local)}")

# ── 5. Rebuild junction matrix (predicted mode, same as _get_junction_targets) ─
_, junc_matrix = junctions_to_junction_matrix(
    all_juncs_local,
    max_splice_sites=positions.shape[1],
    positions=positions,
)
print(f"junc_matrix shape: {junc_matrix.shape}")
print(f"junc_matrix nonzero cells: {(junc_matrix > 0).sum()}")

# ── 6. Compute Pearson r exactly as _extract_splice_pearson_pairs (nonzero) ──
all_pred_nz, all_tgt_nz = [], []

for strand_label, pred_key_slice, tgt_slice, donor_row, accept_row in [
    ("pos", slice(None, n_s),  (slice(None), slice(None), slice(None, n_s)),  0, 1),
    ("neg", slice(n_s, None),  (slice(None), slice(None), slice(n_s, None)),  2, 3),
]:
    pc = pred_counts[:, :, pred_key_slice]   # (D, A, n_s)
    tc = junc_matrix[tgt_slice]              # (D, A, n_s)

    donor_pos  = positions[donor_row]        # (K,) ints
    accept_pos = positions[accept_row]       # (K,)
    valid_d = (donor_pos >= 0).astype(float)
    valid_a = (accept_pos >= 0).astype(float)
    pairs_mask = np.outer(valid_d, valid_a).astype(bool)  # (D, A)
    pairs_mask4 = pairs_mask[:, :, np.newaxis] & np.ones((1, 1, n_s), dtype=bool)  # (D, A, n_s)

    nonzero_mask = pairs_mask4 & (tc > 0)
    n_nz = nonzero_mask.sum()
    print(f"  {strand_label} strand nonzero pairs: {n_nz} / {pairs_mask4.sum()} valid")

    if n_nz > 0:
        all_pred_nz.append(pc[nonzero_mask])
        all_tgt_nz.append(tc[nonzero_mask])

pred_nz = np.concatenate(all_pred_nz)
tgt_nz  = np.concatenate(all_tgt_nz)
r_nz, _ = pearsonr(np.log1p(pred_nz), np.log1p(tgt_nz))
print(f"\nPearson r (nonzero, training-style): {r_nz:.4f}  (n={len(pred_nz)})")

# ── 7. Also compute "full" variant (all valid pairs) ─────────────────────────
all_pred_full, all_tgt_full = [], []
for strand_label, pred_key_slice, tgt_slice, donor_row, accept_row in [
    ("pos", slice(None, n_s),  (slice(None), slice(None), slice(None, n_s)),  0, 1),
    ("neg", slice(n_s, None),  (slice(None), slice(None), slice(n_s, None)),  2, 3),
]:
    pc = pred_counts[:, :, pred_key_slice]
    tc = junc_matrix[tgt_slice]
    donor_pos  = positions[donor_row]
    accept_pos = positions[accept_row]
    valid_d = (donor_pos >= 0).astype(float)
    valid_a = (accept_pos >= 0).astype(float)
    pairs_mask = np.outer(valid_d, valid_a).astype(bool)
    pairs_mask4 = pairs_mask[:, :, np.newaxis] & np.ones((1, 1, n_s), dtype=bool)
    all_pred_full.append(pc[pairs_mask4])
    all_tgt_full.append(tc[pairs_mask4])

pred_full = np.concatenate(all_pred_full)
tgt_full  = np.concatenate(all_tgt_full)
r_full, _ = pearsonr(np.log1p(pred_full), np.log1p(tgt_full))
print(f"Pearson r (full,    training-style): {r_full:.4f}  (n={len(pred_full)})")

# ── 8. Per-sample breakdown ───────────────────────────────────────────────────
print("\nPer-sample nonzero Pearson r:")
for s in range(n_s):
    p_pred, n_pred = [], []
    for strand_label, pred_s, tgt_s, donor_row, accept_row in [
        ("pos", s,      s,      0, 1),
        ("neg", n_s+s,  n_s+s,  2, 3),
    ]:
        pc = pred_counts[:, :, pred_s]
        tc = junc_matrix[:, :, tgt_s]
        d_pos = positions[donor_row];  a_pos = positions[accept_row]
        valid_d = (d_pos >= 0);  valid_a = (a_pos >= 0)
        pmask = np.outer(valid_d, valid_a) & (tc > 0)
        if pmask.any():
            p_pred.append(pc[pmask])
            n_pred.append(tc[pmask])
    if p_pred:
        pv = np.concatenate(p_pred);  nv = np.concatenate(n_pred)
        r_s, _ = pearsonr(np.log1p(pv), np.log1p(nv))
        print(f"  sample {s}: r={r_s:.4f}  n={len(pv)}")

# ── 9. Compare prediction range to target range ───────────────────────────────
print(f"\npred range: [{pred_nz.min():.4f}, {pred_nz.max():.4f}]  mean={pred_nz.mean():.4f}")
print(f"tgt  range: [{tgt_nz.min():.4f},  {tgt_nz.max():.4f}]   mean={tgt_nz.mean():.4f}")

# ── 10. Check whether notebook raw_tidy rows match these nonzero pairs ────────
print("\n--- Cross-check against notebook raw_tidy ---")
# Reconstruct what the notebook would see: per junction row in raw_tidy
# junctions are (d_rel, a_rel, strand, sample_idx) where count = normalised
for s_idx, junc_df in enumerate(all_juncs_local):
    if junc_df.empty:
        continue
    n_s_local = n_s
    for _, row in junc_df.iterrows():
        d_rel = int(row["d_rel"]); a_rel = int(row["a_rel"]); strand = row["strand"]
        cnt_tgt = float(row["count"])
        if strand == "+":
            d_arr = positions[0]; a_arr = positions[1]; ch = s_idx
        else:
            d_arr = positions[2]; a_arr = positions[3]; ch = n_s_local + s_idx
        # find indices
        d_list = np.where(d_arr == d_rel)[0]
        a_list = np.where(a_arr == a_rel)[0]
        if len(d_list) == 0 or len(a_list) == 0:
            print(f"  MISS: sample={s_idx} strand={strand} d_rel={d_rel} a_rel={a_rel} tgt={cnt_tgt:.3f}")
            continue
        d_idx = int(d_list[0]); a_idx = int(a_list[0])
        cnt_pred = float(pred_counts[d_idx, a_idx, ch])
        # also check what the rebuilt matrix says
        mat_cnt = float(junc_matrix[d_idx, a_idx, s_idx if strand == "+" else n_s_local + s_idx])
        if abs(mat_cnt - cnt_tgt) > 1e-4:
            print(f"  MISMATCH: sample={s_idx} strand={strand} d_rel={d_rel} "
                  f"tgt_star={cnt_tgt:.4f}  mat={mat_cnt:.4f}  pred={cnt_pred:.4f}")

print("Cross-check complete.")
