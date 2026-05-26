#!/usr/bin/env python
"""Compare training-style vs inference-style model predictions.

Loads the finetuned checkpoint in both ways and checks if predictions differ.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

PROJECT = "/users/diasfrazer/manglada/projects/alphagenome_finetuning_rna"
RUN     = "debug_splice_junctions__truncrope__origloss__annotated__pretrinit__bfloat16"
DENSITY = "medium"

CKPT_PATH = f"{PROJECT}/results/bsc/finetuning/alphagenome_pytorch/overfitting/single/{DENSITY}/{RUN}/checkpoint_epoch500.pth"
WEIGHTS   = f"{PROJECT}/data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors"
BED_PATH  = f"{PROJECT}/data/prep/overfitting/single/medium.bed"
GENOME    = f"{PROJECT}/data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz"
NPZ_PATH  = f"{PROJECT}/results/finetuning/alphagenome_pytorch/overfitting/single/{DENSITY}/{RUN}/splice_junctions_annotated.npz"
META_PATH = f"{PROJECT}/results/finetuning/alphagenome_pytorch/overfitting/single/{DENSITY}/{RUN}/metadata.json"
SJ_PATHS  = [
    f"{PROJECT}/data/raw/ENA/sf3b1mut/STAR/SRR17111303/paper_pass.SJ.out.tab",
    f"{PROJECT}/data/raw/ENA/sf3b1mut/STAR/SRR17111311/paper_pass.SJ.out.tab",
]

sys.path.insert(0, f"{PROJECT}/src/alphagenome-pytorch/src")
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads, add_head
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    read_star_junctions, junctions_to_junction_matrix, normalize_junctions_per_sample,
)
import pyfaidx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load metadata & positions ──────────────────────────────────────────────
with open(META_PATH) as f:
    meta = json.load(f)
chrom        = meta["chrom"]
padded_start = meta["padded_start"]
padded_end   = meta["padded_end"]
seq_len      = padded_end - padded_start

npz = np.load(NPZ_PATH)
saved_positions   = npz["junction_positions"]   # (4, 512) from run_pretrained_forward_pass
saved_pred_counts = npz["junction_counts"]       # (512, 512, 4) float32

print(f"Window: {chrom}:{padded_start}-{padded_end}  seq_len={seq_len}")
print(f"Valid positions per role (saved): {[(saved_positions[r]>=0).sum() for r in range(4)]}")

# ── Load sequence ──────────────────────────────────────────────────────────
fasta = pyfaidx.Fasta(GENOME)
seq_str = str(fasta[chrom][max(0, padded_start):padded_end]).upper()
if padded_start < 0:
    seq_str = "N" * (-padded_start) + seq_str
seq_tensor = torch.from_numpy(sequence_to_onehot(seq_str)).float().unsqueeze(0).to(device)

# ── Load checkpoint ────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
track_names = ckpt["track_names"]
resolutions = ckpt["resolutions"]
print(f"track_names: {track_names}")
print(f"resolutions: {resolutions}")

# ── Method 1: inference style (model.predict with full model) ──────────────
print("\n=== Method 1: inference style ===")
model1 = AlphaGenome()
model1 = load_trunk(model1, WEIGHTS, exclude_heads=True)
model1 = remove_all_heads(model1)
for modality, names in track_names.items():
    n_tracks = len(names)
    if modality == "splice_junctions":
        n_tracks = n_tracks // 2
    mod_res = resolutions[modality] if isinstance(resolutions, dict) else resolutions
    head = create_finetuning_head(
        assay_type=modality, n_tracks=n_tracks, resolutions=mod_res, num_organisms=1
    )
    add_head(model1, modality, head)
model1.load_state_dict(ckpt["model_state_dict"])
model1 = model1.to(device).eval()

positions_t = torch.from_numpy(saved_positions).long().unsqueeze(0).to(device)  # (1, 4, 512)
with torch.no_grad():
    out1 = model1.predict(seq_tensor, organism_index=0, splice_site_positions=positions_t)
pred1 = out1["splice_sites_junction"]["pred_counts"].squeeze(0).cpu().numpy().astype(np.float32)
print(f"Method 1 pred_counts shape: {pred1.shape}")
print(f"Are method 1 preds same as saved? {np.allclose(pred1, saved_pred_counts, rtol=1e-3, atol=1e-3)}")
print(f"Max diff: {np.abs(pred1 - saved_pred_counts).max():.6f}")

# ── Method 2: training style (separate head + embeddings) ─────────────────
print("\n=== Method 2: training style (separate head, float32) ===")
from alphagenome_pytorch.extensions.finetuning.training import _call_splice_head

# Get the junction head from model1
junc_head = model1.splice_sites_junction_head

# Get embeddings from model (return_embeddings=True)
with torch.no_grad():
    out_emb = model1(seq_tensor, torch.zeros(1, dtype=torch.long, device=device),
                     return_embeddings=True, channels_last=False)
emb_1bp = out_emb.get("embeddings_1bp")
if emb_1bp is None:
    # Try extracting from resolutions
    print("WARNING: embeddings_1bp not in model output, trying resolutions")
    print(f"Output keys: {list(out_emb.keys())}")

print(f"Embeddings 1bp shape: {emb_1bp.shape}")  # (B, C, S)

# Training passes organism idx as zeros
organism_idx = torch.zeros(1, dtype=torch.long, device=device)

# Positions as used in training (clamped)
positions_clamped = positions_t.clamp(min=0)

with torch.no_grad():
    embeddings_dict = {1: emb_1bp}
    pred2_dict = _call_splice_head(
        junc_head, embeddings_dict, organism_idx, positions_t,
        channels_last=False, cls_head=None, junction_top_k=None,
    )

n_s = junc_head._num_tissues
pred2_pos = pred2_dict["pos_counts"].squeeze(0).cpu().numpy().astype(np.float32)  # (D, A, n_s)
pred2_neg = pred2_dict["neg_counts"].squeeze(0).cpu().numpy().astype(np.float32)  # (D, A, n_s)
pred2 = np.concatenate([pred2_pos, pred2_neg], axis=-1)  # (D, A, 2*n_s)
print(f"Method 2 pred_counts shape: {pred2.shape}")
print(f"Are method 2 preds same as method 1? {np.allclose(pred2, pred1, rtol=1e-3, atol=1e-3)}")
print(f"Max diff m2 vs m1: {np.abs(pred2 - pred1).max():.6f}")

# ── Method 3: training style with bfloat16 ─────────────────────────────────
print("\n=== Method 3: training style (bfloat16 autocast) ===")
with torch.no_grad(), torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu",
                                     dtype=torch.bfloat16, enabled=device.type=="cuda"):
    out_emb3 = model1(seq_tensor, torch.zeros(1, dtype=torch.long, device=device),
                      return_embeddings=True, channels_last=False)
emb_1bp3 = out_emb3.get("embeddings_1bp")

with torch.no_grad(), torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu",
                                      dtype=torch.bfloat16, enabled=device.type=="cuda"):
    pred3_dict = _call_splice_head(
        junc_head, {1: emb_1bp3}, organism_idx, positions_t,
        channels_last=False, cls_head=None, junction_top_k=None,
    )
pred3_pos = pred3_dict["pos_counts"].squeeze(0).float().cpu().numpy()
pred3_neg = pred3_dict["neg_counts"].squeeze(0).float().cpu().numpy()
pred3 = np.concatenate([pred3_pos, pred3_neg], axis=-1)
print(f"Method 3 pred shape: {pred3.shape}")
print(f"Are method 3 preds same as method 1? {np.allclose(pred3, pred1, rtol=1e-2, atol=1e-2)}")
print(f"Max diff m3 vs m1: {np.abs(pred3 - pred1).max():.6f}")

# ── Compute Pearson r for methods 1-3 ─────────────────────────────────────
print("\n=== Pearson r computation ===")
# Load STAR junctions and build target matrix
all_juncs = []
for path in SJ_PATHS:
    junc = read_star_junctions(path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc = junc.loc[
        junc["chrom"].str.contains("chr", na=False)
        & junc["strand"].isin(["+", "-"])
    ].drop_duplicates()
    junc["exon_start"] = junc["intron_start"] - 1
    junc["exon_end"]   = junc["intron_end"] + 1
    junc["count"]      = junc["n_uniquely_mapped_reads"]
    junc = normalize_junctions_per_sample(junc)
    all_juncs.append(junc)

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

_, junc_matrix = junctions_to_junction_matrix(
    all_juncs_local, max_splice_sites=saved_positions.shape[1], positions=saved_positions,
)

def compute_pearson_r(pred_counts_full, pos, jmat, n_s):
    all_pred_nz, all_tgt_nz = [], []
    for pred_s, tgt_s, donor_row, accept_row in [
        (slice(None, n_s),  slice(None, n_s),  0, 1),
        (slice(n_s, None),  slice(n_s, None),  2, 3),
    ]:
        pc = pred_counts_full[:, :, pred_s]
        tc = jmat[:, :, tgt_s]
        d_pos = pos[donor_row]; a_pos = pos[accept_row]
        vd = (d_pos >= 0).astype(float); va = (a_pos >= 0).astype(float)
        pmask = np.outer(vd, va).astype(bool)
        pmask4 = pmask[:, :, np.newaxis] & np.ones((1, 1, n_s), dtype=bool)
        nz = pmask4 & (tc > 0)
        if nz.any():
            all_pred_nz.append(pc[nz])
            all_tgt_nz.append(tc[nz])
    if not all_pred_nz:
        return float("nan"), 0
    pred_nz = np.concatenate(all_pred_nz)
    tgt_nz = np.concatenate(all_tgt_nz)
    r, _ = pearsonr(np.log1p(pred_nz), np.log1p(tgt_nz))
    return r, len(pred_nz)

r1, n1 = compute_pearson_r(pred1, saved_positions, junc_matrix, n_s)
r2, n2 = compute_pearson_r(pred2, saved_positions, junc_matrix, n_s)
r3, n3 = compute_pearson_r(pred3, saved_positions, junc_matrix, n_s)
print(f"Method 1 (inference/full model): r={r1:.4f}  n={n1}")
print(f"Method 2 (training-style, fp32): r={r2:.4f}  n={n2}")
print(f"Method 3 (training-style, bf16): r={r3:.4f}  n={n3}")
print(f"Saved npz pred_counts:           r={r1:.4f}  (same as method 1)")

# ── Sample predictions for top junctions ───────────────────────────────────
print("\n=== Top 5 junctions: predictions vs target ===")
all_juncs_both = pd.concat(all_juncs_local)
all_juncs_both = all_juncs_both.sort_values("count", ascending=False)
print("sample  strand  d_rel    a_rel    tgt_norm  pred_m1  pred_m2  pred_m3")
for _, row in all_juncs_both.head(5).iterrows():
    d_rel = int(row["d_rel"]); a_rel = int(row["a_rel"])
    strand = row["strand"]
    s_idx = all_juncs_local.index(all_juncs_local[[i for i, j in enumerate(all_juncs_local) if any((j["d_rel"]==d_rel) & (j["a_rel"]==a_rel) & (j["strand"]==strand))][0]])
    pos_arr = saved_positions[0 if strand=="+" else 2]
    acc_arr = saved_positions[1 if strand=="+" else 3]
    d_idx_arr = np.where(pos_arr == d_rel)[0]
    a_idx_arr = np.where(acc_arr == a_rel)[0]
    if len(d_idx_arr)==0 or len(a_idx_arr)==0:
        print(f"  MISS: d_rel={d_rel} a_rel={a_rel}")
        continue
    d_idx = int(d_idx_arr[0]); a_idx = int(a_idx_arr[0])
    ch = s_idx if strand=="+" else n_s + s_idx
    tgt = float(row["count"])
    p1 = float(pred1[d_idx, a_idx, ch])
    p2 = float(pred2[d_idx, a_idx, ch])
    p3 = float(pred3[d_idx, a_idx, ch])
    print(f"  s={s_idx} {strand}  d={d_rel:6d} a={a_rel:6d}  tgt={tgt:.3f}  p1={p1:.3f}  p2={p2:.3f}  p3={p3:.3f}")
