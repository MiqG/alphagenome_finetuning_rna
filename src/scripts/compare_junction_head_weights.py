"""
Compare pretrained vs fine-tuned splice junction head weights.

Usage:
    python src/scripts/compare_junction_head_weights.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS = Path("results/finetuning/alphagenome_pytorch/overfitting/single/summary/epoch_logs.parquet")
PRETRAINED_WEIGHTS = Path("data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors")
FINETUNED_DIR = Path("results/finetuning/alphagenome_pytorch/overfitting/single/medium")

# ---------------------------------------------------------------------------
# 1. Identify best run at epoch 1000
# ---------------------------------------------------------------------------
logs = pd.read_parquet(LOGS)
print("Columns:", logs.columns.tolist())
print("Runs:", logs["run_name"].unique()[:5])

# Focus on splice junction Pearson at final epoch
sj_col = [c for c in logs.columns if "splice_junction" in c.lower() and "pearson" in c.lower()]
print("Junction Pearson cols:", sj_col)

# Filter to epoch 1000 and find best junction run
final = logs[logs["epoch"] == logs["epoch"].max()].copy()
if sj_col:
    col = sj_col[0]
    best_row = final.nlargest(5, col)[["run_name", "epoch", col]]
    print("\nTop 5 runs at final epoch by junction Pearson r:")
    print(best_row.to_string(index=False))
    best_run = final.nlargest(1, col)["run_name"].iloc[0]
else:
    print("No junction pearson column found, using known best run")
    best_run = "debug_splice_junctions__truncrope__newloss__predicted__randinit__float32"

print(f"\nBest run: {best_run}")

# ---------------------------------------------------------------------------
# 2. Load weights
# ---------------------------------------------------------------------------
print("\nLoading pretrained weights...")
pretrained = load_file(str(PRETRAINED_WEIGHTS))
pretrained_keys = [k for k in pretrained.keys()]
print(f"  Total keys: {len(pretrained_keys)}")

# Find splice junction head keys
sj_keys_pretrained = [k for k in pretrained_keys if "splice_junction" in k.lower() or "splice_site" in k.lower()]
print(f"  Junction/site head keys: {len(sj_keys_pretrained)}")
for k in sj_keys_pretrained[:20]:
    print(f"    {k}: {pretrained[k].shape}")

print(f"\nLoading fine-tuned weights from {best_run}...")
finetuned_path = FINETUNED_DIR / best_run / "best_model.pth"
if not finetuned_path.exists():
    print(f"  Not found: {finetuned_path}")
    # Try float32 variant
    best_run_fp32 = best_run.replace("bfloat16", "float32")
    finetuned_path = FINETUNED_DIR / best_run_fp32 / "best_model.pth"
    print(f"  Trying: {finetuned_path}")
    best_run = best_run_fp32

finetuned_ckpt = torch.load(str(finetuned_path), map_location="cpu", weights_only=False)
if isinstance(finetuned_ckpt, dict) and "model_state_dict" in finetuned_ckpt:
    finetuned = finetuned_ckpt["model_state_dict"]
elif isinstance(finetuned_ckpt, dict) and "state_dict" in finetuned_ckpt:
    finetuned = finetuned_ckpt["state_dict"]
else:
    finetuned = finetuned_ckpt
print(f"  Total keys: {len(finetuned.keys())}")
sj_keys_finetuned = [k for k in finetuned.keys() if "splice_junction" in k.lower() or "splice_site" in k.lower()]
print(f"  Junction/site head keys: {len(sj_keys_finetuned)}")
for k in sj_keys_finetuned[:20]:
    t = finetuned[k]
    print(f"    {k}: {t.shape}")

# ---------------------------------------------------------------------------
# 3. Compare shared keys
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("WEIGHT COMPARISON: pretrained vs fine-tuned splice junction head")
print("="*70)

# Normalize key names (finetuned may have module. prefix)
def strip_prefix(d, prefix="module."):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}

finetuned_clean = strip_prefix(finetuned)

# Find junction head keys in both
common_sj_keys = []
for k in sj_keys_pretrained:
    # Try to find corresponding key in finetuned
    # pretrained uses e.g. "heads.splice_junctions.conv.weight"
    # finetuned may differ if head was rebuilt
    if k in finetuned_clean:
        common_sj_keys.append(k)

print(f"\nShared junction/site keys: {len(common_sj_keys)}")

# Also check all finetuned junction keys
print("\nAll fine-tuned junction/site keys:")
for k in sorted(sj_keys_finetuned):
    t = finetuned[k]
    print(f"  {k}: {t.shape}, dtype={t.dtype}")

# ---------------------------------------------------------------------------
# 4. Detailed comparison for each shared or equivalent key
# Handles n_organisms mismatch by slicing pretrained to organism 0
# ---------------------------------------------------------------------------
print("\n" + "-"*70)
print("Per-parameter comparison (L2 norm, cosine sim, relative change):")
print("(pretrained sliced to organism 0 where shapes differ)")
print("-"*70)
print(f"{'Parameter':<55} {'pre_shape':<22} {'ft_shape':<22} {'L2(pre)':<9} {'L2(ft)':<9} {'rel_change':<12} {'cos_sim':<9}")

results = []
for k_pre in sj_keys_pretrained:
    k_ft = k_pre
    if k_ft not in finetuned_clean:
        continue

    pre = pretrained[k_pre].float()
    ft  = finetuned_clean[k_ft].float()

    # Align n_organisms: slice pretrained organism 0 if needed
    if pre.shape[0] != ft.shape[0]:
        pre = pre[0:1]

    # Align n_tissues for RoPE params: select tissue 139 twice to match T_ft=2
    if pre.ndim == 4 and pre.shape[2] != ft.shape[2]:
        T_ft = ft.shape[2]
        ref = 139 if pre.shape[2] > 139 else 0
        indices = [ref] * T_ft
        pre = pre[:, :, indices, :]  # [1, 2, T_ft, H]

    # Align n_tissues for conv/bias: slice first T_ft tissues
    if pre.ndim in (1, 2, 3) and pre.shape != ft.shape:
        # e.g. usage conv [1, 734, 1536] → [1, 4, 1536]
        slices = tuple(slice(0, s) for s in ft.shape)
        pre = pre[slices]

    if pre.shape != ft.shape:
        print(f"  SHAPE MISMATCH AFTER SLICE {k_pre}: pre={pre.shape} ft={ft.shape}")
        continue

    l2_pre  = pre.norm().item()
    l2_ft   = ft.norm().item()
    delta   = (ft - pre).norm().item()
    rel     = delta / (l2_pre + 1e-12)
    cos_sim = torch.nn.functional.cosine_similarity(
        pre.flatten().unsqueeze(0), ft.flatten().unsqueeze(0)
    ).item()

    short_k = k_pre.split("splice_junctions.")[-1] if "splice_junctions" in k_pre else k_pre.split("splice_site.")[-1]
    pre_shape_orig = tuple(pretrained[k_pre].shape)
    print(f"  {short_k:<55} {str(pre_shape_orig):<22} {str(tuple(ft.shape)):<22} {l2_pre:<9.4f} {l2_ft:<9.4f} {rel:<12.4f} {cos_sim:<9.4f}")

    results.append({
        "key": k_pre,
        "l2_pretrained": l2_pre,
        "l2_finetuned": l2_ft,
        "rel_change": rel,
        "cos_sim": cos_sim,
    })

# ---------------------------------------------------------------------------
# 5. RoPE params: per-tissue-slot breakdown
# ---------------------------------------------------------------------------
print("\n" + "-"*70)
print("RoPE params per-slot analysis (shape: [n_organisms, 2, n_tissues, hidden])")
print("-"*70)

for rope_name in ["pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor"]:
    pre_key = next((k for k in sj_keys_pretrained if rope_name in k), None)
    ft_key  = next((k for k in finetuned_clean.keys() if rope_name in k and "junction" in k), None)
    if pre_key is None or ft_key is None:
        print(f"  {rope_name}: not found (pre={pre_key}, ft={ft_key})")
        continue

    pre = pretrained[pre_key].float()  # [n_org, 2, T_pre, H]
    ft  = finetuned_clean[ft_key].float()  # [n_org, 2, T_ft, H]

    print(f"\n  {rope_name}: pretrained {tuple(pre.shape)} → finetuned {tuple(ft.shape)}")

    T_ft = ft.shape[2]
    # For pretrained, look at tissue slot 139 (the one that was broadcast)
    ref_idx = 139 if pre.shape[2] > 139 else 0
    pre_ref = pre[0, :, ref_idx, :]  # [2, H] — the reference tissue

    for t in range(T_ft):
        ft_t = ft[0, :, t, :]  # [2, H]
        pre_ref_flat = pre_ref.flatten()
        ft_t_flat    = ft_t.flatten()
        cos = torch.nn.functional.cosine_similarity(pre_ref_flat.unsqueeze(0), ft_t_flat.unsqueeze(0)).item()
        delta_norm = (ft_t - pre_ref).norm().item()
        print(f"    slot {t}: cos_sim_vs_pretrained_139={cos:.4f}, ||delta||={delta_norm:.4f}, ||ft||={ft_t.norm().item():.4f}")

# ---------------------------------------------------------------------------
# 6. Conv projection weight (1536→768)
# ---------------------------------------------------------------------------
print("\n" + "-"*70)
print("Conv1d projection weight statistics:")
print("-"*70)
for k in sj_keys_pretrained:
    if "conv" in k and "weight" in k and "splice_junction" in k:
        pre = pretrained[k].float()
        k_ft = k
        if k_ft not in finetuned_clean:
            continue
        ft = finetuned_clean[k_ft].float()
        delta = (ft - pre)
        print(f"  {k}")
        print(f"    pretrained: mean={pre.mean():.5f}, std={pre.std():.5f}, norm={pre.norm():.4f}")
        print(f"    finetuned:  mean={ft.mean():.5f}, std={ft.std():.5f}, norm={ft.norm():.4f}")
        print(f"    delta:      mean={delta.mean():.5f}, std={delta.std():.5f}, norm={delta.norm():.4f}")
        print(f"    rel_change: {delta.norm()/pre.norm():.4f}")

# ---------------------------------------------------------------------------
# 7. Slot symmetry: how similar are the two fine-tuned tissue slots to each other?
# ---------------------------------------------------------------------------
print("\n" + "-"*70)
print("Fine-tuned RoPE slot symmetry (slot 0 vs slot 1 cosine similarity):")
print("-"*70)
for rope_name in ["pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor"]:
    ft_key = next((k for k in finetuned_clean.keys() if rope_name in k and "junction" in k), None)
    if ft_key is None:
        continue
    ft = finetuned_clean[ft_key].float()  # [1, 2, T_ft, H]
    slot0 = ft[0, :, 0, :].flatten()
    slot1 = ft[0, :, 1, :].flatten()
    cos = torch.nn.functional.cosine_similarity(slot0.unsqueeze(0), slot1.unsqueeze(0)).item()
    diff_norm = (ft[0, :, 0, :] - ft[0, :, 1, :]).norm().item()
    print(f"  {rope_name:<20}: cos_sim(slot0,slot1)={cos:.4f}, ||slot0-slot1||={diff_norm:.4f}")

print("\nDone!")
