# Splice Junction Loss Fixes — Implementation Log

## Fixes Applied

### Fix #1: Missing log_normalizer in Ratio Cross-Entropy Loss (CRITICAL)

**File:** `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`

**Problem:** The cross-entropy ratio loss was missing the log-normalization term, computing:
```
ratio_loss = -sum(p_true * log(pred))
```
instead of the correct:
```
ratio_loss = log(sum(pred)) - sum(p_true * log(pred/sum(pred)))
           = log(sum(pred)) - sum(p_true * log(pred))
```

Without the normalizer, the loss can be minimized by making all predictions arbitrarily large, preventing the model from learning correct splicing ratios.

**Solution:** Implemented the JAX cross-entropy loss exactly from `alphagenome_research/model/losses.py` lines 167-184:

```python
# Donor ratio CE (p(acceptor | donor))
t_sum_d = t.sum(dim=1, keepdim=True).clamp(min=eps)
p_true_d = t / t_sum_d
p_masked_d = p + eps
log_norm_d = torch.log(p_masked_d.sum(dim=1, keepdim=True))
log_lik_d = (p_true_d * torch.log(p_masked_d)).sum(dim=1)
ratio_loss_d = log_norm_d - log_lik_d

# Acceptor ratio CE (p(donor | acceptor))
t_sum_a = t.sum(dim=2, keepdim=True).clamp(min=eps)
p_true_a = t / t_sum_a
log_norm_a = torch.log(p_masked_d.sum(dim=2, keepdim=True))
log_lik_a = (p_true_a * torch.log(p_masked_d)).sum(dim=2)
ratio_loss_a = log_norm_a - log_lik_a
```

This correctly computes the cross-entropy of target ratio distributions versus predicted ratio distributions.

### Fix #2: Soft Clipping Applied to Wrong Targets (MEDIUM)

**File:** `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`

**Problem:** Soft clipping (sqrt transformation for outlier reduction) was applied to all targets before the loss computation. This distorted target ratios, e.g., 100 reads became ~14 after soft_clip, changing the expected ratio distribution.

**Solution:** Apply soft clipping **only** to targets used in Poisson marginal loss, **not** to targets used in ratio cross-entropy:

```python
# Soft-clipped version for Poisson marginals only
clipped_target = _soft_clip_counts(target_counts)

# Poisson loss uses clipped targets
d_loss = poisson_loss(
    y_true=clipped_target.sum(dim=2),  # clipped
    y_pred=pred_counts.sum(dim=2),
    ...
)

# Ratio CE uses RAW targets (not clipped) to preserve distributions
if has_reads.any():
    p = pred_counts[has_reads]
    t = target_counts[has_reads]  # RAW, not clipped
    # ... compute ratio losses with t ...
```

This ensures ratio distributions accurately represent the target splicing patterns while still benefiting from outlier reduction in the Poisson marginals.

### Fix #3: Position Padding Not Clamped (LOW)

**File:** `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/training.py`

**Problem:** Junction position arrays are padded with `-1` to max_splice_sites=256. PyTorch's negative indexing interprets `-1` as "last position," causing embeddings to be extracted from position `S-1` instead of masked. Although downstream predictions are zeroed by the mask, gradients still flow through contaminated RoPE computations.

**Solution:** Clamp `-1` padding to 0 before head forward pass:

```python
if isinstance(head, SpliceSitesJunctionHead):
    if positions is None:
        return {}
    # Clamp -1 padding to 0 to avoid PyTorch negative indexing wrapping.
    positions_clamped = positions.clamp(min=0)
    out = head(emb, org, channels_last=channels_last, 
               splice_site_positions=positions_clamped)
```

Padding now uses a safe dummy index (0) that gets masked in output, avoiding gradient contamination.

## Testing & Validation

### Before & After Comparison

**Expected impact:**
1. **Fix #1 (critical):** Junction count predictions should now respond to gradient signals and improve during training
2. **Fix #2 (medium):** Ratio distributions should stabilize as targets are no longer distorted
3. **Fix #4 (low):** Cleaner gradients through RoPE computations

### Validation Steps

1. **Verify loss computation** matches JAX:
   ```bash
   pytest tests/jax_comparison/test_splice_junction_loss.py -v
   ```

2. **Check overfitting on small dataset:**
   ```bash
   snakemake -n  # dry run to check workflow
   # Then run full finetuning and monitor loss curves
   ```

3. **Compare with LDLR example:**
   - Re-run `workflows/examples.smk` after fixes
   - Check if predicted junction counts now improve on the LDLR region
   - Real vs predicted junction correlations should increase

## Reference Implementation

All fixes are derived from the JAX reference implementation:
- **Cross-entropy loss:** `~/repositories/alphagenome_research/src/alphagenome_research/model/losses.py` lines 167-184
- **Junction loss head:** `~/repositories/alphagenome_research/src/alphagenome_research/model/heads.py` lines 998-1053, 1013-1018
- **Soft clipping:** Applied only inside Poisson loss computation

## Related Files Modified

1. `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`
   - `_compute_junction_strand_loss()`: Rewrote ratio loss with log_normalizer and separation of clipping

2. `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/training.py`
   - `_call_splice_head()`: Added position clamping

## Deferred Fix: Data Normalization (Bug #3)

Per-sample normalization (CPM → clip 99.99th percentile → divide by mean) is applied at data loading time and is correct. The JAX code does not show repeated normalization during loss computation, confirming this is a preprocessing step. **No changes made.**

## Next Steps

1. Run full finetuning pipeline with fixes
2. Monitor loss curves (ratio_loss and poisson_loss should both improve)
3. Compare junction count predictions before/after on validation set
4. Check whether model now overfits on training data (expected behavior for finetuning)
