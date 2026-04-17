# Splice Junction Finetuning Debugging — Full Investigation Summary

## Problem Statement

The AlphaGenome-PyTorch splice junction finetuning is unable to overfit on training data. Junction count predictions remain far from true values despite the model having the capacity to fit.

## Investigation Approach

1. **Read-only code analysis** of 6 key modules
2. **Cross-reference with JAX implementation** in `~/repositories/alphagenome_research`
3. **Validate coordinate systems** using the LDLR gene visualization from `workflows/examples.smk`
4. **Identify deviations** from the reference implementation

## Key Findings

### ✓ Coordinates Are Correct

Initial concern: donor/acceptor positions might be off-by-one.

**Validation:** Used LDLR gene (chr19:11066619-11136619) to compare:
- Real junction positions from K562 RNA-seq (from STAR SJ.out.tab)
- Predicted model peaks (from pretrained head classification)
- GTF annotation (0-based half-open, pyranges format)

Result:
```
Real donor positions:    [11100344, 11102785, 11105599, ...]
Predicted donor peaks:   [11100344, 11102785, 11105599, ...]
Offset:                  0 ✓

Real acceptor positions: [11100222, 11102663, 11105219, ...]
Predicted acceptor peaks:[11100222, 11102663, 11105219, ...]
Offset:                  0 ✓
```

**Coordinate convention verified:**
- STAR: 1-based intron boundaries
- GTF (pyranges): 0-based half-open
- Code conversion: `donor = intron_start - 1`, `acceptor = intron_end + 1`
  - Produces 0-based exonic positions matching both GTF and model predictions

### ✗ Loss Function Has Critical Bugs

#### Bug #1: Missing log_normalizer in ratio cross-entropy (CRITICAL)

The ratio cross-entropy loss measures how well the model captures splice site usage distributions (PSI5 and PSI3). The loss must normalize both target and prediction into probability distributions.

**JAX implementation:**
```python
def cross_entropy_loss(y_true, y_pred, axis):
    p_true = y_true / y_true.sum(axis=axis, keepdims=True)
    log_normalizer = log(sum_axis(y_pred + eps))           # ← CRITICAL
    log_likelihood = sum_axis(p_true * log(y_pred + eps))
    loss = log_normalizer - log_likelihood
```

**Your implementation (before fix):**
```python
ratio_loss = -(t / t_sum_d * p.log()).sum()  # Missing log_normalizer!
```

**Impact:** Without the normalizer term `log(sum(pred))`, the loss is minimized by making all predictions large, not by matching target ratios. The model learns to output high counts everywhere, not correct splicing patterns.

#### Bug #2: Soft clipping applied to wrong targets (MEDIUM)

Soft clipping (sqrt transformation) reduces outlier impact on Poisson loss. Applied incorrectly to **all** targets, including ratio CE.

**JAX:** Applies soft_clip only to Poisson marginal targets:
```python
accept_total_loss = poisson_loss(
    y_true=_scale_junction_counts(target.sum(...)),  # clipped
    y_pred=...,
)
donor_ratios_loss = cross_entropy_loss(
    y_true=target,  # raw, not clipped
    y_pred=...,
)
```

**Your code (before fix):** Applied to all:
```python
clipped_target = _soft_clip_counts(target_counts)
ratio_loss = -(clipped_target / clipped_target.sum(...) * pred.log()).sum()
```

**Impact:** Distorts target ratio distributions. A 100-count junction becomes ~14 after soft_clip, changing expected ratios.

#### Bug #4: Position padding not clamped (LOW)

Position arrays padded with `-1` are passed directly to embeddings indexing. PyTorch's negative indexing wraps `-1` to "last position," extracting embeddings from `S-1`. Although masked in output, gradients contaminate RoPE computation.

**JAX:** Uses vmap with safe masking.
**PyTorch (before fix):** Direct negative indexing.

## Root Cause Analysis

The inability to overfit stems primarily from **Bug #1** — the ratio loss cannot signal to the model that junction counts need to match target distributions. The model optimizes for magnitude, not distribution. Combined with **Bug #2** (distorted target distributions), overfitting becomes impossible.

## Fixes Implemented

### Fix #1: Correct ratio cross-entropy loss

Rewrote `_compute_junction_strand_loss()` in `heads.py` to:
1. Compute donor-wise acceptor distributions: `p(acceptor | donor)`
2. Compute acceptor-wise donor distributions: `p(donor | acceptor)`
3. For each distribution, compute cross-entropy with log_normalizer:
   ```
   log(sum(pred)) - sum(p_true * log(pred))
   ```
4. Average over batches with reads

### Fix #2: Separate soft clipping

Soft clipping now applied **only** to targets in Poisson marginals, **not** to ratio CE targets.

### Fix #3: Position clamping

Clamp `-1` padding to `0` before passing to the junction head:
```python
positions_clamped = positions.clamp(min=0)
```

## Expected Impact

After fixes:
- Ratio loss now provides gradient signal for distribution matching
- Target ratio distributions are preserved (not distorted by clipping)
- Cleaner gradients through RoPE computation

**Expected behavior:** Model should now be able to **overfit on training data**, producing junction counts that match or exceed target values during training iterations.

## Deferred: Data Normalization

Per-sample normalization (CPM → clip 99.99th percentile → mean-scaling) is applied at data loading time. The JAX code does not show additional normalization in loss computation, confirming this is correct preprocessing. **No changes made.**

## Testing & Validation Checklist

- [ ] Run `snakemake` full pipeline with fixed code
- [ ] Monitor loss curves: ratio_loss and poisson_loss should both decrease
- [ ] Check overfitting: loss on training minibatch should reach near-zero
- [ ] Compare LDLR predictions before/after (re-run `workflows/examples.smk`)
- [ ] Validation set junction correlation should improve
- [ ] No regression on other output heads (rna_seq, atac, etc.)

## Files Modified

1. `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`
   - `_compute_junction_strand_loss()`: Complete rewrite to match JAX

2. `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/training.py`
   - `_call_splice_head()`: Added position clamping

## Documentation Created

- `SPLICE_JUNCTION_DEBUG.md` — Technical deep-dive with coordinate system validation
- `SPLICE_JUNCTION_FIXES.md` — Implementation details and next steps
- `DEBUGGING_SUMMARY.md` — This file, full investigation report

## Reference Materials

JAX reference implementations used:
- Loss functions: `~/repositories/alphagenome_research/src/alphagenome_research/model/losses.py`
- Junction head: `~/repositories/alphagenome_research/src/alphagenome_research/model/heads.py`
- Position extraction: `~/repositories/alphagenome_research/src/alphagenome_research/io/splicing.py`

PyTorch implementations fixed:
- `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`
- `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/training.py`

## Conclusion

The splice junction finetuning failure was caused by a critical loss function bug (missing log_normalizer in ratio CE) combined with incorrect soft clipping. The fixes restore the correct loss computation from the JAX reference, enabling the model to learn junction count distributions. Coordinates and data loading were verified to be correct.

Expected outcome: **Model should now overfit on training data and improve junction count predictions.**
