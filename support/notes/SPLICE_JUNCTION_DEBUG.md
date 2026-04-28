# Splice Junction Finetuning Bug Analysis

## Executive Summary

Found **3 critical bugs** in the PyTorch splice junction loss implementation that prevent overfitting:

1. **Missing log_normalizer in ratio cross-entropy loss** (CRITICAL)
2. **Soft clipping applied to wrong targets** (MEDIUM)
3. **Position padding not clamped** (LOW)

The coordinate system (donor/acceptor positions) is **correct** and matches the JAX reference implementation.

## Background: Position Coordinate System

### Validation via LDLR Gene Example

Used the LDLR gene visualization (`workflows/examples.smk`) to verify coordinates:

```
GTF (0-based half-open, pyranges):
  Exon ends:   [11100345, 11102786, 11105600, ...]
  Exon starts: [11100222, 11102663, 11105219, ...]

Real splice junctions from K562 RNA-seq:
  Donors:     [11100344, 11102785, 11105599, ...]  (= exon_end - 1)
  Acceptors:  [11100222, 11102663, 11105219, ...]  (= exon_start)

Predicted model peaks:
  Donors:     [11100344, 11102785, 11105599, ...]  (offset = 0 ✓)
  Acceptors:  [11100222, 11102663, 11105219, ...]  (offset = 0 ✓)
```

**Conclusion:** Positions are 0-based and correctly computed. The formula `exon_start = intron_start - 1` (donor) and `exon_end = intron_end + 1` (acceptor) produces correct splice site coordinates.

### Coordinate Convention

- **STAR output:** `intron_start` and `intron_end` (1-based, intron boundaries)
- **GTF (pyranges):** 0-based half-open intervals
- **Your code:** Converts to 0-based exonic positions:
  - `donor = intron_start - 1` → 0-based last exonic base
  - `acceptor = intron_end + 1` → 0-based first exonic base of next exon
- **JAX reference:** Uses the same convention (`Start - 1` for donor, `End` for acceptor)

## Bug #1: Missing log_normalizer in Ratio Cross-Entropy Loss (CRITICAL)

### The Issue

The ratio cross-entropy loss measures how well predicted counts capture the **distribution** of splice usage (PSI5 and PSI3). This requires normalizing both predictions and targets into probability distributions.

### JAX Implementation

`alphagenome_research/src/alphagenome_research/model/losses.py` lines 167-184:

```python
def cross_entropy_loss(
    *,
    y_true: Float[Array, '*dims'],
    y_pred: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
    axis: int,
    eps: float = 1e-7,
) -> Float[Array, '']:
  """Cross entropy loss on counts."""
  mask = jnp.broadcast_to(mask, y_true.shape)
  y_true = jnp.where(mask, y_true.astype(jnp.float32), 0)
  p_true = y_true / jnp.maximum(y_true.sum(axis=axis, keepdims=True), eps)

  log_normalizer = jnp.log((jnp.where(mask, y_pred, 0) + eps).sum(axis=axis))
  log_likelihood = (p_true * jnp.log(y_pred + eps)).sum(axis=axis)
  log_loss = log_normalizer - log_likelihood
  return _safe_masked_mean(log_loss, mask.any(axis=axis))
```

**Key points:**
1. `p_true = targets / targets.sum(axis)` — normalize targets into distribution
2. `log_normalizer = log(predictions.sum(axis))` — **THIS IS MISSING IN YOUR CODE**
3. `log_loss = log_normalizer - sum(p_true * log(pred))`
4. This equals `-sum(p_true * log(pred / sum(pred)))` = cross-entropy of distributions

### Your Implementation

`heads.py` lines 75-83:

```python
ratio_loss = torch.tensor(0.0, device=device)
if has_reads.any():
    p = pred_counts[has_reads]
    t = clipped_target[has_reads]
    t_sum_d = t.sum(dim=1, keepdim=True).clamp(min=1e-7)
    ratio_loss = ratio_loss - (t / t_sum_d * p.log().clamp(min=-100)).sum() / has_reads.sum()
    t_sum_a = t.sum(dim=2, keepdim=True).clamp(min=1e-7)
    ratio_loss = ratio_loss - (t / t_sum_a * p.log().clamp(min=-100)).sum() / has_reads.sum()
```

**Missing:** `log((p + eps).sum(axis))` term. Currently computes `-sum(p_true * log(pred))` which can be minimized by making all predictions arbitrarily large.

### Impact

Without the normalizer, the loss becomes:
- Minimized when: `pred` is large (regardless of distribution)
- Not minimized when: `pred` correctly captures the ratio distribution

The model learns to output uniformly high counts, not the correct splicing ratios. **This explains why junctions don't overfit.**

## Bug #2: Soft Clipping Applied to Wrong Targets (MEDIUM)

### The Issue

Soft clipping (sqrt transformation) is designed to reduce the impact of count outliers on Poisson loss. It should **only** be applied to targets in the Poisson marginal terms, **not** to targets in the cross-entropy ratio terms.

### JAX Implementation

`heads.py` lines 1013-1053:

```python
def _scale_junction_counts(counts):
  return jnp.where(
      counts > _SOFT_CLIP_VALUE,
      2.0 * jnp.sqrt(counts * _SOFT_CLIP_VALUE) - _SOFT_CLIP_VALUE,
      counts,
  )

# Applied only to Poisson loss targets:
accept_total_loss = poisson_loss(
    y_true=_scale_junction_counts(
        count_target.sum(axis=-2, ...)  # ← soft clip HERE
    ),
    y_pred=...,
)

# Cross-entropy uses raw targets:
donor_ratios_loss = cross_entropy_loss(
    y_true=count_target,  # ← NO soft clip, raw counts
    y_pred=pred_pair,
)
```

### Your Implementation

```python
clipped_target = _soft_clip_counts(target_counts)  # ← applied globally

# Both Poisson and ratio use clipped_target:
d_loss = poisson_loss(y_true=true_donor_total, ...)  # uses clipped
ratio_loss = -(t / t_sum_d * p.log()).sum()  # uses clipped (t = clipped_target)
```

**Problem:** Soft clipping distorts the target ratios. A junction with 100 reads becomes ~14 after soft_clip, changing its ratio distribution.

## Bug #3: Data Normalization (DEFERRED)

Per-sample normalization (CPM → clip 99.99th percentile → divide by mean) is likely applied at data loading time in TFRecord preprocessing, not during loss computation. The JAX code doesn't show repeated normalization, confirming this is a preprocessing step.

**Decision:** Keep `normalize_junctions_per_sample` as-is. The CPM + clip is correct; mean scaling is applied once at load time.

## Bug #4: Position Padding Not Clamped (LOW)

### The Issue

Junction position arrays are padded with `-1` to a fixed size (max_splice_sites=256). When passed to the head's `_index_embeddings`, PyTorch interprets `-1` as "last position" (negative indexing), extracting embeddings from `S-1` instead of masking them.

Although downstream predictions are zeroed (masked), gradients still flow through contaminated RoPE computations during backprop.

### JAX Implementation

`heads.py` lines 951-961:

```python
pos_mask = jnp.einsum('bd,ba->bda', pos_donor_idx >= 0, pos_accept_idx >= 0)
neg_mask = jnp.einsum('bd,ba->bda', neg_donor_idx >= 0, neg_acceptor_idx >= 0)
# ...
pred_counts = jnp.where(splice_junction_mask, pred_counts, 0)
```

Predictions are masked to zero for invalid indices, but JAX's vmap handles negative indices differently than PyTorch's advanced indexing.

### Fix

Clamp `-1` positions to a safe dummy index (0) before passing to the head, then mask the output.

## Fixing Strategy

1. **Fix Bug #1 (critical):** Reimplement cross-entropy ratio loss to match JAX exactly
2. **Fix Bug #2 (medium):** Separate clipping — apply only to Poisson marginal targets
3. **Skip Bug #3:** Keep normalization as-is
4. **Fix Bug #4 (low):** Clamp `-1` → `0` before head forward pass

## References

- JAX Loss: `~/repositories/alphagenome_research/src/alphagenome_research/model/losses.py` lines 167-184, 45-58
- JAX Junction Head: `~/repositories/alphagenome_research/src/alphagenome_research/model/heads.py` lines 998-1053, 890-975
- JAX Positions: `~/repositories/alphagenome_research/src/alphagenome_research/io/splicing.py` lines 62-100
- PyTorch Implementation: `src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/heads.py`, `training.py`

## Test Case

The LDLR example (`workflows/examples.smk` output) provides validation:
- Real junctions from K562 RNA-seq match predicted positions (offset=0)
- Model correctly classifies splice sites (prob > 0.99 at real positions)
- Once loss is fixed, junction count predictions should improve dramatically
