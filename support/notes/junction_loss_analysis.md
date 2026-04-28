# Junction Loss Analysis: Paper Pseudocode vs. Current PyTorch Implementation

## Summary

The current PyTorch `_compute_junction_strand_loss` diverges from the paper pseudocode
in three ways. The most important is a CE formula mismatch that causes negative loss
values during training on sparse junction windows (common in SF3B1 data).

---

## Side-by-side comparison

### Cross-entropy (ratio) term

**Paper pseudocode (`multinomial_cross_entropy`):**
```python
pred_ratios = (x + 1e-7) / (x + 1e-7).sum(axis=axis, keepdims=True)
return -(targets * log(pred_ratios)).sum()
```
- `pred_ratios <= 1` always → `log(pred_ratios) <= 0` → `-targets * log(pred_ratios) >= 0`
- **Always non-negative**, even when all targets are zero (0 * anything = 0)

**Current PyTorch (`cross_entropy_loss`):**
```python
log_normalizer  = log((masked_pred + eps).sum(axis))
log_likelihood  = (p_true * log(y_pred + eps)).sum(axis)
log_loss        = log_normalizer - log_likelihood
return safe_masked_mean(log_loss, mask.any(axis))
```
- When all targets are zero: `p_true = 0`, so `log_likelihood = 0`
- But `log_normalizer = log(D * eps)` where D = max_splice_sites = 256
- `log(256 * 1e-7) ≈ -10.6` → **negative loss**

This is the source of the observed negative junction loss. It fires whenever a training
window has valid annotated splice sites but no observed junction counts — common in
sparse datasets like SF3B1 RNA-seq.

Note: the JAX implementation uses the same `cross_entropy_loss` formulation and has
the same theoretical issue, but in practice avoids it because JAX training data
(filtered GTEx/ENCODE) rarely produces windows with valid sites but zero counts.

---

### Poisson (count) term

**Paper pseudocode:**
```python
sum_pred    = x.sum(axis=axis)
sum_targets = soft_clip(targets.sum(axis=axis))
return (sum_pred - sum_targets * log(sum_pred + 1e-7)).sum()
```
- Does **not** subtract the minimum value
- **Can go negative** for well-fitted large-count predictions (expected, not a bug)
- At optimum (`sum_pred = sum_targets = t`): loss = `t - t*log(t)`, negative for `t > e ≈ 2.7`

**Current PyTorch (`poisson_loss`):**
```python
min_value = y_true - y_true * log(y_true + 1e-7)
loss      = (y_pred - y_true * log(y_pred + 1e-7)) - min_value
return safe_masked_mean(loss, mask)
```
- Subtracts minimum value → **always >= 0**
- Same as JAX `poisson_loss`; both differ from the paper pseudocode

---

### Loss weights

| Source | CE weight | Poisson weight |
|--------|-----------|----------------|
| Paper pseudocode | 0.2 | 0.04 |
| JAX `heads.py` | **1.0** | **0.2** |
| Current PyTorch | 0.2 | 0.04 |

PyTorch matches the paper pseudocode. JAX uses weights 5× larger for both terms.

---

## Fix

The minimal fix that matches JAX behaviour: exclude positions with all-zero targets
from the CE mean in `cross_entropy_loss`:

```python
has_target = y_true.sum(dim=axis) > 0
return _safe_masked_mean(log_loss, mask.any(dim=axis) & has_target)
```

This is equivalent to the paper's formulation for the all-zero case (paper returns 0;
fixed PyTorch skips those positions in the mean, also giving 0 contribution).

A stricter alternative: replace `cross_entropy_loss` with the paper's
`multinomial_cross_entropy` directly in `_compute_junction_strand_loss`, which is
always non-negative by construction.
