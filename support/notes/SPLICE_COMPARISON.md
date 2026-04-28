# Splice Junction Handling: PyTorch vs JAX Implementation

## Overview
Both implementations handle splice sites through three complementary prediction heads:
1. **Classification** — which splice sites (5 classes: donor+, acceptor+, donor−, acceptor−, none)
2. **Usage** — per-sample proportions using each splice site
3. **Junction Prediction** — read counts for donor-acceptor pairs

---

## 1. DATA REPRESENTATION

### JAX (`alphagenome_research`)
**From `io/splicing.py`:**
- **Input**: STAR SJ.out.tab junction files + strand information
- **Targets**: 5-dimensional one-hot for classification + per-tissue usage proportions
- **Key Data Structure**: `SpliceSiteAnnotationExtractor`
  - Handles forward (+) and reverse (−) strand junctions separately
  - Converts intron [start, end) to donor (start-1) and acceptor (end) positions
  - Outputs binary masks for splice sites across interval
  - Per-tissue usage extracted from junction counts

**Format**: `(interval_width, 5 + num_tissues)` where:
- Dims 0-3: Classification (donor+, acceptor+, donor−, acceptor−)
- Dim 4: None flag
- Dims 5+: Per-tissue usage (fractional read counts)

### PyTorch (`alphagenome-pytorch`)
**From `extensions/finetuning/datasets.py`:**
- **Input**: STAR SJ.out.tab files (same as JAX)
- **Processing**: `SpliceJunctionDataset` uses helper functions:
  - `junctions_to_classification_array()` — 5-class one-hot union across all samples
  - `junctions_to_usage_array()` — per-sample fractional usage
  - `junctions_to_junction_matrix()` — donor×acceptor read-count pairs

**Format**: `{1: (seq_len, 5 + n_files), "junction_positions": ..., "junction_matrix": ...}`

**Key Difference**: PyTorch also stores:
- `junction_positions` — shape `(4, max_splice_sites)` with donor/acceptor indices
- `junction_matrix` — shape `(max_splice_sites, max_splice_sites, 2*n_samples)` for explicit pairs

---

## 2. LOSS COMPUTATION

### JAX (`model/heads.py`)
```python
# Classification
classification_mask = jnp.any(splice_sites, axis=-1, keepdims=True)
loss = cross_entropy_loss_from_logits(
    y_pred_logits=logits,
    y_true=(1.0 - 1e-7) * splice_sites + 1e-7/num_tracks,  # label smoothing
    mask=classification_mask,
    axis=-1
)

# Usage
loss = binary_crossentropy_from_logits(
    y_pred=logits,
    y_true=jnp.clip(splice_site_usage, 1e-7, 1-1e-7),
    mask=self._get_targets_mask(...)
)
```

### PyTorch (`extensions/finetuning/training.py`)
```python
# Classification  [FIXED in your patch]
target = all_targets[..., :N_CLASSES]  # (batch, seq_len, 5)
mask = target.any(dim=-1, keepdim=True).expand_as(pred)
loss = cross_entropy_loss_from_logits(
    y_pred_logits=pred,
    y_true=(1.0 - 1e-7) * target + 1e-7 / N_CLASSES,
    mask=mask,
    axis=-1
)

# Usage
target = all_targets[..., N_CLASSES:]  # (batch, seq_len, n_samples)
mask = (target > 0).any(dim=-1, keepdim=True).expand_as(pred)
loss = binary_crossentropy_from_logits(
    y_pred=pred,
    y_true=target.float(),
    mask=mask
)
```

**Key Difference**: JAX clips usage targets to `[1e-7, 1-1e-7]`, PyTorch does not (applies clip only in binary_crossentropy_from_logits internally).

---

## 3. HEAD ARCHITECTURE

### PyTorch Heads
All three heads use `MultiOrganismConv1d` for projection:

**Classification & Usage** (Lines 398–505):
```python
class SpliceSitesClassificationHead(nn.Module):
    def __init__(self, in_channels=1536, num_organisms=2):
        self.conv = MultiOrganismConv1d(
            in_channels=1536,
            out_channels=5,  # 5 classes
            num_organisms=num_organisms
        )
    
    def forward(self, embeddings_1bp, organism_index, channels_last=True):
        # Internal NCL: (B, 5, S)
        # Output NLC: (B, S, 5)
```

**Junction Head** (Lines 507+):
- Projects embeddings to hidden dimension (768)
- Applies RoPE (Rotary Position Embedding) at donor/acceptor positions
- Computes `einsum("bdth,bath->bdat")` for donor×acceptor pairs
- Uses softplus activation + separate masking per strand/tissue

### JAX Heads
Similar architecture but:
- Uses Haiku dense layers (`hk.Linear`)
- RoPE parameters learned per organism + strand + tissue
- More explicit tensor shape assertions via `@typing.jaxtyped`

---

## 4. JUNCTION MATRIX COMPUTATION

### PyTorch (`SpliceJunctionDataset.__getitem__`)
```python
# Returns both:
1. targets_dict[1] — classification + usage for loss computation
2. "junction_positions" — (4, max_splice_sites) indices
3. "junction_matrix" — (max_splice_sites, max_splice_sites, 2*n_samples) 
                        read counts for positional loss
```

**Advantage**: Allows joint training on:
- Per-base classification/usage (dense targets)
- Donor×acceptor pairs (sparse, positional targets)

### JAX (`variant_scoring/splice_junction.py`)
- Computes `top_k_splice_sites()` from predictions for inference
- Does NOT return pre-computed junction matrix
- Instead: dynamically extracts pairs during loss computation

**Difference**: JAX extracts tops from predictions, PyTorch pre-computes from ground truth.

---

## 5. TRAINING LOSS DIFFERENCES

### PyTorch Junction Loss (`_compute_junction_strand_loss`)
```python
# Soft-clip counts: counts > 10 → 2*sqrt(counts*10) - 10
clipped_target = _soft_clip_counts(target_counts)

# Poisson loss on donor/acceptor marginals (0.2× weight)
d_loss = poisson_loss(sum over acceptors)
a_loss = poisson_loss(sum over donors)

# Negative log-likelihood on per-junction ratios (1.0× weight)
# ratio_loss = - sum(t_ij / t_i• * log(p_ij))
```

### JAX Junction Loss
- Uses similar soft-clipping strategy (matches PyTorch)
- Separates positive/negative strand computation
- Final weight: `0.2 * (d_loss + a_loss) + ratio_loss`

---

## 6. FORMAT & SHAPE HANDLING

| Aspect | JAX | PyTorch |
|--------|-----|---------|
| **Targets shape** | `(seq_len, 5+n_tissues)` | `(batch, seq_len, 5+n_samples)` |
| **Mask computation** | `jnp.any(..., axis=-1, keepdims=True)` | `.any(dim=-1, keepdim=True)` |
| **Position encoding** | RoPE parameters learned | RoPE parameters learned |
| **Strand handling** | Swapped indices for negative strand | Separate pos/neg masks |
| **Multi-organism** | Per-organism metadata + masks | Per-organism tissue masks |
| **Loss weights** | Static: Poisson 0.2× + ratio 1.0× | Static: same |

---

## 7. KEY IMPLEMENTATION DIFFERENCES

### 1. **Batch Dimension Handling** (Your Bug!)
- **JAX**: Assumes single sample, operates on `(seq_len, channels)`
- **PyTorch**: Must handle batches `(batch, seq_len, channels)`
  - **Bug you fixed**: Slicing `all_targets[:, :N_CLASSES]` used wrong dimension
  - **Correct**: Use `all_targets[..., :N_CLASSES]` for last dimension

### 2. **Target Clipping**
- **JAX**: `splice_site_usage` clipped to `[1e-7, 1-1e-7]` before loss
- **PyTorch**: Clipping only in `binary_crossentropy_from_logits`
- **Impact**: Negligible; both prevent log(0) in BCE

### 3. **Position Indices**
- **JAX**: Inferred from predictions via `approx_max_k()`
- **PyTorch**: Pre-computed from ground truth + stored in dataset
- **Impact**: PyTorch can use true positives for loss; JAX must infer

### 4. **Multi-organism Support**
- **JAX**: Flexible via metadata dictionary
- **PyTorch**: Fixed mask per organism in junction head
- **Current PyTorch**: Hard-coded `num_organisms=1` in finetuning

---

## 8. COMPATIBILITY ASSESSMENT

### Strengths of PyTorch Approach
✓ Pre-computed junction matrix allows true-pair loss supervision  
✓ Batch dimension handled explicitly  
✓ Clear separation of classification/usage/junction losses  
✓ Soft-clipping matches JAX exactly  

### Potential Issues
⚠ Usage targets not clipped before BCE (minor—BCE handles it)  
⚠ Hard-coded `num_organisms=1` limits multi-organism fine-tuning  
⚠ RoPE parameters initialized to zero (may need warm-start)  

### Alignment with JAX
- Core loss computations: **✓ Aligned**
- Shape handling: **✓ Aligned** (after your fix)
- Masking strategy: **✓ Aligned**
- Soft-clipping: **✓ Aligned**
- Position encoding: **✓ Aligned** (RoPE parameters)

---

## Summary

The PyTorch implementation faithfully reproduces the JAX splice heads with one key improvement: **pre-computed junction matrices** enable more direct supervision of donor-acceptor pairs. The bug you fixed (target slicing dimension) was a shape-handling issue specific to batching—JAX avoids it by assuming single samples. Both codebases share the same loss weighting, soft-clipping strategy, and RoPE-based position encoding.

For your fine-tuning pipeline, the main implementation is sound. Consider:
1. ✓ Using junction losses to directly supervise pair predictions
2. Consider clipping usage targets before BCE (cosmetic, but matches JAX exactly)
3. Consider enabling multi-organism support for future flexibility
