#!/usr/bin/env python3
"""Compare three junction loss formulations:

  1. JAX      — cross_entropy_loss (can go negative due to log_normalizer bug)
                + poisson_loss with minimum subtraction (always >= 0)
  2. Paper    — multinomial_cross_entropy (always >= 0)
                + poisson_loss WITHOUT minimum subtraction (can go negative for good fits)
  3. Nonneg   — paper's multinomial_cross_entropy (always >= 0)
                + JAX's poisson_loss with minimum subtraction (always >= 0)
                → total always >= 0, interpretable, safe for early stopping
"""

import numpy as np

EPS = 1e-7
SOFT_CLIP = 10.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def soft_clip(x):
    return np.where(x > SOFT_CLIP, 2.0 * np.sqrt(x * SOFT_CLIP) - SOFT_CLIP, x)


def safe_masked_mean(x, mask):
    mask = np.broadcast_to(mask, x.shape).astype(float)
    return (x * mask).sum() / max(mask.sum(), 1.0)


# ---------------------------------------------------------------------------
# JAX formulation
# ---------------------------------------------------------------------------

def jax_cross_entropy_loss(y_true, y_pred, mask, axis):
    """Matches losses.cross_entropy_loss in alphagenome_research."""
    mask = np.broadcast_to(mask, y_true.shape)
    y_true = np.where(mask, y_true.astype(float), 0.0)
    p_true = y_true / np.maximum(y_true.sum(axis=axis, keepdims=True), EPS)

    masked_pred = np.where(mask, y_pred, 0.0)
    log_normalizer = np.log((masked_pred + EPS).sum(axis=axis))
    log_likelihood = (p_true * np.log(y_pred + EPS)).sum(axis=axis)

    log_loss = log_normalizer - log_likelihood
    return safe_masked_mean(log_loss, mask.any(axis=axis))


def jax_poisson_loss(y_true, y_pred, mask):
    """Matches losses.poisson_loss in alphagenome_research (subtracts minimum)."""
    y_true = np.abs(y_true).astype(float)
    y_pred = y_pred.astype(float)
    log_pred = np.log(y_pred + EPS)
    min_value = y_true - y_true * np.log(y_true + EPS)
    loss = (y_pred - y_true * log_pred) - min_value
    return safe_masked_mean(loss, mask)


def jax_junction_loss(pred, target, mask):
    """Full JAX SpliceSitesJunctionHead.loss (D, A, T arrays, single strand)."""
    # CE ratio terms
    donor_ce    = jax_cross_entropy_loss(target, pred, mask, axis=0)
    acceptor_ce = jax_cross_entropy_loss(target, pred, mask, axis=1)

    # Poisson total terms (sum over valid pairs, then poisson)
    sum_pred_a = (pred * mask).sum(axis=1)    # sum over acceptors → (D, T)
    sum_tgt_a  = soft_clip((target * mask).sum(axis=1))
    sum_pred_d = (pred * mask).sum(axis=0)    # sum over donors    → (A, T)
    sum_tgt_d  = soft_clip((target * mask).sum(axis=0))

    donor_poisson   = jax_poisson_loss(sum_tgt_a, sum_pred_a, mask.any(axis=1))
    acceptor_poisson = jax_poisson_loss(sum_tgt_d, sum_pred_d, mask.any(axis=0))

    # JAX weights: 1.0 * CE + 0.2 * Poisson
    return (donor_ce + acceptor_ce) + 0.2 * (donor_poisson + acceptor_poisson)


# ---------------------------------------------------------------------------
# Paper pseudocode formulation
# ---------------------------------------------------------------------------

def paper_mce(x, targets, axis):
    """multinomial_cross_entropy from paper pseudocode."""
    pred_ratios = (x + EPS) / (x + EPS).sum(axis=axis, keepdims=True)
    return -(targets * np.log(pred_ratios)).sum()


def paper_poisson_loss(x, targets, axis):
    """poisson_loss from paper pseudocode (no minimum subtraction)."""
    sum_pred = x.sum(axis=axis)
    sum_tgt  = soft_clip(targets.sum(axis=axis))
    return (sum_pred - sum_tgt * np.log(sum_pred + EPS)).sum()


def paper_junction_loss(pred, target):
    """junctions_loss from paper pseudocode (no mask needed; padding = 0)."""
    ratios_loss = paper_mce(pred, target, 0) + paper_mce(pred, target, 1)
    counts_loss = paper_poisson_loss(pred, target, 0) + paper_poisson_loss(pred, target, 1)
    # Paper weights: 0.2 * CE + 0.04 * Poisson
    return 0.2 * ratios_loss + 0.04 * counts_loss


# ---------------------------------------------------------------------------
# Nonneg formulation: paper CE + JAX Poisson
# ---------------------------------------------------------------------------

def nonneg_junction_loss(pred, target, mask):
    """paper multinomial_cross_entropy + JAX poisson_loss (both always >= 0)."""
    pred_m   = pred * mask
    target_m = target * mask

    # CE: paper's multinomial_cross_entropy (always >= 0)
    donor_ce    = paper_mce(pred_m, target_m, axis=0)
    acceptor_ce = paper_mce(pred_m, target_m, axis=1)

    # Poisson: JAX's formulation with minimum subtraction (always >= 0)
    sum_pred_a = pred_m.sum(axis=1)
    sum_tgt_a  = soft_clip(target_m.sum(axis=1))
    sum_pred_d = pred_m.sum(axis=0)
    sum_tgt_d  = soft_clip(target_m.sum(axis=0))

    donor_poisson    = jax_poisson_loss(sum_tgt_a, sum_pred_a, mask.any(axis=1))
    acceptor_poisson = jax_poisson_loss(sum_tgt_d, sum_pred_d, mask.any(axis=0))

    # Same CE:Poisson ratio as JAX (5:1), matching paper scale
    return 0.2 * (donor_ce + acceptor_ce) + 0.04 * (donor_poisson + acceptor_poisson)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_case(name, pred, target, mask=None):
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)

    jax_loss   = jax_junction_loss(pred, target, mask)
    paper_loss = paper_junction_loss(pred * mask, target * mask)
    nonneg_loss  = nonneg_junction_loss(pred, target, mask)

    print(f"\n{'='*60}")
    print(f"Case: {name}")
    print(f"  pred shape: {pred.shape}")
    print(f"  target sum: {target.sum():.4f}  pred sum: {pred.sum():.4f}")
    print(f"  JAX loss  : {jax_loss:+.4f}")
    print(f"  Paper loss: {paper_loss:+.4f}")
    print(f"  Nonneg loss: {nonneg_loss:+.4f}")
    for label, val in [("JAX", jax_loss), ("Paper", paper_loss), ("Nonneg", nonneg_loss)]:
        if val < 0:
            print(f"  *** {label} loss is NEGATIVE ***")


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    D, A, T = 8, 8, 2   # small: donors, acceptors, tissues

    # ------------------------------------------------------------------
    # Case 1: all targets zero, valid mask, small predictions (early training)
    # ------------------------------------------------------------------
    pred   = np.full((D, A, T), 0.01)   # small softplus outputs
    target = np.zeros((D, A, T))
    mask   = np.ones((D, A, T), dtype=bool)
    run_case("All targets=0, valid mask, small pred (early training)", pred, target, mask)

    # ------------------------------------------------------------------
    # Case 2: all targets zero, valid mask, larger predictions
    # ------------------------------------------------------------------
    pred   = np.full((D, A, T), 1.0)
    target = np.zeros((D, A, T))
    mask   = np.ones((D, A, T), dtype=bool)
    run_case("All targets=0, valid mask, pred=1.0", pred, target, mask)

    # ------------------------------------------------------------------
    # Case 3: sparse targets (a few junctions), small predictions
    # ------------------------------------------------------------------
    pred   = np.full((D, A, T), 0.01)
    target = np.zeros((D, A, T))
    target[0, 1, 0] = 5.0
    target[2, 3, 1] = 3.0
    mask   = np.ones((D, A, T), dtype=bool)
    run_case("Sparse targets, small pred", pred, target, mask)

    # ------------------------------------------------------------------
    # Case 4: well-fitted large counts (paper Poisson can go negative here)
    # ------------------------------------------------------------------
    pred   = np.zeros((D, A, T))
    target = np.zeros((D, A, T))
    pred[0, 1, 0]   = 20.0
    target[0, 1, 0] = 20.0
    pred[2, 3, 1]   = 15.0
    target[2, 3, 1] = 15.0
    mask   = np.ones((D, A, T), dtype=bool)
    run_case("Well-fitted large counts (good model)", pred, target, mask)

    # ------------------------------------------------------------------
    # Case 5: large D (256 donors, as in real padding), all targets zero
    # ------------------------------------------------------------------
    D2, A2 = 256, 256
    pred   = np.full((D2, A2, T), 0.001)
    target = np.zeros((D2, A2, T))
    mask   = np.ones((D2, A2, T), dtype=bool)
    run_case("D=A=256 (real padding size), all targets=0, small pred", pred, target, mask)

    # ------------------------------------------------------------------
    # Case 6: large D, only a few valid pairs (realistic mask)
    # ------------------------------------------------------------------
    mask2            = np.zeros((D2, A2, T), dtype=bool)
    mask2[:10, :10, :] = True          # only 10 donors × 10 acceptors valid
    pred   = np.full((D2, A2, T), 0.001)
    target = np.zeros((D2, A2, T))
    run_case("D=A=256, sparse mask (10×10 valid), all targets=0", pred, target, mask2)

    print()
