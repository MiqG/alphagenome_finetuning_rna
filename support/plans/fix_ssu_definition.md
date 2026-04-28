# Plan: Fix Splice Site Usage (SSU) Definition

## Problem

`junctions_to_usage_arrays_by_strand` in `star_junctions.py` currently computes:

```python
frac = junc["count"] / pos_total   # pos_total = all junction reads on this strand in window
pos_arr[donor_idx] += frac
pos_arr[accept_idx] += frac
```

This divides by the total junction reads in the entire window, which is not a meaningful
per-site metric. It mixes all donors and acceptors into one denominator and does not reflect
competition between sites sharing a partner.

---

## Correct Formula

AlphaGenome cites SpliSER (Dent et al. 2021) for the SSU definition:

```
SSE = α / (α + β1 + β2)
```

- **α-reads**: split reads using this splice site — **in SJ.out.tab**
- **β1-reads**: reads spanning the site continuously (unspliced) — **require BAM, unavailable**
- **β2-reads**: reads using a competing splice site for the same partner — **derivable from junction data**

Since β1 is unavailable from STAR output, we implement:

```
SSU(D) = α(D) / [α(D) + β2(D)]
       = α(D) / Σ_{A: D→A} acceptor_total(A)

SSU(A) = α(A) / [α(A) + β2(A)]
       = α(A) / Σ_{D: D→A} donor_total(D)
```

Where:
- `α(D)` = sum of counts over all junctions originating from donor D
- `acceptor_total(A)` = sum of counts over all junctions arriving at acceptor A (from any donor)
- `β2(D)` = `Σ_A acceptor_total(A) - α(D)` = reads from other donors competing for D's acceptors

The denominator `Σ_A acceptor_total(A)` equals `α(D) + β2(D)` exactly, giving SSU ∈ (0, 1].

---

## Worked Example

Three junctions on `+` strand, window `[0, 100)`:

| junction | exon_start (1-based) | exon_end (1-based) | count |
|---|---|---|---|
| D1→A1 | 10 | 50 | 30 |
| D2→A1 | 20 | 50 | 70 |
| D1→A2 | 10 | 80 | 20 |

Site totals:
- `donor_total(D1)` = 30 + 20 = 50
- `donor_total(D2)` = 70
- `acceptor_total(A1)` = 30 + 70 = 100
- `acceptor_total(A2)` = 20

Expected SSU values:
- `SSU(D1)` = 50 / (100 + 20) = **0.417**
- `SSU(D2)` = 70 / 100 = **0.700**
- `SSU(A1)` = 100 / (50 + 70) = **0.833**
- `SSU(A2)` = 20 / 50 = **0.400**

All values are in (0, 1]. This is the ground truth for the unit test.

---

## Implementation

### Files to modify

1. **`src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/star_junctions.py`**
   — Replace `junctions_to_usage_arrays_by_strand` (lines 688–753)
   — Remove dead `return arr` at line 753

2. **`src/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/datasets.py`**
   — No changes needed; call site at line 1167 uses unchanged signature

### New function

Replace the entire body of `junctions_to_usage_arrays_by_strand` with:

```python
def junctions_to_usage_arrays_by_strand(
    junc_df: pd.DataFrame,
    chrom: str,
    start: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position Splice Site Usage (SSU) arrays per strand.

    Implements the SpliSER approximation (Dent et al. 2021) without β1 (unspliced reads,
    which require BAM and are absent from SJ.out.tab):

        SSU(D) = α(D) / Σ_{A: D→A} acceptor_total(A)
        SSU(A) = α(A) / Σ_{D: D→A} donor_total(D)

    where α(D) = total junction reads originating from donor D,
          acceptor_total(A) = total junction reads arriving at acceptor A.

    This equals the SpliSER SSE formula α/(α+β2) exactly, bounding all values to (0, 1].

    Args:
        junc_df: Junction DataFrame with columns: chrom, exon_start (1-based),
            exon_end (1-based), strand, count.
        chrom: Chromosome name of the window.
        start: 0-based genomic start of the window.
        seq_len: Length of the window in base pairs.

    Returns:
        Tuple of two float32 arrays of shape (seq_len,):
            - pos_arr: SSU for positive strand
            - neg_arr: SSU for negative strand
    """
    pos_arr = np.zeros(seq_len, dtype=np.float32)
    neg_arr = np.zeros(seq_len, dtype=np.float32)
    end = start + seq_len

    mask = (
        (junc_df["chrom"] == chrom)
        & (junc_df["exon_start"] > start)
        & (junc_df["exon_start"] <= end)
        & (junc_df["exon_end"] > start)
        & (junc_df["exon_end"] <= end)
    )
    local = junc_df.loc[mask].copy()
    if local.empty:
        return pos_arr, neg_arr

    # Compute site-level totals and join back to junction rows
    acc_total = (
        local.groupby(["strand", "exon_end"])["count"]
        .sum().rename("acceptor_total").reset_index()
    )
    don_total = (
        local.groupby(["strand", "exon_start"])["count"]
        .sum().rename("donor_total").reset_index()
    )
    local = local.merge(acc_total, on=["strand", "exon_end"], how="left")
    local = local.merge(don_total, on=["strand", "exon_start"], how="left")

    # SSU per donor: α(D) / Σ_A acceptor_total(A)
    donor_alpha = local.groupby(["strand", "exon_start"])["count"].sum()
    donor_denom = local.groupby(["strand", "exon_start"])["acceptor_total"].sum()
    donor_ssu = (donor_alpha / donor_denom).dropna()

    # SSU per acceptor: α(A) / Σ_D donor_total(D)
    acceptor_alpha = local.groupby(["strand", "exon_end"])["count"].sum()
    acceptor_denom = local.groupby(["strand", "exon_end"])["donor_total"].sum()
    acceptor_ssu = (acceptor_alpha / acceptor_denom).dropna()

    # Scatter SSU values to position arrays
    strand_map = {"+": pos_arr, "-": neg_arr}

    for (strand, pos), val in donor_ssu.items():
        if strand not in strand_map:
            continue
        idx = int(pos) - 1 - start  # 1-based → 0-based relative
        if 0 <= idx < seq_len:
            strand_map[strand][idx] = float(val)

    for (strand, pos), val in acceptor_ssu.items():
        if strand not in strand_map:
            continue
        idx = int(pos) - 1 - start
        if 0 <= idx < seq_len:
            arr = strand_map[strand]
            # A position may be both a donor in one junction and an acceptor in another.
            # Take max to avoid double-counting a shared position.
            arr[idx] = max(arr[idx], float(val))

    return pos_arr, neg_arr
```

---

## Edge Cases

| Case | Behavior |
|---|---|
| Empty window | Return zero arrays immediately |
| Single junction, no competition | `acceptor_total = donor_total = count`; SSU = 1.0 at both sites |
| Junction partially outside window | Filtered by mask; excluded from all denominators |
| Donor connects to multiple acceptors | Denominator sums all `acceptor_total(A)` values; SSU stays ≤ 1 |
| Position is both donor and acceptor | `max()` at scatter step avoids double-counting |
| Strand not in `["+", "-"]` | Skipped by `strand_map` lookup |

---

## Unit Test

Add to `src/alphagenome-pytorch/tests/` (new file `test_ssu_definition.py` or extend existing):

```python
import numpy as np
import pandas as pd
from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    junctions_to_usage_arrays_by_strand,
)

def test_ssu_three_junctions():
    """SSU with two donors competing for the same acceptor."""
    junc_df = pd.DataFrame({
        "chrom":      ["chr1", "chr1", "chr1"],
        "exon_start": [10,     20,     10    ],  # 1-based
        "exon_end":   [50,     50,     80    ],  # 1-based
        "strand":     ["+",    "+",    "+"   ],
        "count":      [30.0,   70.0,   20.0  ],
    })

    pos_arr, neg_arr = junctions_to_usage_arrays_by_strand(
        junc_df, chrom="chr1", start=0, seq_len=100
    )

    assert np.allclose(neg_arr, 0.0)

    # D1 at 0-based index 9 (1-based pos 10)
    assert np.isclose(pos_arr[9],  50 / 120, atol=1e-6), f"SSU(D1)={pos_arr[9]}"
    # D2 at 0-based index 19
    assert np.isclose(pos_arr[19], 70 / 100, atol=1e-6), f"SSU(D2)={pos_arr[19]}"
    # A1 at 0-based index 49
    assert np.isclose(pos_arr[49], 100 / 120, atol=1e-6), f"SSU(A1)={pos_arr[49]}"
    # A2 at 0-based index 79
    assert np.isclose(pos_arr[79], 20 / 50, atol=1e-6), f"SSU(A2)={pos_arr[79]}"
```

---

## Limitations of the Approximation

The omission of β1 (unspliced reads) means SSU will be overestimated for:
- Constitutively spliced exons with few competing junctions (β2 ≈ 0, β1 > 0 → SSU would be 1.0 rather than true value < 1)
- Lowly expressed genes where most reads are unspliced

For the oracle benchmark this is not an issue since the pretrained model's own usage BigWigs are used directly. For real finetuning with SF3B1-mutant data, the overestimation is accepted as the cost of not requiring BAM files at training time.
