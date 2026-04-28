# Plan: `src/scripts/benchmark_ssu_approximation.py`

## Purpose

Benchmark how well the junction-only SSU approximation (no BAM required) correlates with
the SpliSER-style full SSU computed from BAM data. The result informs whether the
approximation is acceptable as a training target for the AlphaGenome finetuning pipeline.

---

## Background: the two SSU definitions

**Full SSU** (SpliSER, Dent et al. 2021):
```
SSU(site) = α / (α + β1 + β2)
```
- **α**: split reads using this splice site — available from SJ.out.tab
- **β1**: reads spanning the site continuously without splicing — require BAM
- **β2**: reads using a competing splice site for the same partner — derivable from junction data

**Approximated SSU** (junction-only, plan `fix_ssu_definition.md`):
```
SSU_approx(D) = α(D) / Σ_{A: D→A} acceptor_total(A)
SSU_approx(A) = α(A) / Σ_{D: D→A} donor_total(D)
```
This equals `α / (α + β2)` exactly — β1 is omitted.

---

## Script location

`src/scripts/benchmark_ssu_approximation.py`

---

## CLI Interface

```
usage: benchmark_ssu_approximation.py
       --bam BAM
       --junctions JUNCTIONS
       --interval INTERVAL
       [--output-dir OUTPUT_DIR]
       [--min-unique-reads N]
       [--mapq N]

arguments:
  --bam               STAR-aligned, coordinate-sorted, indexed BAM file
  --junctions         STAR SJ.out.tab file
  --interval          "chr1:1000000-2000000"  OR  path to a BED file (0-based half-open)
                      If BED: all rows are processed and results concatenated.
  --output-dir        Directory for outputs (default: current directory)
  --min-unique-reads  Minimum n_uniquely_mapped_reads to retain a junction (default: 1)
  --mapq              Minimum MAPQ for β1 reads (default: 30)
```

Interval parsing:
```python
def parse_interval(s: str) -> list[tuple[str, int, int]]:
    """Return list of (chrom, start_0based, end_0based_exclusive).
    
    If s points to a .bed file, read all rows (tab-separated: chrom, start, end).
    Otherwise parse "chrom:start-end" as 1-based inclusive UCSC coords:
        start_0based = int(start) - 1
        end_0based   = int(end)
    """
```

---

## Outputs

### `ssu_comparison.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | str | Chromosome |
| `position` | int64 | 1-based exon coordinate of the splice site |
| `strand` | str | `+` or `-` |
| `role` | str | `donor` or `acceptor` |
| `alpha` | int64 | Split reads using this site (SJ.out.tab) |
| `beta1` | int64 | Reads spanning site continuously without splicing (BAM) |
| `beta2` | int64 | Competing reads (junction data) |
| `ssu_full` | float64 | α / (α + β1 + β2); NaN if denominator == 0 |
| `ssu_approx` | float64 | α / (α + β2); NaN if denominator == 0 |

### `ssu_scatterplot.pdf`

- 2×2 panel grid: rows = strand (`+` / `-`), cols = role (`donor` / `acceptor`)
- x-axis: `ssu_full`, y-axis: `ssu_approx`, both [0, 1]
- Diagonal y=x reference line (dashed grey)
- Points colored by `log10(alpha + 1)` (shared colorbar) to show how correlation varies with read depth
- Pearson r and Spearman r annotated per panel (computed on finite pairs only)
- Panel title: `strand=<s>  role=<r>`; N displayed in annotation box

---

## Algorithm

### Step 1: Load and filter junction data

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "alphagenome-pytorch" / "src"))
from alphagenome_pytorch.extensions.finetuning.star_junctions import read_star_junctions

junctions = read_star_junctions(args.junctions)
# read_star_junctions returns 1-based intron_start, intron_end

# Filter to interval (intron must be fully inside the window)
junctions = junctions[
    (junctions["chrom"] == chrom)
    & (junctions["intron_start"] > start_0)   # 1-based > 0-based start ≡ ≥ start+1
    & (junctions["intron_end"]   <= end_0)
    & (junctions["n_uniquely_mapped_reads"] >= args.min_unique_reads)
    & (junctions["strand"].isin(["+", "-"]))
].copy()

# Derive 1-based exon coordinates (matching dataset convention)
junctions["exon_start"] = junctions["intron_start"] - 1   # donor: last exon base
junctions["exon_end"]   = junctions["intron_end"]   + 1   # acceptor: first exon base
junctions["count"]      = junctions["n_uniquely_mapped_reads"]
```

### Step 2: Compute α per splice site

```python
donor_alpha = (
    junctions.groupby(["chrom", "exon_start", "strand"])["count"]
    .sum().to_dict()
)  # key: (chrom, 1based_pos, strand) → total reads from this donor

acceptor_alpha = (
    junctions.groupby(["chrom", "exon_end", "strand"])["count"]
    .sum().to_dict()
)
```

### Step 3: Compute β2 per splice site (junction-only)

β2(D) = total reads at all acceptors D connects to, minus D's own reads:
```
β2(D) = Σ_{A: D→A} acceptor_total(A) - α(D)
```

```python
# Site-level totals (same as α at each site)
acceptor_total = acceptor_alpha  # dict: (chrom, exon_end, strand) → total
donor_total    = donor_alpha

# β2 per donor
donor_beta2 = {}
for (chrom_j, exon_start, strand), grp in junctions.groupby(
        ["chrom", "exon_start", "strand"]):
    denom = sum(
        acceptor_total.get((chrom_j, row["exon_end"], strand), 0)
        for _, row in grp.iterrows()
    )
    alpha_d = donor_alpha.get((chrom_j, exon_start, strand), 0)
    donor_beta2[(chrom_j, exon_start, strand)] = denom - alpha_d

# β2 per acceptor
acceptor_beta2 = {}
for (chrom_j, exon_end, strand), grp in junctions.groupby(
        ["chrom", "exon_end", "strand"]):
    denom = sum(
        donor_total.get((chrom_j, row["exon_start"], strand), 0)
        for _, row in grp.iterrows()
    )
    alpha_a = acceptor_alpha.get((chrom_j, exon_end, strand), 0)
    acceptor_beta2[(chrom_j, exon_end, strand)] = denom - alpha_a
```

### Step 4: Compute β1 from BAM (batch approach)

Fetch all reads in the interval once, then for each read determine which splice site
positions it spans continuously. This avoids repeated BAM fetches per site.

```python
import pysam

def build_beta1_counts(
    bam_path: str,
    chrom: str,
    start_0: int,
    end_0: int,
    site_positions_0based: set[int],
    mapq_min: int = 30,
) -> dict[int, int]:
    """Return {0-based position → β1 count} for all requested positions.

    A read contributes β1 to a position if:
      - it overlaps the position
      - MAPQ >= mapq_min
      - it is not a PCR/optical duplicate
      - no N CIGAR operation covers the position (i.e., the read is continuous there)
    """
    beta1 = {pos: 0 for pos in site_positions_0based}
    sites_sorted = sorted(site_positions_0based)  # for binary search

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        for read in bam.fetch(chrom, start_0, end_0):
            if read.is_unmapped or read.is_duplicate:
                continue
            if read.mapping_quality < mapq_min:
                continue
            if not read.cigartuples:
                continue

            # Collect intron intervals from N CIGAR ops
            introns = []
            ref_pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:   # N = intron/splice
                    introns.append((ref_pos, ref_pos + length))
                    ref_pos += length
                elif op in (0, 2, 7, 8):  # M, D, =, X consume reference
                    ref_pos += length

            read_start = read.reference_start
            read_end   = read.reference_end  # exclusive

            # Find splice site positions this read overlaps (binary search)
            import bisect
            lo = bisect.bisect_left(sites_sorted, read_start)
            hi = bisect.bisect_right(sites_sorted, read_end - 1)
            overlapping = sites_sorted[lo:hi]

            for site_pos in overlapping:
                # β1 only if NO intron covers this position
                if not any(iv_s <= site_pos < iv_e for iv_s, iv_e in introns):
                    beta1[site_pos] += 1
    finally:
        bam.close()

    return beta1
```

**Coordinate note**: SJ.out.tab exon positions are 1-based; convert before calling:
```python
sites_0based = set()
for chrom_j, pos, strand in donor_alpha:
    if chrom_j == chrom:
        sites_0based.add(pos - 1)
for chrom_j, pos, strand in acceptor_alpha:
    if chrom_j == chrom:
        sites_0based.add(pos - 1)
```

### Step 5: Assemble site table

```python
rows = []

for (chrom_j, exon_start, strand), alpha in donor_alpha.items():
    pos_0 = exon_start - 1
    b1    = beta1_counts.get(pos_0, 0)
    b2    = donor_beta2.get((chrom_j, exon_start, strand), 0)
    d_full   = alpha + b1 + b2
    d_approx = alpha + b2
    rows.append({
        "chrom": chrom_j, "position": exon_start, "strand": strand, "role": "donor",
        "alpha": int(alpha), "beta1": int(b1), "beta2": int(b2),
        "ssu_full":   alpha / d_full   if d_full   > 0 else float("nan"),
        "ssu_approx": alpha / d_approx if d_approx > 0 else float("nan"),
    })

for (chrom_j, exon_end, strand), alpha in acceptor_alpha.items():
    pos_0 = exon_end - 1
    b1    = beta1_counts.get(pos_0, 0)
    b2    = acceptor_beta2.get((chrom_j, exon_end, strand), 0)
    d_full   = alpha + b1 + b2
    d_approx = alpha + b2
    rows.append({
        "chrom": chrom_j, "position": exon_end, "strand": strand, "role": "acceptor",
        "alpha": int(alpha), "beta1": int(b1), "beta2": int(b2),
        "ssu_full":   alpha / d_full   if d_full   > 0 else float("nan"),
        "ssu_approx": alpha / d_approx if d_approx > 0 else float("nan"),
    })

df = pd.DataFrame(rows).drop_duplicates(subset=["chrom", "position", "strand", "role"])
```

### Step 6: Write parquet

```python
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_dir / "ssu_comparison.parquet", index=False)
print(f"Wrote {len(df)} splice sites → {out_dir}/ssu_comparison.parquet")
```

### Step 7: Generate scatterplot

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

strands = ["+", "-"]
roles   = ["donor", "acceptor"]

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
fig.suptitle("SSU: junction-only approximation vs BAM ground truth", fontsize=13)

vmax = np.log10(df["alpha"].max() + 1) if not df.empty else 1
sc_ref = None

for row_i, strand in enumerate(strands):
    for col_i, role in enumerate(roles):
        ax = axes[row_i][col_i]
        sub = df[
            (df["strand"] == strand) & (df["role"] == role)
        ].dropna(subset=["ssu_full", "ssu_approx"])

        color_vals = np.log10(sub["alpha"].values + 1)
        sc = ax.scatter(
            sub["ssu_full"], sub["ssu_approx"],
            c=color_vals, vmin=0, vmax=vmax,
            cmap="viridis", s=12, alpha=0.6, linewidths=0,
        )
        sc_ref = sc  # for colorbar

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, alpha=0.5)

        n = len(sub)
        if n >= 3:
            r_p, _ = pearsonr(sub["ssu_full"], sub["ssu_approx"])
            r_s, _ = spearmanr(sub["ssu_full"], sub["ssu_approx"])
            ax.text(
                0.05, 0.95,
                f"Pearson r = {r_p:.3f}\nSpearman r = {r_s:.3f}\nN = {n}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
            )

        ax.set_title(f"strand={strand}  role={role}", fontsize=10)
        ax.set_xlabel("SSU full (BAM)")
        ax.set_ylabel("SSU approx (junction-only)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

if sc_ref is not None:
    fig.colorbar(sc_ref, ax=axes, label="log10(α + 1)", shrink=0.55, pad=0.02)

fig.savefig(out_dir / "ssu_scatterplot.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Wrote figure → {out_dir}/ssu_scatterplot.pdf")
```

---

## Internal function structure

| Function | Inputs | Returns |
|----------|--------|---------|
| `parse_args()` | — | `argparse.Namespace` |
| `parse_interval(s)` | str | `list[(chrom, start_0, end_0)]` |
| `load_and_filter_junctions(path, chrom, start_0, end_0, min_reads)` | — | `pd.DataFrame` |
| `compute_alpha_beta2(junctions)` | DataFrame | `(donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2)` dicts |
| `build_beta1_counts(bam_path, chrom, start_0, end_0, sites_0, mapq)` | — | `dict[int, int]` |
| `assemble_site_table(donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2, beta1_counts)` | — | `pd.DataFrame` |
| `plot_scatterplot(df, out_path)` | — | writes PDF |
| `main()` | — | orchestrates all steps |

---

## Key gotchas

### Coordinate system
- `read_star_junctions` returns **1-based** `intron_start` / `intron_end`
- Exon positions throughout this script are **1-based** (`exon_start = intron_start - 1`, `exon_end = intron_end + 1`)
- `pysam.fetch` and CIGAR traversal use **0-based** coordinates — convert with `pos_0 = exon_pos_1based - 1`

### β2 = 0 sites
When a donor connects only to one acceptor and no other donor shares that acceptor,
`β2 = 0` and `ssu_approx = 1.0` always regardless of β1. These sites will cluster at
`ssu_approx = 1.0` in the plot and inflate the apparent correlation. They can be
identified by `beta2 == 0` in the parquet; consider annotating them separately.

### Strand-agnostic β1
β1 is counted without strand filtering (no XS tag check). This slightly over-counts β1
for sites on the non-dominant strand. Acceptable for a first benchmark; add `--stranded`
flag as a future extension.

### Performance
Batch BAM fetch is O(reads_in_interval × sites_per_read_footprint). For a 1 Mb window
with ~50 k reads and ~1 k sites, the inner loop is manageable in pure Python. Use
`bisect` to narrow per-read site candidates (already included in the plan). If slow,
vectorize using `numpy` interval intersection.

### Multi-interval BED input
Process each BED row independently, append DataFrames, then call
`drop_duplicates(subset=["chrom", "position", "strand", "role"])` before writing parquet
and plotting. Keeps memory bounded when processing many intervals.

---

## Dependencies

All available in the existing `alphagenome_pytorch` conda environment:

```
pysam        # BAM traversal
scipy        # pearsonr, spearmanr
matplotlib   # plotting
pandas
numpy
pyarrow      # parquet I/O
```
