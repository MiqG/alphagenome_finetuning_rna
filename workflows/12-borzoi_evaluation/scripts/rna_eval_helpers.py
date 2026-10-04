"""
RNA-seq evaluation helpers copied verbatim from src/scripts/collect_predictions.py.

collect_predictions.py imports torch/alphagenome_pytorch at module level, which
must not be loaded into the TensorFlow process that runs Borzoi inference, so
the numpy/pandas-only helpers it uses for the RNA-seq parquets are duplicated
here. Do not edit: check_helpers_sync.py (run in the alphagenome_pytorch env)
fails if any of these drift from the originals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyranges as pr

_EPS = 1e-8


def merge_gene_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """Union-merge overlapping exon intervals within each gene.

    Different transcripts of a gene often have overlapping-but-not-identical exon
    spans (alternative splice sites, retained introns, alternative UTRs). Without
    this, summing over raw per-transcript exon rows double-counts bases covered by
    more than one transcript's exon, which biases gene-level means toward
    constitutive exons and deviates from AlphaGenome's own union-of-exons gene mask.
    """
    gr = pr.PyRanges(exons[["Chromosome", "Start", "End", "Strand", "gene_id"]])
    merged = gr.merge(strand=True, by="gene_id").df
    gene_name = exons.drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
    merged["gene_name"] = merged["gene_id"].map(gene_name)
    return merged


def compute_gene_interval_map(
    exons: pd.DataFrame,
    test_intervals: pd.DataFrame,
) -> dict[str, tuple[int, str, str]]:
    """Return {gene_id: (iv_idx, strand, gene_name)} for genes with ≥50% exons in one test interval.

    Exons are 0-based half-open [Start, End). Only first qualifying interval per gene kept.
    """
    gene_total_bp = (
        exons.groupby("gene_id")
        .apply(lambda g: int((g["End"] - g["Start"]).sum()))
        .to_dict()
    )
    gene_strand = exons.drop_duplicates("gene_id").set_index("gene_id")["Strand"].to_dict()
    gene_name = exons.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].to_dict()

    exons_by_chrom: dict[str, pd.DataFrame] = {
        chrom: df for chrom, df in exons.groupby("Chromosome")
    }

    gene_interval_map: dict[str, tuple[int, str, str]] = {}

    for iv_idx, row in test_intervals.iterrows():
        chrom, iv_start, iv_end = row["chrom"], int(row["start"]), int(row["end"])
        chrom_exons = exons_by_chrom.get(chrom)
        if chrom_exons is None:
            continue

        overlapping = chrom_exons[
            (chrom_exons["Start"] < iv_end) & (chrom_exons["End"] > iv_start)
        ].copy()
        if overlapping.empty:
            continue

        overlapping["ov_bp"] = (
            overlapping["End"].clip(upper=iv_end) - overlapping["Start"].clip(lower=iv_start)
        ).clip(lower=0)

        by_gene = overlapping.groupby("gene_id")["ov_bp"].sum()
        for gid, ov_bp in by_gene.items():
            total = gene_total_bp.get(gid, 0)
            if total > 0 and ov_bp / total >= 0.5 and gid not in gene_interval_map:
                gene_interval_map[gid] = (iv_idx, gene_strand.get(gid, "+"), gene_name.get(gid, gid))

    return gene_interval_map


def get_exon_mean_pred(
    rna_pred: np.ndarray,  # (seq_len, n_tracks)
    exon_rows: pd.DataFrame,  # exons for this gene (0-based half-open)
    window_start: int,
    seq_len: int,
) -> np.ndarray | None:
    """Mean predicted rna_seq coverage over all exon positions, per track."""
    sums = np.zeros(rna_pred.shape[1], dtype=np.float64)
    count = 0
    for _, ex in exon_rows.iterrows():
        rel_s = max(0, int(ex["Start"]) - window_start)
        rel_e = min(seq_len, int(ex["End"]) - window_start)
        if rel_e <= rel_s:
            continue
        sums += rna_pred[rel_s:rel_e].sum(axis=0)
        count += rel_e - rel_s
    if count == 0:
        return None
    return sums / count


def get_exon_mean_pred_binned(
    rna_pred_binned: np.ndarray,  # (n_bins, n_tracks)
    exon_rows: pd.DataFrame,
    window_start: int,
    seq_len: int,
    bin_size: int = 32,
) -> np.ndarray | None:
    """Mean predicted rna_seq coverage over exon bins, per track.

    Exon coordinates are floor-divided into bin_size-bp bins before averaging,
    matching Borzoi's 32 bp output resolution.
    """
    n_bins = rna_pred_binned.shape[0]
    sums = np.zeros(rna_pred_binned.shape[1], dtype=np.float64)
    count = 0
    for _, ex in exon_rows.iterrows():
        rel_s = max(0, int(ex["Start"]) - window_start)
        rel_e = min(seq_len, int(ex["End"]) - window_start)
        if rel_e <= rel_s:
            continue
        bin_s = rel_s // bin_size
        bin_e = min(((rel_e - 1) // bin_size) + 1, n_bins)
        if bin_e <= bin_s:
            continue
        sums += rna_pred_binned[bin_s:bin_e].sum(axis=0)
        count += bin_e - bin_s
    if count == 0:
        return None
    return sums / count


def get_exon_mean_obs(
    bigwigs: list,
    exon_rows: pd.DataFrame,
    chrom: str,
    chrom_sizes: dict[str, int],
    window_start: int,
    seq_len: int,
) -> np.ndarray | None:
    """Mean observed bigwig coverage over all exon positions, per track.

    Clipped to the same [window_start, window_start + seq_len) model window as
    get_exon_mean_pred, so genes only partially covered by the window (allowed by
    compute_gene_interval_map's >=50% criterion) are compared over the same set of
    bases on both sides rather than obs pulling in out-of-window bp that pred never sees.
    """
    n_tracks = len(bigwigs)
    sums = np.zeros(n_tracks, dtype=np.float64)
    count = 0
    window_end = window_start + seq_len
    for _, ex in exon_rows.iterrows():
        ex_start, ex_end = int(ex["Start"]), int(ex["End"])
        chrom_len = chrom_sizes.get(chrom, int(1e10))
        ex_start = max(ex_start, window_start)
        ex_end = min(ex_end, chrom_len, window_end)
        if ex_end <= ex_start:
            continue
        for t, bw in enumerate(bigwigs):
            vals = bw.values(chrom, ex_start, ex_end)
            if vals is None:
                continue
            arr = np.array(vals, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0)
            sums[t] += arr.sum()
        count += ex_end - ex_start
    if count == 0:
        return None
    return sums / count


class ProfileCorrAccumulator:
    """Online Pearson r accumulator over a stream of (pred, obs) position batches.

    Maintains sufficient statistics in O(n_tracks) space regardless of how many
    positions are seen. Call update() with matching arrays of shape (m, n_tracks),
    then result() once all intervals are processed.
    """

    def __init__(self, n_tracks: int) -> None:
        z = np.zeros(n_tracks, dtype=np.float64)
        self.n      = np.zeros(n_tracks, dtype=np.int64)
        self.sum_x  = z.copy()
        self.sum_y  = z.copy()
        self.sum_xx = z.copy()
        self.sum_yy = z.copy()
        self.sum_xy = z.copy()

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """x, y: (m, n_tracks) float64."""
        self.n      += x.shape[0]
        self.sum_x  += x.sum(axis=0)
        self.sum_y  += y.sum(axis=0)
        self.sum_xx += (x ** 2).sum(axis=0)
        self.sum_yy += (y ** 2).sum(axis=0)
        self.sum_xy += (x * y).sum(axis=0)

    def result(self) -> np.ndarray:
        """Return per-track Pearson r, shape (n_tracks,). NaN where undefined."""
        n = self.n.astype(np.float64)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den = np.sqrt(
            np.maximum(0.0, n * self.sum_xx - self.sum_x ** 2)
            * np.maximum(0.0, n * self.sum_yy - self.sum_y ** 2)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def pearson_r_per_track(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-track Pearson r for a single batch. x, y: (n_positions, n_tracks) float64.

    Returns shape (n_tracks,), NaN where undefined (zero variance or no positions).
    """
    if x.shape[0] == 0:
        return np.full(x.shape[1], np.nan)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    num = (xc * yc).sum(axis=0)
    den = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pearson_r_pooled(x: np.ndarray, y: np.ndarray) -> float:
    """Scalar Pearson r over two arrays of matching shape, flattened together
    (pools all axes, e.g. positions and tracks, into one correlation)."""
    xf = x.reshape(-1).astype(np.float64)
    yf = y.reshape(-1).astype(np.float64)
    if xf.size == 0:
        return float("nan")
    xc = xf - xf.mean()
    yc = yf - yf.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if den <= 0:
        return float("nan")
    return float((xc * yc).sum() / den)


def jsd_per_track(pred: np.ndarray, obs: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Per-track Jensen-Shannon divergence between position-normalized profiles.

    pred, obs: (n_positions, n_tracks) raw (non-negative) signal. Each track's
    profile is normalized into a probability distribution over positions.
    """
    p = obs / (obs.sum(axis=0, keepdims=True) + eps)
    q = pred / (pred.sum(axis=0, keepdims=True) + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)), axis=0)
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)), axis=0)
    return 0.5 * (kl_pm + kl_qm)


def fetch_obs_window(
    bigwigs: list,
    chrom: str,
    window_start: int,
    window_end: int,
    chrom_sizes: dict,
) -> np.ndarray:
    """Return (window_end - window_start, n_tracks) float64 coverage.

    Positions beyond chromosome end are filled with 0.
    """
    seq_len = window_end - window_start
    out = np.zeros((seq_len, len(bigwigs)), dtype=np.float64)
    chrom_len = chrom_sizes.get(chrom, int(1e10))
    safe_end = min(window_end, chrom_len)
    n_valid = safe_end - window_start
    if n_valid <= 0:
        return out
    for t, bw in enumerate(bigwigs):
        vals = bw.values(chrom, window_start, safe_end)
        if vals is None:
            continue
        arr = np.array(vals, dtype=np.float64)
        np.nan_to_num(arr, copy=False, nan=0.0)
        out[:n_valid, t] = arr
    return out


def accumulator_to_df(
    acc: ProfileCorrAccumulator,
    samples: list,
) -> pd.DataFrame:
    r = acc.result()
    rows = []
    for t_idx, r_val in enumerate(r):
        sample_idx = t_idx // 2
        strand = "forward" if t_idx % 2 == 0 else "reverse"
        rows.append({
            "track_idx":   t_idx,
            "track_name":  samples[sample_idx],
            "strand":      strand,
            "pearson_r":   float(r_val),
            "n_positions": int(acc.n[t_idx]),
        })
    return pd.DataFrame(rows)
