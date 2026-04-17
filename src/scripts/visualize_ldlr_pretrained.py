#!/usr/bin/env python
"""Visualize AlphaGenome pretrained predictions vs real K562 data for LDLR.

Plots predicted and real RNA-seq, splice donor/acceptor classification,
splice site usage, and splice junctions for chr19:11066619-11136619
(positive strand). Real data loaded using the same approach as the
AlphaGenome finetuning dataloader.

Usage:
    python src/scripts/visualize_ldlr_pretrained.py \
        --weights data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors \
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \
        --track-metadata data/raw/articles/Avsec2026/alphagenome_pytorch/track_metadata.parquet \
        --bigwig-fwd data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.forward.bw \
        --bigwig-rev data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.reverse.bw \
        --star-junctions data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.SJ.out.tab \
        --output results/examples/alphagenome_pytorch/ldlr_hepg2_pretrained.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# AlphaGenome imports
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

# Dataloader utilities — same logic used during finetuning
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "alphagenome-pytorch" / "src"))
from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    read_star_junctions,
    normalize_junctions_per_sample,
    junctions_to_classification_array,
    junctions_to_usage_arrays_by_strand,
    junctions_to_junction_matrix,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CHROM     = "chr19"
ROI_START = 11_066_619
ROI_END   = 11_136_619   # 70 kb window

SEQUENCE_LENGTH = 1_048_576   # AlphaGenome requires exactly 1 Mb


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def center_window(roi_start: int, roi_end: int, seq_len: int) -> tuple[int, int]:
    center = (roi_start + roi_end) // 2
    half   = seq_len // 2
    start  = center - half
    return start, start + seq_len


def load_sequence(chrom: str, start: int, end: int, fasta_path: str) -> torch.Tensor:
    try:
        import pyfaidx
    except ImportError:
        raise ImportError("pyfaidx is required: pip install pyfaidx")
    fasta   = pyfaidx.Fasta(fasta_path)
    seq_str = str(fasta[chrom][max(0, start):end]).upper()
    if start < 0:
        seq_str = "N" * (-start) + seq_str
    return torch.from_numpy(sequence_to_onehot(seq_str)).float().unsqueeze(0)  # (1, S, 4)


def load_bigwig_signal(bw_path: str, chrom: str, start: int, end: int) -> np.ndarray:
    """Load 1 bp-resolution signal from a BigWig, matching dataloader behaviour.

    NaN → 0.0 (identical to GenomicDataset._get_signal).
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required: pip install pyBigWig")
    bw  = pyBigWig.open(bw_path)
    sig = np.array(bw.stats(chrom, start, end, type="mean", nBins=end - start),
                   dtype=np.float32)
    bw.close()
    return np.nan_to_num(sig, nan=0.0)


def load_junctions(sj_path: str) -> pd.DataFrame:
    """Read and normalise a STAR SJ.out.tab file as the dataloader does."""
    junc = read_star_junctions(sj_path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc["count"]      = junc["n_uniquely_mapped_reads"].astype(float)
    junc["exon_start"] = junc["intron_start"] - 1   # 1-based donor exon end
    junc["exon_end"]   = junc["intron_end"]   + 1   # 1-based acceptor exon start
    junc = normalize_junctions_per_sample(junc)
    return junc


def find_hepg2_indices(
    metadata_path: str | None,
    output_type: str,
    strand: str,
) -> list[int]:
    """Return within-head track indices for HepG2 + strand from parquet metadata."""
    if metadata_path is None or not Path(metadata_path).exists():
        return []
    df      = pd.read_parquet(metadata_path)
    head_df = df[
        (df["output_type"] == output_type) & (df["organism"] == "human")
    ].reset_index(drop=True)
    mask = (
        (head_df["ontology_curie"] == "EFO:0001187")  # HepG2
        & (head_df["track_strand"] == strand)
    )
    return head_df.index[mask].tolist()


def draw_arcs(ax, junctions, cmap, color_label):
    """Draw arc plot. junctions: list of (donor_pos, acceptor_pos, count)."""
    if not junctions:
        ax.text(0.5, 0.5, "No junctions in ROI", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        return
    max_cnt = junctions[0][2]
    for d_pos, a_pos, cnt in junctions[:50]:
        left, right = min(d_pos, a_pos), max(d_pos, a_pos)
        if right == left:
            continue
        height = 0.2 + 0.8 * (cnt / max_cnt)
        ax.annotate(
            "",
            xy=(right, 0), xycoords="data",
            xytext=(left, 0), textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=cmap(0.4 + 0.6 * cnt / max_cnt),
                connectionstyle=f"arc3,rad=-{height:.2f}",
                lw=0.8, alpha=0.7,
            ),
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args):
    seq_start, seq_end = center_window(ROI_START, ROI_END, SEQUENCE_LENGTH)
    roi_s = ROI_START - seq_start
    roi_e = ROI_END   - seq_start
    roi_len   = ROI_END - ROI_START
    positions = np.arange(ROI_START, ROI_END)

    print(f"Region of interest : {CHROM}:{ROI_START:,}-{ROI_END:,} ({roi_len:,} bp)")
    print(f"Model input window : {CHROM}:{seq_start:,}-{seq_end:,} ({SEQUENCE_LENGTH:,} bp)")

    # ------------------------------------------------------------------ #
    # 1. Load sequence
    # ------------------------------------------------------------------ #
    print("\nLoading genome sequence …")
    seq_tensor = load_sequence(CHROM, seq_start, seq_end, args.genome)
    print(f"  Shape: {seq_tensor.shape}")

    # ------------------------------------------------------------------ #
    # 2. Load real K562 data (same logic as finetuning dataloader)
    # ------------------------------------------------------------------ #
    print("\nLoading real K562 data …")

    # BigWig: raw 1 bp signal, NaN→0 (matches GenomicDataset._get_signal)
    real_rna_fwd = load_bigwig_signal(args.bigwig_fwd, CHROM, ROI_START, ROI_END)
    real_rna_rev = load_bigwig_signal(args.bigwig_rev, CHROM, ROI_START, ROI_END)
    print(f"  BigWig fwd max: {real_rna_fwd.max():.3f}  rev max: {real_rna_rev.max():.3f}")

    # STAR junctions: read → filter (≥1 unique read) → add exon coords → CPM + clip + scale
    junc_df = load_junctions(args.star_junctions)
    print(f"  Junctions loaded: {len(junc_df)}")

    # Classification array — same 5-class encoding as the model head
    cls_arr = junctions_to_classification_array(
        [junc_df], CHROM, seq_start, SEQUENCE_LENGTH
    )  # (S, 5)
    real_donor_pos    = cls_arr[roi_s:roi_e, 0]   # binary: 1 where donor+
    real_acceptor_pos = cls_arr[roi_s:roi_e, 1]   # binary: 1 where acceptor+

    # Splice site usage by strand
    real_usage_pos, _ = junctions_to_usage_arrays_by_strand(
        junc_df, CHROM, seq_start, SEQUENCE_LENGTH
    )  # (S,) each
    real_usage_roi = real_usage_pos[roi_s:roi_e]

    # Junction matrix — extract positive-strand junctions in ROI
    ssp, junc_matrix = junctions_to_junction_matrix(
        [junc_df], cls_arr, CHROM, seq_start, SEQUENCE_LENGTH
    )
    # ssp: (4, 256) — row 0 = pos donors (0-based relative), row 1 = pos acceptors
    # junc_matrix: (256, 256, 2) — [:, :, 0] = pos strand sample 0
    pos_donors_rel    = ssp[0]   # relative to seq_start
    pos_acceptors_rel = ssp[1]

    real_junctions = []
    for di, d_rel in enumerate(pos_donors_rel):
        if d_rel < 0:
            break
        d_genomic = int(d_rel) + seq_start
        if not (ROI_START <= d_genomic < ROI_END):
            continue
        for ai, a_rel in enumerate(pos_acceptors_rel):
            if a_rel < 0:
                break
            a_genomic = int(a_rel) + seq_start
            if not (ROI_START <= a_genomic < ROI_END):
                continue
            cnt = float(junc_matrix[di, ai, 0])
            if cnt > 0:
                real_junctions.append((d_genomic, a_genomic, cnt))
    real_junctions.sort(key=lambda x: -x[2])
    print(f"  Real junctions in ROI: {len(real_junctions)}")

    # ------------------------------------------------------------------ #
    # 3. Run model inference
    # ------------------------------------------------------------------ #
    print(f"\nLoading AlphaGenome from {args.weights} …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = AlphaGenome.from_pretrained(args.weights, device=device)
    model.eval()

    print("Running inference …")
    with torch.no_grad():
        outputs = model.predict(seq_tensor.to(device), organism_index=0)
    print("  Output keys:", list(outputs.keys()))

    # ------------------------------------------------------------------ #
    # 4. Extract predicted signals
    # ------------------------------------------------------------------ #

    # RNA-seq
    rna_full = outputs["rna_seq"][1].squeeze(0).cpu().numpy()   # (S, 768)
    rna_roi  = rna_full[roi_s:roi_e]
    hepg2_rna_idx = find_hepg2_indices(args.track_metadata, "rna_seq", "+")
    if hepg2_rna_idx:
        pred_rna = rna_roi[:, hepg2_rna_idx].mean(axis=1)
        rna_label = f"HepG2 RNA-seq (+) — mean of {len(hepg2_rna_idx)} tracks"
    else:
        print("  [WARN] No HepG2 RNA-seq metadata — using mean of all tracks.")
        pred_rna  = rna_roi.mean(axis=1)
        rna_label = "RNA-seq (all tracks mean)"

    # Splice classification
    cls_full  = outputs["splice_sites_classification"]["probs"].squeeze(0).cpu().numpy()
    cls_roi   = cls_full[roi_s:roi_e]
    pred_donor_pos    = cls_roi[:, 0]
    pred_acceptor_pos = cls_roi[:, 1]

    # Splice usage
    usage_full = outputs["splice_sites_usage"]["predictions"].squeeze(0).cpu().numpy()
    usage_roi  = usage_full[roi_s:roi_e]
    hepg2_usage_idx = find_hepg2_indices(args.track_metadata, "splice_sites_usage", "+")
    if hepg2_usage_idx:
        pred_usage  = usage_roi[:, hepg2_usage_idx].mean(axis=1)
        usage_label = f"HepG2 splice usage (+) — mean of {len(hepg2_usage_idx)} tracks"
    else:
        print("  [WARN] No HepG2 splice usage metadata — using mean of all tracks.")
        pred_usage  = usage_roi.mean(axis=1)
        usage_label = "Splice usage (all tracks mean)"

    # Predicted junctions
    junc_data   = outputs.get("splice_sites_junction")
    pred_junctions = []
    if junc_data is not None:
        pssp        = junc_data["splice_site_positions"].squeeze(0).cpu().numpy()  # (4, P)
        pred_counts = junc_data["pred_counts"].squeeze(0).cpu().numpy()            # (P, P, 2T)
        n_tissues   = pred_counts.shape[2] // 2
        jc_fwd      = pred_counts[:, :, :n_tissues].sum(axis=2)                   # (P, P)

        d_genomic_all = pssp[0] + seq_start
        a_genomic_all = pssp[1] + seq_start

        in_roi_d = (d_genomic_all >= ROI_START) & (d_genomic_all < ROI_END)
        in_roi_a = (a_genomic_all >= ROI_START) & (a_genomic_all < ROI_END)
        for di in np.where(in_roi_d)[0]:
            for ai in np.where(in_roi_a)[0]:
                cnt = jc_fwd[di, ai]
                if cnt > 0:
                    pred_junctions.append((d_genomic_all[di], a_genomic_all[ai], cnt))
        pred_junctions.sort(key=lambda x: -x[2])
    print(f"  Predicted junctions in ROI: {len(pred_junctions)}")

    # ------------------------------------------------------------------ #
    # 5. Save parquet outputs
    # ------------------------------------------------------------------ #
    out_dir = Path(os.path.dirname(args.output) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pos_df = pd.DataFrame({
        "chrom":                    CHROM,
        "position":                 positions,
        # predicted
        "pred_rna_seq_hepg2_plus":  pred_rna,
        "pred_donor_plus_prob":     pred_donor_pos,
        "pred_acceptor_plus_prob":  pred_acceptor_pos,
        "pred_splice_usage_hepg2_plus": pred_usage,
        # real
        "real_rna_seq_fwd":         real_rna_fwd,
        "real_donor_plus":          real_donor_pos,
        "real_acceptor_plus":       real_acceptor_pos,
        "real_splice_usage_plus":   real_usage_roi,
    })
    per_pos_path = out_dir / "ldlr_hepg2_pretrained_per_position.parquet"
    per_pos_df.to_parquet(per_pos_path, index=False)
    print(f"\n  Per-position data → {per_pos_path}")

    pred_junc_df = pd.DataFrame(pred_junctions,  columns=["donor_pos", "acceptor_pos", "pred_count"])
    pred_junc_df.insert(0, "chrom", CHROM)
    real_junc_df = pd.DataFrame(real_junctions,  columns=["donor_pos", "acceptor_pos", "real_count"])
    real_junc_df.insert(0, "chrom", CHROM)

    pred_junc_df.to_parquet(out_dir / "ldlr_hepg2_pretrained_junctions_pred.parquet", index=False)
    real_junc_df.to_parquet(out_dir / "ldlr_hepg2_pretrained_junctions_real.parquet", index=False)
    print(f"  Predicted junctions → {out_dir}/ldlr_hepg2_pretrained_junctions_pred.parquet")
    print(f"  Real junctions      → {out_dir}/ldlr_hepg2_pretrained_junctions_real.parquet")

    # ------------------------------------------------------------------ #
    # 6. Plot — 8 rows: pred + real side-by-side for each signal type
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 20))
    row_labels = [
        "RNA-seq\n(predicted)", "RNA-seq\n(real K562)",
        "Donor+\n(pred prob)", "Donor+\n(real sites)",
        "Acceptor+\n(pred prob)", "Acceptor+\n(real sites)",
        "Splice usage\n(predicted)", "Splice usage\n(real K562)",
        "Junctions\n(predicted)", "Junctions\n(real K562)",
    ]
    n_rows = len(row_labels)
    gs     = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)
    axes   = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    def _style(ax, ylabel, xlim, ylim=None, last=False):
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if not last:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(f"Genomic position ({CHROM})", fontsize=9)

    xlim = (ROI_START, ROI_END)

    # Row 0: Predicted RNA-seq
    axes[0].fill_between(positions, pred_rna, color="steelblue", alpha=0.8)
    axes[0].set_title(rna_label, fontsize=7, color="steelblue")
    _style(axes[0], "RNA-seq\n(pred)", xlim)

    # Row 1: Real RNA-seq (forward strand bigwig)
    axes[1].fill_between(positions, real_rna_fwd, color="steelblue", alpha=0.4, label="fwd")
    axes[1].fill_between(positions, -real_rna_rev, color="steelblue", alpha=0.4, label="rev")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("Real K562 RNA-seq (fwd above / rev below)", fontsize=7, color="steelblue")
    _style(axes[1], "RNA-seq\n(real)", xlim)

    # Row 2: Predicted splice donor
    axes[2].fill_between(positions, pred_donor_pos, color="forestgreen", alpha=0.8)
    axes[2].set_title("Predicted splice donor probability (positive strand)", fontsize=7, color="forestgreen")
    _style(axes[2], "Donor+\n(pred)", xlim, ylim=(0, 1))

    # Row 3: Real splice donor sites (binary from STAR)
    donor_sites = np.where(real_donor_pos > 0)[0]
    axes[3].vlines(positions[donor_sites], 0, 1, color="forestgreen", alpha=0.8, lw=1)
    axes[3].set_title("Real splice donor sites (STAR, positive strand)", fontsize=7, color="forestgreen")
    _style(axes[3], "Donor+\n(real)", xlim, ylim=(0, 1.2))

    # Row 4: Predicted splice acceptor
    axes[4].fill_between(positions, pred_acceptor_pos, color="darkorange", alpha=0.8)
    axes[4].set_title("Predicted splice acceptor probability (positive strand)", fontsize=7, color="darkorange")
    _style(axes[4], "Acceptor+\n(pred)", xlim, ylim=(0, 1))

    # Row 5: Real splice acceptor sites
    accept_sites = np.where(real_acceptor_pos > 0)[0]
    axes[5].vlines(positions[accept_sites], 0, 1, color="darkorange", alpha=0.8, lw=1)
    axes[5].set_title("Real splice acceptor sites (STAR, positive strand)", fontsize=7, color="darkorange")
    _style(axes[5], "Acceptor+\n(real)", xlim, ylim=(0, 1.2))

    # Row 6: Predicted splice usage
    axes[6].fill_between(positions, pred_usage, color="mediumpurple", alpha=0.8)
    axes[6].set_title(usage_label, fontsize=7, color="mediumpurple")
    _style(axes[6], "Usage\n(pred)", xlim)

    # Row 7: Real splice usage
    axes[7].fill_between(positions, real_usage_roi, color="mediumpurple", alpha=0.4)
    axes[7].set_title("Real splice site usage (STAR, positive strand)", fontsize=7, color="mediumpurple")
    _style(axes[7], "Usage\n(real)", xlim)

    # Row 8: Predicted junctions (arc plot)
    axes[8].set_ylim(0, 1.1)
    axes[8].set_yticks([])
    axes[8].set_title("Predicted splice junctions (positive strand)", fontsize=7)
    draw_arcs(axes[8], pred_junctions, plt.cm.Blues, "pred")
    _style(axes[8], "Junctions\n(pred)", xlim)

    # Row 9: Real junctions (arc plot)
    axes[9].set_ylim(0, 1.1)
    axes[9].set_yticks([])
    axes[9].set_title("Real splice junctions (STAR K562, positive strand)", fontsize=7)
    draw_arcs(axes[9], real_junctions, plt.cm.Greens, "real")
    _style(axes[9], "Junctions\n(real)", xlim, last=True)

    plt.suptitle(
        f"AlphaGenome pretrained — LDLR gene  ({CHROM}:{ROI_START:,}–{ROI_END:,})\n"
        f"Predicted (HepG2) vs Real (K562) · positive strand",
        fontsize=12, fontweight="bold", y=1.005,
    )

    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"\n✓ PDF saved to {args.output}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualize AlphaGenome pretrained predictions vs real K562 data for LDLR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights",        required=True, help="Pretrained AlphaGenome weights")
    parser.add_argument("--genome",         required=True, help="Reference genome FASTA (.gz ok)")
    parser.add_argument("--track-metadata", default=None,  help="track_metadata.parquet for HepG2 track selection")
    parser.add_argument("--bigwig-fwd",     required=True, help="Forward-strand BigWig (K562)")
    parser.add_argument("--bigwig-rev",     required=True, help="Reverse-strand BigWig (K562)")
    parser.add_argument("--star-junctions", required=True, help="STAR SJ.out.tab file (K562)")
    parser.add_argument("--output",         default="results/examples/alphagenome_pytorch/ldlr_hepg2_pretrained.pdf",
                        help="Output PDF path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
