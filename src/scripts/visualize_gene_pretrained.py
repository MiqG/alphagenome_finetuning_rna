#!/usr/bin/env python
"""Visualize AlphaGenome pretrained predictions vs real K562 data for a gene of interest.

Plots predicted and real RNA-seq, splice donor/acceptor classification,
splice site usage, and splice junctions. Real data loaded using the same
approach as the AlphaGenome finetuning dataloader.

Usage:
    python src/scripts/visualize_gene_pretrained.py \
        --gene-name LDLR \
        --chrom chr19 \
        --roi-start 11066619 \
        --roi-end 11136619 \
        --strand + \
        --weights data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors \
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \
        --bigwig-fwd data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.forward.bw \
        --bigwig-rev data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.reverse.bw \
        --star-junctions data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.SJ.out.tab \
        --track-index-rna 119 \
        --track-index-splice-usage 139 \
        --track-index-splice-junction 139 \
        --output results/examples/alphagenome_pytorch/ldlr_k562_pretrained.pdf
"""


"""
DEBUG:

gene_name="LDLR"
chrom="chr19"
roi_start=11066619
roi_end=11136619
strand="+"
weights="data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors"
genome="data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz"
bigwig_fwd="data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.forward.bw"
bigwig_rev="data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.Aligned.sortedByCoord.out.filtered.reverse.bw"
star_junctions="data/raw/ENA/sf3b1mut/STAR/SRR2103591/second_pass.SJ.out.tab"
track_index_rna=119
track_index_splice_usage=139
track_index_splice_junction=139
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

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "alphagenome-pytorch" / "src"))
from alphagenome_pytorch.extensions.finetuning.star_junctions import (
    read_star_junctions,
    normalize_junctions_per_sample,
    junctions_to_classification_array,
    junctions_to_usage_arrays_by_strand,
    junctions_to_junction_matrix,
)

SEQUENCE_LENGTH = 1_048_576
_NUCLEOTIDES = np.array(["A", "C", "G", "T", "N"])


def center_window(roi_start: int, roi_end: int, seq_len: int) -> tuple[int, int]:
    center = (roi_start + roi_end) // 2
    half = seq_len // 2
    start = center - half
    return start, start + seq_len


def load_sequence(chrom: str, start: int, end: int, fasta_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Return (one_hot_tensor (1,S,4), nucleotide_array (S,) of letters)."""
    try:
        import pyfaidx
    except ImportError:
        raise ImportError("pyfaidx is required: pip install pyfaidx")
    fasta = pyfaidx.Fasta(fasta_path)
    seq_str = str(fasta[chrom][max(0, start):end]).upper()
    if start < 0:
        seq_str = "N" * (-start) + seq_str
    onehot = sequence_to_onehot(seq_str)                        # (S, 4)
    nucleotides = np.array(list(seq_str), dtype="U1")
    tensor = torch.from_numpy(onehot).float().unsqueeze(0)      # (1, S, 4)
    return tensor, nucleotides


def load_bigwig_signal(bw_path: str, chrom: str, start: int, end: int) -> np.ndarray:
    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required: pip install pyBigWig")
    bw = pyBigWig.open(bw_path)
    sig = np.array(bw.stats(chrom, start, end, type="mean", nBins=end - start), dtype=np.float32)
    bw.close()
    return np.nan_to_num(sig, nan=0.0)


def load_junctions(sj_path: str) -> pd.DataFrame:
    junc = read_star_junctions(sj_path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc["count"] = junc["n_uniquely_mapped_reads"].astype(float)
    junc["exon_start"] = junc["intron_start"] - 1
    junc["exon_end"] = junc["intron_end"] + 1
    junc = normalize_junctions_per_sample(junc)
    return junc


def draw_arcs(ax, junctions, cmap):
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


def run(args):
    chrom = args.chrom
    roi_start = args.roi_start
    roi_end = args.roi_end
    strand = args.strand
    gene_name = args.gene_name
    genome = args.genome
    bigwig_fwd = args.bigwig_fwd
    bigwig_rev = args.bigwig_rev
    star_junctions = args.star_junctions
    weights = args.weights
    track_index_rna = args.track_index_rna
    track_index_splice_usage = args.track_index_splice_usage
    track_index_splice_junction = args.track_index_splice_junction
    output = args.output

    seq_start, seq_end = center_window(roi_start, roi_end, SEQUENCE_LENGTH)
    roi_s = roi_start - seq_start
    roi_e = roi_end - seq_start
    positions = np.arange(roi_start, roi_end)

    print(f"Gene               : {gene_name}  (strand {strand})")
    print(f"Region of interest : {chrom}:{roi_start:,}-{roi_end:,} ({roi_end-roi_start:,} bp)")
    print(f"Model input window : {chrom}:{seq_start:,}-{seq_end:,} ({SEQUENCE_LENGTH:,} bp)")

    # ------------------------------------------------------------------ #
    # 1. Sequence
    # ------------------------------------------------------------------ #
    print("\nLoading genome sequence …")
    seq_tensor, nucleotides_full = load_sequence(chrom, seq_start, seq_end, genome)
    nucleotides_roi = nucleotides_full[roi_s:roi_e]
    print(f"  Shape: {seq_tensor.shape}")

    # ------------------------------------------------------------------ #
    # 2. Real data
    # ------------------------------------------------------------------ #
    print("\nLoading real K562 data …")
    real_rna_fwd = load_bigwig_signal(bigwig_fwd, chrom, roi_start, roi_end)
    real_rna_rev = load_bigwig_signal(bigwig_rev, chrom, roi_start, roi_end)

    junc_df = load_junctions(star_junctions)
    print(f"  Junctions loaded: {len(junc_df)}")

    cls_arr = junctions_to_classification_array([junc_df], chrom, seq_start, SEQUENCE_LENGTH)
    real_donor_pos = cls_arr[roi_s:roi_e, 0]
    real_acceptor_pos = cls_arr[roi_s:roi_e, 1]
    real_donor_neg = cls_arr[roi_s:roi_e, 2]
    real_acceptor_neg = cls_arr[roi_s:roi_e, 3]

    real_usage_pos, real_usage_neg = junctions_to_usage_arrays_by_strand(
        junc_df, chrom, seq_start, SEQUENCE_LENGTH
    )
    real_usage_roi = real_usage_pos[roi_s:roi_e] if strand == "+" else real_usage_neg[roi_s:roi_e]

    ssp, junc_matrix = junctions_to_junction_matrix([junc_df], cls_arr, chrom, seq_start, SEQUENCE_LENGTH)
    pos_donors_rel = ssp[0]
    pos_acceptors_rel = ssp[1]
    neg_donors_rel = ssp[2]
    neg_acceptors_rel = ssp[3]

    def _extract_junctions(donors_rel, acceptors_rel, mat_col):
        junctions = []
        for di, d_rel in enumerate(donors_rel):
            if d_rel < 0:
                break
            d_genomic = int(d_rel) + seq_start
            if not (roi_start <= d_genomic < roi_end):
                continue
            for ai, a_rel in enumerate(acceptors_rel):
                if a_rel < 0:
                    break
                a_genomic = int(a_rel) + seq_start
                if not (roi_start <= a_genomic < roi_end):
                    continue
                cnt = float(junc_matrix[di, ai, mat_col])
                if cnt > 0:
                    junctions.append((d_genomic+1, a_genomic+1, cnt))
        junctions.sort(key=lambda x: -x[2])
        return junctions

    real_junctions = _extract_junctions(pos_donors_rel, pos_acceptors_rel, 0)
    print(f"  Real junctions in ROI: {len(real_junctions)}")

    # ------------------------------------------------------------------ #
    # 3. Model inference
    # ------------------------------------------------------------------ #
    print(f"\nLoading AlphaGenome from {weights} …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = AlphaGenome.from_pretrained(weights, device=device)
    model.eval()

    print("Running inference …")
    with torch.no_grad():
        outputs = model.predict(seq_tensor.to(device), organism_index=0)

    # ------------------------------------------------------------------ #
    # 4. Extract predicted signals using K562 track indices
    # ------------------------------------------------------------------ #
    idx_rna = track_index_rna
    idx_usage = track_index_splice_usage
    idx_junction = track_index_splice_junction

    # RNA-seq: use both fwd (+) and rev (-) K562 tracks
    # rna_seq track layout: idx_rna = fwd (+), idx_rna + 271 = rev (-) based on metadata
    rna_full = outputs["rna_seq"][1].squeeze(0).cpu().numpy()   # (S, 768)
    rna_roi = rna_full[roi_s:roi_e]
    pred_rna = rna_roi[:, idx_rna]
    rna_label = f"K562 RNA-seq (track {idx_rna}, + strand, polyA)"

    # Splice classification
    cls_full = outputs["splice_sites_classification"]["probs"].squeeze(0).cpu().numpy()
    cls_roi = cls_full[roi_s:roi_e]
    pred_donor_pos = cls_roi[:, 0]
    pred_acceptor_pos = cls_roi[:, 1]
    pred_donor_neg = cls_roi[:, 2]
    pred_acceptor_neg = cls_roi[:, 3]

    # Splice usage
    usage_full = outputs["splice_sites_usage"]["predictions"].squeeze(0).cpu().numpy()
    usage_roi = usage_full[roi_s:roi_e]
    pred_usage = usage_roi[:, idx_usage]
    usage_label = f"K562 splice usage (track {idx_usage}, + strand, polyA)"

    # Predicted junctions
    pred_junctions = []
    junc_data = outputs.get("splice_sites_junction")
    if junc_data is not None:
        pssp = junc_data["splice_site_positions"].squeeze(0).cpu().numpy()
        pred_counts = junc_data["pred_counts"].squeeze(0).cpu().numpy()   # (P, P, 2T)
        n_tissues = pred_counts.shape[2] // 2
        jc_fwd = pred_counts[:, :, idx_junction]                          # single tissue fwd

        d_genomic_all = pssp[0] + seq_start
        a_genomic_all = pssp[1] + seq_start
        in_roi_d = (pssp[0] >= 0) & (d_genomic_all >= roi_start) & (d_genomic_all < roi_end)
        in_roi_a = (pssp[1] >= 0) & (a_genomic_all >= roi_start) & (a_genomic_all < roi_end)
        for di in np.where(in_roi_d)[0]:
            for ai in np.where(in_roi_a)[0]:
                cnt = float(jc_fwd[di, ai])
                if cnt > 0:
                    pred_junctions.append((int(d_genomic_all[di]+1), int(a_genomic_all[ai]+1), cnt))
        pred_junctions.sort(key=lambda x: -x[2])
    print(f"  Predicted junctions in ROI: {len(pred_junctions)}")

    # ------------------------------------------------------------------ #
    # 5. Save parquets (with nucleotide column)
    # ------------------------------------------------------------------ #
    out_dir = Path(os.path.dirname(output) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{gene_name.lower()}_k562_pretrained"

    per_pos_df = pd.DataFrame({
        "chrom":                        chrom,
        "position":                     positions+1,
        "nucleotide":                   nucleotides_roi,
        "pred_rna_seq_k562_plus":       pred_rna,
        "pred_donor_plus_prob":         pred_donor_pos,
        "pred_acceptor_plus_prob":      pred_acceptor_pos,
        "pred_donor_neg_prob":          pred_donor_neg,
        "pred_acceptor_neg_prob":       pred_acceptor_neg,
        "pred_splice_usage_k562_plus":  pred_usage,
        "real_rna_seq_fwd":             real_rna_fwd,
        "real_rna_seq_rev":             real_rna_rev,
        "real_donor_plus":              real_donor_pos,
        "real_acceptor_plus":           real_acceptor_pos,
        "real_donor_neg":               real_donor_neg,
        "real_acceptor_neg":            real_acceptor_neg,
        "real_splice_usage":            real_usage_roi,
    })
    per_pos_path = out_dir / f"{prefix}_per_position.parquet"
    per_pos_df.to_parquet(per_pos_path, index=False)
    print(f"\n  Per-position data → {per_pos_path}")

    pred_junc_df = pd.DataFrame(pred_junctions, columns=["donor_pos", "acceptor_pos", "pred_count"])
    pred_junc_df.insert(0, "chrom", chrom)
    real_junc_df = pd.DataFrame(real_junctions, columns=["donor_pos", "acceptor_pos", "real_count"])
    real_junc_df.insert(0, "chrom", chrom)
    pred_junc_df.to_parquet(out_dir / f"{prefix}_junctions_pred.parquet", index=False)
    real_junc_df.to_parquet(out_dir / f"{prefix}_junctions_real.parquet", index=False)

    # ------------------------------------------------------------------ #
    # 6. Plot
    # ------------------------------------------------------------------ #
    plot_strand = strand
    if plot_strand == "+":
        real_donor = real_donor_pos
        real_acceptor = real_acceptor_pos
        pred_donor = pred_donor_pos
        pred_acceptor = pred_acceptor_pos
    else:
        real_donor = real_donor_neg
        real_acceptor = real_acceptor_neg
        pred_donor = pred_donor_neg
        pred_acceptor = pred_acceptor_neg

    n_rows = 10
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    xlim = (roi_start, roi_end)

    def _style(ax, ylabel, ylim=None, last=False):
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if not last:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(f"Genomic position ({chrom})", fontsize=9)

    axes[0].fill_between(positions, pred_rna, color="steelblue", alpha=0.8)
    axes[0].set_title(rna_label, fontsize=7, color="steelblue")
    _style(axes[0], "RNA-seq\n(pred)")

    axes[1].fill_between(positions, real_rna_fwd, color="steelblue", alpha=0.5, label="fwd")
    axes[1].fill_between(positions, -real_rna_rev, color="steelblue", alpha=0.3, label="rev")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("Real K562 RNA-seq (fwd above / rev below)", fontsize=7, color="steelblue")
    _style(axes[1], "RNA-seq\n(real)")

    axes[2].fill_between(positions, pred_donor, color="forestgreen", alpha=0.8)
    axes[2].set_title(f"Predicted splice donor probability ({plot_strand} strand)", fontsize=7, color="forestgreen")
    _style(axes[2], f"Donor{plot_strand}\n(pred)", ylim=(0, 1))

    donor_sites = np.where(real_donor > 0)[0]
    axes[3].vlines(positions[donor_sites], 0, 1, color="forestgreen", alpha=0.8, lw=1)
    axes[3].set_title(f"Real splice donor sites (STAR, {plot_strand} strand)", fontsize=7, color="forestgreen")
    _style(axes[3], f"Donor{plot_strand}\n(real)", ylim=(0, 1.2))

    axes[4].fill_between(positions, pred_acceptor, color="darkorange", alpha=0.8)
    axes[4].set_title(f"Predicted splice acceptor probability ({plot_strand} strand)", fontsize=7, color="darkorange")
    _style(axes[4], f"Acceptor{plot_strand}\n(pred)", ylim=(0, 1))

    acceptor_sites = np.where(real_acceptor > 0)[0]
    axes[5].vlines(positions[acceptor_sites], 0, 1, color="darkorange", alpha=0.8, lw=1)
    axes[5].set_title(f"Real splice acceptor sites (STAR, {plot_strand} strand)", fontsize=7, color="darkorange")
    _style(axes[5], f"Acceptor{plot_strand}\n(real)", ylim=(0, 1.2))

    axes[6].fill_between(positions, pred_usage, color="mediumpurple", alpha=0.8)
    axes[6].set_title(usage_label, fontsize=7, color="mediumpurple")
    _style(axes[6], "Usage\n(pred)")

    axes[7].fill_between(positions, real_usage_roi, color="mediumpurple", alpha=0.4)
    axes[7].set_title(f"Real splice site usage (STAR, {plot_strand} strand)", fontsize=7, color="mediumpurple")
    _style(axes[7], "Usage\n(real)")

    axes[8].set_ylim(0, 1.1)
    axes[8].set_yticks([])
    axes[8].set_title("Predicted splice junctions (positive strand)", fontsize=7)
    draw_arcs(axes[8], pred_junctions, plt.cm.Blues)
    _style(axes[8], "Junctions\n(pred)")

    axes[9].set_ylim(0, 1.1)
    axes[9].set_yticks([])
    axes[9].set_title("Real splice junctions (STAR K562, positive strand)", fontsize=7)
    draw_arcs(axes[9], real_junctions, plt.cm.Greens)
    _style(axes[9], "Junctions\n(real)", last=True)

    plt.suptitle(
        f"AlphaGenome pretrained — {gene_name}  ({chrom}:{roi_start:,}–{roi_end:,})\n"
        f"K562 tracks (rna_seq:{idx_rna}, splice_usage:{idx_usage}, splice_junction:{idx_junction})"
        f" · {plot_strand} strand",
        fontsize=11, fontweight="bold", y=1.005,
    )

    plt.savefig(output, bbox_inches="tight", dpi=150)
    print(f"\n✓ PDF saved to {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AlphaGenome pretrained predictions vs real K562 data for a gene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gene-name",    required=True, help="Gene name (for labels/filenames)")
    parser.add_argument("--chrom",        required=True, help="Chromosome (e.g. chr19)")
    parser.add_argument("--roi-start",    required=True, type=int, help="ROI start (0-based)")
    parser.add_argument("--roi-end",      required=True, type=int, help="ROI end (exclusive)")
    parser.add_argument("--strand",       default="+", choices=["+", "-"], help="Gene strand for plot orientation")
    parser.add_argument("--weights",      required=True, help="Pretrained AlphaGenome weights")
    parser.add_argument("--genome",       required=True, help="Reference genome FASTA (.gz ok)")
    parser.add_argument("--bigwig-fwd",   required=True, help="Forward-strand BigWig (K562)")
    parser.add_argument("--bigwig-rev",   required=True, help="Reverse-strand BigWig (K562)")
    parser.add_argument("--star-junctions", required=True, help="STAR SJ.out.tab file (K562)")
    parser.add_argument("--track-index-rna",             type=int, default=119, help="K562 rna_seq track index")
    parser.add_argument("--track-index-splice-usage",    type=int, default=139, help="K562 splice_usage track index")
    parser.add_argument("--track-index-splice-junction", type=int, default=139, help="K562 splice_junction track index")
    parser.add_argument("--output",       required=True, help="Output PDF path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
