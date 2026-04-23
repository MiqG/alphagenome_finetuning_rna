#!/usr/bin/env python
"""Visualize overfitting predictions vs real tracks at gene level.

For each protein-coding gene overlapping intervals in a BED file, generates
one PDF per sample with 10 rows (pred+real for RNA-seq, splice donor/acceptor,
splice usage, and splice junctions). Structure mirrors visualize_ldlr_pretrained.py.

BigWig files must be supplied as fwd/rev pairs in sample order:
    s1_fwd.bw s1_rev.bw [s2_fwd.bw s2_rev.bw ...]
STAR junction files must be supplied one per sample in the same order.

Usage:
    python src/scripts/visualize_overfit.py \
        --checkpoint best_model.pth \
        --bed overfit.bed \
        --genome hg38.fa.gz \
        --gtf gencode.v46.annotation.gtf.gz \
        --bigwig s1_fwd.bw s1_rev.bw s2_fwd.bw s2_rev.bw \
        --star-junctions s1.SJ.out.tab s2.SJ.out.tab \
        --sequence-length 1048576 \
        --output-dir results/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_genes_from_gtf(gtf_path: str) -> pd.DataFrame:
    """Load protein-coding gene coordinates from GTF file."""
    import gzip
    genes = []
    open_fn = gzip.open if gtf_path.endswith('.gz') else open
    with open_fn(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9 or cols[2] != 'gene':
                continue
            attrs = cols[8]
            gene_name, gene_type = 'N/A', 'N/A'
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name "'):
                    gene_name = attr.split('"')[1]
                elif attr.startswith('gene_type "'):
                    gene_type = attr.split('"')[1]
            if gene_type != 'protein_coding':
                continue
            genes.append({
                'chrom': cols[0],
                'start': int(cols[3]),
                'end': int(cols[4]),
                'strand': cols[6],
                'gene_name': gene_name,
                'gene_type': gene_type,
            })
    return pd.DataFrame(genes)


def find_overlapping_genes(
    interval: Tuple[str, int, int],
    genes_df: pd.DataFrame,
) -> pd.DataFrame:
    chrom, start, end = interval
    return genes_df[
        (genes_df['chrom'] == chrom) &
        (genes_df['start'] <= end) &
        (genes_df['end'] >= start)
    ].sort_values('start').reset_index(drop=True)


def pad_gene_to_sequence_length(
    gene_start: int,
    gene_end: int,
    sequence_length: int,
) -> Tuple[int, int]:
    gene_len = gene_end - gene_start
    if gene_len >= sequence_length:
        # Center-crop to sequence_length
        center = (gene_start + gene_end) // 2
        return max(0, center - sequence_length // 2), center - sequence_length // 2 + sequence_length
    padding = sequence_length - gene_len
    pad_left = padding // 2
    pad_right = padding - pad_left
    padded_start = max(0, gene_start - pad_left)
    padded_end = gene_end + pad_right
    if padded_start == 0:
        padded_end = min(padded_end + (gene_start - pad_left), sequence_length)
    return padded_start, padded_end


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[AlphaGenome, dict]:
    """Load fine-tuned AlphaGenome checkpoint, reconstructing custom heads from metadata.

    Supports both full finetuning and LoRA checkpoints. LoRA config is read from
    config.json in the same directory as the checkpoint.
    """
    import json
    from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads, add_head
    from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load training config from config.json in the same directory (if present)
    config_path = Path(checkpoint_path).parent / "config.json"
    train_config = {}
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        print(f"  Training config: mode={train_config.get('mode', 'full')}")

    modalities = checkpoint.get("modality", [])
    resolutions = checkpoint.get("resolutions", {})
    track_names = checkpoint.get("track_names", {})
    if isinstance(modalities, str):
        modalities = [modalities]
    if not isinstance(track_names, dict):
        track_names = {modalities[0]: track_names} if modalities else {}
    if not isinstance(resolutions, dict):
        resolutions = {m: resolutions for m in modalities}

    model = AlphaGenome(num_organisms=2)
    model = remove_all_heads(model)

    for modality in modalities:
        names = track_names.get(modality, [])
        n_tracks = len(names)
        if n_tracks == 0:
            continue
        res = list(resolutions.get(modality, (1,)))
        # track_names for splice_junctions stores both strands (2 per sample),
        # but the head expects num_tissues = num_samples (same as finetune.py line 912).
        if modality == "splice_junctions":
            n_tracks = n_tracks // 2
        head = create_finetuning_head(assay_type=modality, n_tracks=n_tracks, resolutions=res, num_organisms=1)
        add_head(model, modality, head)
        print(f"  Head: {modality}  tracks={n_tracks}  resolutions={res}")

    # Apply LoRA/LoCon adapters before loading state dict if checkpoint was LoRA-trained
    mode = train_config.get("mode", "full")
    if mode == "lora":
        from alphagenome_pytorch.extensions.finetuning.adapters import apply_lora, apply_locon
        lora_rank = train_config.get("lora_rank") or 0
        lora_alpha = train_config.get("lora_alpha", 16)
        lora_targets = (train_config.get("lora_targets") or "").split(",")
        if lora_rank > 0 and any(lora_targets):
            print(f"  Applying LoRA: rank={lora_rank}, alpha={lora_alpha}, targets={lora_targets}")
            model = apply_lora(model, lora_targets, rank=lora_rank, alpha=lora_alpha)
        locon_targets_str = train_config.get("locon_targets") or ""
        if locon_targets_str:
            locon_rank = train_config.get("locon_rank", 4)
            locon_alpha = train_config.get("locon_alpha", 1)
            locon_targets = locon_targets_str.split(",")
            print(f"  Applying LoCon: rank={locon_rank}, alpha={locon_alpha}, targets={locon_targets}")
            model = apply_locon(model, locon_targets, rank=locon_rank, alpha=locon_alpha)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_sequence(chrom: str, start: int, end: int, fasta) -> torch.Tensor:
    seq_str = str(fasta[chrom][max(0, start):end]).upper()
    if start < 0:
        seq_str = "N" * (-start) + seq_str
    return torch.from_numpy(sequence_to_onehot(seq_str)).float().unsqueeze(0)  # (1, S, 4)


def load_bigwig_signal(bw, chrom: str, start: int, end: int) -> np.ndarray:
    sig = np.array(bw.stats(chrom, start, end, type="mean", nBins=end - start), dtype=np.float32)
    return np.nan_to_num(sig, nan=0.0)


def load_junctions(sj_path: str) -> pd.DataFrame:
    junc = read_star_junctions(sj_path)
    junc = junc.loc[junc["n_uniquely_mapped_reads"] >= 1].copy()
    junc["count"] = junc["n_uniquely_mapped_reads"].astype(float)
    junc["exon_start"] = junc["intron_start"] - 1
    junc["exon_end"] = junc["intron_end"] + 1
    junc = normalize_junctions_per_sample(junc)
    return junc


def compute_correlation(real: np.ndarray, pred: np.ndarray) -> float:
    mask = real > 0
    if mask.sum() < 2:
        return np.nan
    try:
        corr, _ = pearsonr(real[mask], pred[mask])
        return corr
    except:
        return np.nan


def compute_correlation_all(real: np.ndarray, pred: np.ndarray) -> float:
    """Pearson R over all positions (not masked to nonzero real)."""
    if len(real) < 2:
        return np.nan
    try:
        corr, _ = pearsonr(real.astype(float), pred.astype(float))
        return corr
    except:
        return np.nan


def _corr_str(r: float) -> str:
    return f"r={r:.3f}" if not np.isnan(r) else "r=n/a"


def compute_junction_correlation(
    pred_junctions: list, real_junctions: list
) -> float:
    """Spearman R over union of (donor, acceptor) pairs; missing entries set to 0.

    Using Spearman (rank-based) avoids scale mismatch between normalized real
    counts and raw softplus predicted counts. The union+0-fill penalizes both
    false positives (pred>0, real=0) and false negatives (pred=0, real>0).
    """
    if not real_junctions:
        return np.nan
    real_map = {(int(d), int(a)): c for d, a, c in real_junctions}
    pred_map = {(int(d), int(a)): c for d, a, c in pred_junctions}
    keys = set(real_map) | set(pred_map)
    if len(keys) < 2:
        return np.nan
    real_vec = np.array([real_map.get(k, 0.0) for k in keys])
    pred_vec = np.array([pred_map.get(k, 0.0) for k in keys])
    try:
        corr, _ = spearmanr(real_vec, pred_vec)
        return float(corr)
    except:
        return np.nan


def draw_arcs(ax, junctions: list, cmap) -> None:
    if not junctions:
        ax.text(0.5, 0.5, "No junctions", ha="center", va="center",
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


def plot_gene_sample(
    pdf_path: str,
    gene_name: str,
    chrom: str,
    gene_start: int,
    gene_end: int,
    strand: str,
    sample_name: str,
    positions: np.ndarray,
    # predicted
    pred_rna_fwd: np.ndarray,
    pred_rna_rev: np.ndarray,
    pred_donor: np.ndarray | None,
    pred_acceptor: np.ndarray | None,
    pred_usage: np.ndarray,
    pred_junctions: list,
    # real
    real_rna_fwd: np.ndarray,
    real_rna_rev: np.ndarray,
    real_donor: np.ndarray | None,
    real_acceptor: np.ndarray | None,
    real_usage: np.ndarray,
    real_junctions: list,
    rna_corr: float,
    donor_corr: float = float("nan"),
    acceptor_corr: float = float("nan"),
    usage_corr: float = float("nan"),
    junction_corr: float = float("nan"),
) -> None:
    n_rows = 10
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    xlim = (gene_start, gene_end)

    def _style(ax, ylabel, ylim=None, last=False):
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if not last:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(f"Genomic position ({chrom})", fontsize=9)

    # Row 0: Predicted RNA-seq (fwd above, rev below)
    axes[0].fill_between(positions, pred_rna_fwd, color="steelblue", alpha=0.8, label="fwd")
    axes[0].fill_between(positions, -pred_rna_rev, color="steelblue", alpha=0.5, label="rev")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_title(f"Predicted RNA-seq (fwd above / rev below)  ·  {_corr_str(rna_corr)}", fontsize=7, color="steelblue")
    _style(axes[0], "RNA-seq\n(pred)")

    # Row 1: Real RNA-seq
    axes[1].fill_between(positions, real_rna_fwd, color="steelblue", alpha=0.4, label="fwd")
    axes[1].fill_between(positions, -real_rna_rev, color="steelblue", alpha=0.4, label="rev")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("Real RNA-seq (fwd above / rev below)", fontsize=7, color="steelblue")
    _style(axes[1], "RNA-seq\n(real)")

    # Row 2: Predicted splice donor
    if pred_donor is not None:
        axes[2].fill_between(positions, pred_donor, color="forestgreen", alpha=0.8)
    axes[2].set_title(f"Predicted splice donor probability (+ strand)  ·  {_corr_str(donor_corr)}", fontsize=7, color="forestgreen")
    _style(axes[2], "Donor+\n(pred)", ylim=(0, 1))

    # Row 3: Real splice donor sites (binary from STAR)
    if real_donor is not None:
        donor_sites = np.where(real_donor > 0)[0]
        if len(donor_sites) > 0:
            axes[3].vlines(positions[donor_sites], 0, 1, color="forestgreen", alpha=0.8, lw=1)
    axes[3].set_title("Real splice donor sites (STAR, + strand)", fontsize=7, color="forestgreen")
    _style(axes[3], "Donor+\n(real)", ylim=(0, 1.2))

    # Row 4: Predicted splice acceptor
    if pred_acceptor is not None:
        axes[4].fill_between(positions, pred_acceptor, color="darkorange", alpha=0.8)
    axes[4].set_title(f"Predicted splice acceptor probability (+ strand)  ·  {_corr_str(acceptor_corr)}", fontsize=7, color="darkorange")
    _style(axes[4], "Acceptor+\n(pred)", ylim=(0, 1))

    # Row 5: Real splice acceptor sites
    if real_acceptor is not None:
        accept_sites = np.where(real_acceptor > 0)[0]
        if len(accept_sites) > 0:
            axes[5].vlines(positions[accept_sites], 0, 1, color="darkorange", alpha=0.8, lw=1)
    axes[5].set_title("Real splice acceptor sites (STAR, + strand)", fontsize=7, color="darkorange")
    _style(axes[5], "Acceptor+\n(real)", ylim=(0, 1.2))

    # Row 6: Predicted splice usage
    axes[6].fill_between(positions, pred_usage, color="mediumpurple", alpha=0.8)
    axes[6].set_title(f"Predicted splice site usage  ·  {_corr_str(usage_corr)}", fontsize=7, color="mediumpurple")
    _style(axes[6], "Usage\n(pred)")

    # Row 7: Real splice usage
    axes[7].fill_between(positions, real_usage, color="mediumpurple", alpha=0.4)
    axes[7].set_title("Real splice site usage (STAR)", fontsize=7, color="mediumpurple")
    _style(axes[7], "Usage\n(real)")

    # Row 8: Predicted junctions (arc plot)
    axes[8].set_ylim(0, 1.1)
    axes[8].set_yticks([])
    axes[8].set_title(f"Predicted splice junctions  ·  rho={junction_corr:.3f}" if not np.isnan(junction_corr) else "Predicted splice junctions  ·  rho=n/a", fontsize=7)
    draw_arcs(axes[8], pred_junctions, plt.cm.Blues)
    _style(axes[8], "Junctions\n(pred)")

    # Row 9: Real junctions (arc plot)
    axes[9].set_ylim(0, 1.1)
    axes[9].set_yticks([])
    axes[9].set_title("Real splice junctions (STAR)", fontsize=7)
    draw_arcs(axes[9], real_junctions, plt.cm.Greens)
    _style(axes[9], "Junctions\n(real)", last=True)

    plt.suptitle(
        f"{gene_name}  ·  {chrom}:{gene_start:,}–{gene_end:,} ({strand})\n"
        f"Sample: {sample_name}",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualize overfitting predictions at gene level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--bed", required=True, help="BED file with intervals")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA (.gz ok)")
    parser.add_argument("--gtf", required=True, help="GTF annotation (.gz ok)")
    parser.add_argument(
        "--bigwig", nargs="+", required=True,
        help="BigWig files in sample order: s1_fwd s1_rev [s2_fwd s2_rev ...]",
    )
    parser.add_argument(
        "--star-junctions", nargs="*", default=[],
        help="STAR SJ.out.tab files, one per sample in the same order",
    )
    parser.add_argument("--sequence-length", type=int, default=1048576)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_strands = 2  # always fwd + rev
    if len(args.bigwig) % n_strands != 0:
        raise ValueError(f"Expected even number of BigWig files (fwd+rev pairs), got {len(args.bigwig)}")
    n_samples = len(args.bigwig) // n_strands
    if args.star_junctions and len(args.star_junctions) != n_samples:
        raise ValueError(f"Expected {n_samples} STAR junction files, got {len(args.star_junctions)}")

    # Infer sample names from parent directory of the fwd bigwig of each sample
    sample_names = [Path(args.bigwig[2 * i]).parent.name for i in range(n_samples)]
    print(f"Samples ({n_samples}): {', '.join(sample_names)}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, _ = load_model(args.checkpoint, device)

    # Load protein-coding genes
    print(f"\nLoading protein-coding genes from: {args.gtf}")
    genes_df = load_genes_from_gtf(args.gtf)
    print(f"  {len(genes_df)} protein-coding genes")

    # Read intervals
    print(f"\nReading intervals from: {args.bed}")
    intervals = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])
    print(f"  {len(intervals)} intervals")

    # Open file handles (kept open for all iterations)
    import pyfaidx
    import pyBigWig
    fasta = pyfaidx.Fasta(args.genome)
    bw_handles = [pyBigWig.open(f) for f in args.bigwig]  # 2*n_samples handles, fwd/rev interleaved

    all_stats = []
    tsv_path = os.path.join(args.output_dir, "summary_stats.tsv")
    tsv_header_written = False

    for interval_idx, (_, interval_row) in enumerate(intervals.iterrows()):
        chrom = interval_row["chrom"]
        interval_start = int(interval_row["start"])
        interval_end = int(interval_row["end"])
        print(f"\n[{interval_idx + 1}/{len(intervals)}] {chrom}:{interval_start}-{interval_end}")

        genes = find_overlapping_genes((chrom, interval_start, interval_end), genes_df)
        print(f"  {len(genes)} protein-coding genes")
        if len(genes) == 0:
            continue

        try:
            padded_start, padded_end = pad_gene_to_sequence_length(
                interval_start, interval_end, args.sequence_length
            )
            print(f"  Padded window: {chrom}:{padded_start:,}-{padded_end:,}")

            # Pre-load STAR junctions per sample — needed before forward pass to
            # supply real splice site positions to the junction head
            junc_dfs = []
            real_cls_arrs = []
            real_usage_arrs_full = []
            real_junc_matrices = []
            real_ssps = []
            for s_idx in range(n_samples):
                if s_idx < len(args.star_junctions):
                    junc_df = load_junctions(args.star_junctions[s_idx])
                    cls_arr_real = junctions_to_classification_array(
                        [junc_df], chrom, padded_start, args.sequence_length
                    )
                    real_usage_pos, _ = junctions_to_usage_arrays_by_strand(
                        junc_df, chrom, padded_start, args.sequence_length
                    )
                    ssp_real, junc_matrix = junctions_to_junction_matrix(
                        [junc_df], cls_arr_real, chrom, padded_start, args.sequence_length
                    )
                    junc_dfs.append(junc_df)
                    real_cls_arrs.append(cls_arr_real)
                    real_usage_arrs_full.append(real_usage_pos)
                    real_junc_matrices.append(junc_matrix)
                    real_ssps.append(ssp_real)
                else:
                    junc_dfs.append(None)
                    real_cls_arrs.append(None)
                    real_usage_arrs_full.append(None)
                    real_junc_matrices.append(None)
                    real_ssps.append(None)

            # ---------------------------------------------------------------- #
            # Per-sample forward pass, supplying real STAR junction positions
            # so the junction head predicts CPMs at observed splice sites.
            # RNA-seq / classification / usage outputs are sequence-only and
            # identical across samples — extracted from the first pass only.
            # ---------------------------------------------------------------- #
            seq_tensor = load_sequence(chrom, padded_start, padded_end, fasta)
            outputs_per_sample = []
            for s_idx in range(n_samples):
                ssp = real_ssps[s_idx]
                ssp_tensor = (
                    torch.from_numpy(ssp).long().unsqueeze(0).to(device)
                    if ssp is not None else None
                )
                with torch.no_grad():
                    out = model.predict(
                        seq_tensor.to(device),
                        organism_index=torch.tensor([0], device=device),
                        splice_site_positions=ssp_tensor,
                    )
                outputs_per_sample.append(out)
                torch.cuda.empty_cache()

            # Full-window arrays (keep on CPU as numpy) — same for all samples
            out0 = outputs_per_sample[0]
            rna_full = out0["rna_seq"][1].squeeze(0).cpu().numpy()  # (S, n_tracks)

            cls_full = None
            pred_donor_full = None
            pred_acceptor_full = None
            if "splice_sites_classification" in out0:
                cls_full = out0["splice_sites_classification"]["probs"].squeeze(0).cpu().numpy()
                pred_donor_full = cls_full[:, 0] if cls_full.shape[1] > 0 else None
                pred_acceptor_full = cls_full[:, 1] if cls_full.shape[1] > 1 else None

            usage_full = None
            if "splice_sites_usage" in out0:
                usage_full = out0["splice_sites_usage"]["predictions"].squeeze(0).cpu().numpy()  # (S, n_tracks)

            # ---------------------------------------------------------------- #
            # Subset outputs per gene
            # ---------------------------------------------------------------- #
            for _, gene_row in genes.iterrows():
                gene_name = gene_row["gene_name"]
                gene_start = int(gene_row["start"])
                gene_end = int(gene_row["end"])
                strand = gene_row["strand"]
                gene_len = gene_end - gene_start

                if gene_len < 1000:
                    continue

                print(f"  {gene_name} ({strand})  {gene_start:,}-{gene_end:,}  ({gene_len:,}bp)")

                # Offsets within padded window — skip genes outside the cropped window
                off_s = gene_start - padded_start
                off_e = off_s + gene_len
                if off_s < 0 or off_e > (padded_end - padded_start):
                    print(f"    Skipping {gene_name}: outside padded window")
                    continue
                positions = np.arange(gene_start, gene_end)

                # Subset predicted arrays to gene
                rna_roi = rna_full[off_s:off_e]
                pred_donor_roi = pred_donor_full[off_s:off_e] if pred_donor_full is not None else None
                pred_acceptor_roi = pred_acceptor_full[off_s:off_e] if pred_acceptor_full is not None else None
                usage_roi = usage_full[off_s:off_e] if usage_full is not None else None

                for s_idx, sample_name in enumerate(sample_names):
                    fwd_track = 2 * s_idx
                    rev_track = 2 * s_idx + 1

                    # Predicted RNA-seq
                    pred_rna_fwd = rna_roi[:, fwd_track] if rna_roi.shape[1] > fwd_track else np.zeros(gene_len, np.float32)
                    pred_rna_rev = rna_roi[:, rev_track] if rna_roi.shape[1] > rev_track else np.zeros(gene_len, np.float32)

                    # Real RNA-seq from BigWig
                    real_rna_fwd = load_bigwig_signal(bw_handles[fwd_track], chrom, gene_start, gene_end)
                    real_rna_rev = load_bigwig_signal(bw_handles[rev_track], chrom, gene_start, gene_end)

                    rna_corr = compute_correlation(real_rna_fwd, pred_rna_fwd)

                    # Donor / acceptor / usage / junction correlations computed after real data is loaded
                    donor_corr = float("nan")
                    acceptor_corr = float("nan")
                    usage_corr = float("nan")
                    junction_corr = float("nan")

                    # Predicted splice usage (fwd track)
                    if usage_roi is not None and usage_roi.shape[1] > fwd_track:
                        pred_usage = usage_roi[:, fwd_track]
                    else:
                        pred_usage = np.zeros(gene_len, np.float32)

                    # Predicted junctions for this sample (from per-sample forward pass
                    # with real STAR positions supplied as splice_site_positions)
                    junc_data = outputs_per_sample[s_idx].get("splice_sites_junction")
                    pred_junctions: list = []
                    if junc_data is not None:
                        pssp = junc_data["splice_site_positions"].squeeze(0).cpu().numpy()  # (4, P)
                        pred_counts = junc_data["pred_counts"].squeeze(0).cpu().numpy()     # (P, P, 2T)
                        n_tissues = pred_counts.shape[2] // 2
                        jc_fwd = pred_counts[:, :, s_idx] if pred_counts.shape[2] > s_idx else pred_counts[:, :, :n_tissues].sum(axis=2)
                        d_genomic_all = pssp[0] + padded_start
                        a_genomic_all = pssp[1] + padded_start
                        valid_d = (pssp[0] >= 0) & (d_genomic_all >= gene_start) & (d_genomic_all < gene_end)
                        valid_a = (pssp[1] >= 0) & (a_genomic_all >= gene_start) & (a_genomic_all < gene_end)
                        for di in np.where(valid_d)[0]:
                            for ai in np.where(valid_a)[0]:
                                cnt = float(jc_fwd[di, ai])
                                if cnt > 0:
                                    pred_junctions.append((d_genomic_all[di], a_genomic_all[ai], cnt))
                        pred_junctions.sort(key=lambda x: -x[2])

                    # Real junctions and splice arrays from STAR
                    real_donor_arr = None
                    real_acceptor_arr = None
                    real_usage_arr = np.zeros(gene_len, np.float32)
                    real_junctions: list = []

                    if real_cls_arrs[s_idx] is not None:
                        real_donor_arr = real_cls_arrs[s_idx][off_s:off_e, 0]
                        real_acceptor_arr = real_cls_arrs[s_idx][off_s:off_e, 1]
                        real_usage_arr = real_usage_arrs_full[s_idx][off_s:off_e]

                        ssp_real = real_ssps[s_idx]
                        junc_matrix = real_junc_matrices[s_idx]
                        for di, d_rel in enumerate(ssp_real[0]):
                            if d_rel < 0:
                                break
                            d_genomic = int(d_rel) + padded_start
                            if not (gene_start <= d_genomic < gene_end):
                                continue
                            for ai, a_rel in enumerate(ssp_real[1]):
                                if a_rel < 0:
                                    break
                                a_genomic = int(a_rel) + padded_start
                                if not (gene_start <= a_genomic < gene_end):
                                    continue
                                cnt = float(junc_matrix[di, ai, 0])
                                if cnt > 0:
                                    real_junctions.append((d_genomic, a_genomic, cnt))
                        real_junctions.sort(key=lambda x: -x[2])

                    # Compute correlations now that real arrays are ready
                    if pred_donor_roi is not None and real_donor_arr is not None:
                        donor_corr = compute_correlation_all(real_donor_arr, pred_donor_roi)
                    if pred_acceptor_roi is not None and real_acceptor_arr is not None:
                        acceptor_corr = compute_correlation_all(real_acceptor_arr, pred_acceptor_roi)
                    usage_corr = compute_correlation(real_usage_arr, pred_usage)
                    junction_corr = compute_junction_correlation(pred_junctions, real_junctions)

                    # Save parquet
                    parquet_data: dict = {
                        "chrom":             chrom,
                        "position":          positions,
                        "gene_name":         gene_name,
                        "strand":            strand,
                        "sample":            sample_name,
                        "pred_rna_fwd":      pred_rna_fwd,
                        "pred_rna_rev":      pred_rna_rev,
                        "real_rna_fwd":      real_rna_fwd,
                        "real_rna_rev":      real_rna_rev,
                        "pred_splice_usage": pred_usage,
                        "real_splice_usage": real_usage_arr,
                    }
                    if pred_donor_roi is not None:
                        parquet_data["pred_donor_prob"] = pred_donor_roi
                    if pred_acceptor_roi is not None:
                        parquet_data["pred_acceptor_prob"] = pred_acceptor_roi
                    if real_donor_arr is not None:
                        parquet_data["real_donor_sites"] = real_donor_arr
                    if real_acceptor_arr is not None:
                        parquet_data["real_acceptor_sites"] = real_acceptor_arr

                    stem = f"{gene_name}_{gene_start}_{gene_end}_{sample_name}"
                    pd.DataFrame(parquet_data).to_parquet(
                        os.path.join(args.output_dir, f"{stem}.parquet"), index=False
                    )
                    if pred_junctions:
                        (pd.DataFrame(pred_junctions, columns=["donor_pos", "acceptor_pos", "pred_count"])
                           .assign(chrom=chrom)
                           .to_parquet(os.path.join(args.output_dir, f"{stem}_pred_junctions.parquet"), index=False))
                    if real_junctions:
                        (pd.DataFrame(real_junctions, columns=["donor_pos", "acceptor_pos", "real_count"])
                           .assign(chrom=chrom)
                           .to_parquet(os.path.join(args.output_dir, f"{stem}_real_junctions.parquet"), index=False))

                    # Plot
                    pdf_path = os.path.join(args.output_dir, f"{stem}.pdf")
                    plot_gene_sample(
                        pdf_path=pdf_path,
                        gene_name=gene_name,
                        chrom=chrom,
                        gene_start=gene_start,
                        gene_end=gene_end,
                        strand=strand,
                        sample_name=sample_name,
                        positions=positions,
                        pred_rna_fwd=pred_rna_fwd,
                        pred_rna_rev=pred_rna_rev,
                        pred_donor=pred_donor_roi,
                        pred_acceptor=pred_acceptor_roi,
                        pred_usage=pred_usage,
                        pred_junctions=pred_junctions,
                        real_rna_fwd=real_rna_fwd,
                        real_rna_rev=real_rna_rev,
                        real_donor=real_donor_arr,
                        real_acceptor=real_acceptor_arr,
                        real_usage=real_usage_arr,
                        real_junctions=real_junctions,
                        rna_corr=rna_corr,
                        donor_corr=donor_corr,
                        acceptor_corr=acceptor_corr,
                        usage_corr=usage_corr,
                        junction_corr=junction_corr,
                    )
                    print(
                        f"    {sample_name}"
                        f"  rna={_corr_str(rna_corr)}"
                        f"  donor={_corr_str(donor_corr)}"
                        f"  acceptor={_corr_str(acceptor_corr)}"
                        f"  usage={_corr_str(usage_corr)}"
                        f"  junc={_corr_str(junction_corr)}"
                        f"  pred_junc={len(pred_junctions)}  real_junc={len(real_junctions)}"
                        f"  → {pdf_path}"
                    )

                    row = {
                        "chrom":                  chrom,
                        "gene_name":              gene_name,
                        "gene_start":             gene_start,
                        "gene_end":               gene_end,
                        "strand":                 strand,
                        "gene_length":            gene_len,
                        "sample":                 sample_name,
                        "rna_seq_correlation":    rna_corr,
                        "donor_correlation":      donor_corr,
                        "acceptor_correlation":   acceptor_corr,
                        "usage_correlation":      usage_corr,
                        "junction_correlation":   junction_corr,
                        "n_pred_junctions":       len(pred_junctions),
                        "n_real_junctions":       len(real_junctions),
                    }
                    all_stats.append(row)
                    with open(tsv_path, "a") as tsv_f:
                        if not tsv_header_written:
                            tsv_f.write("\t".join(row.keys()) + "\n")
                            tsv_header_written = True
                        tsv_f.write("\t".join(str(v) for v in row.values()) + "\n")

            del rna_full, cls_full, usage_full, outputs_per_sample
            del pred_donor_full, pred_acceptor_full
            del junc_dfs, real_cls_arrs, real_usage_arrs_full, real_junc_matrices, real_ssps

        except Exception as e:
            import traceback
            print(f"  Error on interval: {e}")
            traceback.print_exc()
            continue

    for bw in bw_handles:
        bw.close()

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_path = os.path.join(args.output_dir, "summary_stats.parquet")
        stats_df.to_parquet(stats_path, index=False)
        print(f"\n✓ Summary: {stats_path}  ({len(all_stats)} gene×sample records)")
        print("\nTop genes by RNA-seq correlation (fwd):")
        print(stats_df.nlargest(10, "rna_seq_correlation")[
            ["gene_name", "sample", "rna_seq_correlation"]].to_string(index=False))

    print(f"\n✓ Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
