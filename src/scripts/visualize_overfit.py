#!/usr/bin/env python
"""Visualize overfitting predictions vs real tracks.

Loads a finetuned checkpoint and plots predicted vs real bigwig/splice tracks
for intervals in a BED file. Creates a multi-page PDF (one page per interval).

Usage:
    python scripts/visualize_overfit.py \\
        --checkpoint best_model.pth \\
        --bed overfit.bed \\
        --genome hg38.fa \\
        --bigwig sample1_fwd.bw sample1_rev.bw \\
        --star-junctions sample1_fwd.tab sample1_rev.tab \\
        --sequence-length 1048576 \\
        --output tracks.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# AlphaGenome imports
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset, SpliceJunctionDataset


def load_model_with_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[AlphaGenome, dict]:
    """Load a finetuned checkpoint and rebuild model with heads.

    Args:
        checkpoint_path: Path to best_model.pth
        device: Device to load on

    Returns:
        (model, checkpoint_dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild model with full state
    model = AlphaGenome(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Modality: {checkpoint.get('modality')}")
    print(f"  Resolutions: {checkpoint.get('resolutions')}")

    return model, checkpoint


def load_interval_bigwig(
    interval: Tuple[str, int, int],
    bigwig_files: list[str],
) -> np.ndarray:
    """Load bigwig signal for one interval.

    Args:
        interval: (chrom, start, end)
        bigwig_files: List of bigwig files

    Returns:
        Array of shape (seq_len, n_tracks)
    """
    chrom, start, end = interval
    seq_len = end - start

    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required. Install via: pip install pyBigWig")

    tracks = []
    for bw_file in bigwig_files:
        bw = pyBigWig.open(bw_file)
        try:
            values = np.array(bw.stats(chrom, start, end, type="mean", nBins=seq_len))
            values = np.nan_to_num(values, nan=0.0)
            tracks.append(values)
        finally:
            bw.close()

    return np.stack(tracks, axis=-1) if tracks else np.zeros((seq_len, 0))


def load_interval_sequence(
    interval: Tuple[str, int, int],
    genome_fasta: str,
) -> torch.Tensor:
    """Load sequence for one interval.

    Args:
        interval: (chrom, start, end)
        genome_fasta: Path to genome FASTA

    Returns:
        One-hot tensor (seq_len, 4)
    """
    chrom, start, end = interval

    try:
        import pyfaidx
    except ImportError:
        raise ImportError("pyfaidx is required. Install via: pip install pyfaidx")

    fasta = pyfaidx.Fasta(genome_fasta)
    seq_str = str(fasta[chrom][max(0, start):end])

    if start < 0:
        seq_str = "N" * (-start) + seq_str

    from alphagenome_pytorch.utils.sequence import sequence_to_onehot
    return torch.from_numpy(sequence_to_onehot(seq_str)).float()


def plot_interval(
    axes: list,
    chrom: str,
    start: int,
    end: int,
    sequence: torch.Tensor,
    real_tracks: np.ndarray,
    predictions: dict,
    title: str = "",
):
    """Plot one interval with real vs predicted tracks.

    Args:
        axes: List of matplotlib axes (6 total)
        chrom, start, end: Interval coordinates
        sequence: One-hot sequence (seq_len, 4)
        real_tracks: Real bigwig tracks (seq_len, n_tracks)
        predictions: Dict of prediction tensors from model
        title: Title for the plot
    """
    seq_len = sequence.shape[0]
    positions = np.arange(seq_len) + start

    # Row 0: Real RNA-seq
    ax = axes[0]
    if real_tracks.shape[1] > 0:
        real_signal = real_tracks[:, 0]  # First track as example
        ax.fill_between(positions, real_signal, alpha=0.6, color="steelblue", label="Real")
        ax.set_ylabel("Real RNA-seq")
        ax.set_xlim(start, end)
        ax.set_ylim(0, np.percentile(real_signal[real_signal > 0], 95) if (real_signal > 0).any() else 1)

    # Row 1: Predicted RNA-seq
    ax = axes[1]
    if "rna_seq" in predictions:
        pred_dict = predictions["rna_seq"]
        # Use 1bp resolution if available, else 128bp
        resolution = 1 if 1 in pred_dict else 128
        pred_tensor = pred_dict[resolution]

        # Convert (B, S, C) or (B, C, S) to (S,)
        if pred_tensor.ndim > 1:
            pred_signal = pred_tensor.squeeze(0).squeeze(-1).cpu().numpy()
        else:
            pred_signal = pred_tensor.squeeze(0).cpu().numpy()

        # Adjust positions if using 128bp resolution
        if resolution == 128:
            step = seq_len // len(pred_signal)
            positions_pred = np.arange(len(pred_signal)) * step + start
        else:
            positions_pred = positions

        ax.fill_between(positions_pred, pred_signal, alpha=0.6, color="coral", label="Predicted")
        ax.set_ylabel("Pred RNA-seq")
        ax.set_xlim(start, end)

    # Row 2-3: Splice classification (real vs pred)
    ax = axes[2]
    ax.set_ylabel("Splice\nClass (Real)")
    ax.set_xlim(start, end)
    ax.text(0.5, 0.5, "Splice classification\n(if available)", ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray')

    ax = axes[3]
    ax.set_ylabel("Splice\nClass (Pred)")
    ax.set_xlim(start, end)
    if "splice_sites_classification" in predictions:
        probs = predictions["splice_sites_classification"][1].squeeze(0).cpu().numpy()
        pred_class = probs.argmax(axis=-1)
        mask = probs.max(axis=-1) > 0.5  # Only plot confident predictions
        ax.scatter(positions[mask], pred_class[mask] / 5.0, alpha=0.5, s=10, color="coral")
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.text(0.5, 0.5, "(not available)", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')

    # Row 4-5: Splice usage
    ax = axes[4]
    ax.set_ylabel("Splice\nUsage (Real)")
    ax.set_xlim(start, end)
    ax.text(0.5, 0.5, "Splice usage\n(if available)", ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray')

    ax = axes[5]
    ax.set_ylabel("Splice\nUsage (Pred)")
    ax.set_xlabel(f"Position ({chrom})")
    ax.set_xlim(start, end)
    if "splice_sites_usage" in predictions:
        preds = predictions["splice_sites_usage"][1].squeeze(0).cpu().numpy()
        usage_max = preds.max(axis=-1)
        ax.fill_between(positions, usage_max, alpha=0.5, color="coral")
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "(not available)", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')

    # Title
    plt.suptitle(title, fontsize=12, fontweight="bold", y=0.98)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize overfitting predictions vs real tracks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--bed", type=str, required=True, help="BED file with intervals")
    parser.add_argument("--genome", type=str, required=True, help="Reference genome FASTA")
    parser.add_argument(
        "--bigwig",
        type=str,
        nargs="+",
        required=True,
        help="BigWig signal files",
    )
    parser.add_argument(
        "--star-junctions",
        type=str,
        nargs="*",
        default=[],
        help="Optional STAR junction files",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=131072,
        help="Sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PDF path",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_with_checkpoint(args.checkpoint, device=device)

    # Read intervals
    print(f"\nReading intervals from: {args.bed}")
    intervals = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])
    print(f"Found {len(intervals)} intervals")

    # Create visualization
    print(f"\nCreating visualization PDF: {args.output}")
    with PdfPages(args.output) as pdf:
        for idx, (_, row) in enumerate(intervals.iterrows()):
            chrom, start, end = row["chrom"], int(row["start"]), int(row["end"])
            print(f"  [{idx + 1}/{len(intervals)}] Processing {chrom}:{start}-{end}")

            try:
                # Load sequence and bigwig
                sequence = load_interval_sequence((chrom, start, end), args.genome)
                real_tracks = load_interval_bigwig((chrom, start, end), args.bigwig)

                # Run inference
                with torch.no_grad():
                    seq_batch = sequence.unsqueeze(0).to(device)  # (1, seq_len, 4)
                    organism_idx = torch.tensor([0], device=device)
                    outputs = model.predict(seq_batch, organism_index=organism_idx)

                # Create figure
                fig = plt.figure(figsize=(14, 8))
                gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.5)
                axes = [fig.add_subplot(gs[i]) for i in range(6)]

                # Plot
                plot_interval(
                    axes,
                    chrom, start, end,
                    sequence, real_tracks, outputs,
                    title=f"{chrom}:{start}-{end}",
                )

                # Save page
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n✓ Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
