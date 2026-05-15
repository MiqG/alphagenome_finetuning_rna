#!/usr/bin/env python
"""Run pretrained AlphaGenome forward pass on a BED interval, saving all outputs.

Reads the first interval from the BED file, pads it to sequence_length, runs a
single forward pass of the pretrained model, and writes each output modality as
a separate compressed .npz file:

  rna_seq.npz              — (seq_len, n_tracks) float32
  splice_classification.npz — (seq_len, 5) float32  [Donor+, Acceptor+, Donor-, Acceptor-, None]
  splice_usage.npz         — (seq_len, n_tracks) float32
  splice_junctions.npz     — junction_positions (4, P) int32 + junction_counts (P, P, 2T) float32
  metadata.json            — window coordinates (chrom, padded_start, padded_end, iv_start, iv_end)

Usage:
    python src/scripts/run_pretrained_forward_pass.py \
        --weights data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors \
        --bed data/prep/overfitting/single/high.bed \
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \
        --sequence-length 1048576 \
        --output-dir data/prep/overfitting/single/high
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot


def pad_interval(start: int, end: int, seq_len: int) -> tuple[int, int]:
    length = end - start
    if length >= seq_len:
        center = (start + end) // 2
        return max(0, center - seq_len // 2), center - seq_len // 2 + seq_len
    pad = seq_len - length
    pad_l = pad // 2
    padded_start = max(0, start - pad_l)
    padded_end = padded_start + seq_len
    return padded_start, padded_end


def load_sequence(chrom: str, start: int, end: int, fasta) -> torch.Tensor:
    seq_str = str(fasta[chrom][max(0, start):end]).upper()
    if start < 0:
        seq_str = "N" * (-start) + seq_str
    return torch.from_numpy(sequence_to_onehot(seq_str)).float().unsqueeze(0)  # (1, S, 4)


def main():
    parser = argparse.ArgumentParser(
        description="Run pretrained AlphaGenome forward pass and save all outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Pretrained AlphaGenome .safetensors")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint .pth (model_state_dict loaded on top of pretrained weights)")
    parser.add_argument("--bed", required=True, help="BED file with interval to predict on")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA (.gz ok)")
    parser.add_argument("--sequence-length", type=int, default=1_048_576)
    parser.add_argument("--output-dir", required=True, help="Directory to write output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.checkpoint:
        from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads, add_head
        from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head

        print(f"Loading fine-tuned checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        track_names: dict = ckpt["track_names"]   # {modality: [name, ...]}
        resolutions: dict = ckpt["resolutions"]    # {modality: (int, ...)}

        # Rebuild model with the same head architecture as during fine-tuning,
        # then load the full state dict (trunk + custom heads).
        print(f"Rebuilding model from pretrained trunk {args.weights}")
        model = AlphaGenome()
        model = load_trunk(model, args.weights, exclude_heads=True)
        model = remove_all_heads(model)

        for modality, names in track_names.items():
            n_tracks = len(names)
            if modality == "splice_junctions":
                n_tracks = n_tracks // 2
            mod_res = resolutions[modality] if isinstance(resolutions, dict) else resolutions
            head = create_finetuning_head(
                assay_type=modality,
                n_tracks=n_tracks,
                resolutions=mod_res,
                num_organisms=1,
            )
            add_head(model, modality, head)
            print(f"  Added {modality} head: {len(names)} tracks at resolutions {mod_res}")

        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
    else:
        print(f"Loading model from {args.weights}")
        model = AlphaGenome.from_pretrained(args.weights, device=device)

    model.eval()

    import pyfaidx
    fasta = pyfaidx.Fasta(args.genome)

    intervals = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])
    row = intervals.iloc[0]
    chrom = row["chrom"]
    iv_start, iv_end = int(row["start"]), int(row["end"])
    padded_start, padded_end = pad_interval(iv_start, iv_end, args.sequence_length)
    print(f"Interval: {chrom}:{iv_start}-{iv_end}  padded: {padded_start}-{padded_end}")

    seq_tensor = load_sequence(chrom, padded_start, padded_end, fasta).to(device)

    with torch.no_grad():
        outputs = model.predict(seq_tensor, organism_index=0)

    # rna_seq: (seq_len, n_tracks)
    rna_arr = outputs["rna_seq"][1].squeeze(0).cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(args.output_dir, "rna_seq.npz"), arr=rna_arr)
    print(f"  rna_seq: {rna_arr.shape}")

    # splice_sites_classification: (seq_len, n_classes)
    if "splice_sites_classification" in outputs:
        cls_arr = outputs["splice_sites_classification"]["probs"].squeeze(0).cpu().numpy().astype(np.float32)
        np.savez_compressed(os.path.join(args.output_dir, "splice_classification.npz"), arr=cls_arr)
        print(f"  splice_classification: {cls_arr.shape}")
    else:
        np.savez_compressed(os.path.join(args.output_dir, "splice_classification.npz"))
        print("  splice_classification: not in outputs")

    # splice_sites_usage: (seq_len, n_tracks)
    if "splice_sites_usage" in outputs:
        usg_arr = outputs["splice_sites_usage"]["predictions"].squeeze(0).cpu().numpy().astype(np.float32)
        np.savez_compressed(os.path.join(args.output_dir, "splice_usage.npz"), arr=usg_arr)
        print(f"  splice_usage: {usg_arr.shape}")
    else:
        np.savez_compressed(os.path.join(args.output_dir, "splice_usage.npz"))
        print("  splice_usage: not in outputs")

    # splice_sites_junction: positions (4, P) + counts (P, P, 2T)
    junc_data = outputs.get("splice_sites_junction")
    if junc_data is not None:
        positions = junc_data["splice_site_positions"].squeeze(0).cpu().numpy().astype(np.int32)
        counts = junc_data["pred_counts"].squeeze(0).cpu().numpy().astype(np.float32)
        np.savez_compressed(
            os.path.join(args.output_dir, "splice_junctions.npz"),
            junction_positions=positions,
            junction_counts=counts,
        )
        print(f"  splice_junctions: positions={positions.shape}  counts={counts.shape}")
    else:
        np.savez_compressed(os.path.join(args.output_dir, "splice_junctions.npz"))
        print("  splice_junctions: not in outputs")

    metadata = {
        "chrom": chrom,
        "iv_start": iv_start,
        "iv_end": iv_end,
        "padded_start": padded_start,
        "padded_end": padded_end,
        "sequence_length": args.sequence_length,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
