#!/usr/bin/env python
"""Run pretrained AlphaGenome forward pass on a BED interval, saving all outputs.

Reads the first interval from the BED file, pads it to sequence_length, runs a
single forward pass of the pretrained model, and writes each output modality as
a separate compressed .npz file:

  rna_seq.npz                   — (seq_len, n_tracks) float32
  splice_classification.npz     — (seq_len, 5) float32  [Donor+, Acceptor+, Donor-, Acceptor-, None]
  splice_usage.npz              — (seq_len, n_tracks) float32
  splice_junctions.npz          — junction_positions (4, P) int32 + junction_counts (P, P, 2T) float32
                                  positions selected by the classification head (predicted)
  splice_junctions_annotated.npz — same format, but positions are taken from --star-junctions files
                                  first; remaining slots filled with top-predicted positions.
                                  Only written when --star-junctions is provided.
  metadata.json                 — window coordinates (chrom, padded_start, padded_end, iv_start, iv_end)

Usage:
    python src/scripts/run_pretrained_forward_pass.py \
        --weights data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors \
        --bed data/prep/overfitting/single/high.bed \
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \
        --sequence-length 1048576 \
        --output-dir data/prep/overfitting/single/high \
        --star-junctions path/to/sample1.SJ.out.tab path/to/sample2.SJ.out.tab
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


def read_star_junctions(path: str) -> pd.DataFrame:
    df = pd.read_table(
        path,
        header=None,
        names=[
            "chrom", "intron_start", "intron_end", "strand_code",
            "intron_motif", "annotated", "n_uniquely_mapped_reads",
            "n_multi_mapped_reads", "max_overhang",
        ],
    )
    df["strand"] = df["strand_code"].astype(str).map({"1": "+", "2": "-"})
    df = df.dropna(subset=["strand"])
    # 1-based exon coordinates (last base of upstream exon / first base of downstream exon)
    df["exon_start"] = df["intron_start"] - 1
    df["exon_end"] = df["intron_end"] + 1
    return df[["chrom", "exon_start", "exon_end", "strand", "n_uniquely_mapped_reads"]]


def build_annotated_positions(
    star_junction_files: list[str],
    chrom: str,
    window_start: int,      # 0-based padded window start
    seq_len: int,
    iv_start: int,          # 0-based original interval start (for junction filtering)
    iv_end: int,            # 0-based original interval end (for junction filtering)
    max_sites: int = 512,
    min_unique_reads: int = 1,
) -> np.ndarray:
    """Build a (4, max_sites) int32 position array from STAR junction files.

    Mirrors the annotated mode used during finetuning (SplicingDataset): positions
    are derived exclusively from observed STAR splice sites within [iv_start, iv_end),
    with no fallback to model predictions. Remaining slots are padded with -1.

    Roles: 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-
    STAR exon coords are 1-based; 0-based relative = exon_coord - 1 - window_start
    """
    role_positions: list[set[int]] = [set(), set(), set(), set()]

    for path in star_junction_files:
        df = read_star_junctions(path)
        df = df[
            (df["chrom"] == chrom)
            & (df["n_uniquely_mapped_reads"] >= min_unique_reads)
            & (df["exon_start"] > iv_start)
            & (df["exon_end"] <= iv_end)
        ]
        for _, row in df.iterrows():
            d_rel = int(row["exon_start"]) - 1 - window_start
            a_rel = int(row["exon_end"]) - 1 - window_start
            strand = row["strand"]
            if strand == "+":
                if 0 <= d_rel < seq_len:
                    role_positions[0].add(d_rel)
                if 0 <= a_rel < seq_len:
                    role_positions[1].add(a_rel)
            else:
                if 0 <= d_rel < seq_len:
                    role_positions[2].add(d_rel)
                if 0 <= a_rel < seq_len:
                    role_positions[3].add(a_rel)

    result = np.full((4, max_sites), -1, dtype=np.int32)
    for role_idx, annot_set in enumerate(role_positions):
        selected = sorted(annot_set)[:max_sites]
        result[role_idx, : len(selected)] = selected

    return result


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
    parser.add_argument(
        "--star-junctions", nargs="+", default=None,
        help="STAR SJ.out.tab files used to seed annotated junction positions. "
             "When provided, also writes splice_junctions_annotated.npz with positions "
             "taken from these files first, remaining slots filled with top-predicted.",
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16"],
        help="Inference dtype. Use bfloat16 for models trained with --dtype bfloat16 "
             "to match training-time precision (default: float32).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.dtype == "bfloat16" and device.type == "cuda"
    print(f"Device: {device}  dtype: {args.dtype}  bfloat16: {use_bf16}")

    from alphagenome_pytorch.config import DtypePolicy
    dtype_policy = DtypePolicy.mixed_precision() if use_bf16 else DtypePolicy.full_float32()

    if args.checkpoint:
        from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads, add_head
        from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head

        print(f"Loading fine-tuned checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        track_names: dict = ckpt["track_names"]   # {modality: [name, ...]}
        resolutions: dict = ckpt["resolutions"]    # {modality: (int, ...)}

        print(f"Rebuilding model from pretrained trunk {args.weights}")
        model = AlphaGenome(dtype_policy=dtype_policy)
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
        model = AlphaGenome.from_pretrained(args.weights, device=device, dtype_policy=dtype_policy)

    model.eval()

    import pyfaidx
    fasta = pyfaidx.Fasta(args.genome)

    intervals = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])
    row = intervals.iloc[0]
    chrom = row["chrom"]
    iv_start, iv_end = int(row["start"]), int(row["end"])
    padded_start, padded_end = pad_interval(iv_start, iv_end, args.sequence_length)
    seq_len = padded_end - padded_start
    print(f"Interval: {chrom}:{iv_start}-{iv_end}  padded: {padded_start}-{padded_end}")

    seq_tensor = load_sequence(chrom, padded_start, padded_end, fasta).to(device)

    with torch.no_grad():
        outputs = model.predict(seq_tensor, organism_index=0)

    # rna_seq: (seq_len, n_tracks)
    rna_arr = outputs["rna_seq"][1].squeeze(0).cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(args.output_dir, "rna_seq.npz"), arr=rna_arr)
    print(f"  rna_seq: {rna_arr.shape}")

    # splice_sites_classification: (seq_len, 5)
    cls_arr = None
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

    # splice_junctions — predicted positions (existing behaviour)
    junc_data = outputs.get("splice_sites_junction")
    if junc_data is not None:
        positions = junc_data["splice_site_positions"].squeeze(0).cpu().numpy().astype(np.int32)
        counts = junc_data["pred_counts"].squeeze(0).cpu().numpy().astype(np.float32)
        np.savez_compressed(
            os.path.join(args.output_dir, "splice_junctions.npz"),
            junction_positions=positions,
            junction_counts=counts,
        )
        print(f"  splice_junctions (predicted): positions={positions.shape}  counts={counts.shape}")
    else:
        np.savez_compressed(os.path.join(args.output_dir, "splice_junctions.npz"))
        print("  splice_junctions: not in outputs")

    # splice_junctions_annotated — positions from STAR files only (interval-restricted)
    if args.star_junctions and junc_data is not None:
        annot_positions_np = build_annotated_positions(
            star_junction_files=args.star_junctions,
            chrom=chrom,
            window_start=padded_start,
            seq_len=seq_len,
            iv_start=iv_start,
            iv_end=iv_end,
            max_sites=512,
        )
        n_annot = [(annot_positions_np[r] >= 0).sum() for r in range(4)]
        print(f"  annotated positions per role [D+, A+, D-, A-]: {n_annot}")

        annot_positions_t = torch.from_numpy(annot_positions_np).long().unsqueeze(0).to(device)  # (1, 4, 512)
        with torch.no_grad():
            annot_outputs = model.predict(seq_tensor, organism_index=0, splice_site_positions=annot_positions_t)

        annot_junc = annot_outputs.get("splice_sites_junction")
        if annot_junc is not None:
            # Use the input positions directly (model output positions may be cast to float in bfloat16 mode)
            annot_pos = annot_positions_np.astype(np.int32)
            annot_counts = annot_junc["pred_counts"].squeeze(0).cpu().numpy().astype(np.float32)
            np.savez_compressed(
                os.path.join(args.output_dir, "splice_junctions_annotated.npz"),
                junction_positions=annot_pos,
                junction_counts=annot_counts,
            )
            print(f"  splice_junctions (annotated): positions={annot_pos.shape}  counts={annot_counts.shape}")

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
