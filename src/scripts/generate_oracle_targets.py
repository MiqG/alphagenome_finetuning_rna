#!/usr/bin/env python
"""Generate oracle training targets from pretrained AlphaGenome predictions.

Runs the pretrained AlphaGenome model on intervals from a BED file and writes
HepG2 predictions as:
  - BigWig files for rna_seq (fwd + rev strand)
  - BigWig files for splice_sites_usage (fwd + rev strand)
  - STAR SJ.out.tab file for splice junctions (used to derive splice_site
    classification and usage targets during finetuning)

The generated files are consumed directly by finetune.py to train a new head
that must recover the pretrained model's own outputs — a self-consistency
capacity benchmark.

Usage:
    python src/scripts/generate_oracle_targets.py \
        --weights data/raw/articles/Avsec2026/alphagenome_pytorch/fold-1/model_fold_1.safetensors \
        --track-metadata data/raw/articles/Avsec2026/alphagenome_pytorch/track_metadata.parquet \
        --bed support/overfit.bed \
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \
        --sequence-length 1048576 \
        --output-dir support/oracle/hepg2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

# ------------------------------------------------------------------ #
# Track index helpers
# ------------------------------------------------------------------ #

def within_head_indices(metadata_path: str, output_type: str, strand: str, ontology: str) -> list[int]:
    """Return within-head track indices for a given output_type / strand / cell line."""
    df = pd.read_parquet(metadata_path)
    head_df = df[(df["output_type"] == output_type) & (df["organism"] == "human")].reset_index(drop=True)
    mask = (head_df["ontology_curie"] == ontology) & (head_df["track_strand"] == strand)
    return head_df.index[mask].tolist()


# ------------------------------------------------------------------ #
# Sequence helpers
# ------------------------------------------------------------------ #

def load_sequence(chrom: str, start: int, end: int, fasta) -> torch.Tensor:
    seq_str = str(fasta[chrom][max(0, start):end]).upper()
    if start < 0:
        seq_str = "N" * (-start) + seq_str
    return torch.from_numpy(sequence_to_onehot(seq_str)).float().unsqueeze(0)  # (1, S, 4)


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


# ------------------------------------------------------------------ #
# BigWig writing
# ------------------------------------------------------------------ #

def get_chrom_sizes(genome_path: str) -> dict[str, int]:
    """Read chromosome sizes from .fai index or pyfaidx."""
    fai = genome_path + ".fai"
    if not os.path.exists(fai) and genome_path.endswith(".gz"):
        fai = genome_path[:-3] + ".fai"
    if os.path.exists(fai):
        sizes = {}
        with open(fai) as f:
            for line in f:
                parts = line.split()
                sizes[parts[0]] = int(parts[1])
        return sizes
    # fallback: use pyfaidx
    import pyfaidx
    fa = pyfaidx.Fasta(genome_path)
    return {k: len(fa[k]) for k in fa.keys()}


def create_bigwig_writer(path: str, chrom_sizes: dict[str, int]):
    import pyBigWig
    bw = pyBigWig.open(path, "w")
    # write header with chromosomes that have data (add all for compatibility)
    bw.addHeader(list(chrom_sizes.items()))
    return bw


def write_signal_to_bw(bw, chrom: str, start: int, values: np.ndarray) -> None:
    """Write 1-bp resolution float32 signal to an open pyBigWig handle."""
    end = start + len(values)
    vals = values.astype(np.float64)
    # pyBigWig expects list of (chrom, start, end, value) — but addEntries is faster
    chroms = [chrom] * len(vals)
    starts = list(range(start, end))
    ends = list(range(start + 1, end + 1))
    bw.addEntries(chroms, starts, ends=ends, values=vals.tolist())


# ------------------------------------------------------------------ #
# STAR SJ.out.tab writing
# ------------------------------------------------------------------ #

STAR_COLUMNS = [
    "chrom", "intron_start", "intron_end", "strand",
    "motif", "annotated", "n_uniquely_mapped_reads",
    "n_multi_mapped_reads", "max_spliced_overhang",
]


def junction_output_to_star_rows(
    junc_data: dict,
    seq_start: int,
    chrom: str,
    hepg2_tissue_idx: int,
    min_count_fraction: float = 0.5,
    count_scale: float = 1000.0,
) -> list[tuple]:
    """Convert model junction output to STAR SJ.out.tab rows (both strands).

    Predicted counts are in AlphaGenome's normalized space:
        target = clip(CPM, 99.99th pct) / mean_nonzero_CPM
    To anchor read counts to this scale, the mean nonzero prediction for the
    target tissue (both strands combined) is computed per interval and used as
    the denominator:
        n_reads = round(cnt / mean_nonzero * count_scale)
    A junction at the mean predicted value therefore yields exactly count_scale
    reads, so after normalize_junctions_per_sample the recovered normalized
    values approximate the original model predictions.

    Args:
        junc_data: dict with keys "splice_site_positions" (4, P) and
                   "pred_counts" (P, P, 2T).
        seq_start: genomic start of the padded window.
        chrom: chromosome name.
        hepg2_tissue_idx: within-head tissue index for HepG2 junctions.
        min_count_fraction: fraction of mean_nonzero below which a junction
            is excluded (replaces a fixed absolute threshold).
        count_scale: integer read count assigned to a junction at
            mean_nonzero predicted value.

    Returns:
        List of 9-tuples in STAR SJ.out.tab format (tab-delimited).
    """
    pssp = junc_data["splice_site_positions"].squeeze(0).cpu().numpy()  # (4, P)
    pred_counts = junc_data["pred_counts"].squeeze(0).cpu().numpy()     # (P, P, 2T)
    n_tissues = pred_counts.shape[2] // 2
    t_idx = min(hepg2_tissue_idx, n_tissues - 1)

    # Mean nonzero across both strands for the target tissue in this interval.
    # Used to convert from prediction space to a count scale where count_scale
    # reads represents the typical expressed junction.
    both_strands = np.concatenate([
        pred_counts[:, :, t_idx].ravel(),
        pred_counts[:, :, n_tissues + t_idx].ravel(),
    ])
    nonzero_vals = both_strands[both_strands > 0]
    mean_nonzero = float(nonzero_vals.mean()) if len(nonzero_vals) > 0 else 1.0
    min_count = min_count_fraction * mean_nonzero

    rows = []

    # pssp rows: 0=pos_donors, 1=pos_acceptors, 2=neg_donors, 3=neg_acceptors
    # pred_counts last dim: 0..T-1 = pos strand, T..2T-1 = neg strand
    # STAR strand codes: 1 = positive, 2 = negative
    # For pos strand: donor < acceptor genomically; intron = [donor+1, acceptor]
    # For neg strand: donor > acceptor genomically (5' ss is downstream on genome);
    #   intron coordinates are still [lower+1, higher] = [acceptor+1, donor]
    for strand_code, d_row, a_row, count_offset in [
        (1, 0, 1, 0),           # positive strand
        (2, 2, 3, n_tissues),   # negative strand
    ]:
        jc = pred_counts[:, :, count_offset + t_idx]
        donor_rel = pssp[d_row]
        accept_rel = pssp[a_row]

        for di in range(len(donor_rel)):
            if donor_rel[di] < 0:
                break
            d_genomic = int(donor_rel[di]) + seq_start
            for ai in range(len(accept_rel)):
                if accept_rel[ai] < 0:
                    break
                a_genomic = int(accept_rel[ai]) + seq_start
                cnt = float(jc[di, ai])
                if cnt < min_count:
                    continue
                if d_genomic == a_genomic:
                    continue
                # intron coords always [lower+1, higher] (1-based, inclusive)
                lo, hi = min(d_genomic, a_genomic), max(d_genomic, a_genomic)
                intron_start = lo + 1
                intron_end = hi
                n_reads = max(1, int(round(cnt / mean_nonzero * count_scale)))
                rows.append((chrom, intron_start, intron_end, strand_code, 0, 0, n_reads, 0, 0))

    return rows


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Generate HepG2 oracle targets from pretrained AlphaGenome",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Pretrained AlphaGenome .safetensors")
    parser.add_argument("--track-metadata", required=True, help="track_metadata.parquet")
    parser.add_argument("--bed", required=True, help="BED file with intervals to predict on")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA (.gz ok)")
    parser.add_argument("--sequence-length", type=int, default=1_048_576)
    parser.add_argument("--output-dir", required=True, help="Directory to write oracle files")
    parser.add_argument("--ontology", default="EFO:0001187", help="Ontology CURIE for target cell line")
    parser.add_argument("--junction-min-count-fraction", type=float, default=0.5,
                        help="Min junction count as a fraction of the interval mean nonzero prediction")
    parser.add_argument("--junction-count-scale", type=float, default=1000.0,
                        help="Read count assigned to a junction at the mean nonzero predicted value")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve within-head track indices for HepG2
    # ------------------------------------------------------------------ #
    rna_fwd_idx  = within_head_indices(args.track_metadata, "rna_seq",            "+", args.ontology)
    rna_rev_idx  = within_head_indices(args.track_metadata, "rna_seq",            "-", args.ontology)
    usg_fwd_idx  = within_head_indices(args.track_metadata, "splice_sites_usage", "+", args.ontology)
    usg_rev_idx  = within_head_indices(args.track_metadata, "splice_sites_usage", "-", args.ontology)
    junc_idx     = within_head_indices(args.track_metadata, "splice_sites_junction", ".", args.ontology)

    if not rna_fwd_idx:
        raise ValueError("No HepG2 rna_seq + track found in metadata")
    if not rna_rev_idx:
        raise ValueError("No HepG2 rna_seq - track found in metadata")

    rna_fwd  = rna_fwd_idx[0]
    rna_rev  = rna_rev_idx[0]
    usg_fwd  = usg_fwd_idx[0] if usg_fwd_idx else None
    usg_rev  = usg_rev_idx[0] if usg_rev_idx else None
    junc_t   = junc_idx[0] if junc_idx else None

    print(f"HepG2 within-head indices:")
    print(f"  rna_seq fwd={rna_fwd}  rev={rna_rev}")
    print(f"  splice_usage fwd={usg_fwd}  rev={usg_rev}")
    print(f"  splice_junction tissue={junc_t}")

    # ------------------------------------------------------------------ #
    # Load model and genome
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Loading model from {args.weights} …")
    model = AlphaGenome.from_pretrained(args.weights, device=device)
    model.eval()

    import pyfaidx
    fasta = pyfaidx.Fasta(args.genome)

    # ------------------------------------------------------------------ #
    # Read intervals
    # ------------------------------------------------------------------ #
    intervals = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])
    print(f"\n{len(intervals)} intervals from {args.bed}")

    # Accumulate signal per chrom → dict[chrom] -> list[(start, array)]
    rna_fwd_signal: dict[str, list] = {}
    rna_rev_signal: dict[str, list] = {}
    usg_fwd_signal: dict[str, list] = {}
    usg_rev_signal: dict[str, list] = {}
    all_junc_rows: list[tuple] = []
    all_splice_site_rows: list[tuple] = []  # (chrom, genomic_pos, site_type, prob)
    npz_manifest: list[dict] = []  # records for DistillationDataset index

    for _, row in intervals.iterrows():
        chrom = row["chrom"]
        iv_start = int(row["start"])
        iv_end   = int(row["end"])
        seq_start, seq_end = pad_interval(iv_start, iv_end, args.sequence_length)
        print(f"  {chrom}:{iv_start}-{iv_end}  →  padded {seq_start}-{seq_end}")

        seq_tensor = load_sequence(chrom, seq_start, seq_end, fasta).to(device)

        with torch.no_grad():
            outputs = model.predict(seq_tensor, organism_index=0)

        # RNA-seq (1bp resolution)
        rna_out = outputs["rna_seq"][1].squeeze(0).cpu().numpy()  # (S, n_tracks)
        off_s = iv_start - seq_start
        off_e = off_s + (iv_end - iv_start)

        rna_fwd_signal.setdefault(chrom, []).append((iv_start, rna_out[off_s:off_e, rna_fwd]))
        rna_rev_signal.setdefault(chrom, []).append((iv_start, rna_out[off_s:off_e, rna_rev]))

        # Splice usage
        usg_out = None
        if "splice_sites_usage" in outputs:
            usg_out = outputs["splice_sites_usage"]["predictions"].squeeze(0).cpu().numpy()
            if usg_fwd is not None:
                usg_fwd_signal.setdefault(chrom, []).append((iv_start, usg_out[off_s:off_e, usg_fwd]))
            if usg_rev is not None:
                usg_rev_signal.setdefault(chrom, []).append((iv_start, usg_out[off_s:off_e, usg_rev]))

        # Splice classification
        cls_out = None
        if "splice_sites_classification" in outputs:
            cls_out = outputs["splice_sites_classification"]["probs"].squeeze(0).cpu().numpy()
            # Extract positions where each class probability > 0.5 and accumulate
            # a genomic coordinate table across intervals.
            # Columns: 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-, 4=None
            for class_idx, site_type in enumerate(["Donor+", "Acceptor+", "Donor-", "Acceptor-"]):
                pos_rel = np.where(cls_out[:, class_idx] > 0.5)[0]
                if len(pos_rel) == 0:
                    continue
                genomic_pos = pos_rel.astype(np.int64) + seq_start
                probs = cls_out[pos_rel, class_idx].astype(np.float32)
                for gp, pr in zip(genomic_pos, probs):
                    nt = str(fasta[chrom][int(gp):int(gp) + 1]).upper()
                    all_splice_site_rows.append((chrom, int(gp), site_type, float(pr), nt))

        # Splice junctions
        junc_data = outputs.get("splice_sites_junction")
        junc_rows_interval = []
        if junc_t is not None and junc_data is not None:
            junc_rows_interval = junction_output_to_star_rows(
                junc_data, seq_start, chrom, junc_t,
                min_count_fraction=args.junction_min_count_fraction,
                count_scale=args.junction_count_scale,
            )
            all_junc_rows.extend(junc_rows_interval)
            print(f"    {len(junc_rows_interval)} junctions extracted")

        # ------------------------------------------------------------------ #
        # Save raw distillation .npz for this interval (full padded window)
        # The DistillationDataset loads these and slices to the interval.
        # ------------------------------------------------------------------ #
        npz_arrays: dict[str, np.ndarray] = {
            "chrom":     np.array(chrom),
            "seq_start": np.array(seq_start),
            "seq_end":   np.array(seq_end),
            "iv_start":  np.array(iv_start),
            "iv_end":    np.array(iv_end),
            # rna_seq: full padded window, shape (seq_len, n_tracks)
            "rna_seq":   rna_out.astype(np.float32),
        }
        if cls_out is not None:
            npz_arrays["splice_sites_classification"] = cls_out.astype(np.float32)
        if usg_out is not None:
            npz_arrays["splice_sites_usage"] = usg_out.astype(np.float32)
        if junc_data is not None:
            pssp = junc_data["splice_site_positions"].squeeze(0).cpu().numpy()  # (4, P)
            pred_counts = junc_data["pred_counts"].squeeze(0).cpu().numpy()     # (P, P, 2T)
            npz_arrays["junction_positions"] = pssp.astype(np.int32)
            npz_arrays["junction_counts"]    = pred_counts.astype(np.float32)

        npz_name = f"interval_{chrom}_{iv_start}_{iv_end}.npz"
        npz_path = os.path.join(args.output_dir, npz_name)
        np.savez_compressed(npz_path, **npz_arrays)
        print(f"    saved {npz_path}")
        npz_manifest.append({
            "chrom": chrom, "iv_start": iv_start, "iv_end": iv_end,
            "seq_start": seq_start, "seq_end": seq_end,
            "npz": npz_name,
        })

        del outputs, seq_tensor
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Write BigWig files
    # ------------------------------------------------------------------ #
    print("\nReading chromosome sizes …")
    chrom_sizes = get_chrom_sizes(args.genome)

    def _write_bw(signal_dict: dict, path: str) -> None:
        import pyBigWig
        bw = pyBigWig.open(path, "w")
        bw.addHeader(list(chrom_sizes.items()))
        for chrom, entries in sorted(signal_dict.items()):
            for start, vals in sorted(entries, key=lambda x: x[0]):
                end = start + len(vals)
                vals64 = vals.astype(np.float64)
                bw.addEntries(
                    [chrom] * len(vals64),
                    list(range(start, end)),
                    ends=list(range(start + 1, end + 1)),
                    values=vals64.tolist(),
                )
        bw.close()
        print(f"  wrote {path}")

    rna_fwd_path = os.path.join(args.output_dir, "oracle_rna_fwd.bw")
    rna_rev_path = os.path.join(args.output_dir, "oracle_rna_rev.bw")
    _write_bw(rna_fwd_signal, rna_fwd_path)
    _write_bw(rna_rev_signal, rna_rev_path)

    if usg_fwd_signal:
        usg_fwd_path = os.path.join(args.output_dir, "oracle_usage_fwd.bw")
        usg_rev_path = os.path.join(args.output_dir, "oracle_usage_rev.bw")
        _write_bw(usg_fwd_signal, usg_fwd_path)
        _write_bw(usg_rev_signal, usg_rev_path)

    # ------------------------------------------------------------------ #
    # Write STAR junction file
    # ------------------------------------------------------------------ #
    junc_path = os.path.join(args.output_dir, "oracle_junctions.SJ.out.tab")
    junc_df = pd.DataFrame(all_junc_rows, columns=STAR_COLUMNS)
    junc_df = junc_df.sort_values(["chrom", "intron_start"]).drop_duplicates(
        subset=["chrom", "intron_start", "intron_end", "strand"]
    )
    junc_df.to_csv(junc_path, sep="\t", index=False, header=False)
    print(f"  wrote {junc_path}  ({len(junc_df)} junctions)")

    junc_start_plus1_path = os.path.join(args.output_dir, "oracle_junctions_start_plus1.SJ.out.tab")
    junc_start_plus1_df = junc_df.copy()
    junc_start_plus1_df["intron_start"] = junc_start_plus1_df["intron_start"] + 1
    junc_start_plus1_df.to_csv(junc_start_plus1_path, sep="\t", index=False, header=False)
    print(f"  wrote {junc_start_plus1_path}  ({len(junc_start_plus1_df)} junctions)")

    # ------------------------------------------------------------------ #
    # Write splice site position table
    # ------------------------------------------------------------------ #
    ss_path = os.path.join(args.output_dir, "oracle_splice_sites.parquet")
    ss_df = pd.DataFrame(all_splice_site_rows, columns=["chrom", "position", "site_type", "probability", "nucleotide"])
    ss_df = ss_df.sort_values(["chrom", "position"]).drop_duplicates(subset=["chrom", "position", "site_type"])
    ss_df.to_parquet(ss_path, index=False)
    print(f"  wrote {ss_path}  ({len(ss_df)} splice sites, prob>0.5)")

    # ------------------------------------------------------------------ #
    # Write distillation manifest (index of .npz files)
    # ------------------------------------------------------------------ #
    manifest_path = os.path.join(args.output_dir, "distillation_manifest.parquet")
    pd.DataFrame(npz_manifest).to_parquet(manifest_path, index=False)
    print(f"  wrote {manifest_path}  ({len(npz_manifest)} intervals)")

    print(f"\nOracle targets written to {args.output_dir}/")
    print("  oracle_rna_fwd.bw")
    print("  oracle_rna_rev.bw")
    if usg_fwd_signal:
        print("  oracle_usage_fwd.bw")
        print("  oracle_usage_rev.bw")
    print("  oracle_junctions.SJ.out.tab")
    print("  oracle_junctions_start_plus1.SJ.out.tab")
    print("  oracle_splice_sites.parquet")
    print("  distillation_manifest.parquet")
    print(f"  interval_*.npz  ({len(npz_manifest)} files)")


if __name__ == "__main__":
    main()
