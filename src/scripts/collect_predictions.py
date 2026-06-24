#!/usr/bin/env python
"""
Collect finetuned AlphaGenome predictions on held-out test intervals.

Runs single-GPU inference on every interval in --test-bed and writes five
prediction parquets consumed by compute_eval_metrics.py:

  rna_seq_per_gene.parquet    — per-gene exon-mean coverage (pred + obs, per track)
  splice_site_scores.parquet  — per-annotated-position classification probs (all splice site
                                positions from RNA-seq / GTF; negatives are other annotated
                                classes, matching the publication's auPRC definition)
  ssu_scores.parquet          — per-SSU-position usage predictions
  junction_scores.parquet     — per-junction count predictions (true + sampled false)
  psi_scores.parquet          — PSI5/PSI3 predictions (chr2 intervals only)

Usage:
    python src/scripts/collect_predictions.py \\
        --pretrained-weights data/raw/.../model_fold_1.safetensors \\
        --checkpoint results/finetuning/.../best_model.pth \\
        --test-bed data/prep/finetuning/alphagenome/FOLD_1/test.bed \\
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \\
        --gtf-parquet data/raw/GENCODE/release_46/gencode.v46.annotation.gtf.parquet \\
        --bigwigs bw1.bw bw2.bw bw3.bw bw4.bw \\
        --ssu-parquets ssu1.parquet ssu2.parquet \\
        --star-junctions sj1.tab sj2.tab \\
        --samples SRR17111303 SRR17111311 \\
        --output-dir results/evaluation/.../predictions
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.transfer import (
    add_head,
    load_trunk,
    remove_all_heads,
)
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

_EPS = 1e-8
_MAX_SPLICE_SITES = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pretrained-weights", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-bed", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--gtf-parquet", required=True)
    p.add_argument("--bigwigs", nargs="+", required=True,
                   help="Bigwig files in order: sample0/forward, sample0/reverse, sample1/forward, ...")
    p.add_argument("--ssu-parquets", nargs="+", required=True,
                   help="SSU parquet files, one per sample, same order as --bigwigs pairs")
    p.add_argument("--star-junctions", nargs="+", required=True,
                   help="STAR SJ.out.tab files, one per sample")
    p.add_argument("--samples", nargs="+", required=True,
                   help="Sample IDs (e.g. SRR17111303 SRR17111311), matching order of --ssu-parquets / --star-junctions")
    p.add_argument("--sequence-length", type=int, default=1_048_576)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (kept for reproducibility of any future sampling)")
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"],
        help="Inference dtype. Must match training dtype (bfloat16 for all finetuned runs here).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_finetuned_model(
    pretrained_weights: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: str = "bfloat16",
):
    """Reconstruct finetuned model from pretrained trunk + checkpoint heads.

    dtype must match the training dtype (bfloat16 for all runs in this project).
    A mismatch causes degraded correlation, as documented in run_pretrained_forward_pass.py.
    """
    from alphagenome_pytorch.config import DtypePolicy

    use_bf16 = dtype == "bfloat16" and device.type == "cuda"
    dtype_policy = DtypePolicy.mixed_precision() if use_bf16 else DtypePolicy.full_float32()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    track_names: dict = ckpt["track_names"]
    resolutions: dict = ckpt["resolutions"]

    model = AlphaGenome(dtype_policy=dtype_policy)
    model = load_trunk(model, pretrained_weights, exclude_heads=True)
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

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, track_names


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def read_star_junctions(path: str, sample_id: str, sample_idx: int) -> pd.DataFrame:
    """Parse STAR SJ.out.tab into a DataFrame with 1-based exon coordinates."""
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
    # 1-based exon coordinates (last upstream exon base / first downstream exon base)
    df["donor_pos"] = df["intron_start"] - 1   # 1-based donor
    df["acceptor_pos"] = df["intron_end"] + 1  # 1-based acceptor
    df["sample_id"] = sample_id
    df["sample_idx"] = sample_idx
    return df[["chrom", "donor_pos", "acceptor_pos", "strand",
               "n_uniquely_mapped_reads", "sample_id", "sample_idx"]]


def pad_interval(start: int, end: int, seq_len: int) -> tuple[int, int]:
    if end - start >= seq_len:
        center = (start + end) // 2
        return max(0, center - seq_len // 2), center - seq_len // 2 + seq_len
    pad = seq_len - (end - start)
    padded_start = max(0, start - pad // 2)
    return padded_start, padded_start + seq_len


def build_annotated_positions(
    merged_junctions: pd.DataFrame,
    chrom: str,
    window_start: int,
    seq_len: int,
    iv_start: int,
    iv_end: int,
) -> np.ndarray:
    """Build (4, MAX_SPLICE_SITES) int32 position array from merged STAR junctions.

    Roles: 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-
    Positions are 0-based relative to window_start.
    Padding value: -1.
    """
    role_positions: list[set[int]] = [set(), set(), set(), set()]

    df = merged_junctions[
        (merged_junctions["chrom"] == chrom)
        & (merged_junctions["n_uniquely_mapped_reads"] >= 1)
        & (merged_junctions["donor_pos"] > iv_start)
        & (merged_junctions["acceptor_pos"] <= iv_end + 1)
    ]
    for _, row in df.iterrows():
        d_rel = int(row["donor_pos"]) - 1 - window_start
        a_rel = int(row["acceptor_pos"]) - 1 - window_start
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

    result = np.full((4, _MAX_SPLICE_SITES), -1, dtype=np.int32)
    for role_idx, pos_set in enumerate(role_positions):
        selected = sorted(pos_set)[:_MAX_SPLICE_SITES]
        result[role_idx, : len(selected)] = selected
    return result


# ---------------------------------------------------------------------------
# Gene-interval assignment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-gene coverage helpers
# ---------------------------------------------------------------------------

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


def get_exon_mean_obs(
    bigwigs: list,
    exon_rows: pd.DataFrame,
    chrom: str,
    chrom_sizes: dict[str, int],
) -> np.ndarray | None:
    """Mean observed bigwig coverage over all exon positions, per track."""
    n_tracks = len(bigwigs)
    sums = np.zeros(n_tracks, dtype=np.float64)
    count = 0
    for _, ex in exon_rows.iterrows():
        ex_start, ex_end = int(ex["Start"]), int(ex["End"])
        chrom_len = chrom_sizes.get(chrom, int(1e10))
        ex_end = min(ex_end, chrom_len)
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


# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

def compute_psi_from_matrix(
    counts_mat: np.ndarray,  # (K, K)
) -> tuple[np.ndarray, np.ndarray]:
    """PSI5 and PSI3 matrices from a count matrix."""
    d_total = counts_mat.sum(axis=1, keepdims=True) + _EPS  # (K, 1)
    a_total = counts_mat.sum(axis=0, keepdims=True) + _EPS  # (1, K)
    psi5 = counts_mat / d_total
    psi3 = counts_mat / a_total
    return psi5, psi3


def compute_obs_psi(
    obs_rows: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    """Observed PSI5 and PSI3 from raw STAR counts.

    Returns (obs_by_da, donor_totals, acceptor_totals) as dicts keyed by 1-based positions.
    """
    obs_by_da: dict[tuple[int, int], int] = {}
    donor_total: dict[int, int] = {}
    acceptor_total: dict[int, int] = {}

    for _, row in obs_rows.iterrows():
        d = int(row["donor_pos"])
        a = int(row["acceptor_pos"])
        cnt = int(row["n_uniquely_mapped_reads"])
        obs_by_da[(d, a)] = obs_by_da.get((d, a), 0) + cnt
        donor_total[d] = donor_total.get(d, 0) + cnt
        acceptor_total[a] = acceptor_total.get(a, 0) + cnt

    return obs_by_da, donor_total, acceptor_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}  dtype: {}".format(device, args.dtype))

    # --- Load model ---
    print("Loading model...")
    model, track_names = load_finetuned_model(
        args.pretrained_weights, args.checkpoint, device, dtype=args.dtype
    )
    n_rna_tracks = len(track_names["rna_seq"])
    n_ssu_tracks = len(track_names["splice_usage"])
    n_junc_samples = len(track_names["splice_junctions"]) // 2
    print("  rna_seq tracks={}, ssu_tracks={}, junc_samples={}".format(n_rna_tracks, n_ssu_tracks, n_junc_samples))

    # Strand-matched (track_idx, bw_idx) per strand:
    # Bigwigs ordered [s0/forward, s0/reverse, s1/forward, s1/reverse, ...]
    # Track ordering mirrors bigwig ordering.
    forward_track_indices = list(range(0, n_rna_tracks, 2))   # 0, 2, ...
    reverse_track_indices = list(range(1, n_rna_tracks, 2))   # 1, 3, ...

    # --- Load genome ---
    print("Loading genome...")
    fasta = pyfaidx.Fasta(args.genome)

    # --- Load bigwigs ---
    bigwigs = [pyBigWig.open(p) for p in args.bigwigs]
    # Build chrom sizes from first bigwig
    chrom_sizes: dict[str, int] = {}
    if bigwigs:
        for chrom, size in bigwigs[0].chroms().items():
            chrom_sizes[chrom] = size

    # --- Load GTF exons ---
    print("Loading GTF exons...")
    gtf = pd.read_parquet(
        args.gtf_parquet,
        columns=["Chromosome", "Start", "End", "Strand", "Feature", "gene_id", "gene_name"],
    )
    exons = (
        gtf[gtf["Feature"] == "exon"][
            ["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    exons_by_gene: dict[str, pd.DataFrame] = {
        gid: df for gid, df in exons.groupby("gene_id")
    }
    # GTF splice sites set: {(chrom, pos_1based, strand, role)}
    # Donor (+): End (0-based exclusive = last exon base 0-based = End-1, 1-based = End)
    # Acceptor (+): Start (0-based = first exon base 0-based, 1-based = Start+1)
    # For - strand: donor is Start+1 (1-based) and acceptor is End (1-based) — mirrors STAR convention
    gtf_splice_sites: set[tuple[str, int, str, str]] = set()
    for _, ex in exons.iterrows():
        chrom, start, end, strand = ex["Chromosome"], int(ex["Start"]), int(ex["End"]), ex["Strand"]
        if strand == "+":
            gtf_splice_sites.add((chrom, end, "+", "donor"))       # 1-based End = last exon base +1 ...
            gtf_splice_sites.add((chrom, start + 1, "+", "acceptor"))  # 1-based Start+1
        else:
            gtf_splice_sites.add((chrom, start + 1, "-", "donor"))
            gtf_splice_sites.add((chrom, end, "-", "acceptor"))

    # --- Load STAR junctions ---
    print("Loading STAR junctions...")
    assert len(args.star_junctions) == len(args.samples), \
        "--star-junctions and --samples must have the same length"

    star_per_sample: list[pd.DataFrame] = [
        read_star_junctions(path, sid, idx)
        for idx, (path, sid) in enumerate(zip(args.star_junctions, args.samples))
    ]
    star_all = pd.concat(star_per_sample, ignore_index=True)
    # Merged across samples for building position tensors
    star_merged = star_all.drop_duplicates(
        subset=["chrom", "donor_pos", "acceptor_pos", "strand"]
    ).reset_index(drop=True)

    # --- Load SSU parquets ---
    print("Loading SSU parquets...")
    assert len(args.ssu_parquets) == len(args.samples)
    ssu_per_sample: list[pd.DataFrame] = []
    for idx, (path, sid) in enumerate(zip(args.ssu_parquets, args.samples)):
        df = pd.read_parquet(
            path,
            columns=["chrom", "strand", "role", "exon_pos", "ssu_spliser"],
        )
        df = df[df["ssu_spliser"].notna()].reset_index(drop=True)
        df["sample_id"] = sid
        df["sample_idx"] = idx
        ssu_per_sample.append(df)
    ssu_all = pd.concat(ssu_per_sample, ignore_index=True)

    # Union of SSU positions (across samples) for splice site classification
    ssu_positions = ssu_all.drop_duplicates(subset=["chrom", "exon_pos", "strand", "role"])

    # --- Load test intervals ---
    test_intervals = pd.read_csv(
        args.test_bed, sep="\t", header=None, names=["chrom", "start", "end"]
    )
    n_intervals = len(test_intervals)
    print("Test intervals: {}".format(n_intervals))

    # --- Gene-interval assignment ---
    print("Computing gene-interval assignments...")
    gene_interval_map = compute_gene_interval_map(exons, test_intervals)
    print("  Genes assigned: {}".format(len(gene_interval_map)))

    # Build iv_idx → list of gene_ids for fast lookup
    from collections import defaultdict
    genes_per_interval: dict[int, list[str]] = defaultdict(list)
    for gid, (iv_idx, _, _) in gene_interval_map.items():
        genes_per_interval[iv_idx].append(gid)

    # --- Inference loop ---
    rna_rows: list[dict] = []
    splice_site_rows: list[dict] = []
    ssu_rows: list[dict] = []
    junction_rows: list[dict] = []
    junction_total_rows: list[dict] = []
    psi_rows: list[dict] = []

    for iv_idx, iv_row in test_intervals.iterrows():
        chrom = iv_row["chrom"]
        iv_start = int(iv_row["start"])
        iv_end = int(iv_row["end"])

        if iv_idx % 100 == 0:
            print("  Interval {}/{}: {}:{}-{}".format(iv_idx + 1, n_intervals, chrom, iv_start, iv_end))

        window_start, window_end = pad_interval(iv_start, iv_end, args.sequence_length)
        seq_len = window_end - window_start

        # Load sequence
        raw_seq = str(fasta[chrom][max(0, window_start):window_end]).upper()
        if window_start < 0:
            raw_seq = "N" * (-window_start) + raw_seq
        seq_tensor = (
            torch.from_numpy(sequence_to_onehot(raw_seq))
            .float()
            .unsqueeze(0)
            .to(device)
        )

        # Build annotated junction positions
        positions_np = build_annotated_positions(
            star_merged, chrom, window_start, seq_len, iv_start, iv_end
        )
        positions_t = torch.from_numpy(positions_np).long().unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model.predict(
                seq_tensor, organism_index=0, splice_site_positions=positions_t
            )

        # Extract outputs to CPU numpy
        rna_pred = outputs["rna_seq"][1].squeeze(0).cpu().float().numpy()         # (seq_len, n_rna_tracks)
        cls_probs = outputs["splice_sites_classification"]["probs"].squeeze(0).cpu().float().numpy()  # (seq_len, 5)
        usage_pred = outputs["splice_sites_usage"]["predictions"].squeeze(0).cpu().float().numpy()    # (seq_len, n_ssu_tracks)
        pred_counts = outputs["splice_sites_junction"]["pred_counts"].squeeze(0).cpu().float().numpy()  # (K, K, 2*n_junc_samples)

        # Build position→index lookups
        pos_lookup: list[dict[int, int]] = []
        for role in range(4):
            pos_lookup.append({
                int(positions_np[role, i]): i
                for i in range(positions_np.shape[1])
                if positions_np[role, i] >= 0
            })

        # ── RNA-seq gene expression ────────────────────────────────────────
        for gid in genes_per_interval.get(iv_idx, []):
            gene_exons = exons_by_gene.get(gid)
            if gene_exons is None:
                continue
            _, g_strand, g_name = gene_interval_map[gid]

            # Strand-matched track indices
            if g_strand == "+":
                track_indices = forward_track_indices
            else:
                track_indices = reverse_track_indices

            pred_means = get_exon_mean_pred(rna_pred, gene_exons, window_start, seq_len)
            obs_means = get_exon_mean_obs(bigwigs, gene_exons, chrom, chrom_sizes)
            if pred_means is None or obs_means is None:
                continue

            for t_idx in track_indices:
                if t_idx >= len(args.bigwigs):
                    continue
                # t_idx is 0,2,4,... for forward or 1,3,5,... for reverse;
                # both map to sample index 0,1,2,... via integer division by 2.
                sample_idx_for_track = t_idx // 2
                sample_id = args.samples[sample_idx_for_track]
                rna_rows.append({
                    "gene_id": gid,
                    "gene_name": g_name,
                    "chrom": chrom,
                    "strand": g_strand,
                    "interval_idx": iv_idx,
                    "track_idx": t_idx,
                    "track_name": sample_id,
                    "pred_log_mean": float(np.log1p(pred_means[t_idx])),
                    "obs_log_mean": float(np.log1p(obs_means[t_idx])),
                })

        # ── Splice site classification ─────────────────────────────────────
        # Collect every annotated splice site position in this interval.
        # Negatives for each class are the other annotated classes (not background),
        # matching the publication's auPRC definition.
        iv_ssu_pos = ssu_positions[
            (ssu_positions["chrom"] == chrom)
            & (ssu_positions["exon_pos"] > window_start)
            & (ssu_positions["exon_pos"] <= window_end)
        ]

        for _, ssu_row in iv_ssu_pos.iterrows():
            pos_1based = int(ssu_row["exon_pos"])
            rel_pos = pos_1based - 1 - window_start
            if not (0 <= rel_pos < seq_len):
                continue
            probs = cls_probs[rel_pos]
            splice_site_rows.append({
                "chrom": chrom,
                "pos_1based": pos_1based,
                "strand": ssu_row["strand"],
                "role": ssu_row["role"],
                "pred_donor_pos": float(probs[0]),
                "pred_acceptor_pos": float(probs[1]),
                "pred_donor_neg": float(probs[2]),
                "pred_acceptor_neg": float(probs[3]),
                "pred_no_site": float(probs[4]),
                "label_rnaseq": 1,
                "label_gtf": int(
                    (chrom, pos_1based, ssu_row["strand"], ssu_row["role"]) in gtf_splice_sites
                ),
            })

        # ── SSU predictions ────────────────────────────────────────────────
        iv_ssu_all = ssu_all[
            (ssu_all["chrom"] == chrom)
            & (ssu_all["exon_pos"] > window_start)
            & (ssu_all["exon_pos"] <= window_end)
        ]
        for _, ssu_row in iv_ssu_all.iterrows():
            rel_pos = int(ssu_row["exon_pos"]) - 1 - window_start
            if not (0 <= rel_pos < seq_len):
                continue
            s_idx = int(ssu_row["sample_idx"])
            strand = ssu_row["strand"]
            # Track layout: [s0_pos, s1_pos, ..., s0_neg, s1_neg, ...]
            t_idx = s_idx if strand == "+" else n_ssu_tracks // 2 + s_idx
            ssu_rows.append({
                "chrom": chrom,
                "exon_pos_1based": int(ssu_row["exon_pos"]),
                "strand": strand,
                "role": ssu_row["role"],
                "sample_id": ssu_row["sample_id"],
                "pred_ssu": float(usage_pred[rel_pos, t_idx]),
                "obs_ssu": float(ssu_row["ssu_spliser"]),
            })

        # ── Junction predictions ───────────────────────────────────────────
        iv_star = star_all[
            (star_all["chrom"] == chrom)
            & (star_all["donor_pos"] > iv_start)
            & (star_all["acceptor_pos"] <= iv_end + 1)
        ]

        for strand_name, d_role, a_role, ch_offset in [("+", 0, 1, 0), ("-", 2, 3, n_junc_samples)]:
            n_d = int((positions_np[d_role] >= 0).sum())
            n_a = int((positions_np[a_role] >= 0).sum())
            if n_d == 0 or n_a == 0:
                continue

            obs_s = iv_star[iv_star["strand"] == strand_name]

            for s_idx, sample_id in enumerate(args.samples):
                channel = ch_offset + s_idx
                pred_mat = pred_counts[:n_d, :n_a, channel]  # (n_d, n_a)

                # Build ground-truth matrix from STAR observations
                gt_mat = np.zeros((n_d, n_a), dtype=np.int32)
                obs_s_sample = obs_s[obs_s["sample_idx"] == s_idx]
                for _, jrow in obs_s_sample.iterrows():
                    d_rel = int(jrow["donor_pos"]) - 1 - window_start
                    a_rel = int(jrow["acceptor_pos"]) - 1 - window_start
                    di = pos_lookup[d_role].get(d_rel)
                    ai = pos_lookup[a_role].get(a_rel)
                    if di is not None and ai is not None and di < n_d and ai < n_a:
                        gt_mat[di, ai] = int(jrow["n_uniquely_mapped_reads"])

                # Store pairs with pred > 0 OR obs > 0; record n_total for auPRC denominator
                n_total = n_d * n_a
                informative = (pred_mat > 0) | (gt_mat > 0)
                d_indices, a_indices = np.where(informative)

                d_pos_arr = positions_np[d_role, :n_d]  # 0-based relative positions
                a_pos_arr = positions_np[a_role, :n_a]

                for di, ai in zip(d_indices.tolist(), a_indices.tolist()):
                    junction_rows.append({
                        "interval_idx": int(iv_idx),
                        "chrom": chrom,
                        "donor_pos_1based": window_start + int(d_pos_arr[di]) + 1,
                        "acceptor_pos_1based": window_start + int(a_pos_arr[ai]) + 1,
                        "strand": strand_name,
                        "sample_id": sample_id,
                        "pred_count": float(pred_mat[di, ai]),
                        "obs_count": int(gt_mat[di, ai]),
                    })

                junction_total_rows.append({
                    "interval_idx": int(iv_idx),
                    "chrom": chrom,
                    "strand": strand_name,
                    "sample_id": sample_id,
                    "n_valid_pairs": n_total,
                })

        # ── PSI (chr2 only) ────────────────────────────────────────────────
        if chrom == "chr2":
            for s_idx, sample_id in enumerate(args.samples):
                for strand_name, d_role, a_role, ch_offset in [("+", 0, 1, 0), ("-", 2, 3, n_junc_samples)]:
                    n_d = int((positions_np[d_role] >= 0).sum())
                    n_a = int((positions_np[a_role] >= 0).sum())
                    if n_d == 0 or n_a == 0:
                        continue
                    counts_mat = pred_counts[:n_d, :n_a, ch_offset + s_idx]  # (n_d, n_a)
                    pred_psi5, pred_psi3 = compute_psi_from_matrix(counts_mat)

                    # Observed PSI from STAR for this sample/strand/interval
                    obs_rows_s = iv_star[
                        (iv_star["strand"] == strand_name)
                        & (iv_star["sample_idx"] == s_idx)
                    ]
                    obs_by_da, donor_total, acceptor_total = compute_obs_psi(obs_rows_s)

                    for (d_1, a_1), obs_cnt in obs_by_da.items():
                        if obs_cnt == 0:
                            continue
                        d_rel = d_1 - 1 - window_start
                        a_rel = a_1 - 1 - window_start
                        d_idx = pos_lookup[d_role].get(d_rel)
                        a_idx = pos_lookup[a_role].get(a_rel)
                        if d_idx is None or a_idx is None:
                            continue

                        psi_rows.append({
                            "chrom": chrom,
                            "donor_pos_1based": d_1,
                            "acceptor_pos_1based": a_1,
                            "strand": strand_name,
                            "sample_id": sample_id,
                            "pred_psi5": float(pred_psi5[d_idx, a_idx]),
                            "obs_psi5": float(obs_cnt / (donor_total[d_1] + _EPS)),
                            "pred_psi3": float(pred_psi3[d_idx, a_idx]),
                            "obs_psi3": float(obs_cnt / (acceptor_total[a_1] + _EPS)),
                        })

    # --- Write parquets ---
    print("Writing parquets...")
    kw = dict(index=False, compression="zstd")

    pd.DataFrame(rna_rows).to_parquet(
        os.path.join(args.output_dir, "rna_seq_per_gene.parquet"), **kw
    )
    pd.DataFrame(splice_site_rows).to_parquet(
        os.path.join(args.output_dir, "splice_site_scores.parquet"), **kw
    )
    pd.DataFrame(ssu_rows).to_parquet(
        os.path.join(args.output_dir, "ssu_scores.parquet"), **kw
    )
    # junction_scores: pairs with pred>0 OR obs>0 (informative subset of full K×K)
    pd.DataFrame(junction_rows).to_parquet(
        os.path.join(args.output_dir, "junction_scores.parquet"), **kw
    )
    # junction_totals: n_valid_pairs per interval/strand/sample for correct auPRC denominator
    pd.DataFrame(junction_total_rows).to_parquet(
        os.path.join(args.output_dir, "junction_totals.parquet"), **kw
    )
    pd.DataFrame(psi_rows).to_parquet(
        os.path.join(args.output_dir, "psi_scores.parquet"), **kw
    )

    for bw in bigwigs:
        bw.close()

    print("\nDone. Outputs written to {}".format(args.output_dir))
    print("  rna_seq rows: {}".format(len(rna_rows)))
    print("  splice_site rows: {}".format(len(splice_site_rows)))
    print("  ssu rows: {}".format(len(ssu_rows)))
    print("  junction rows (pred>0 or obs>0): {}".format(len(junction_rows)))
    print("  psi rows: {}".format(len(psi_rows)))


if __name__ == "__main__":
    main()
