#!/usr/bin/env python
"""
Collect finetuned AlphaGenome (JAX/alphagenome_ft) predictions on held-out
test intervals.

JAX analog of src/scripts/collect_predictions.py (the PyTorch version). Runs
single-GPU inference on every interval in --test-bed and writes the same
parquet outputs consumed by src/scripts/compute_eval_metrics.py. All
framework-agnostic post-processing (gene/exon assignment, profile-correlation
accumulators, PSI/junction bookkeeping) is imported directly from
collect_predictions.py rather than duplicated — only model loading and the
per-interval forward pass differ (torch -> alphagenome_ft/JAX).

Checkpoint is an orbax directory saved by
workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py (e.g.
.../checkpoints/jax/probing_epoch10/last), not a single-file .pth like the
PyTorch checkpoints.

Resumable: writes per-interval partial results under
<output-dir>/.progress/ (not a declared Snakemake output — see this repo's
CLAUDE.md note on never declaring resumable state as a rule's output) so a
run killed by a SLURM time limit can pick back up close to where it left off
on the next invocation, before the final parquets are written.

Usage:
    python workflows/10-jax_finetuning/scripts/collect_predictions_jax.py \\
        --checkpoint-path-file data/raw/.../kaggle_fold_1/checkpoint_path.txt \\
        --checkpoint-dir results/xinming/.../checkpoints/jax/probing_epoch10/last \\
        --mode linear-probe \\
        --test-bed data/prep/finetuning/alphagenome/FOLD_1/test.bed \\
        --train-bed data/prep/finetuning/alphagenome/FOLD_1/train.bed \\
        --track-means-samples 1000 \\
        --genome data/raw/GENCODE/release_46/GRCh38.primary_assembly.genome.fa.gz \\
        --gtf-parquet data/raw/GENCODE/release_46/gencode.v46.annotation.gtf.parquet \\
        --bigwigs bw1.bw bw2.bw bw3.bw bw4.bw \\
        --ssu-parquets ssu1.parquet ssu2.parquet \\
        --star-junctions sj1.tab sj2.tab \\
        --samples SRR17111303 SRR17111311 \\
        --max-splice-sites 256 \\
        --output-dir results/evaluation/.../predictions
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyBigWig
import pyfaidx

# --- Reuse framework-agnostic helpers from the PyTorch script -------------
# collect_predictions.py imports torch/alphagenome_pytorch at module level;
# both are present in envs/alphagenome.yaml (this script's own conda env),
# so this import is safe even though we never use torch ourselves.
_SRC_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "src", "scripts",
)
sys.path.insert(0, _SRC_SCRIPTS_DIR)
from collect_predictions import (  # noqa: E402
    read_star_junctions,
    pad_interval,
    centered_window,
    build_annotated_positions,
    merge_gene_exons,
    compute_gene_interval_map,
    get_exon_mean_pred,
    get_exon_mean_pred_binned,
    get_exon_mean_obs,
    compute_psi_from_matrix,
    compute_obs_psi,
    ProfileCorrAccumulator,
    pearson_r_per_track,
    pearson_r_pooled,
    jsd_per_track,
    fetch_obs_window,
    accumulator_to_df,
)

_EPS = 1e-8
_CENTRAL_LENGTH = 196_608
_CENTRAL_BINS_32BP = _CENTRAL_LENGTH // 32  # 6144


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint-path-file", required=True,
                   help="Text file containing the absolute path to the base Kaggle "
                        "AlphaGenome-JAX checkpoint (see finetune_alphagenome_jax.py's "
                        "download_alphagenome_jax_weights rule).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Orbax finetuned checkpoint directory (e.g. .../probing_epoch10/last).")
    p.add_argument("--mode", choices=["linear-probe", "lora"], default="linear-probe")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", default="q_proj,v_proj")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None,
                    help="Compute dtype for heads (must match what the checkpoint "
                         "was actually trained under). Default: auto-detected from "
                         "config.json alongside --checkpoint-dir's parent (written "
                         "by finetune_alphagenome_jax.py); if that file doesn't "
                         "exist (e.g. a checkpoint trained before this dtype-"
                         "selection feature existed, when heads always silently "
                         "ran float32 regardless of any setting), falls back to "
                         "float32 to match that historical behavior, not bfloat16. "
                         "Pass this flag explicitly only to deliberately override.")
    p.add_argument("--test-bed", required=True)
    p.add_argument("--train-bed", required=True,
                   help="Training-fold BED used to compute each rna_seq track's "
                        "nonzero_mean, the same way finetune_alphagenome_jax.py "
                        "does at training time (must match exactly -- same bed, "
                        "same bigwigs, same --sequence-length, same "
                        "--track-means-samples -- since track_means is a plain "
                        "Haiku array, not a checkpointed parameter, and must be "
                        "recomputed identically at eval time or every rna_seq "
                        "prediction is silently rescaled by the wrong factor).")
    p.add_argument("--track-means-samples", type=int, default=None,
                   help="Must match the --track-means-samples value used to train "
                        "this checkpoint (config['finetuning']['alphagenome_ft']"
                        "['sf3b1mut']['track_means_samples']). Default None means "
                        "'all' train-bed windows, matching finetune_alphagenome_jax.py.")
    p.add_argument("--genome", required=True)
    p.add_argument("--gtf-parquet", required=True)
    p.add_argument("--bigwigs", nargs="+", required=True,
                   help="Bigwig files in order: sample0/forward, sample0/reverse, sample1/forward, ...")
    p.add_argument("--ssu-parquets", nargs="+", required=True,
                   help="SSU parquet files, one per sample, same order as --bigwigs pairs")
    p.add_argument("--star-junctions", nargs="+", required=True,
                   help="STAR SJ.out.tab files, one per sample")
    p.add_argument("--samples", nargs="+", required=True,
                   help="Sample IDs, matching order of --ssu-parquets / --star-junctions")
    p.add_argument("--sequence-length", type=int, default=1_048_576)
    p.add_argument("--max-splice-sites", type=int, default=256,
                   help="Must match the value used to train this checkpoint "
                        "(config['finetuning']['alphagenome_ft']['sf3b1mut']['max_splice_sites']).")
    p.add_argument("--junction-position-source", choices=["annotated", "predicted"], default="annotated")
    p.add_argument("--organism", default="HOMO_SAPIENS")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (kept for reproducibility of any future sampling)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (JAX/alphagenome_ft)
# ---------------------------------------------------------------------------

def load_finetuned_model_jax(
    checkpoint_dir: str,
    base_checkpoint_path: str,
    mode: str,
    sequence_length: int,
    star_junctions: list[str],
    max_splice_sites: int,
    junction_position_source: str,
    bigwigs: list[str],
    track_means: list[float],
    ssu_parquets: list[str] | None = None,
    gtf_parquet: str | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_targets: str = "q_proj,v_proj",
    dtype: str = "bfloat16",
):
    """Reconstruct the finetuned JAX model from an orbax checkpoint directory.

    Mirrors workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py's
    main() model-construction path (same 4 custom heads, same LoRA backbone
    patch timing).

    Does NOT use `alphagenome_ft.load_checkpoint()` for the restore step.
    xinming's checkpoints were trained with alphagenome-ft==0.1.7 (see
    results/xinming/.../metadata/environment/uv.alphagenome-ft.lock), but
    this env has 0.1.9 installed (editable, from the sibling repo's
    splice-finetuning branch). Commit 211f613 ("Add real backbone LoRA
    support...") added the install_backbone_patches/LoRA machinery but never
    updated CustomAlphaGenomeModel._checkpoint_slice_trees / the
    load_checkpoint()-internal merge_head_params helper (both in
    alphagenome_ft/custom_model.py) to know about q_lora/v_lora params — so
    for a LoRA run, load_checkpoint()'s restore_target never asks for them
    (orbax then errors: "Source: MISSING" for every mha_block*/{q,v}_lora
    leaf, since they exist on disk but aren't in the requested target), and
    even if that were bypassed, its merge step only copies back head/*
    keys, silently dropping the LoRA weights. The Haiku scope naming itself
    is unaffected by the version skew (confirmed identical between the
    checkpoint's on-disk metadata and this env's freshly-built template),
    so instead we build the template ourselves via create_model_with_heads
    (same call load_checkpoint() makes internally) and restore+merge with
    no key filtering at all: request `target=None` (accepts whatever
    structure is on disk, skipping orbax's target-vs-source structural
    validation) and unconditionally `dict.update()` every returned flat key
    into the template's params/state. This has no knowledge of "heads" vs
    "lora" vs anything else — it just overwrites whatever the checkpoint
    actually contains (heads and, for LoRA runs, backbone LoRA deltas
    together), leaving the rest of the freshly-initialized template alone.
    """
    from alphagenome_ft import create_model_with_heads
    from alphagenome_ft import lora as lora_lib
    from alphagenome_ft.finetune import config as ft_config
    from alphagenome_ft.finetune.train import register_predefined_heads

    lora_enabled = mode == "lora"
    detach_backbone = not lora_enabled
    install_backbone_patches = None
    if lora_enabled:
        lora_cfg = lora_lib.BackboneLoRAConfig.from_pytorch_style_targets(
            lora_targets.split(","), rank=lora_rank, alpha=lora_alpha,
        )

        def install_backbone_patches() -> None:
            lora_lib.install_mha_backbone_lora(lora_cfg)

    head_ids = {
        "splice_sites_classification": "splice_site",
        "splice_sites_usage": "splice_usage",
        "splice_sites_junction": "splice_junctions",
    }

    heads_cfg = []
    for kind, head_id in head_ids.items():
        entry = {
            "id": head_id,
            "source": "predefined",
            "kind": kind,
            "star_junctions": star_junctions,
            "max_splice_sites": max_splice_sites,
        }
        if ssu_parquets is not None:
            entry["ssu"] = ssu_parquets
        if gtf_parquet is not None:
            entry["gtf"] = gtf_parquet
        if kind == "splice_sites_junction":
            entry["junction_position_source"] = junction_position_source
            if junction_position_source == "predicted":
                entry["classification_head_id"] = head_ids["splice_sites_classification"]
        heads_cfg.append(entry)

    # rna_seq head targets: num_tracks is derived as len(targets) by
    # ft_config.prepare_head_specs (each entry must point to a real,
    # existing file — see _parse_targets), so this must have exactly one
    # entry per bigwig for the head shape to match the checkpoint's real
    # n_rna_tracks.
    #
    # nonzero_mean must be passed here, matching finetune_alphagenome_jax.py's
    # own heads_cfg construction exactly. track_means is a plain Haiku array
    # (not hk.get_parameter), so it's never part of _params/_state and never
    # reaches the orbax checkpoint -- config.py's prepare_head_specs falls
    # back to jnp.ones(num_tracks) (heads.py::_get_track_means) whenever no
    # target carries nonzero_mean, which silently rescales every rna_seq
    # prediction by the wrong per-track factor at inference. See caller for
    # how track_means is (re)computed identically to training time.
    heads_cfg.append({
        "id": "rna_seq",
        "source": "predefined",
        "kind": "rna_seq",
        "targets": [
            {"path": str(bw), "nonzero_mean": mean}
            for bw, mean in zip(bigwigs, track_means)
        ],
        "resolutions": [1],
    })

    specs = ft_config.prepare_head_specs({"heads": heads_cfg}, organism="HOMO_SAPIENS")
    ft_config.validate_head_specs(specs)
    register_predefined_heads(specs)

    custom_head_ids = [spec.head_id for spec in specs]
    model = create_model_with_heads(
        heads=custom_head_ids,
        checkpoint_path=base_checkpoint_path,
        init_seq_len=sequence_length,
        detach_backbone=detach_backbone,
        gradient_checkpointing=False,
        install_backbone_patches=install_backbone_patches,
        dtype=dtype,
    )

    import jax
    import orbax.checkpoint as ocp
    from pathlib import Path

    checkpoint_path = Path(checkpoint_dir).resolve() / "checkpoint"
    checkpointer = ocp.StandardCheckpointer()
    restore_result = checkpointer.restore(str(checkpoint_path), target=None)
    if isinstance(restore_result, (tuple, list)) and len(restore_result) == 2:
        loaded_params, loaded_state = restore_result
    elif isinstance(restore_result, (tuple, list)) and len(restore_result) == 1:
        loaded_params, loaded_state = restore_result[0], {}
    else:
        loaded_params, loaded_state = restore_result, {}

    if isinstance(loaded_params, dict):
        model._params.update(loaded_params)
    if isinstance(loaded_state, dict) and loaded_state:
        if not isinstance(model._state, dict):
            model._state = {}
        model._state.update(loaded_state)

    if dtype == "bfloat16":
        # loaded_params (real values restored from the orbax checkpoint file)
        # unconditionally overwrite the fresh create_model_with_heads(...,
        # dtype="bfloat16") call's already-correctly-cast head params above
        # -- if the checkpoint was saved under a different dtype (e.g. any
        # checkpoint trained before this dtype-selection feature existed,
        # when heads always silently ran float32), that overwrite silently
        # reverts heads back to float32 regardless of the dtype requested
        # here. Re-cast after the update so the requested dtype always wins.
        from alphagenome_ft.custom_model import _cast_head_params_to_bfloat16
        model._params = _cast_head_params_to_bfloat16(model._params)

    device = model._device_context._device
    model._params = jax.device_put(model._params, device)
    model._state = jax.device_put(model._state, device)

    return model


def run_forward_pass(model, seq_arr, organism_index_value, positions_np):
    """One-interval forward pass through the JAX finetuned model.

    Returns (rna_pred, cls_probs, usage_pred, pred_counts) as numpy arrays,
    matching the shapes produced by collect_predictions.py's torch path:
      rna_pred:    (seq_len, n_rna_tracks)
      cls_probs:   (seq_len, 5)
      usage_pred:  (seq_len, n_ssu_tracks)
      pred_counts: (K, K, 2*n_junc_samples)
    """
    import jax
    import jax.numpy as jnp

    seq_batch = jnp.asarray(seq_arr)[None]  # (1, seq_len, 4)
    organism_index = jnp.full((1,), organism_index_value, dtype=jnp.int32)
    negative_strand_mask = jnp.zeros((1,), dtype=jnp.bool_)
    strand_reindexing = None
    if hasattr(model._base_model, "_metadata"):
        first_org = list(model._base_model._metadata.keys())[0]
        strand_reindexing = jax.device_put(
            model._base_model._metadata[first_org].strand_reindexing
        )
    if strand_reindexing is None:
        strand_reindexing = jnp.array([], dtype=jnp.int32)
    positions_batch = jnp.asarray(positions_np, dtype=jnp.int32)[None]  # (1, 4, K)

    predictions = model._predict(
        model._params,
        model._state,
        seq_batch,
        organism_index,
        negative_strand_mask=negative_strand_mask,
        strand_reindexing=strand_reindexing,
        splice_site_positions=positions_batch,
    )

    rna_pred = np.asarray(predictions["rna_seq"]["predictions_1bp"])[0]
    cls_probs = np.asarray(predictions["splice_site"]["predictions"])[0]
    usage_pred = np.asarray(predictions["splice_usage"]["predictions"])[0]
    pred_counts = np.asarray(predictions["splice_junctions"]["predictions"])[0]
    return rna_pred, cls_probs, usage_pred, pred_counts


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _progress_dir(output_dir: str) -> str:
    d = os.path.join(output_dir, ".progress")
    os.makedirs(d, exist_ok=True)
    return d


def _load_progress(output_dir: str):
    """Return (done_interval_indices: set[int], row_lists: dict, accumulators: dict) from
    any prior partial run, or empty defaults if none exists."""
    progress_dir = _progress_dir(output_dir)
    state_path = os.path.join(progress_dir, "state.pkl")
    if not os.path.exists(state_path):
        return set(), None
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return set(state["done_intervals"]), state


def _save_progress(output_dir: str, done_intervals: set, row_lists: dict, accumulators: dict) -> None:
    progress_dir = _progress_dir(output_dir)
    state_path = os.path.join(progress_dir, "state.pkl")
    tmp_path = state_path + ".tmp"
    state = {
        "done_intervals": done_intervals,
        "row_lists": row_lists,
        "accumulators": accumulators,
    }
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, state_path)  # atomic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.checkpoint_path_file) as f:
        base_checkpoint_path = f.read().strip()

    if args.dtype is None:
        # --checkpoint-dir is the orbax "last"/"best" leaf; config.json (if
        # this checkpoint was trained after the dtype-selection feature
        # landed) lives one level up, alongside it -- see
        # finetune_alphagenome_jax.py's config.json write.
        config_path = os.path.join(os.path.dirname(os.path.normpath(args.checkpoint_dir)), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                trained_dtype = json.load(f).get("dtype")
            args.dtype = trained_dtype or "float32"
            print("Auto-detected dtype={} from {}".format(args.dtype, config_path))
        else:
            args.dtype = "float32"
            print("No config.json found at {} -- this checkpoint predates the "
                  "dtype-selection feature, when heads always silently ran "
                  "float32 regardless of any setting. Defaulting eval dtype to "
                  "float32 to match that historical training behavior (not "
                  "bfloat16). Pass --dtype explicitly to override.".format(config_path))

    # rna_seq track_means must be recomputed exactly as at training time (see
    # load_finetuned_model_jax's rna_seq heads_cfg comment) -- same bigwigs,
    # same bed, same sequence_length, same subsetting, or every rna_seq
    # prediction below is silently rescaled by the wrong per-track factor.
    from pathlib import Path as _Path
    from finetune_alphagenome_jax import _compute_track_means
    track_means = _compute_track_means(
        args.bigwigs, _Path(args.train_bed), args.sequence_length, args.track_means_samples,
    )

    # --- Load model ---
    print("Loading JAX model from checkpoint: {}".format(args.checkpoint_dir))
    model = load_finetuned_model_jax(
        args.checkpoint_dir,
        base_checkpoint_path,
        args.mode,
        args.sequence_length,
        args.star_junctions,
        args.max_splice_sites,
        args.junction_position_source,
        args.bigwigs,
        track_means,
        ssu_parquets=args.ssu_parquets,
        gtf_parquet=None,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        dtype=args.dtype,
    )

    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model
    from alphagenome_research.model import one_hot_encoder

    organism_enum = getattr(ag_dna_model.Organism, args.organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

    n_rna_tracks = len(args.bigwigs)
    n_ssu_tracks = 2 * len(args.samples)
    n_junc_samples = len(args.samples)
    print("  rna_seq tracks={}, ssu_tracks={}, junc_samples={}".format(
        n_rna_tracks, n_ssu_tracks, n_junc_samples))

    # Profile correlation accumulators (online Pearson r across all test intervals)
    acc_names = [
        "acc_exon_1bp", "acc_full_1bp", "acc_exon_32bp", "acc_full_32bp",
        "acc_central_1bp", "acc_central_32bp",
    ]

    forward_track_indices = list(range(0, n_rna_tracks, 2))
    reverse_track_indices = list(range(1, n_rna_tracks, 2))

    # --- Load genome ---
    print("Loading genome...")
    fasta = pyfaidx.Fasta(args.genome)

    # --- Load bigwigs ---
    bigwigs = [pyBigWig.open(p) for p in args.bigwigs]
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
    exons["Strand"] = exons["Strand"].astype(str)
    exons = merge_gene_exons(exons)
    exons_by_gene: dict[str, pd.DataFrame] = {
        gid: df for gid, df in exons.groupby("gene_id")
    }
    gtf_splice_sites: set[tuple[str, int, str, str]] = set()
    for _, ex in exons.iterrows():
        chrom, start, end, strand = ex["Chromosome"], int(ex["Start"]), int(ex["End"]), ex["Strand"]
        if strand == "+":
            gtf_splice_sites.add((chrom, end, "+", "donor"))
            gtf_splice_sites.add((chrom, start + 1, "+", "acceptor"))
        else:
            gtf_splice_sites.add((chrom, start + 1, "-", "donor"))
            gtf_splice_sites.add((chrom, end, "-", "acceptor"))

    # --- Load STAR junctions ---
    print("Loading STAR junctions...")
    from collect_predictions import normalize_junctions_per_sample  # re-exported from alphagenome_pytorch dep

    assert len(args.star_junctions) == len(args.samples)
    star_per_sample: list[pd.DataFrame] = []
    for idx, (path, sid) in enumerate(zip(args.star_junctions, args.samples)):
        junc = read_star_junctions(path, sid, idx)
        junc = junc[junc["n_uniquely_mapped_reads"] >= 1].copy()
        junc["count"] = junc["n_uniquely_mapped_reads"].astype(float)
        junc = normalize_junctions_per_sample(junc)
        star_per_sample.append(junc)
    star_all = pd.concat(star_per_sample, ignore_index=True)
    star_merged = star_all.drop_duplicates(
        subset=["chrom", "donor_pos", "acceptor_pos", "strand"]
    ).reset_index(drop=True)

    # --- Load SSU parquets ---
    print("Loading SSU parquets...")
    assert len(args.ssu_parquets) == len(args.samples)
    ssu_per_sample: list[pd.DataFrame] = []
    for idx, (path, sid) in enumerate(zip(args.ssu_parquets, args.samples)):
        schema_cols = set(pq.read_schema(path).names)
        if "ssu_spliser" not in schema_cols:
            raise ValueError("SSU parquet {} is missing ssu_spliser column".format(path))
        df = pd.read_parquet(path, columns=["chrom", "strand", "role", "exon_pos", "alpha_juncs", "ssu_spliser"])
        df = df[df["ssu_spliser"].notna()].reset_index(drop=True)
        df["sample_id"] = sid
        df["sample_idx"] = idx
        ssu_per_sample.append(df)
    ssu_all = pd.concat(ssu_per_sample, ignore_index=True)
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

    from collections import defaultdict
    genes_per_interval: dict[int, list[str]] = defaultdict(list)
    for gid, (iv_idx, _, _) in gene_interval_map.items():
        genes_per_interval[iv_idx].append(gid)

    # --- Resume state ---
    done_intervals, prior_state = _load_progress(args.output_dir)
    if prior_state is not None:
        print("Resuming: {} / {} intervals already done.".format(len(done_intervals), n_intervals))
        row_lists = prior_state["row_lists"]
        accumulators = {name: prior_state["accumulators"][name] for name in acc_names}
    else:
        row_lists = {
            "rna_rows": [], "rna_rows_32bp": [], "splice_site_rows": [], "ssu_rows": [],
            "junction_rows": [], "junction_total_rows": [], "psi_rows": [],
            "interval_corr_rows": [], "pooled_profile_metrics_rows": [], "track_totals_rows": [],
        }
        accumulators = {name: ProfileCorrAccumulator(n_rna_tracks) for name in acc_names}

    (acc_exon_1bp, acc_full_1bp, acc_exon_32bp, acc_full_32bp,
     acc_central_1bp, acc_central_32bp) = (accumulators[name] for name in acc_names)

    rna_rows = row_lists["rna_rows"]
    rna_rows_32bp = row_lists["rna_rows_32bp"]
    splice_site_rows = row_lists["splice_site_rows"]
    ssu_rows = row_lists["ssu_rows"]
    junction_rows = row_lists["junction_rows"]
    junction_total_rows = row_lists["junction_total_rows"]
    psi_rows = row_lists["psi_rows"]
    interval_corr_rows = row_lists["interval_corr_rows"]
    pooled_profile_metrics_rows = row_lists["pooled_profile_metrics_rows"]
    track_totals_rows = row_lists["track_totals_rows"]

    _SAVE_EVERY = 200

    # --- Inference loop ---
    for iv_idx, iv_row in test_intervals.iterrows():
        if iv_idx in done_intervals:
            continue

        chrom = iv_row["chrom"]
        iv_start = int(iv_row["start"])
        iv_end = int(iv_row["end"])

        if iv_idx % 100 == 0:
            print("  Interval {}/{}: {}:{}-{}".format(iv_idx + 1, n_intervals, chrom, iv_start, iv_end))

        window_start, window_end = pad_interval(iv_start, iv_end, args.sequence_length)
        seq_len = window_end - window_start

        raw_seq = str(fasta[chrom][max(0, window_start):window_end]).upper()
        if window_start < 0:
            raw_seq = "N" * (-window_start) + raw_seq
        seq_arr = encoder.encode(raw_seq)  # (seq_len, 4)

        # build_annotated_positions is imported as-is (hardcoded to a 512-wide
        # position array internally); slice to args.max_splice_sites, which is
        # exactly equivalent to capping at that width directly since positions
        # are selected in ascending sorted order.
        positions_np_full = build_annotated_positions(
            star_merged, chrom, window_start, seq_len, iv_start, iv_end
        )
        positions_np = positions_np_full[:, : args.max_splice_sites]

        rna_pred, cls_probs, usage_pred, pred_counts = run_forward_pass(
            model, seq_arr, organism_index_value, positions_np
        )

        _n_full_bins = seq_len // 32
        rna_pred_32bp = rna_pred[:_n_full_bins * 32].reshape(_n_full_bins, 32, -1).mean(axis=1)

        obs_window = fetch_obs_window(bigwigs, chrom, window_start, window_end, chrom_sizes)
        obs_window_32bp = obs_window[:_n_full_bins * 32].reshape(_n_full_bins, 32, -1).mean(axis=1)

        acc_full_1bp.update(np.log1p(rna_pred), np.log1p(obs_window))
        acc_full_32bp.update(np.log1p(rna_pred_32bp), np.log1p(obs_window_32bp))

        interval_r_1bp = pearson_r_per_track(np.log1p(rna_pred), np.log1p(obs_window))
        interval_r_32bp = pearson_r_per_track(np.log1p(rna_pred_32bp), np.log1p(obs_window_32bp))

        central = centered_window(chrom, iv_start, iv_end, _CENTRAL_LENGTH, chrom_sizes)
        central_r_1bp = np.full(n_rna_tracks, np.nan)
        central_r_32bp = np.full(n_rna_tracks, np.nan)
        n_central_1bp = 0
        n_central_32bp = 0

        profile_pearson_track_central = np.full(n_rna_tracks, np.nan)
        jsd_track_central = np.full(n_rna_tracks, np.nan)
        profile_pearson_pooled_central = float("nan")
        jsd_pooled_central = float("nan")
        pred_sum_central = np.full(n_rna_tracks, np.nan)
        obs_sum_central = np.full(n_rna_tracks, np.nan)

        if central is not None:
            _, c_start, c_end = central
            rel_s = c_start - window_start
            rel_e = c_end - window_start
            if 0 <= rel_s and rel_e <= seq_len:
                central_pred_1bp = rna_pred[rel_s:rel_e]
                central_obs_1bp = obs_window[rel_s:rel_e]
                central_pred_32bp = central_pred_1bp.reshape(_CENTRAL_BINS_32BP, 32, -1).mean(axis=1)
                central_obs_32bp = central_obs_1bp.reshape(_CENTRAL_BINS_32BP, 32, -1).mean(axis=1)
                central_log_pred_1bp = np.log1p(central_pred_1bp)
                central_log_obs_1bp = np.log1p(central_obs_1bp)
                central_log_pred_32bp = np.log1p(central_pred_32bp)
                central_log_obs_32bp = np.log1p(central_obs_32bp)
                acc_central_1bp.update(central_log_pred_1bp, central_log_obs_1bp)
                acc_central_32bp.update(central_log_pred_32bp, central_log_obs_32bp)
                central_r_1bp = pearson_r_per_track(central_log_pred_1bp, central_log_obs_1bp)
                central_r_32bp = pearson_r_per_track(central_log_pred_32bp, central_log_obs_32bp)
                n_central_1bp = _CENTRAL_LENGTH
                n_central_32bp = _CENTRAL_BINS_32BP

                profile_pearson_track_central = pearson_r_per_track(central_pred_1bp, central_obs_1bp)
                profile_pearson_pooled_central = pearson_r_pooled(central_pred_1bp, central_obs_1bp)
                jsd_track_central = jsd_per_track(central_pred_1bp, central_obs_1bp)
                jsd_pooled_central = float(np.nanmean(jsd_track_central))
                pred_sum_central = central_pred_1bp.sum(axis=0)
                obs_sum_central = central_obs_1bp.sum(axis=0)

        profile_pearson_track_full = pearson_r_per_track(rna_pred, obs_window)
        profile_pearson_pooled_full = pearson_r_pooled(rna_pred, obs_window)
        jsd_track_full = jsd_per_track(rna_pred, obs_window)
        jsd_pooled_full = float(np.nanmean(jsd_track_full))
        pred_sum_full = rna_pred.sum(axis=0)
        obs_sum_full = obs_window.sum(axis=0)

        pooled_profile_metrics_rows.append({
            "interval_idx": int(iv_idx),
            "chrom": chrom,
            "start": iv_start,
            "end": iv_end,
            "profile_pearson_r_full": float(profile_pearson_pooled_full),
            "profile_pearson_r_central": float(profile_pearson_pooled_central),
            "jsd_full": float(jsd_pooled_full),
            "jsd_central": float(jsd_pooled_central),
            "n_positions_full": seq_len,
            "n_positions_central": n_central_1bp,
        })

        for t_idx in range(n_rna_tracks):
            sample_idx_for_track = t_idx // 2
            track_name = args.samples[sample_idx_for_track]
            strand = "forward" if t_idx % 2 == 0 else "reverse"
            interval_corr_rows.append({
                "interval_idx": int(iv_idx),
                "chrom": chrom,
                "start": iv_start,
                "end": iv_end,
                "track_idx": t_idx,
                "track_name": track_name,
                "strand": strand,
                "pearson_r_1bp": float(interval_r_1bp[t_idx]),
                "pearson_r_32bp": float(interval_r_32bp[t_idx]),
                "n_positions_1bp": seq_len,
                "n_positions_32bp": _n_full_bins,
                "pearson_r_central_1bp": float(central_r_1bp[t_idx]),
                "pearson_r_central_32bp": float(central_r_32bp[t_idx]),
                "n_positions_central_1bp": n_central_1bp,
                "n_positions_central_32bp": n_central_32bp,
                "profile_pearson_r_raw_full": float(profile_pearson_track_full[t_idx]),
                "profile_pearson_r_raw_central": float(profile_pearson_track_central[t_idx]),
                "jsd_full": float(jsd_track_full[t_idx]),
                "jsd_central": float(jsd_track_central[t_idx]),
            })
            track_totals_rows.append({
                "interval_idx": int(iv_idx),
                "chrom": chrom,
                "track_idx": t_idx,
                "track_name": track_name,
                "strand": strand,
                "pred_sum_full": float(pred_sum_full[t_idx]),
                "obs_sum_full": float(obs_sum_full[t_idx]),
                "pred_sum_central": float(pred_sum_central[t_idx]),
                "obs_sum_central": float(obs_sum_central[t_idx]),
            })

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
            track_indices = forward_track_indices if g_strand == "+" else reverse_track_indices

            pred_means = get_exon_mean_pred(rna_pred, gene_exons, window_start, seq_len)
            pred_means_32bp = get_exon_mean_pred_binned(rna_pred_32bp, gene_exons, window_start, seq_len)
            obs_means = get_exon_mean_obs(bigwigs, gene_exons, chrom, chrom_sizes, window_start, seq_len)
            if pred_means is None or obs_means is None:
                continue

            for _, ex in gene_exons.iterrows():
                rel_s = max(0, int(ex["Start"]) - window_start)
                rel_e = min(seq_len, int(ex["End"]) - window_start)
                if rel_e <= rel_s:
                    continue
                acc_exon_1bp.update(
                    np.log1p(rna_pred[rel_s:rel_e].astype(np.float64)),
                    np.log1p(obs_window[rel_s:rel_e]),
                )
                bin_s = rel_s // 32
                bin_e = min(((rel_e - 1) // 32) + 1, _n_full_bins)
                if bin_e > bin_s:
                    acc_exon_32bp.update(
                        np.log1p(rna_pred_32bp[bin_s:bin_e].astype(np.float64)),
                        np.log1p(obs_window_32bp[bin_s:bin_e]),
                    )

            for t_idx in track_indices:
                if t_idx >= len(args.bigwigs):
                    continue
                sample_idx_for_track = t_idx // 2
                sample_id = args.samples[sample_idx_for_track]
                _row = {
                    "gene_id": gid,
                    "gene_name": g_name,
                    "chrom": chrom,
                    "strand": g_strand,
                    "interval_idx": iv_idx,
                    "track_idx": t_idx,
                    "track_name": sample_id,
                    "obs_log_mean": float(np.log1p(obs_means[t_idx])),
                }
                rna_rows.append({**_row, "pred_log_mean": float(np.log1p(pred_means[t_idx]))})
                if pred_means_32bp is not None:
                    rna_rows_32bp.append({**_row, "pred_log_mean": float(np.log1p(pred_means_32bp[t_idx]))})

        # ── Splice site classification ─────────────────────────────────────
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
            t_idx = s_idx * 2 if strand == "+" else s_idx * 2 + 1
            ssu_rows.append({
                "chrom": chrom,
                "exon_pos_1based": int(ssu_row["exon_pos"]),
                "strand": strand,
                "role": ssu_row["role"],
                "sample_id": ssu_row["sample_id"],
                "alpha_juncs": int(ssu_row["alpha_juncs"]),
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
                pred_mat = pred_counts[:n_d, :n_a, channel]

                gt_mat = np.zeros((n_d, n_a), dtype=np.float32)
                gt_mat_raw = np.zeros((n_d, n_a), dtype=np.int32)
                obs_s_sample = obs_s[obs_s["sample_idx"] == s_idx]
                for _, jrow in obs_s_sample.iterrows():
                    d_rel = int(jrow["donor_pos"]) - 1 - window_start
                    a_rel = int(jrow["acceptor_pos"]) - 1 - window_start
                    di = pos_lookup[d_role].get(d_rel)
                    ai = pos_lookup[a_role].get(a_rel)
                    if di is not None and ai is not None and di < n_d and ai < n_a:
                        gt_mat[di, ai] = float(jrow["count"])
                        gt_mat_raw[di, ai] = int(jrow["n_uniquely_mapped_reads"])

                n_total = n_d * n_a
                informative = (pred_mat > 0) | (gt_mat > 0)
                d_indices, a_indices = np.where(informative)

                d_pos_arr = positions_np[d_role, :n_d]
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
                        "obs_count": float(gt_mat[di, ai]),
                        "obs_count_raw": int(gt_mat_raw[di, ai]),
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
                    counts_mat = pred_counts[:n_d, :n_a, ch_offset + s_idx]
                    pred_psi5, pred_psi3 = compute_psi_from_matrix(counts_mat)

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

        done_intervals.add(int(iv_idx))
        if len(done_intervals) % _SAVE_EVERY == 0:
            print("  Checkpointing progress: {}/{} intervals done.".format(len(done_intervals), n_intervals))
            _save_progress(args.output_dir, done_intervals, row_lists, accumulators)

    # Final progress save (covers the tail not aligned to _SAVE_EVERY)
    _save_progress(args.output_dir, done_intervals, row_lists, accumulators)

    # --- Write final parquets ---
    print("Writing parquets...")
    kw = dict(index=False, compression="zstd")

    pd.DataFrame(rna_rows).to_parquet(os.path.join(args.output_dir, "rna_seq_per_gene.parquet"), **kw)
    pd.DataFrame(rna_rows_32bp).to_parquet(os.path.join(args.output_dir, "rna_seq_per_gene_32bp.parquet"), **kw)
    pd.DataFrame(splice_site_rows).to_parquet(os.path.join(args.output_dir, "splice_site_scores.parquet"), **kw)
    pd.DataFrame(ssu_rows).to_parquet(os.path.join(args.output_dir, "ssu_scores.parquet"), **kw)
    pd.DataFrame(junction_rows).to_parquet(os.path.join(args.output_dir, "junction_scores.parquet"), **kw)
    pd.DataFrame(junction_total_rows).to_parquet(os.path.join(args.output_dir, "junction_totals.parquet"), **kw)
    pd.DataFrame(psi_rows).to_parquet(os.path.join(args.output_dir, "psi_scores.parquet"), **kw)
    accumulator_to_df(acc_exon_1bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_exon_1bp.parquet"), **kw)
    accumulator_to_df(acc_full_1bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_full_1bp.parquet"), **kw)
    accumulator_to_df(acc_exon_32bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_exon_32bp.parquet"), **kw)
    accumulator_to_df(acc_full_32bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_full_32bp.parquet"), **kw)
    accumulator_to_df(acc_central_1bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_central_1bp.parquet"), **kw)
    accumulator_to_df(acc_central_32bp, args.samples).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_central_32bp.parquet"), **kw)
    pd.DataFrame(interval_corr_rows).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_corr_per_interval.parquet"), **kw)
    pd.DataFrame(pooled_profile_metrics_rows).to_parquet(
        os.path.join(args.output_dir, "rna_seq_profile_metrics_per_interval.parquet"), **kw)
    pd.DataFrame(track_totals_rows).to_parquet(
        os.path.join(args.output_dir, "rna_seq_track_totals_per_interval.parquet"), **kw)

    for bw in bigwigs:
        bw.close()

    print("\nDone. Outputs written to {}".format(args.output_dir))
    print("  rna_seq rows (1bp): {}".format(len(rna_rows)))
    print("  splice_site rows: {}".format(len(splice_site_rows)))
    print("  ssu rows: {}".format(len(ssu_rows)))
    print("  junction rows: {}".format(len(junction_rows)))
    print("  psi rows: {}".format(len(psi_rows)))


if __name__ == "__main__":
    main()
