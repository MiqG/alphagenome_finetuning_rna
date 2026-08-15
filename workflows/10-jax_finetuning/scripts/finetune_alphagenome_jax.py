"""Splice-modality probing (frozen-backbone) finetuning with alphagenome_ft.

JAX/alphagenome_research analogue of alphagenome-pytorch's
"randinit__newloss__annotated__frozen__multigpu_ddp" paper run (see
workflows/05-full_finetuning/Snakefile and
plans/10-jax-alphagenome_ft-reproduction.md for the settings this mirrors).

Scope: reproduces all 4 modalities the PyTorch probing run trains jointly —
rna_seq (bigwig-derived) plus the three splice heads (STAR/SSU-derived) — by
combining alphagenome_ft.finetune.data.BigWigDataModule and
finetune.splice_data.SpliceDataModule via the local CombinedDataModule below.
Neither data module has ever been combined like this in alphagenome_ft
before; see plans/10-jax-alphagenome_ft-reproduction.md ("rna_seq joint
training: design") for why this is safe (identical batch schema, and
identical-by-construction window lists/order/seed make both modules' index
shuffles align in lock-step, verified by reading both iter_batches directly
rather than assumed) and for the CombinedDataModule docstring below for the
runtime safety check that would catch it immediately if that ever stopped
holding.

Gradient accumulation: alphagenome_ft.finetune.train.train() originally took
one full optimizer step per data_module batch with no accumulation loop, so it
could not reproduce the PyTorch run's effective batch of 64
(batch=1 x 4 GPUs x grad_accum=16) without risking OOM at 1Mb sequence length
on a single device. train() now accepts gradient_accumulation_steps (see
plans/10-jax-alphagenome_ft-reproduction.md) — use --batch-size 1
--gradient-accumulation-steps 64 with --num-devices 1 to match that effective
batch on this cluster's single non-MIG H100.

Junction loss note: alphagenome_research's real SpliceSitesJunctionHead.loss
has no PyTorch-style original/normalized/sparse switch — it's a single fixed
formula that already matches the PyTorch port's "normalized" variant, so
there is intentionally no --junction-loss flag here.

RoPE zero-init dead-gradient bug (--rope-init): SpliceSitesJunctionHead's RoPE
scale/offset parameter ("embeddings") is zero-initialized in the real JAX head
(hk.get_parameter(..., init=jnp.zeros)). Predicted junction counts are a
bilinear product of donor and acceptor logits, both of which are exactly zero
whenever this parameter is exactly zero — so the gradient of that product
w.r.t. either logit is proportional to the *other* logit, which is also zero.
That is a stable zero-gradient fixed point: nothing moves, ever, when training
this head from scratch. Confirmed empirically (see
plans/10-jax-alphagenome_ft-reproduction.md) — every splice_junctions
parameter, including the non-zero-initialized multi_organism_linear/w, showed
exactly zero change after 5 real training steps with a nonzero loss, while the
splice_site head's params moved normally in the same run. This is the exact
same bug alphagenome-pytorch's --rope-init flag documents and works around
(zeros is explicitly "the original buggy JAX init, for ablation only";
truncated_normal is its default). alphagenome_ft calls the real JAX head
directly with no equivalent override, so --rope-init here manually re-inits
the four RoPE embeddings parameters after model construction, before training.
"""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path

import numpy as np


def _load_interval_list(bed_path: Path, window_size: int):
    """Load a plain 3-column (chrom, start, end) BED into genome.Interval list.

    Unlike alphagenome_ft.finetune.data.load_intervals_from_bed (which expects
    one combined BED with a 4th train/valid/test split column), this repo's
    fold BEDs (data/prep/finetuning/alphagenome/FOLD_1/{train,valid,test}.bed)
    are already split into one file per split, with no split column — so the
    4-column loader silently drops every row (len(parts) < 4) if pointed at
    them directly.
    """
    from alphagenome_ft.finetune.data import build_interval

    intervals = []
    opened = gzip.open if str(bed_path).endswith(".gz") else open
    with opened(bed_path, "rt") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start_str, end_str = line.split()[:3]
            intervals.append(build_interval(
                chromosome=chrom,
                start=int(float(start_str)),
                end=int(float(end_str)),
                window_size=window_size,
            ))
    if not intervals:
        raise ValueError(f"No intervals parsed from {bed_path}.")
    return intervals


def _compute_track_means(
    bigwig_files, bed_path: Path, sequence_length: int, max_samples: int | None,
) -> list[float]:
    """Compute nonzero_mean per rna_seq track, matching alphagenome-pytorch's
    datasets.py::compute_track_means exactly (same centering/expansion logic,
    same deterministic every-Nth subsetting, same nonzero-mean formula,
    resolution 1) — ported line-for-line rather than approximated, since this
    directly feeds a real, active part of training dynamics (see
    plans/10-jax-alphagenome_ft-reproduction.md, "track-means-samples").

    Unlike PyTorch's version this has no strand_pair_groups support: the
    probing run's workflow (workflows/05-full_finetuning/Snakefile) never
    passes --strand-pairs for the rna_seq modality, so PyTorch's own
    modality_strand_pairs['rna_seq'] is empty there too — nothing to mirror
    for this specific run.
    """
    import pyBigWig

    raw_intervals = []
    opened = gzip.open if str(bed_path).endswith(".gz") else open
    with opened(bed_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start_str, end_str = line.split()[:3]
            raw_intervals.append((chrom, int(float(start_str)), int(float(end_str))))

    bws = [pyBigWig.open(str(p)) for p in bigwig_files]
    n_tracks = len(bws)
    try:
        chrom_sizes = dict(bws[0].chroms())
        half_len = sequence_length // 2
        valid_positions = []
        for chrom, start, end in raw_intervals:
            if chrom not in chrom_sizes:
                continue
            center = (start + end) // 2
            final_start = center - half_len
            final_end = center + half_len
            if final_start < 0 or final_end > chrom_sizes[chrom]:
                continue
            valid_positions.append((chrom, final_start, final_end))

        if max_samples is not None and len(valid_positions) > max_samples:
            step = len(valid_positions) // max_samples
            valid_positions = valid_positions[::step][:max_samples]

        if not valid_positions:
            raise ValueError("No valid positions found for computing track means.")

        sums = np.zeros(n_tracks, dtype=np.float64)
        counts = np.zeros(n_tracks, dtype=np.int64)
        for chrom, start, end in valid_positions:
            for i, bw in enumerate(bws):
                values = bw.values(chrom, start, end, numpy=True)
                values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
                nonzero = values[values != 0]
                sums[i] += nonzero.sum()
                counts[i] += len(nonzero)
    finally:
        for bw in bws:
            bw.close()

    means = np.where(counts > 0, sums / counts, 1.0)
    print(f"Computed nonzero_mean per rna_seq track ({len(valid_positions)} "
          f"sampled windows): {means}", flush=True)
    return means.tolist()


class CombinedDataModule:
    """Zips a BigWigDataModule (rna_seq) and a SpliceDataModule (3 splice
    heads) into one joint-modality batch per step, matching the PyTorch
    probing run's --modality bigwig ... --modality splicing ... (all 4 heads
    trained together).

    train() only ever reads `_intervals`/`_batch_size`/`_drop_last` and calls
    `iter_batches` on whatever data_module it's given (see
    alphagenome_ft.finetune.train.train) - this wrapper needs no changes to
    either underlying data module.

    Safety: both underlying modules must be constructed from the identical
    window list/order (see module docstring and the plan doc for why this
    makes their independent index shuffles align in lock-step). Rather than
    just trust that, `iter_batches` asserts the two modules' `sequences`
    arrays are byte-identical every single batch - same windows extracted via
    the same FASTA must produce the same encoded sequence, so any mismatch
    (a future alphagenome_ft change reordering internally, a filtering
    difference introduced later, etc.) surfaces immediately as a loud error
    instead of silently training on misaligned targets.
    """

    def __init__(self, bigwig_module, splice_module):
        for split in ("train", "valid"):
            n_bw = len(bigwig_module._intervals.get(split, ()))
            n_sp = len(splice_module._intervals.get(split, ()))
            if n_bw != n_sp:
                raise ValueError(
                    f"CombinedDataModule: {split} window count mismatch "
                    f"(bigwig={n_bw}, splice={n_sp}) - the two modules were "
                    f"not built from the same window list, so their "
                    f"per-batch shuffles cannot be assumed to align."
                )
        self._bigwig = bigwig_module
        self._splice = splice_module
        self._intervals = splice_module._intervals
        self._batch_size = splice_module._batch_size
        self._drop_last = splice_module._drop_last

    def iter_batches(self, split: str, *, seed: int | None = None, skip_batches: int = 0):
        for bw_batch, sp_batch in zip(
            self._bigwig.iter_batches(split, seed=seed, skip_batches=skip_batches),
            self._splice.iter_batches(split, seed=seed, skip_batches=skip_batches),
        ):
            if not np.array_equal(bw_batch["sequences"], sp_batch["sequences"]):
                raise RuntimeError(
                    "CombinedDataModule: bigwig and splice batches disagree on "
                    "'sequences' for the same batch index - the two data "
                    "modules' window order has desynchronized. Refusing to "
                    "train on what would be misaligned targets."
                )
            combined = dict(sp_batch)
            combined["targets_rna_seq"] = bw_batch["targets_rna_seq"]
            yield combined


_JUNCTION_ROPE_SUBMODULES = (
    "pos_donor_logits", "pos_acceptor_logits", "neg_donor_logits", "neg_acceptor_logits",
)


def _reinit_junction_rope_embeddings(model, head_id: str, std: float, seed: int) -> None:
    """Replace a SpliceSitesJunctionHead's zero-initialized RoPE "embeddings"
    parameter with small truncated-normal noise.

    model._params is a plain flat {module_path: {param_name: array}} dict
    (confirmed by direct inspection, not a Haiku FlatMapping requiring
    hk.data_structures round-tripping — that round-trip was tried first and
    silently restructured the tree in a way parameter_utils.get_head_parameter_paths
    no longer recognized, breaking --heads-only optimizer masking entirely).

    See the module docstring for why this reinit is needed: at exact zero,
    predicted junction counts (a bilinear donor*acceptor product) have an
    exactly-zero gradient w.r.t. this parameter, so training from scratch
    never moves it.
    """
    import jax

    target_module_paths = {f"head/{head_id}/{sm}" for sm in _JUNCTION_ROPE_SUBMODULES}
    missing = target_module_paths - set(model._params)
    if missing:
        raise KeyError(
            f"Expected RoPE submodule(s) {sorted(missing)} not found in model "
            f"params — alphagenome_ft/alphagenome_research's "
            f"SpliceSitesJunctionHead parameter naming may have changed; "
            f"update _JUNCTION_ROPE_SUBMODULES."
        )

    key = jax.random.PRNGKey(seed)
    for module_path in sorted(target_module_paths):
        key, subkey = jax.random.split(key)
        old = model._params[module_path]["embeddings"]
        model._params[module_path] = dict(model._params[module_path])
        model._params[module_path]["embeddings"] = std * jax.random.truncated_normal(
            subkey, lower=-2.0, upper=2.0, shape=old.shape, dtype=old.dtype,
        )


_PRETRAINED_SPLICE_SITE_KEY = "alphagenome/head/splice_sites_classification/multi_organism_linear"


def _init_splice_site_from_pretrained(model, head_id: str, organism_index: int = 0) -> None:
    """Initialize a custom splice_site head from the pretrained model's own
    standard splice-site classification head, matching alphagenome-pytorch's
    ``--pretrained-head-samples "splice_site:0"`` (see
    workflows/05-full_finetuning/Snakefile).

    alphagenome-pytorch's transfer.py comment for this modality: "Fixed
    5-class output: copy full pretrained weight matrix directly" — unlike
    other modalities' per-track slicing, splice_site's classification output
    isn't per-tissue, so there is nothing to select a track of; PyTorch's
    ``:0`` there is an *organism* index (``sd[pt_key][organism_idx:organism_idx+1]``),
    not a tissue/track index, and this mirrors exactly that.

    In this JAX port, `create_model_with_heads`'s param-merging keeps the
    pretrained model's full param tree in `model._params` even for standard
    heads never used by our forward pass (confirmed by reading
    `merge_params` in alphagenome_ft/custom_model.py directly - it appends
    "any keys only in pretrained" after merging our custom heads' keys), so
    the pretrained splice_sites_classification head's weights are already
    sitting in `model._params` unused, under a different module path than
    our own custom-named head.

    Both are the same predefined head kind (`splice_sites_classification`),
    but NOT the same shape: the pretrained model's own head is multi-organism
    ({'b': (2, 5), 'w': (2, 1536, 5)}, confirmed by direct checkpoint
    inspection), while our custom head — built for a single
    --organism — is single-organism ({'b': (1, 5), 'w': (1, 1536, 5)},
    confirmed by a real create_model_with_heads() build). Slice the
    pretrained tensor down to `organism_index` (0 = human) before copying,
    matching PyTorch's own organism_idx slice exactly rather than assuming
    the shapes already match.
    """
    dst_key = f"head/{head_id}/multi_organism_linear"
    if _PRETRAINED_SPLICE_SITE_KEY not in model._params:
        raise KeyError(
            f"Expected pretrained standard head at "
            f"'{_PRETRAINED_SPLICE_SITE_KEY}' not found in model params — "
            f"alphagenome_research's splice_sites_classification head "
            f"parameter naming may have changed."
        )
    if dst_key not in model._params:
        raise KeyError(
            f"Expected custom head at '{dst_key}' not found in model params."
        )
    src_full = model._params[_PRETRAINED_SPLICE_SITE_KEY]
    sliced = {
        k: v[organism_index:organism_index + 1] for k, v in src_full.items()
    }
    sliced_shapes = {k: v.shape for k, v in sliced.items()}
    dst_shapes = {k: v.shape for k, v in model._params[dst_key].items()}
    if sliced_shapes != dst_shapes:
        raise ValueError(
            f"Shape mismatch initializing '{dst_key}' from pretrained "
            f"'{_PRETRAINED_SPLICE_SITE_KEY}'[organism_index={organism_index}]: "
            f"{dst_shapes} vs {sliced_shapes}."
        )
    model._params[dst_key] = sliced


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                         help="Local AlphaGenome JAX checkpoint dir (see "
                              "download_alphagenome_jax_weights.py — do not "
                              "point this at Kaggle directly from a "
                              "network-isolated SLURM node).")
    parser.add_argument("--genome", required=True, type=Path, help="Reference FASTA.")
    parser.add_argument("--train-bed", required=True, type=Path)
    parser.add_argument("--val-bed", required=True, type=Path)
    parser.add_argument("--bigwig", required=True, nargs="+",
                         help="Bigwig files driving a single rna_seq head's "
                              "targets (one track per file), matching the "
                              "PyTorch probing run's --modality rna_seq "
                              "--bigwig ... (2 samples x fwd/rev strand = 4 "
                              "files/tracks there).")
    parser.add_argument("--track-means-samples", type=int, default=None,
                         help="Number of --train-bed windows to sample when "
                              "computing each rna_seq bigwig track's "
                              "nonzero_mean (default: all). Matches "
                              "alphagenome-pytorch's --track-means-samples — "
                              "the real predefined rna_seq head "
                              "(alphagenome_research.model.heads) rescales "
                              "predictions/targets by this on every forward "
                              "pass when present; omit only to fall back to "
                              "no scaling (all-ones), which is NOT what the "
                              "probing run does.")
    parser.add_argument("--star-junctions", required=True, nargs="+",
                         help="STAR SJ.out.tab files, one per sample.")
    parser.add_argument("--ssu", nargs="+", default=None,
                         help="Optional per-sample SSU parquet files, same "
                              "order as --star-junctions.")
    parser.add_argument("--gtf", default=None,
                         help="Optional canonical splice-site GTF/parquet "
                              "(annotation-only sites, zero usage).")
    parser.add_argument("--junction-position-source", choices=["annotated", "predicted"],
                         default="annotated")
    parser.add_argument("--rope-init", choices=["truncated_normal", "zeros"],
                         default="truncated_normal",
                         help="How to initialize SpliceSitesJunctionHead's RoPE "
                              "scale/offset ('embeddings') parameter. zeros is "
                              "the real JAX head's own init and has a dead-"
                              "gradient bug when training from scratch (see "
                              "module docstring) — use truncated_normal unless "
                              "specifically running the zeros ablation.")
    parser.add_argument("--rope-init-std", type=float, default=0.02,
                         help="Stddev for --rope-init truncated_normal.")
    parser.add_argument("--sequence-length", type=int, default=1048576)
    parser.add_argument("--max-splice-sites", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4,
                         help="Global batch size, sharded across --num-devices. "
                              "Combined with --gradient-accumulation-steps, the "
                              "effective batch size per optimizer step is "
                              "batch_size * gradient_accumulation_steps, matching "
                              "the PyTorch run's batch=1 x num_gpus x grad_accum "
                              "scheme.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                         help="Number of --batch-size batches to average grads "
                              "over per optimizer step. Use this (rather than a "
                              "larger --batch-size) to match the PyTorch run's "
                              "effective batch of 64 without risking OOM at 1Mb "
                              "sequence length: e.g. --batch-size 1 "
                              "--gradient-accumulation-steps 64 on a single GPU "
                              "reproduces the same 64-example gradient average as "
                              "the PyTorch probing run's batch=1 x 4 GPUs x "
                              "grad_accum=16.")
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                         help="Mirrors alphagenome-pytorch's --gradient-checkpointing: "
                              "wrap the backbone forward pass in jax.checkpoint so its "
                              "activations are recomputed on the backward pass instead "
                              "of retained, trading compute for memory. Currently a "
                              "no-op for this probing run: --rope-init/--resume both "
                              "already imply detach_backbone=True (heads-only, frozen "
                              "trunk), and no backward pass ever reaches the backbone "
                              "in that case -- same reason alphagenome-pytorch's own "
                              "--gradient-checkpointing is inert for its frozen-backbone "
                              "path (torch.no_grad() there means torch.utils.checkpoint "
                              "has nothing to recompute either). Wired here for parity "
                              "and for future non-frozen modes (e.g. real backbone LoRA).")
    parser.add_argument("--max-train-steps", type=int, default=None,
                         help="Optional global cap on optimizer updates, for "
                              "quick smoke-test runs before a full finetune.")
    parser.add_argument("--save-every-steps", type=int, default=None,
                         help="Also save a 'last' checkpoint (+ opt_state, "
                              "train_state.json) every this many optimizer "
                              "steps, not just at epoch end. Mirrors "
                              "alphagenome-pytorch's --save-every-steps — "
                              "important here since epochs take ~10h and the "
                              "gpu partition's MIG slice caps wall-time at "
                              "12h, so epoch-end-only checkpointing risks "
                              "losing a whole epoch's progress per kill.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                         help="Clip gradients to this global norm before the "
                              "optimizer update. Matches alphagenome-pytorch's "
                              "--max-grad-norm (also hardcoded to 1.0 there). "
                              "Pass 0 or a negative value to disable clipping.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--organism", default="HOMO_SAPIENS")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto",
                         help="'auto' (default): if <output-dir>/<run-name>/last "
                              "already has a train_state.json (from a previous, "
                              "possibly-preempted invocation with the same "
                              "--output-dir/--run-name), resume model weights "
                              "from it via alphagenome_ft.load_checkpoint and "
                              "continue epoch/global_step bookkeeping from "
                              "there instead of --rope-init reinit + fresh "
                              "pretrained weights. 'none' always starts fresh. "
                              "Also restores optimizer (Adam) state from an "
                              "opt_state sidecar there, if present. Mirrors "
                              "alphagenome-pytorch's --resume auto.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Imports deferred past argparse so --help works without a full JAX install.
    from alphagenome_ft import create_model_with_heads, load_checkpoint
    from alphagenome_ft.finetune import config as ft_config
    from alphagenome_ft.finetune.data import BigWigDataModule
    from alphagenome_ft.finetune.splice_data import SpliceDataModule
    from alphagenome_ft.finetune.train import register_predefined_heads, train as run_train

    random.seed(args.seed)
    np.random.seed(args.seed)

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
            "star_junctions": args.star_junctions,
            "max_splice_sites": args.max_splice_sites,
        }
        if args.ssu is not None:
            entry["ssu"] = args.ssu
        if args.gtf is not None:
            entry["gtf"] = args.gtf
        if kind == "splice_sites_junction":
            entry["junction_position_source"] = args.junction_position_source
            if args.junction_position_source == "predicted":
                entry["classification_head_id"] = head_ids["splice_sites_classification"]
        heads_cfg.append(entry)

    track_means = _compute_track_means(
        args.bigwig, args.train_bed, args.sequence_length, args.track_means_samples,
    )
    heads_cfg.append({
        "id": "rna_seq",
        "source": "predefined",
        "kind": "rna_seq",
        "targets": [
            {"path": str(bw), "nonzero_mean": mean}
            for bw, mean in zip(args.bigwig, track_means)
        ],
    })

    specs = ft_config.prepare_head_specs(
        {"heads": heads_cfg}, organism=args.organism,
    )
    ft_config.validate_head_specs(specs)
    register_predefined_heads(specs)

    checkpoint_dir = args.output_dir / args.run_name
    resume_dir = checkpoint_dir / "last"
    do_resume = args.resume == "auto" and (resume_dir / "train_state.json").exists()

    if do_resume:
        print(f"Found existing checkpoint at {resume_dir} — resuming from it "
              f"(skipping fresh pretrained-weight load and --rope-init reinit, "
              f"both of which would clobber already-trained head weights).")
        model = load_checkpoint(
            resume_dir,
            base_checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            detach_backbone=True,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        print("Loading pretrained AlphaGenome JAX model from local checkpoint "
              f"cache: {args.checkpoint_path}")
        model = create_model_with_heads(
            heads=[spec.head_id for spec in specs],
            checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            # heads_only=True in run_train() below only zeroes the backbone's
            # optimizer updates -- without this, jax.grad still backprops
            # through the full ~450M-param frozen trunk every step, which is
            # almost certainly why the first real run OOM'd on a single 80GB
            # GPU at batch_size=1: peak memory was consistent with training
            # the whole model, not just the small splice heads.
            detach_backbone=True,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        if args.rope_init == "truncated_normal":
            print(f"Re-initializing junction head RoPE embeddings "
                  f"(std={args.rope_init_std}) to avoid the zero-init dead-gradient "
                  f"bug — see module docstring.")
            _reinit_junction_rope_embeddings(
                model, head_ids["splice_sites_junction"], std=args.rope_init_std, seed=args.seed,
            )

        print("Initializing splice_site head from the pretrained model's own "
              "standard splice-site classification head (matches "
              "alphagenome-pytorch's --pretrained-head-samples splice_site:0; "
              "see _init_splice_site_from_pretrained docstring).")
        _init_splice_site_from_pretrained(model, head_ids["splice_sites_classification"])

    intervals = {
        "train": _load_interval_list(args.train_bed, window_size=args.sequence_length),
        "valid": _load_interval_list(args.val_bed, window_size=args.sequence_length),
    }

    # Pre-filter to chromosomes common to all --bigwig files using
    # BigWigDataModule's own helper, and construct BOTH data modules from
    # this identical, already-filtered interval dict (not the raw intervals
    # above) — this is what makes CombinedDataModule's lock-step zip safe;
    # see its docstring and the plan doc.
    rna_seq_spec = next(spec for spec in specs if spec.head_id == "rna_seq")
    intervals = BigWigDataModule._filter_intervals_by_bigwig_chromosomes(
        intervals, [rna_seq_spec],
    )

    # SpliceDataModule's own head-kind vocabulary ("splice_sites",
    # "splice_site_usage", "splice_junctions") differs from
    # finetune.config's SPLICE_KINDS naming
    # ("splice_sites_classification"/"splice_sites_usage"/"splice_sites_junction")
    # used above for prepare_head_specs — map explicitly rather than assuming
    # they line up.
    data_head_kinds = {
        "splice_sites": head_ids["splice_sites_classification"],
        "splice_site_usage": head_ids["splice_sites_usage"],
        "splice_junctions": head_ids["splice_sites_junction"],
    }

    splice_module = SpliceDataModule(
        intervals=intervals,
        fasta_path=args.genome,
        star_junction_files=args.star_junctions,
        head_kinds=data_head_kinds,
        batch_size=args.batch_size,
        shuffle=True,
        ssu_files=args.ssu,
        gtf_file=args.gtf,
        max_splice_sites=args.max_splice_sites,
        drop_last=args.num_devices > 1,
        emit_raw_junction_events=(args.junction_position_source == "predicted"),
        # Always False, not a CLI flag: CombinedDataModule requires both
        # underlying modules to share the identical window list (see its
        # docstring and the plan doc), and this also matches the PyTorch
        # probing run, which trains over the entire FOLD_1 split with no
        # junction-presence filter (README.md: 41,699 train / 6,323 val
        # intervals).
        filter_to_junctions=False,
    )
    bigwig_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=args.genome,
        head_specs=[rna_seq_spec],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.num_devices > 1,
    )
    data_module = CombinedDataModule(bigwig_module, splice_module)

    run_train(
        model,
        data_module,
        specs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        heads_only=True,
        checkpoint_dir=checkpoint_dir,
        organism=args.organism,
        num_devices=args.num_devices,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from=resume_dir if do_resume else None,
        save_every_steps=args.save_every_steps,
        gradient_clip_global_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        verbose=True,
    )

    # Distinct from checkpoint_dir/{last,best} (which --resume auto reads/
    # writes across invocations): this is the Snakemake rule's declared
    # output. Keeping them separate means --rerun-incomplete only ever
    # deletes this marker on a killed/incomplete run, never the resumable
    # checkpoint state - see the "Redo" section of the plan doc for why the
    # opposite (declaring last/ itself as the output) silently destroyed a
    # real run's progress.
    (checkpoint_dir / "training_complete.marker").touch()
    print(f"Done! Checkpoints written under {checkpoint_dir}")


if __name__ == "__main__":
    main()
