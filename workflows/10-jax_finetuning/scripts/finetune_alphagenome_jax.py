"""Splice-modality probing (frozen-backbone) finetuning with alphagenome_ft.

JAX/alphagenome_research analogue of alphagenome-pytorch's
"randinit__newloss__annotated__frozen__multigpu_ddp" paper run (see
workflows/05-full_finetuning/Snakefile and
plans/10-jax-alphagenome_ft-reproduction.md for the settings this mirrors).

Scope note: this reproduces the three splice heads only
(splice_sites_classification, splice_sites_usage, splice_sites_junction) via
alphagenome_ft.finetune.splice_data.SpliceDataModule. Unlike the PyTorch run,
rna_seq is NOT trained jointly here — alphagenome_ft has no data module that
feeds both bigwig-derived (rna_seq) and STAR-derived (splice) targets in the
same batch/optimizer step yet. Adding that is tracked as follow-up work in
the plan doc, not done here.

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
    parser.add_argument("--max-train-steps", type=int, default=None,
                         help="Optional global cap on optimizer updates, for "
                              "quick smoke-test runs before a full finetune.")
    parser.add_argument("--filter-to-junctions", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Discard intervals with no complete splice "
                              "junction (default True; SpliceDataModule's own "
                              "default). Useful to disable for tiny debug "
                              "interval sets that may not contain a junction.")
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
                              "Mirrors alphagenome-pytorch's --resume auto. "
                              "Optimizer state (Adam moments) is never resumed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Imports deferred past argparse so --help works without a full JAX install.
    from alphagenome_ft import create_model_with_heads, load_checkpoint
    from alphagenome_ft.finetune import config as ft_config
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
        )

        if args.rope_init == "truncated_normal":
            print(f"Re-initializing junction head RoPE embeddings "
                  f"(std={args.rope_init_std}) to avoid the zero-init dead-gradient "
                  f"bug — see module docstring.")
            _reinit_junction_rope_embeddings(
                model, head_ids["splice_sites_junction"], std=args.rope_init_std, seed=args.seed,
            )

    intervals = {
        "train": _load_interval_list(args.train_bed, window_size=args.sequence_length),
        "valid": _load_interval_list(args.val_bed, window_size=args.sequence_length),
    }

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

    data_module = SpliceDataModule(
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
        filter_to_junctions=args.filter_to_junctions,
    )

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
        verbose=True,
    )

    print(f"Done! Checkpoints written under {checkpoint_dir}")


if __name__ == "__main__":
    main()
