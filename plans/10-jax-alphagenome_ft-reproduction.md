# Plan: reproduce paper probing/LoRA runs with `alphagenome_ft` (JAX)

## Equivalence review (2026-08-13): systematic pass against workflows/05-full_finetuning/Snakefile

Went through every flag in the PyTorch probing run's actual shell command
(`workflows/05-full_finetuning/Snakefile`'s `torchrun ... scripts/finetune.py`
invocation) one at a time and checked the JAX driver's equivalent. Fixed real
gaps found (see commits in both repos, same date):

- **`--pretrained-head-samples "...splice_site:0"`**: implemented
  (`_init_splice_site_from_pretrained`).
- **`--resume auto`**: implemented, and extended beyond PyTorch's own scope —
  restores optimizer (Adam) state too (`opt_state` sidecar), and resumes
  mid-epoch via `--save-every-steps` (see below), not just at epoch
  boundaries.
- **`--save-every-steps 250`**: implemented (JAX had none before — only
  epoch-boundary checkpoints, which combined with ~10h epochs on a
  wall-time-limited partition meant losing a whole epoch's progress per
  kill, not just progress since the last checkpoint).
- **`--max-grad-norm 1.0`**: implemented. Found `alphagenome_ft.finetune.
  train.train()` had a *local* `create_optimizer` that duplicated and
  shadowed the module-level import of `alphagenome_ft.optimizer_utils.
  create_optimizer` — the local one had no clipping support, so
  `train()` had no way to request it at all. Removed the local duplicate;
  the (already clip-capable) import is what runs now.
- **`--gradient-checkpointing`**: implemented (`gradient_checkpointing`
  param, `jax.checkpoint` wrapping the backbone). Confirmed inert for this
  specific run in both ports (frozen-backbone path never backprops into the
  trunk at all — PyTorch's `torch.no_grad()`, JAX's `detach_backbone`); kept
  for parity and for future non-frozen modes.
- **`--modality rna_seq --bigwig ...`** (joint 4-modality training):
  implemented via the new `CombinedDataModule` (see below in this doc for
  the original design writeup).
- **Snakemake output/checkpoint design**: fixed to mirror this same
  PyTorch workflow's pattern (final-artifact-only `output:`, not the
  resumable checkpoint directory itself) — was backwards before and
  `--rerun-incomplete` silently destroyed a real run's progress as a
  result. See the "Problem 2" section below for the incident.

**Confirmed equivalent without any code change** (traced the actual value/
mechanism on both sides rather than assuming):

- `--min-alpha-juncs 0`: this value means "disable alpha-based usage-loss
  masking" on the PyTorch side (`training.py`); JAX's real
  `SpliceSitesUsageHead.loss` has no such masking to begin with, so "off"
  already matches "off." Would need real implementation work only if a
  *nonzero* min-alpha-juncs run were ever needed.
- `{lr_schedule_args}` (`--warmup-steps 0 --lr-schedule constant` for the
  probing run): JAX's `create_optimizer` takes a plain scalar
  `learning_rate` with no schedule concept at all — which is already
  exactly "constant, no warmup." Only a gap for a *different* run that
  needed an actual schedule (e.g. cosine/warmup).
- `--no-val-pearson`: PyTorch-only metric: JAX's `train()` never computes a
  Pearson correlation validation metric to begin with, so there's nothing
  to disable.
- `--modality-weights "...:1.0,...:1.0,...:1.0,...:1.0"` (all 1.0): JAX's
  `train()` sums per-head losses unweighted — already exactly equivalent to
  all-1.0 weights. Would need a real weights parameter only for a
  differently-weighted run.
- `--num-workers`: PyTorch DataLoader worker count, no JAX equivalent needed
  (different data-loading architecture; doesn't affect training outputs,
  only throughput).

**Known gap, NOT fixed — real, and feasibility is unclear:**

- **`--track-means-samples 1000`**: PyTorch computes real per-track nonzero
  means from 1000 sampled windows (`compute_track_means`) and the rna_seq
  head *divides its output by `track_means * resolution` on every forward
  pass* (`alphagenome_pytorch/heads.py` — `GenomeTracksHead`, confirmed by
  reading the actual tensor op, not just the docstring) — a real, active
  part of training dynamics for this run, not a no-op default (default
  when omitted is `torch.ones(...)`, i.e. no scaling, but the probing run
  does *not* omit it).

  `alphagenome_ft`'s config schema has a matching-looking `nonzero_mean`
  field per track (`finetune/config.py`, with a docstring example showing
  exactly this use case) — but grepped the entire `alphagenome_ft`
  repository for `nonzero_mean`: it is parsed into `TrackInfo` and
  **never read again anywhere**. It's a documented but unimplemented field,
  not a working equivalent.

  Not fixed this pass because the feasibility itself is unknown:
  `alphagenome_ft` calls `alphagenome_research`'s real predefined rna_seq
  head directly (no local reimplementation, consistent with every other
  head in this codebase) — unlike PyTorch's own `GenomeTracksHead`
  reimplementation, there may be no hook in the *real* DeepMind head class
  to inject an equivalent per-track output scaling at all. Needs its own
  investigation (does the real head accept anything like this natively?
  if not, is monkey-patching/wrapping its forward pass even feasible,
  the way `detach_backbone`/`gradient_checkpointing` wrap `forward_trunk`?)
  before attempting an implementation — flagging explicitly rather than
  either silently skipping it or guessing at a fix.

## Redo (2026-08-13): make the probing run genuinely equivalent to PyTorch's

The first real submission attempt surfaced two problems, both now understood
and being fixed — this section documents the redo. Read bottom-to-top for
history; this is the current state of intent.

### Problem 1 (fixed): OOM was `detach_backbone` missing, not a hardware limit

Already root-caused and fixed in a prior pass (see the "Fix real OOM cause"
work further down / the corresponding commits in both repos):
`create_model_with_heads`/`load_checkpoint` now take `detach_backbone` and
`gradient_checkpointing`, matching what `alphagenome-pytorch`'s
`training.py` already does for its frozen-backbone path
(`torch.no_grad()` + `.detach()`). Confirmed working: a real submission
trained cleanly (step-by-step loss decreasing, no OOM) before being stopped
for the reasons below.

### Problem 2 (being fixed now): the run wasn't actually equivalent to PyTorch, and the checkpoint/output design was wrong

Two gaps identified when comparing directly against
`workflows/05-full_finetuning/Snakefile` (the PyTorch run this is supposed to
reproduce):

1. **`rna_seq` was never trained jointly.** PyTorch's probing run
   (`randinit__newloss__annotated__frozen__multigpu_ddp`) trains 4 modalities
   per batch: `rna_seq` (bigwig-derived) + `splice_site`/`splice_usage`/
   `splice_junctions` (STAR/SSU-derived) — see `--modality-weights
   "rna_seq:1.0,splice_site:1.0,splice_usage:1.0,splice_junctions:1.0"` and
   the two `--modality ... --bigwig ...` / `--modality ... --star-junctions
   ...` flag groups in `workflows/05-full_finetuning/Snakefile`. The JAX
   driver only ever trained the 3 splice heads — this was a real,
   consequential scope cut (flagged in the original plan as "tracked as
   follow-up work, not done here"), not a hidden bug, but it means the run
   wasn't comparable to the paper run it's supposed to cross-check. Fixing
   this now (see "rna_seq joint training" below).

2. **Checkpoint/output design was backwards relative to what workflow 05
   already does correctly.** The JAX rule declared its Snakemake `output:`
   as the mutable, continuously-overwritten `last/` checkpoint directory
   that `--resume auto` also depends on for its own bookkeeping. Combined
   with `--rerun-incomplete` (used in every submission command in this repo,
   per `README.md`), this is actively dangerous: when a run gets killed
   mid-epoch (e.g. a partition wall-time limit) and Snakemake retries the
   rule, `--rerun-incomplete` deletes the declared-incomplete output
   directory *before* rerunning — wiping the checkpoint the resume logic
   needed. Confirmed on disk: after job `27358332` was killed at the `gpu`
   partition's 12h QOS limit mid-epoch-2, Snakemake's retry (job `27373239`)
   found no `last/train_state.json`, fell back to a cold start, and reran
   `--rope-init` reinit — silently destroying epoch 1's trained weights. The
   run was manually stopped once this was discovered.

   `workflows/05-full_finetuning/Snakefile` never has this problem because
   its declared `output:` is `checkpoint_epoch{N}.pth` — the **final**
   epoch's checkpoint, written exactly once, at the very end, by a script
   that manages its *own* intermediate/resumable state (`--resume auto`,
   per-epoch checkpoints, `epoch_log.csv`) entirely outside anything
   Snakemake tracks or can delete. Fix: mirror this pattern exactly — change
   the JAX rule's output to a final-epoch marker distinct from the directory
   `--resume auto` reads/writes, so `--rerun-incomplete` can never touch the
   resumable state.

### rna_seq joint training: design

`alphagenome_ft` has two independent data modules with **identical batch
schemas** (`sequences`, `negative_strand_mask`, `targets_{head_id}` — see
`BigWigDataModule._make_batch` and `SpliceDataModule._make_batch`, both in
the `alphagenome_ft` repo): `BigWigDataModule` (bigwig-driven, used for
`rna_seq`) and `SpliceDataModule` (STAR/SSU-driven, used for the 3 splice
heads). Neither has ever been combined into one joint-modality batch before
(this is exactly the gap the original plan flagged).

Verified this is safely composable **without** modifying either class's
internals, by construction rather than by hoping their independent shuffles
happen to line up:

- Both classes' `iter_batches` shuffle via `np.random.default_rng(seed)`
  over `np.arange(len(windows))` — the resulting permutation depends only on
  `(len(windows), seed)`, not on window content. So two data modules built
  from the *same window list, same order, same length* and driven with the
  *same seed* per epoch will always yield batch `k` over the *same windows*,
  in lock-step, with no risk of silent misalignment — confirmed by reading
  both `iter_batches` implementations directly (`alphagenome_ft/finetune/
  data.py` and `alphagenome_ft/finetune/splice_data.py`).
- `SpliceDataModule(..., filter_to_junctions=False)` is confirmed (read the
  `__init__` directly) to store `intervals` completely unmodified in that
  case — no other filtering happens internally.
- `BigWigDataModule.__init__` filters intervals to chromosomes common to all
  configured bigwigs. Apply that same filter to the interval list *before*
  constructing both modules (reusing `BigWigDataModule
  ._get_common_bigwig_chromosomes`), so both modules end up with the
  identical filtered list.
- A `CombinedDataModule` wrapper (new, in the driver script — thin
  composition, not a library change) owns one `BigWigDataModule` (rna_seq)
  and one `SpliceDataModule` (3 splice heads) built this way, exposes
  `_intervals`/`_batch_size`/`_drop_last` (train() already reads these
  directly off whatever `data_module` it's given), and its own
  `iter_batches` zips the two sub-iterators, asserting `sequences` arrays
  are identical between them every batch as a cheap, strong runtime
  correctness check (they must be — same windows, same FASTA) before
  merging in `targets_rna_seq` alongside the splice targets.
- `--filter-to-junctions` therefore also needs to default to False for this
  run to be a fair comparison anyway: PyTorch's full run trains over the
  *entire* FOLD_1 split (41,699 train / 6,323 val intervals per `README.md`)
  with no junction-presence filtering, so removing JAX's junction filter
  isn't just an engineering convenience for the lock-step alignment — it's
  also more faithful to what the PyTorch run actually does.
- Modality weights: PyTorch's probing run uses `1.0` for all 4 heads, and
  JAX's `train()` already sums per-head losses unweighted (equivalent to
  all-1.0 weights) — no gap for *this specific* run; a real per-head-weight
  parameter would only be needed for a differently-weighted run.

### Pretrained head init (`splice_site:0`) — now implemented

PyTorch's probing run uses `--pretrained-head-samples
"rna_seq:NA,splice_usage:NA,splice_junctions:NA,splice_site:0"` — the
`splice_site` head's weights are initialized from the pretrained model's own
standard splice-site head, not randomly (the other three heads stay
randomly initialized either way — "NA"). Originally flagged as a deferred
gap (`alphagenome_ft` had no equivalent mechanism), now implemented as
`_init_splice_site_from_pretrained` in the driver script.

Checked directly against `transfer.py`: for `splice_site` specifically,
PyTorch's `:0` is an *organism* index (`sd[pt_key][organism_idx:organism_idx+1]`),
not a tissue/track index — the classification output is a fixed 5-class
head, not per-tissue, so "Fixed 5-class output: copy full pretrained weight
matrix directly" (PyTorch's own comment) is describing an organism-index
slice, not a track selection.

On the JAX side: `create_model_with_heads`'s param-merging keeps the
pretrained model's full param tree in `model._params` even for standard
heads our forward pass never touches (confirmed by reading `merge_params`
directly), so the pretrained `splice_sites_classification` head's weights
were already present, unused, at
`alphagenome/head/splice_sites_classification/multi_organism_linear`. Two
things had to be verified empirically rather than assumed, both via a real
`create_model_with_heads()` build submitted through SLURM (a `salloc
--no-shell` + repeated `srun --jobid=...` allocation, not the login node):
1. That key really exists post-merge with the expected shape
   (`{'b': (2, 5), 'w': (2, 1536, 5)}` — organism axis first).
2. Our own custom head's matching parameter is **not** the same shape —
   built for a single `--organism`, it's single-organism
   (`{'b': (1, 5), 'w': (1, 1536, 5)}`). First implementation attempt
   assumed the shapes matched and copied wholesale; this failed loudly (the
   function's own shape-check raised) rather than silently training on a
   mismatched copy. Fixed by slicing the pretrained tensor to
   `organism_index` (default 0 = human) before copying — which also makes
   this a more faithful mirror of PyTorch's own `organism_idx` slice than
   the original "copy both organisms" plan would have been.

Verified end-to-end (SLURM, not login node): building the real model, the
params visibly change from their random init, and the new value matches
`pretrained[organism_index=0]` exactly.

## Dependency upgrade (2026-08-06/07): alphagenome/alphagenome_research pinned to latest

Per explicit direction to use the latest versions of everything, upgraded in
the `alphagenome` conda env:

- `alphagenome`: `0.6.1` → `0.7.0` (latest `main`).
- `alphagenome_research`: `0.1.0` (commit `a5162ce`, 2026-03-02) →
  `0.3.0` (commit `1e55dcf`, latest `main` at the time). This latest commit
  is what actually carries the RoPE dead-gradient fix (see below) — an
  intermediate pin was tried first and abandoned once "use latest
  everything" was clarified as the actual intent.
- `protobuf`: `7.34.0` → `7.35.1` (latest `alphagenome_research`'s generated
  proto code requires a newer runtime than was installed).

This surfaced real API breaks between `alphagenome_ft` (last updated against
the older API) and the new packages — all fixed directly in the
`alphagenome_ft` repo (editable install, so these are real source edits, not
environment hacks):

1. **`alphagenome_research.io.fasta` was relocated** to `alphagenome.io.fasta`
   (upstream commit `d9186331fa`, "Migrate to use alphagenome
   FastaExtractor", 2026-06-29). Fixed the two import sites:
   `alphagenome_ft/finetune/data.py` and `alphagenome_ft/finetune/splice_data.py`.
2. **Every trunk-forward Haiku module now requires a keyword-only
   `is_training: bool` argument** it didn't need before
   (`SequenceEncoder`, `TransformerTower`, `SequenceDecoder`,
   `embeddings.OutputEmbedder`) — confirmed against the real
   `AlphaGenomeModel.forward_trunk` source as the authoritative reference.
   Fixed 8 call sites across `alphagenome_ft/custom_forward.py`,
   `alphagenome_ft/custom_model.py` (2 encoder-only-mode call sites), and
   `tests/test_model_predictions.py` — all pass `is_training=False` since
   every one of these call sites is an inference/embedding-extraction path
   (no training-time stochastic behavior desired). Also fixed
   `OutputEmbedder`'s second call: its `skip_x` parameter is keyword-only
   (`*, is_training, skip_x=None`), not positional — an easy mistake since
   the old code passed it positionally when it happened to be the last
   argument accepted at all.
3. Also caught, unrelated to `alphagenome_ft` itself: the newer
   `alphagenome_research` added a mandatory calibration-scores fetch from
   `gs://alphagenome/data/hg38/calibration_scores.pb` during model creation.
   This failed with an SSL cert error specific to this login node
   (`/etc/ssl/certs/ca-certificates.crt` doesn't exist here). Fixed by
   setting `CURL_CA_BUNDLE`/`SSL_CERT_FILE` to the conda env's own
   `ssl/cacert.pem` in every SLURM job script that creates a model — needs
   to be added to the real `finetune_alphagenome_jax.py`/Snakefile too, not
   just the scratch debug scripts (see Follow-up below).

**Verification**: ran `alphagenome_ft`'s own test suite on SLURM (not the
login node — see the operational note below). Final state: all
`is_training`/`skip_x` fixes confirmed (`test_custom_forward_matches_standard_forward`
now passes). One remaining failure, `test_wrapped_model_predictions_match_base`,
is NOT an API-compat issue — it's a real numeric divergence between the
"wrapped" (`create_model_with_heads`) and "base" (`create_from_kaggle`
directly) prediction paths for the *standard* pretrained heads: ATAC/DNase/
RNA_seq differ by ~0.008–0.016 (plausibly ordinary bf16/algorithm-selection
noise), but CHIP_HISTONE (~2.0) and CHIP_TF (~1.0) differ by amounts far
too large to be numerical noise. **Not investigated further** — CHIP_HISTONE/
CHIP_TF are standard heads we don't use (our splice-finetuning heads are
`splice_site`/`splice_usage`/`splice_junctions`), so this doesn't block Phase
A, but it's a real, currently-unexplained behavior difference in
`alphagenome_ft`'s wrapped-model code path worth investigating before anyone
relies on it for ATAC/CHIP predictions specifically.

**Operational note**: this debugging pass required many SLURM round-trips
(each ~35 min for the full `test_model_predictions.py`, since it builds
several full ~450M-param models) — the H100 partition here is shared and
frequently fully occupied for many hours at a stretch (including by the
user's own long-running `asowalk_ag03` job), so getting a traceback for one
fix, patching, and resubmitting is a slow iteration loop. Bare `python -c
"..."` signature/source inspection (no model creation, no GPU compute
needed) was still run through SLURM rather than the login node throughout,
per explicit instruction after an earlier stray login-node pytest run
consumed excessive shared resources.

**Follow-up — done**: folded the `CURL_CA_BUNDLE`/`SSL_CERT_FILE` env vars into
both real rules in `workflows/10-jax_finetuning/Snakefile`
(`download_alphagenome_jax_weights` and `jax_full_finetune`), not just
scratch debug scripts.

**Final re-verification**: re-ran the actual splice-finetuning smoke test
(5 steps, 128kb window on the curated high-junction-density interval) through
`finetune_alphagenome_jax.py` itself — not just `alphagenome_ft`'s own test
suite — on the fully-upgraded packages. Confirms the main training path
(`create_model_with_heads` → `train(heads_only=True)`) is unaffected by the
`is_training` API breaks (those only lived in `alphagenome_ft`'s manual
forward-pass reimplementations, `custom_forward.py` and the encoder-only test
file, not in the main `AlphaGenomeModel.forward_trunk` path our driver script
actually uses). All three splice heads train correctly end-to-end:
`splice_junctions` validation loss moves every epoch
(`12.2388 → 12.2206 → 12.2025 → 12.1842 → 12.1659`), matching
`splice_site`/`splice_usage`'s normal behavior. Phase A is now validated
against the latest package versions, not just the original (older) pins.

One unrelated infrastructure note hit along the way: this cluster's
`gpu_diasfrazer` partition's MIG slices (`1g.24gb` etc.) intermittently go
into a broken state — `nvidia-smi` reports MIG mode enabled but "No MIG
devices found," while SLURM still hands out a MIG UUID gres that doesn't
correspond to any real device, causing `CUDA_ERROR_NO_DEVICE` at model
creation. Not a software bug; requesting the full `gpu:h100:1` (non-MIG)
worked around it. Keep this in mind for future debug-scale runs on this
partition — if a MIG-slice job fails with `CUDA_ERROR_NO_DEVICE` immediately
at device init (not during actual computation), suspect stale MIG state
before suspecting the code.

## Implementation status (2026-08-05)

Phase A (probing) scaffolding has been written AND debugged end-to-end on
SLURM (small-scale smoke tests, not the full FOLD_1 run yet):

- `config/config.yaml` → new `finetuning.alphagenome_ft` block.
- `workflows/10-jax_finetuning/Snakefile` — `download_alphagenome_jax_weights`
  (login-node, Kaggle credentials) + `jax_full_finetune` (SLURM GPU) rules.
- `workflows/10-jax_finetuning/scripts/download_alphagenome_jax_weights.py`
- `workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py`

Debugging notes (all fixed in the files above, not just noted):

- **`/tmp` is node-local, not shared.** SLURM job scripts, log paths, and any
  data files a job reads/writes must live under shared project storage
  (`~/projects/...`) — a compute node's slurmd cannot see the login node's
  `/tmp`, so `#SBATCH --output=/tmp/...` silently produces no log file at all
  on the node the job actually runs on. All debug scaffolding was moved to
  `results/finetuning/alphagenome_ft/debug_*/`.
- **`load_intervals_from_bed` (4-column, single-file) is the wrong loader for
  this repo's fold BEDs** (`data/prep/finetuning/alphagenome/FOLD_1/{train,valid,test}.bed`
  are 3-column, one file per split, no split column) — it silently drops
  every row. `finetune_alphagenome_jax.py` now has its own `_load_interval_list`.
- **Kaggle checkpoint model version is `fold_1`, not `all_folds`** — confirmed
  earlier (see below), and confirmed the download/train/inspect scripts all
  agree on this now.
- **Found and fixed a real dead-gradient bug in `SpliceSitesJunctionHead`'s
  RoPE zero-init** — see its own subsection below. This blocked the
  `splice_junctions` metric specifically; `splice_site`/`splice_usage` were
  unaffected and trained normally throughout.

Verified via two SLURM debug jobs on the curated `data/prep/overfitting/single/high.bed`
interval (chosen because — unlike arbitrary FOLD_1 windows, which are mostly
junction-free at debug-sized sub-windows — it's independently known via
`interval_ranking.tsv.gz`/`splice_junctions.npz` to contain real, dense
junction signal): after the RoPE-init fix, 5 training steps at a 128kb
recentered window moved `splice_site`, `splice_usage`, AND `splice_junctions`
validation loss every single epoch (previously `splice_junctions` was frozen
at exactly the same value for all 5 epochs — see below).

Not yet done: evaluation script (step 4 below), the full-scale FOLD_1 run at
the real 1Mb sequence length (blocked on H100 queue availability during this
session — the only 1g.24gb/1g.12gb MIG slices free don't have enough memory
for 1Mb sequences, and `alphagenome_ft`'s train() has no gradient
checkpointing to reduce that), and the overfitting-single/dev-style sanity
checks from step 5 at full scale.

## Critical bug found and fixed: SpliceSitesJunctionHead RoPE zero-init has a
## dead gradient when training from scratch

While investigating why a debug run's `splice_junctions` validation loss was
exactly constant across 5 epochs (while `splice_site`/`splice_usage` improved
normally), traced it to `alphagenome_research.model.heads.SpliceSitesJunctionHead._apply_rope`:
the RoPE scale/offset parameter (Haiku param name `embeddings`) is
zero-initialized (`hk.get_parameter(..., init=jnp.zeros)`). Predicted junction
counts are a **bilinear product** of donor and acceptor logits
(`einsum('bdtc,batc->bdat', donor_logits, accept_logits)`), and both logits
are exactly zero whenever `embeddings` is exactly zero (since
`x = scale * embedding + offset` with `scale=offset=0` gives `x=0`
regardless of the actual embedding content). The gradient of a product
w.r.t. one factor is proportional to the *other* factor — so when both start
at exactly zero, gradient w.r.t. either is also exactly zero. That's a
stable fixed point: nothing ever moves, at any step, when training this head
from scratch.

Confirmed empirically with two SLURM jobs comparing a fresh vs. 5-epoch-trained
checkpoint's raw parameter values (see `results/finetuning/alphagenome_ft/debug_logs/inspect_param_delta.py`,
not checked in — scratch diagnostic): **every** `splice_junctions` parameter,
including `multi_organism_linear/w` (which has ordinary nonzero init and is
only dragged into the dead point via the bilinear product), showed
`|delta_vs_fresh|_sum = 0` exactly after 5 real training steps with a nonzero
loss. The `splice_site` head's equivalent params moved normally in the same
run (`|delta|=3.83` for its `w`).

This is **the exact same bug** `alphagenome-pytorch/CLAUDE.md`'s `--rope-init`
flag already documents and works around on the PyTorch side:
> `--rope-init` (`truncated_normal` default matches the JAX pretrained
> distribution; `zeros` replicates **the original buggy JAX init**, for
> ablation only)

`alphagenome_ft` calls the real JAX head directly (no local reimplementation
of the loss/head math — consistent with everything else found earlier in
this plan), so it has no equivalent override; the bug is live in
`alphagenome_research` as installed. Since our probing run trains from
scratch (`randinit`), this would have silently produced a `splice_junctions`
metric that never moves off its random-init value for the entire run,
without erroring or otherwise signaling anything was wrong.

**Fix applied**: `finetune_alphagenome_jax.py` now has `--rope-init`
(`truncated_normal` default, `zeros` for ablation, mirroring the PyTorch
flag's names/semantics) and a `_reinit_junction_rope_embeddings` helper that
replaces the four RoPE `embeddings` parameters
(`pos_donor_logits`/`pos_acceptor_logits`/`neg_donor_logits`/`neg_acceptor_logits`)
with small truncated-normal noise (std configurable via `--rope-init-std`,
default 0.02) right after model construction, before training. Verified this
resolves the dead gradient: with the fix, `splice_junctions` validation loss
moved every epoch across 5 debug steps (`12.0348 → 12.0158 → 11.9969 →
11.9778 → 11.9586`), matching `splice_site`/`splice_usage`'s normal behavior.

Implementation note for future readers: the fix initially tried rebuilding
`model._params` via `hk.data_structures.to_mutable_dict`/`to_immutable_dict`
around the targeted parameter update — this silently restructured the tree
into a form `parameter_utils.get_head_parameter_paths` no longer recognized,
breaking `--heads-only` optimizer masking entirely (`create_optimizer` raised
"No trainable head parameters matched"). `model._params` is a plain flat
`{module_path: {param_name: array}}` dict in this codebase, not a Haiku
`FlatMapping` requiring that round-trip — the working fix just does plain
dict copy-and-reassign on the two affected levels.

Two more gaps surfaced while writing the driver script/Snakefile (beyond the
LoRA and junction-loss ones already resolved above) — both are documented
inline in the new files, repeated here for visibility:

1. **Pretrained weights come from Kaggle, and must be `fold_1` not
   `all_folds`.** `alphagenome_ft.create_model_with_heads` loads weights via
   `alphagenome_research.model.dna_model.create_from_kaggle`
   (`kagglehub.model_download('google/alphagenome/jax/{version}')`), not the
   HuggingFace path the PyTorch pipeline uses. Two consequences:
   - SLURM GPU compute nodes on this cluster generally lack internet/Kaggle
     credentials, so `download_alphagenome_jax_weights.py` caches the
     checkpoint once via `kagglehub` on the login node, and the training job
     is pointed at the resulting local directory through
     `create_model_with_heads(checkpoint_path=...)` (confirmed this
     parameter exists and skips Kaggle entirely when set — read directly
     from `alphagenome_ft/custom_model.py`).
   - **Initially used `all_folds` as the default model version — wrong,
     caught mid-session by the user** ("no, all folds no, fold 1"). The
     PyTorch pipeline uses `model_fold_1.safetensors`, i.e. the checkpoint
     pretrained *holding out* FOLD_1 regions, specifically so it doesn't leak
     our FOLD_1 test intervals into the backbone's own pretraining data.
     `all_folds` would reintroduce that leakage. Confirmed `ModelVersion.FOLD_1`
     exists (`alphagenome.models.dna_model.ModelVersion`) and is now the
     default everywhere (`config.yaml`, the download script, the Snakefile).
2. **No gradient accumulation in `alphagenome_ft.finetune.train.train()`.**
   Each yielded batch is one full optimizer step; there's no accumulation
   loop. The PyTorch probing run's effective batch size is 64
   (`batch_size=1 × 4 GPUs × gradient_accumulation_steps=16`) at a 1Mb
   sequence length — reproducing that exactly in JAX would mean a batch of
   64 length-1Mb sequences per step, likely OOM. `finetune_alphagenome_jax.py`
   currently defaults to `--batch-size 4 --num-devices 4` (1 example/device,
   no accumulation) as a starting point; this is a real, currently-unresolved
   deviation from the PyTorch run's optimization dynamics, not just a
   compute-budget knob — flag it explicitly in any results write-up rather
   than treating the two runs as batch-size-equivalent.

## Goal

Add a new workflow (`workflows/10-jax_finetuning/`, mirroring the numbering of
the existing `alphagenome-pytorch` workflows) that reproduces, with the JAX
wrapper package `alphagenome_ft` (`splice-finetuning` branch,
`/users/diasfrazer/manglada/repositories/alphagenome_ft`), the two runs used
for the `figures/paper.ipynb` plots:

- **AlphaGenome (probing)** — linear-probe / frozen-backbone finetuning.
- **AlphaGenome (LoRA)** — low-rank-adapter finetuning.

This gives a JAX/`alphagenome_research`-based cross-check of the same
splicing fine-tuning results currently reported only from the PyTorch port.

## Source of truth for the target run settings

Read directly out of this repo (not re-derived):

- `workflows/05-full_finetuning/Snakefile` — the two `ALL_RUNS` entries.
- `workflows/09-submission/rules/paper.smk` / `prepare_paper_metrics.py` —
  confirms these are literally the two runs `paper.ipynb` plots (`AG_PROBING_RUN`,
  `AG_LORA_RUN`).
- `config/config.yaml` → `finetuning.alphagenome.sf3b1mut` — shared
  hyperparameters (`lr`, `sequence_length`, `epochs`, etc.).

### Run 1 — probing (`randinit__newloss__annotated__frozen__multigpu_ddp`)

| Setting | Value |
|---|---|
| mode | linear-probe (frozen trunk) |
| modalities | `rna_seq`, `splice_site`, `splice_usage`, `splice_junctions` (weights all 1.0) |
| pretrained head init | `rna_seq:NA, splice_usage:NA, splice_junctions:NA, splice_site:0` (splice-site head initialized from pretrained track 0; everything else random) |
| RoPE init | `truncated_normal` |
| junction loss | `normalized` |
| junction position source | `annotated` (STAR-derived, not predicted) |
| min-alpha-juncs | 0 |
| weight decay | 0 |
| lr schedule | constant, no warmup (frozen-trunk default) |
| epochs | 10, lr 1e-4, seq length 1,048,576, batch size 1, grad-accum 16, 4 GPUs DDP |

### Run 2 — LoRA (`randinit__newloss__annotated__lora__largegpu__nowarmup`)

Same modalities/pretrained-head/RoPE/junction-loss/junction-position-source
as Run 1, plus:

| Setting | Value |
|---|---|
| mode | LoRA, rank 8, alpha 16, targets `q_proj,v_proj` (transformer attention projections) |
| lr schedule | **constant, 0 warmup steps** (explicitly matched to the probing run's schedule — this is the `__nowarmup` variant, chosen in the PyTorch repo to isolate whether LoRA's slower early-epoch SSU convergence was a lr-schedule artifact rather than intrinsic to LoRA; see the comment above this entry in `workflows/05-full_finetuning/Snakefile`) |
| weight decay | 0.0 (also matched to probing, not the original LoRA run's 0.1) |
| batch size / grad-accum | 1 / 32 (2x the probing run's grad-accum, on 2 GPUs instead of 4) |
| GPUs | 2x `gpu:7g.80gb` (MIG slices), vs. 4 full GPUs for probing |

**Important**: use the `__nowarmup` variant, not the earlier (commented-out)
`randinit__newloss__annotated__lora__largegpu` entry — the `__nowarmup`
version is the one actually evaluated in `workflows/06-evaluation/Snakefile`
and consumed by `paper.ipynb` (`LORA_EVAL_DIR` / `AG_RUNS["AlphaGenome (LoRA)"]`).

## Decisions (resolved 2026-08-05)

- **Scope**: Phase A (probing reproduction) only, for now. Phase B (true
  backbone LoRA) is explicitly deferred — revisit only after the probing
  run's results are compared against the PyTorch probing run.
- **Driver script location**: `workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py`,
  in a `scripts/` subdirectory alongside the workflow's `rules/` subdirectory —
  i.e. `workflows/10-jax_finetuning/{Snakefile, rules/, scripts/finetune_alphagenome_jax.py}`,
  not a script living in `alphagenome_finetuning_rna/src/scripts/` and not an
  addition to the `alphagenome_ft` package itself.
- **Junction loss parity — investigated, resolved, no work needed**: read
  `alphagenome_research.model.heads.SpliceSitesJunctionHead.loss` directly
  (`conda run -n alphagenome python -c "import inspect; from
  alphagenome_research.model import heads; print(inspect.getsource(...))"`).
  Finding: the **real JAX head has no `original`/`normalized`/`sparse` switch
  at all** — it always computes exactly one fixed loss:
  `donor_ratios_loss + acceptor_ratios_loss + 0.2 * (accept_total_loss + donor_total_loss)`,
  where the ratio terms call `losses.cross_entropy_loss` directly (the same
  function whose PyTorch port was the subject of this session's earlier bug
  fix) and the totals use `losses.poisson_loss` on soft-clipped summed counts.
  The PyTorch port's `--junction-loss original/normalized/sparse` three-way
  flag is a **PyTorch-only extension with no JAX-side equivalent** — `normalized`
  is simply the name the PyTorch port gave to matching this one real formula.
  Consequence for the driver script: **there is nothing to configure** —
  `alphagenome_ft`/`alphagenome_research`'s junction loss always matches what
  the paper's `--junction-loss normalized` runs used; no CLI flag for this is
  needed in `finetune_alphagenome_jax.py`, and no mismatch is possible here by
  construction. This is also a nice independent corroboration that the
  `cross_entropy_loss` formula fixed earlier this session is now exactly right
  ([the fix in `alphagenome-pytorch/src/alphagenome_pytorch/losses.py`] matches
  this real head's `losses.cross_entropy_loss` call verbatim).

## Critical blocker: `alphagenome_ft`'s "LoRA" is not backbone LoRA

Investigated `alphagenome_ft` (`splice-finetuning` branch) directly —
`README.md` §Workflow 3, `docs/lora_adapters.md`, `alphagenome_ft/lora.py`,
`alphagenome_ft/parameter_utils.py::freeze_except_lora`, and
`alphagenome_ft/finetune/train.py`.

Finding: `alphagenome_ft`'s LoRA support is **not** the same intervention as
the PyTorch port's `--mode lora --lora-targets q_proj,v_proj`. In
`alphagenome_ft`:

- The backbone (`sequence_encoder`, `transformer_tower`, `sequence_decoder`)
  is always fully frozen (`freeze_backbone` + `heads_only=True` optimizer
  masking in `train()`).
- "LoRA" means building a **custom head** (`CustomHead` subclass) whose own
  linear projection is a `lora.LoRALinear` reading frozen 1bp/128bp backbone
  embeddings — i.e. a richer/adapter-augmented *probe*, not adapters injected
  into the transformer's own attention `q_proj`/`v_proj` weights. The
  backbone's attention computation itself never changes.
- There is no code path in `alphagenome_ft` or `alphagenome_research` (as
  installed) that patches LoRA into the trunk's attention layers — doing that
  would mean writing new Haiku modules that wrap/replace
  `alphagenome_research`'s attention block, since predefined heads
  (`rna_seq`, `splice_sites_*`) are the real DeepMind head classes, not
  reimplemented locally (confirmed via `custom_model.py::create_loss_fn_for_head`,
  which calls `head.loss(...)` on the actual `alphagenome_research.model.heads`
  classes).

Consequence: an apples-to-apples reproduction of the PyTorch **backbone**
LoRA run is **not possible with `alphagenome_ft` as it stands today** without
new engineering (forking/monkey-patching the JAX attention module to accept
LoRA adapters on `q_proj`/`v_proj`, analogous to the PyTorch repo's LoCon/LoRA
work). That is a nontrivial addition, out of scope for "run the same two
configs."

### Resolution: Phase A only, for now

Per the decision above, this plan covers **Phase A only**: reproduce the
**probing** run exactly — `alphagenome_ft` already supports
linear-probe/frozen-backbone finetuning end-to-end via
`train(..., heads_only=True)`. This is the direct JAX analogue of
`randinit__newloss__annotated__frozen__multigpu_ddp`. Phase B (real backbone
LoRA, or accepting `alphagenome_ft`'s LoRA-head as a labeled analogue) is
deliberately out of scope here and not scheduled — revisit only after Phase
A's result is compared against the PyTorch probing run.

## Second gap: no CLI/orchestration script yet in `alphagenome_ft`

Unlike `alphagenome-pytorch` (which ships `scripts/finetune.py` /
`agt finetune`), `alphagenome_ft`'s `splice-finetuning` branch is a library
only — `finetune/train.py::train()` plus `finetune/config.py` head-spec
parsing, driven so far only from notebooks/tests
(`tests/test_splice_finetuning.py`). There is no `argparse` entrypoint that
takes `--star-junctions`, `--bigwig`, `--mode`, etc. the way the PyTorch
`finetune.py` does.

This plan therefore includes writing a small driver script,
`workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py` (in a
`scripts/` subdirectory alongside the workflow's `rules/` subdirectory) —
inside *this* repo (`alphagenome_finetuning_rna`), not inside `alphagenome_ft`
itself. Keep `alphagenome_ft` a pure library, per its own README's design, and
put the Snakemake-facing CLI glue in the workflow folder.

## Step-by-step plan

1. **Environment**: add `envs/alphagenome_ft.yaml` — installs `alphagenome_ft`
   (editable, from the sibling repo path) plus its real deps (`alphagenome`,
   `alphagenome_research`, `jax[cuda]`, `haiku`, `optax`, `wandb` optional).
   Keep separate from `envs/alphagenome_pytorch.yaml` (different framework,
   avoid dependency conflicts between torch and jax[cuda] in one env).

2. **Driver script** (`workflows/10-jax_finetuning/scripts/finetune_alphagenome_jax.py`):
   - loads STAR junctions / SSU parquets / bigwigs the same way
     `collect_predictions.py` does today (reuse `SAMPLES`, fold BEDs from
     `data/prep/finetuning/alphagenome/FOLD_1/{train,valid,test}.bed`);
   - builds the `heads:` config (`alphagenome_ft.finetune.config.prepare_head_specs`)
     for `rna_seq` (predefined) + the three splice predefined heads, with
     `junction_position_source="annotated"` and `min_unique_reads`/`min_alpha_juncs`
     equivalents wired to CLI flags;
   - constructs `SpliceDataModule`/`BigWigDataModule`, `CustomAlphaGenomeModel`,
     calls `register_predefined_heads`, then `train(...)` with `heads_only=True`
     for the probing run.
   - Junction loss needs no flag/config (see Decisions above — the real head
     has exactly one fixed formula, already matching what we want).

3. **Snakemake workflow** (`workflows/10-jax_finetuning/Snakefile`, with a
   `rules/` subdirectory alongside the `scripts/` subdirectory holding the
   driver script): one rule for the
   `probing` run (Phase A only), invoking the driver script with the settings
   tabulated above, writing
   checkpoints to `results/finetuning/alphagenome_ft/full/{run_name}/`.
   Keep output layout parallel to `results/finetuning/alphagenome_pytorch/full/`
   so `06-evaluation`-style downstream scripts can be reused/extended rather
   than rewritten.

4. **Evaluation**: extend or clone `src/scripts/collect_predictions.py` +
   `compute_eval_metrics.py` to accept an `alphagenome_ft` checkpoint (its
   `save_checkpoint(..., save_full_model=False)` format differs from the
   PyTorch port's `.pth`) — likely a new `collect_predictions_jax.py` given
   the prediction code paths are framework-specific, but keep the *output*
   parquet schema identical so `prepare_paper_metrics.py` and `paper.ipynb`
   can plot both without changes to plotting code.

5. **Sanity checks before the full runs**:
   - Reuse the existing `03-overfitting_single` / `04-overfitting_dev`
     pattern: first overfit `alphagenome_ft` on a single interval / small dev
     set with the probing config, confirm loss goes to ~0 and predictions
     match targets, *before* burning GPU time on the full FOLD_1 run.
   - Cross-check `alphagenome_ft`'s splice classification/usage/junction
     target construction (`finetune/splice_data.py`,
     `finetune/star_junctions.py`) against `alphagenome-pytorch`'s
     `compute_ssu.py`/`get_star_junctions.py` outputs on the same STAR files —
     confirm they agree on e.g. number of splice sites and junction counts
     for one test interval, so a downstream metric mismatch can be
     attributed to model behavior rather than to a data-pipeline difference
     between the two ports.

6. **Compute**: probing run is 4-GPU DDP-equivalent in the PyTorch config;
   `alphagenome_ft`'s multi-GPU path is `jax.pmap`-based single-host DDP
   (`train(..., num_devices=N)`), which should map directly. Match
   `gres`/`partition`/`runtime` SLURM resources to the existing
   `workflows/05-full_finetuning/Snakefile` `full_finetune` rule as a
   starting point, adjusting for JAX's typically different memory profile.

All three open questions from the original draft of this plan (Phase B scope,
driver script location, junction-loss parity) are resolved — see Decisions
above. No open questions remain before implementation can start.
