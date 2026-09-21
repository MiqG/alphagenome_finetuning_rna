"""
JAX finetuning workflow — evaluation of xinming's probing/lora runs.

Xinming reran this workflow's finetuning on their own cluster (ours was too
slow) and shipped back a results package, unpacked at
results/xinming/20260918/runs/jax_historical_reference/ (supersedes the
earlier results/xinming/alphagenome_finetuning_downstream_2026-08-31/
package: that path has since moved to results/xinming/20260831/... and its
JAX checkpoint is an older training run -- confirmed via train_state.json's
best_value/epochs_since_improvement differing from the new package's, and
file timestamps 2026-08-29 vs 2026-09-05 -- despite the new package's own
README describing its JAX checkpoints as "not newly retrained"). It
contains one final (epoch 10) orbax checkpoint per run — "probing" and
"lora" — each saved as both a "best" (best val loss) and "last" (final
epoch) dir; this workflow evaluates the "last" checkpoint of each, matching
the single final checkpoint each PyTorch run (workflows/06-evaluation/
Snakefile) evaluates.

For each (run, subset) pair, runs:
  1. collect_predictions  — single-GPU JAX inference on <subset>.bed
  2. compute_metrics      — computes all paper metrics from the parquets
                             (framework-agnostic script, shared with the
                             PyTorch evaluation workflow)

Subsets:
  test          — held-out test intervals
  train_sample  — seeded random sample of train intervals, same size as test

Outputs:
  results/evaluation/alphagenome_ft/full/{run_name}/{subset}/predictions/*.parquet
  results/evaluation/alphagenome_ft/full/{run_name}/{subset}/metrics.parquet

See ../Snakefile for how this is included and wired into rule all.
"""

import os

DATA_DIR       = config["rnaseq"]["sf3b1mut"]["path"]
FOLDS_DIR      = config["finetuning"]["alphagenome"]["folds_dir"]
FOLD           = config["preprocessing"]["overfitting"]["fold"]
SAMPLES        = config["preprocessing"]["overfitting"]["samples"]
BIGWIG_STRANDS = ["forward", "reverse"]

# Xinming's results package — see module docstring. Only the "last" (final
# epoch 10) checkpoint of each run is evaluated here.
XINMING_DIR = "results/xinming/20260918/runs/jax_historical_reference"
XINMING_CHECKPOINT_SUBDIR = {
    SF3B1_FT_CFG["runs"]["probing_jax"]["run_name"]: "probing",
    SF3B1_FT_CFG["runs"]["lora_jax"]["run_name"]: "lora",
}

EVAL_OUTPUT_DIR = "results/evaluation/alphagenome_ft/full"

# SCRIPTS_DIR (workflow-local scripts only, e.g. collect_predictions_jax.py)
# is defined by rules/finetune.smk (included before this file — see
# ../Snakefile). The finetuning driver itself now lives in alphagenome_ft
# (config["finetuning"]["alphagenome_ft"]["finetune_script"]), not here.
COLLECT_SCRIPT = os.path.join(SCRIPTS_DIR, "collect_predictions_jax.py")
METRICS_SCRIPT = "src/scripts/compute_eval_metrics.py"

# Subsets to evaluate: test (held-out) + train_sample (overfitting check)
EVAL_SUBSETS = ["test", "train_sample"]

# Map subset -> BED file
SUBSET_BED = {
    "test":         os.path.join(FOLDS_DIR, FOLD, "test.bed"),
    "train_sample": os.path.join(FOLDS_DIR, FOLD, "train_sample.bed"),
}

# run_name -> run config (mode, lora params) — same dict finetune.smk's
# ALL_RUNS/_mode_args use, so the model reconstructed for evaluation exactly
# matches the one xinming trained.
EVAL_RUNS = list(XINMING_CHECKPOINT_SUBDIR.keys())

# Job-level parallelism for the two collect_predictions jobs (probing_jax,
# lora_jax): pin each to a different physical GPU so they run concurrently
# instead of both queueing for the single GPU_CFG target, mirroring
# workflows/11-xinming_pytorch_evaluation/Snakefile's GPU_POOL pattern.
# Only 2 entries needed since there are only 2 EVAL_RUNS. Reuses
# finetune.smk's GPU_PRESETS so each tier's runtime/qos stays consistent
# with the partition's actual cap (the "gpu" partition's 7g.80gb rejects
# --time beyond ~12h -- see finetune.smk's GPU_PRESETS comment).
_JAX_GPU_POOL = [GPU_PRESETS["h100"], GPU_PRESETS["mig"]]
_JAX_GPU_ASSIGNMENT = dict(zip(EVAL_RUNS, _JAX_GPU_POOL))


def _jax_gpu_partition(wildcards):
    return _JAX_GPU_ASSIGNMENT[wildcards.run_name]["partition"]


def _jax_gpu_gres(wildcards):
    return "gpu:{}:1".format(_JAX_GPU_ASSIGNMENT[wildcards.run_name]["gres_type"])


def _jax_gpu_runtime(wildcards):
    return _JAX_GPU_ASSIGNMENT[wildcards.run_name]["runtime"]


def _jax_gpu_qos(wildcards):
    return _JAX_GPU_ASSIGNMENT[wildcards.run_name]["qos"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bigwigs(wildcards):
    return [
        os.path.join(DATA_DIR, "STAR", sample,
                     "paper_pass.Aligned.sortedByCoord.out.filtered.{}.bw".format(strand))
        for sample in SAMPLES
        for strand in BIGWIG_STRANDS
    ]


def _ssu_parquets(wildcards):
    return [
        os.path.join(DATA_DIR, "STAR", sample, "paper_pass.ssu.parquet")
        for sample in SAMPLES
    ]


def _star_junctions(wildcards):
    return [
        os.path.join(DATA_DIR, "STAR", sample, "paper_pass.SJ.out.tab")
        for sample in SAMPLES
    ]


def _interval_bed(wildcards):
    return SUBSET_BED[wildcards.subset]


def _checkpoint_dir(wildcards):
    return os.path.join(
        XINMING_DIR,
        XINMING_CHECKPOINT_SUBDIR[wildcards.run_name], "last",
    )


# Target list for the top-level Snakefile's rule all (see ../Snakefile).
EVAL_TARGETS = expand(
    os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "metrics.parquet"),
    run_name=EVAL_RUNS,
    subset=EVAL_SUBSETS,
)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule collect_predictions:
    """Single-GPU JAX inference on interval BED — writes prediction parquets."""
    wildcard_constraints:
        run_name = "|".join(EVAL_RUNS),
        subset   = "|".join(EVAL_SUBSETS),
    input:
        checkpoint_path_file = os.path.join(CHECKPOINT_CACHE_DIR, "checkpoint_path.txt"),
        checkpoint_marker    = lambda wc: os.path.join(_checkpoint_dir(wc), "train_state.json"),
        interval_bed     = _interval_bed,
        train_bed        = os.path.join(FOLDS_DIR, FOLD, "train.bed"),
        genome           = config["gencode"]["paths"]["fasta"],
        gtf_parquet      = config["gencode"]["paths"]["gtf_parquet"],
        bigwigs          = _bigwigs,
        ssu_parquets     = _ssu_parquets,
        star_junctions   = _star_junctions,
    output:
        rna_seq          = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "rna_seq_per_gene.parquet"),
        splice_site      = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "splice_site_scores.parquet"),
        ssu              = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "ssu_scores.parquet"),
        junctions        = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "junction_scores.parquet"),
        psi              = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "psi_scores.parquet"),
        junction_totals  = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "junction_totals.parquet"),
    params:
        checkpoint_dir  = _checkpoint_dir,
        output_dir      = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.run_name, wildcards.subset, "predictions"),
        samples         = " ".join(SAMPLES),
        sequence_length = SF3B1_FT_CFG["sequence_length"],
        track_means_samples = SF3B1_FT_CFG["track_means_samples"],
        mode_args       = _mode_args,
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "{subset}", "collect_predictions.tsv")
    threads: 20
    resources:
        runtime   = _jax_gpu_runtime,
        memory    = 128,
        gres      = _jax_gpu_gres,
        partition = _jax_gpu_partition,
        qos       = _jax_gpu_qos,
    conda:
        "alphagenome"
    # Generous: covers the case where JAX inference on the "mig" GPU target
    # (12h QOS cap) needs more than one SLURM allocation to finish — the
    # script's own interval-level progress checkpointing (see
    # collect_predictions_jax.py, and the CLAUDE.md Snakemake gotcha this
    # follows) makes repeated attempts resume close to where they left off
    # instead of restarting from scratch. Harmless on "h100" (no time
    # limit): real PyTorch collect_predictions runs for this exact eval
    # took ~3-3.5h (see results/evaluation/alphagenome_pytorch/full/
    # benchmarks/*/collect_predictions.tsv), so this is expected to finish
    # in one attempt there.
    retries: 5
    shell:
        """
        set -eo pipefail

        # See finetune.smk's download_alphagenome_jax_weights rule for why
        # this is needed (real AlphaGenomeModel construction also fetches
        # calibration scores from GCS, hitting the same missing-system-CA-
        # bundle issue).
        export CURL_CA_BUNDLE="${{CONDA_PREFIX}}/ssl/cacert.pem"
        export SSL_CERT_FILE="${{CONDA_PREFIX}}/ssl/cacert.pem"

        # kagglehub (pulled in transitively by alphagenome_research's
        # dna_model import) drags in IPython -> sqlite3, whose compiled
        # extension is linked against a newer libstdc++ ABI (CXXABI_1.3.15)
        # than the one this compute node's system /lib64/libstdc++.so.6
        # provides. The conda env ships a new-enough libstdc++ itself —
        # just needs to be found before the system one.
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}"

        # JAX/XLA preallocates a fixed fraction (default 75%) of total GPU
        # memory as an arena on first use, regardless of actual working set
        # -- nvidia-smi then reports that whole arena as "used". Disable
        # preallocation so usage reflects real requirements (needed to know
        # whether this genuinely needs an H100-class GPU or could fit the
        # gpu partition's 80GB MIG slice with real headroom).
        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        # Unbuffered stdout: without -u, Python fully block-buffers stdout
        # when it's not a tty (i.e. redirected to this SLURM log), so the
        # script's progress prints ("Interval N/M: ...") don't actually
        # appear until the internal buffer fills or the process exits --
        # making a job that's progressing normally look silently stuck.
        python -u {COLLECT_SCRIPT} \
            --checkpoint-path-file {input.checkpoint_path_file} \
            --checkpoint-dir {params.checkpoint_dir} \
            --test-bed {input.interval_bed} \
            --train-bed {input.train_bed} \
            --track-means-samples {params.track_means_samples} \
            --genome {input.genome} \
            --gtf-parquet {input.gtf_parquet} \
            --bigwigs {input.bigwigs} \
            --ssu-parquets {input.ssu_parquets} \
            --star-junctions {input.star_junctions} \
            --samples {params.samples} \
            --sequence-length {params.sequence_length} \
            {params.mode_args} \
            --output-dir {params.output_dir}

        echo "Done collecting predictions for {wildcards.run_name} subset {wildcards.subset}"
        """


rule compute_metrics:
    """Compute all evaluation metrics from prediction parquets.

    Framework-agnostic (pandas/scipy/sklearn only) — identical to
    workflows/06-evaluation/Snakefile's rule of the same name, just
    repointed at EVAL_OUTPUT_DIR.
    """
    wildcard_constraints:
        run_name = "|".join(EVAL_RUNS),
        subset   = "|".join(EVAL_SUBSETS),
    input:
        rna_seq         = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "rna_seq_per_gene.parquet"),
        splice_site     = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "splice_site_scores.parquet"),
        ssu             = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "ssu_scores.parquet"),
        junctions       = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "junction_scores.parquet"),
        psi             = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "psi_scores.parquet"),
        junction_totals = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "predictions", "junction_totals.parquet"),
    output:
        metrics_json    = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "metrics.json"),
        metrics_parquet = os.path.join(EVAL_OUTPUT_DIR, "{run_name}", "{subset}", "metrics.parquet"),
    params:
        predictions_dir = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.run_name, wildcards.subset, "predictions"),
        output_dir      = lambda wildcards: os.path.join(EVAL_OUTPUT_DIR, wildcards.run_name, wildcards.subset),
    benchmark:
        os.path.join(EVAL_OUTPUT_DIR, "benchmarks", "{run_name}", "{subset}", "compute_metrics.tsv")
    threads: 32
    resources:
        runtime   = int(0.5 * 60),
        gres      = "none",
        partition = "genoa64",
        qos       = "normal",
        memory    = 64,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {METRICS_SCRIPT} \
            --predictions-dir {params.predictions_dir} \
            --output-dir {params.output_dir} \
            --min-junction-counts 5

        echo "Done computing metrics for {wildcards.run_name} subset {wildcards.subset}"
        """
