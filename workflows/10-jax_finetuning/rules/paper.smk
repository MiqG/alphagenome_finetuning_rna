"""
paper.smk — precompute the three metrics tables used by figures/paper_jax.ipynb.

Combines all 6 models in the JAX-vs-PyTorch, local-vs-xinming comparison via
prepare_paper_metrics.py's --extra-model flag (no --ag-probing-*/--ag-lora-*
here, since none of the 4 PyTorch variants are the "default" AG_MODEL_LABELS
pair -- every label is spelled out explicitly so local vs xinming is visible
in the data itself, not just notebook cosmetics).

Duplicated verbatim in workflows/11-xinming_pytorch_evaluation/rules/paper.smk
(same outputs) so either workflow can produce the comparison once its own
evaluation half is done -- mirrors this repo's existing convention of
duplicating small rule sets across sibling workflows (see compute_metrics in
workflows/06-evaluation/Snakefile vs workflows/11-xinming_pytorch_evaluation/
Snakefile).

Run/path constants here mirror the PAL_MODELS config cell in
figures/paper_jax.ipynb -- keep the two in sync when the comparison changes.
"""

import os
import shlex

PREPARE_SCRIPT   = "workflows/09-submission/scripts/prepare_paper_metrics.py"
PAPER_OUTPUT_DIR = "results/paper"

# label -> predictions dir, for all 6 models in the comparison.
JAX_PAPER_MODELS = {
    "AlphaGenome-PyTorch, local (probing)":
        os.path.join("results", "bsc", "evaluation", "alphagenome_pytorch", "full",
                      "randinit__newloss__annotated__frozen__multigpu_ddp", "epoch10", "test", "predictions"),
    "AlphaGenome-PyTorch, local (LoRA)":
        os.path.join("results", "evaluation", "alphagenome_pytorch", "full",
                      "randinit__newloss__annotated__lora__largegpu__nowarmup", "epoch10", "test", "predictions"),
    "AlphaGenome-PyTorch, xinming (probing)":
        os.path.join("results", "evaluation", "alphagenome_pytorch_xinming", "full",
                      "randinit__newloss__annotated__frozen__multigpu_ddp", "epoch10", "test", "predictions"),
    "AlphaGenome-PyTorch, xinming (LoRA, epoch7 partial)":
        os.path.join("results", "evaluation", "alphagenome_pytorch_xinming", "full",
                      "randinit__newloss__annotated__lora__largegpu__nowarmup", "epoch7", "test", "predictions"),
    "AlphaGenome-JAX (probing)":
        os.path.join("results", "evaluation", "alphagenome_ft", "full",
                      "randinit__annotated__frozen__probing_jax", "test", "predictions"),
    "AlphaGenome-JAX (LoRA)":
        os.path.join("results", "evaluation", "alphagenome_ft", "full",
                      "randinit__annotated__frozen__lora_jax", "test", "predictions"),
}

TEST_BED        = os.path.join(config["finetuning"]["alphagenome"]["folds_dir"],
                                config["preprocessing"]["overfitting"]["fold"], "test.bed")
SEQUENCE_LENGTH = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"]

_EXTRA_MODEL_ARGS = " ".join(
    "--extra-model {}".format(shlex.quote("{}={}".format(label, pred_dir)))
    for label, pred_dir in JAX_PAPER_MODELS.items()
)


PAPER_JAX_TARGETS = [
    os.path.join(PAPER_OUTPUT_DIR, "gene_expr_metrics_jax.parquet"),
    os.path.join(PAPER_OUTPUT_DIR, "ssu_metrics_jax.parquet"),
    os.path.join(PAPER_OUTPUT_DIR, "junc_metrics_jax.parquet"),
]


rule all_paper_jax:
    input:
        PAPER_JAX_TARGETS,


rule prepare_gene_expr_metrics_jax:
    """Gene expression evaluation across all 6 models, local+xinming x PyTorch+JAX."""
    input:
        [os.path.join(pred_dir, "rna_seq_per_gene.parquet") for pred_dir in JAX_PAPER_MODELS.values()],
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "gene_expr_metrics_jax.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_gene_expr_metrics_jax.tsv")
    threads: 4
    resources:
        runtime   = int(0.5 * 60),
        gres      = "none",
        partition = "genoa64",
        memory    = 32,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {PREPARE_SCRIPT} \
            --figure gene_expr \
            {_EXTRA_MODEL_ARGS} \
            --output {output.metrics}

        echo "Done preparing gene expression metrics (JAX comparison)"
        """


rule prepare_ssu_metrics_jax:
    """Splice site usage across all 6 models, general/shared/WT-specific/K700E-specific."""
    input:
        [os.path.join(pred_dir, "ssu_scores.parquet") for pred_dir in JAX_PAPER_MODELS.values()],
        test_bed = TEST_BED,
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "ssu_metrics_jax.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_ssu_metrics_jax.tsv")
    threads: 8
    resources:
        runtime   = int(1 * 60),
        gres      = "none",
        partition = "genoa64",
        memory    = 64,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {PREPARE_SCRIPT} \
            --figure ssu \
            {_EXTRA_MODEL_ARGS} \
            --test-bed {TEST_BED} \
            --sequence-length {SEQUENCE_LENGTH} \
            --output {output.metrics}

        echo "Done preparing SSU metrics (JAX comparison)"
        """


rule prepare_junction_metrics_jax:
    """Splice junction counts across all 6 models, general/shared/WT-specific/K700E-specific."""
    input:
        [os.path.join(pred_dir, "junction_scores.parquet") for pred_dir in JAX_PAPER_MODELS.values()],
        test_bed = TEST_BED,
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "junc_metrics_jax.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_junction_metrics_jax.tsv")
    threads: 8
    resources:
        runtime   = int(1 * 60),
        gres      = "none",
        partition = "genoa64",
        memory    = 64,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        python {PREPARE_SCRIPT} \
            --figure junctions \
            {_EXTRA_MODEL_ARGS} \
            --test-bed {TEST_BED} \
            --sequence-length {SEQUENCE_LENGTH} \
            --output {output.metrics}

        echo "Done preparing junction metrics (JAX comparison)"
        """
