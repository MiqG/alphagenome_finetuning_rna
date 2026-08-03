"""
paper.smk — precompute the three metrics tables used by figures/paper.ipynb.

Run/path constants here mirror the `AG_RUNS` / `PANGOLIN_RUN_NAMES` config cell
in figures/paper.ipynb -- keep the two in sync when the comparison changes
(e.g. a new LoRA checkpoint).
"""

import os

PREPARE_SCRIPT = "workflows/09-submission/scripts/prepare_paper_metrics.py"
PAPER_OUTPUT_DIR = "results/paper"

AG_PROBING_EVAL_DIR = os.path.join("results", "bsc", "evaluation", "alphagenome_pytorch", "full")
AG_PROBING_RUN      = "randinit__newloss__annotated__frozen__multigpu_ddp"

AG_LORA_EVAL_DIR = os.path.join("results", "evaluation", "alphagenome_pytorch", "full")
AG_LORA_RUN      = "randinit__newloss__annotated__lora__largegpu__nowarmup"

PANGOLIN_EVAL_DIR    = os.path.join("results", "evaluation", "pangolin", "full")
PANGOLIN_PROBING_RUN = "annotated__frozen__1gpu"
PANGOLIN_FULL_RUN    = "annotated__full__1gpu"
PANGOLIN_EPOCH       = 5

EPOCH  = 10
SUBSET = "test"

TEST_BED        = os.path.join(config["finetuning"]["alphagenome"]["folds_dir"],
                                config["preprocessing"]["overfitting"]["fold"], "test.bed")
SEQUENCE_LENGTH = config["finetuning"]["alphagenome"]["sf3b1mut"]["sequence_length"]


def _ag_pred_dir(eval_dir, run_name):
    return os.path.join(eval_dir, run_name, "epoch{}".format(EPOCH), SUBSET, "predictions")


rule all_paper:
    input:
        os.path.join(PAPER_OUTPUT_DIR, "gene_expr_metrics.parquet"),
        os.path.join(PAPER_OUTPUT_DIR, "ssu_metrics.parquet"),
        os.path.join(PAPER_OUTPUT_DIR, "junc_metrics.parquet"),


rule prepare_gene_expr_metrics:
    """Gene expression evaluation: profile per-interval / accumulated / gene mean exonic, 1bp+32bp."""
    input:
        ag_probing_rna = os.path.join(_ag_pred_dir(AG_PROBING_EVAL_DIR, AG_PROBING_RUN), "rna_seq_per_gene.parquet"),
        ag_lora_rna    = os.path.join(_ag_pred_dir(AG_LORA_EVAL_DIR, AG_LORA_RUN), "rna_seq_per_gene.parquet"),
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "gene_expr_metrics.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_gene_expr_metrics.tsv")
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
            --ag-probing-eval-dir {AG_PROBING_EVAL_DIR} \
            --ag-probing-run {AG_PROBING_RUN} \
            --ag-lora-eval-dir {AG_LORA_EVAL_DIR} \
            --ag-lora-run {AG_LORA_RUN} \
            --epoch {EPOCH} \
            --subset {SUBSET} \
            --output {output.metrics}

        echo "Done preparing gene expression metrics"
        """


rule prepare_ssu_metrics:
    """Splice site usage: common sites across AlphaGenome + Pangolin, general/shared/WT-specific/K700E-specific."""
    input:
        ag_probing_ssu = os.path.join(_ag_pred_dir(AG_PROBING_EVAL_DIR, AG_PROBING_RUN), "ssu_scores.parquet"),
        ag_lora_ssu    = os.path.join(_ag_pred_dir(AG_LORA_EVAL_DIR, AG_LORA_RUN), "ssu_scores.parquet"),
        pg_probing_ssu = os.path.join(PANGOLIN_EVAL_DIR, PANGOLIN_PROBING_RUN, "epoch{}".format(PANGOLIN_EPOCH), SUBSET, "predictions", "ssu_scores.parquet"),
        pg_full_ssu    = os.path.join(PANGOLIN_EVAL_DIR, PANGOLIN_FULL_RUN, "epoch{}".format(PANGOLIN_EPOCH), SUBSET, "predictions", "ssu_scores.parquet"),
        test_bed       = TEST_BED,
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "ssu_metrics.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_ssu_metrics.tsv")
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
            --ag-probing-eval-dir {AG_PROBING_EVAL_DIR} \
            --ag-probing-run {AG_PROBING_RUN} \
            --ag-lora-eval-dir {AG_LORA_EVAL_DIR} \
            --ag-lora-run {AG_LORA_RUN} \
            --pangolin-eval-dir {PANGOLIN_EVAL_DIR} \
            --pangolin-probing-run {PANGOLIN_PROBING_RUN} \
            --pangolin-full-run {PANGOLIN_FULL_RUN} \
            --pangolin-epoch {PANGOLIN_EPOCH} \
            --epoch {EPOCH} \
            --subset {SUBSET} \
            --test-bed {TEST_BED} \
            --sequence-length {SEQUENCE_LENGTH} \
            --output {output.metrics}

        echo "Done preparing SSU metrics"
        """


rule prepare_junction_metrics:
    """Splice junction counts: AlphaGenome only, general/shared/WT-specific/K700E-specific."""
    input:
        ag_probing_junc = os.path.join(_ag_pred_dir(AG_PROBING_EVAL_DIR, AG_PROBING_RUN), "junction_scores.parquet"),
        ag_lora_junc    = os.path.join(_ag_pred_dir(AG_LORA_EVAL_DIR, AG_LORA_RUN), "junction_scores.parquet"),
        test_bed        = TEST_BED,
    output:
        metrics = os.path.join(PAPER_OUTPUT_DIR, "junc_metrics.parquet"),
    benchmark:
        os.path.join(PAPER_OUTPUT_DIR, "benchmarks", "prepare_junction_metrics.tsv")
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
            --ag-probing-eval-dir {AG_PROBING_EVAL_DIR} \
            --ag-probing-run {AG_PROBING_RUN} \
            --ag-lora-eval-dir {AG_LORA_EVAL_DIR} \
            --ag-lora-run {AG_LORA_RUN} \
            --epoch {EPOCH} \
            --subset {SUBSET} \
            --test-bed {TEST_BED} \
            --sequence-length {SEQUENCE_LENGTH} \
            --output {output.metrics}

        echo "Done preparing junction metrics"
        """
