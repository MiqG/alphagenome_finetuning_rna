# Vendored Borzoi evaluation code

Unmodified copies of two scripts from
https://github.com/abuendia/alphagenome-pytorch (branch `abuen/borzoi-eval`),
pinned at commit `3ad490bf8dd235805e1b5bd215bced6697b1747b`:

- `evaluate_borzoi_checkpoint.py`
- `compute_gene_metrics.py` (imported by the script above as a sibling module)

Do not edit these. The workflow wrappers in `src/scripts/collect_predictions_borzoi.py`
and `src/scripts/compute_metrics_borzoi.py` import their functions, so the
numerics stay identical to the upstream evaluation. To update, re-copy from a new
upstream commit and update the hash here.
