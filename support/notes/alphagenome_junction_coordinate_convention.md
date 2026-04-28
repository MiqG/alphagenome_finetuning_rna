# AlphaGenome Junction Coordinate Convention — Open Question

## Background

`src/scripts/generate_oracle_targets.py` generates oracle training targets by running
the pretrained AlphaGenome model on intervals from `support/overfit.bed`. Among the
outputs is a STAR-format `oracle_junctions.SJ.out.tab` file derived from the model's
`splice_sites_junction` head.

## Observed coordinate inconsistency

When inspecting the outputs in IGV:

- **Splice site parquet** (`oracle_splice_sites.parquet`): positions are stored 0-based.
  IGV (1-based) therefore displays them 1 bp before the actual site. Adding 1 to all
  positions puts donors at the last exon base and acceptors at the first exon base —
  the canonical biological definition.
- **Junction file** (`oracle_junctions.SJ.out.tab`): `intron_start` is computed as
  `lo + 1` where `lo` is the 0-based donor genomic position. In STAR's 1-based format
  this resolves to the **last exon base**, not the first intronic base. Adding 1 would
  give the correct biological intron start.

The script already writes a companion file `oracle_junctions_start_plus1.SJ.out.tab`
with `intron_start += 1`, and `benchmark_pretrained_oracle.smk` tests both under the
`data_like` (original) and `data_like_plus1` (corrected) target types.

## The open question

The coordinate shift in `data_like` could be either:

1. **A script bug** — the 0-based → 1-based conversion was applied only once (for the
   last-exon → first-intron step) but the second +1 needed for STAR's 1-based system
   was missed. In this case `data_like_plus1` has the correct targets.

2. **A faithful mirror of the model's internal convention** — the model itself defines
   donor/acceptor positions with a different offset from the biological standard. If the
   head being trained already operates in that internal coordinate frame, the shifted
   targets would be the ones it can actually learn.

## Diagnostic experiment

`benchmark_pretrained_oracle.smk` runs three conditions:

| `target_type`      | Junction file used              | Expected if hypothesis 1 | Expected if hypothesis 2 |
|--------------------|---------------------------------|--------------------------|--------------------------|
| `distillation`     | .npz raw tensors (no conversion)| overfits                 | overfits                 |
| `data_like`        | `oracle_junctions.SJ.out.tab`   | fails or slower          | overfits                 |
| `data_like_plus1`  | `_start_plus1.SJ.out.tab`       | overfits                 | fails or slower          |

**Key interpretation:** if the model achieves overfitting in `data_like` and
`distillation` but NOT in `data_like_plus1`, it is evidence that the model's internal
junction convention matches the original (shifted) file — i.e., hypothesis 2 — and
that `data_like_plus1` is actually misaligned with the model's representations despite
being biologically correct.

## Next steps

- Run the benchmark and compare learning curves and final loss across the three conditions.
- If hypothesis 2 is confirmed, audit the AlphaGenome training code to identify the
  exact donor/acceptor position convention used during pretraining.
- Decide whether to keep `data_like` as the canonical junction target for real
  finetuning runs, or to fix the convention end-to-end.
