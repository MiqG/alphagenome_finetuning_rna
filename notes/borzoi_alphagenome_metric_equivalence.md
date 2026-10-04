# Making AlphaGenome evaluation metrics equivalent to the Borzoi evaluation

Status: notes only. Nothing here has been applied to the AlphaGenome pipeline yet.
Borzoi is being added first as a separate, additive workflow (see workflows/12-borzoi_evaluation).

Compared:
- AlphaGenome side: `src/scripts/collect_predictions.py` + `src/scripts/compute_eval_metrics.py`
- Borzoi side: `evaluate_borzoi_checkpoint.py` + `compute_gene_metrics.py`
  (abuendia/alphagenome-pytorch, branch `abuen/borzoi-eval`)

## What already matches (no change needed)

- Central window: both use `center = (start + end) // 2` and a 196,608 bp window,
  so the scored central regions are identical.
- JSD: per-track normalization over positions, eps 1e-8, averaged over tracks.
- Count Pearson: one Pearson r pooling all (region, track) total-signal pairs.
- Accumulated per-track Pearson on log1p signal over all regions and positions
  (our `rna_seq_profile_corr_central_1bp` equals Borzoi's `track_pearson_r_accumulated`).
- Gene metrics: union of exons per gene (our `merge_gene_exons`), >=50% unique exon bp
  criterion, strand-matched track per sample, log1p of exon-mean coverage,
  quantile normalization + gene centering, and the three paper correlations.
  Tie handling in quantile normalization differs slightly (ours rounds average ranks);
  negligible with only 2 sample tracks.
- Units at 1 bp: Borzoi 1 bp prediction = bin sum / 32 (per-base coverage), the same
  scale as AlphaGenome's 1 bp output and the raw bigwigs.

## What differs and what to change in our pipeline

1. Gene window (largest difference).
   - `test.bed` intervals are 1,048,576 bp. We select genes (>=50% exon bp) and compute
     exon means over the whole 1 Mb window. Borzoi only predicts the central 196,608 bp
     and does both steps inside that window.
   - Change: add `--gene-score-window-bp` (default off, so old behavior is unchanged) to
     `collect_predictions.py`. When set to 196608: (a) `compute_gene_interval_map` uses the
     centered windows instead of the BED intervals; (b) `get_exon_mean_pred` and
     `get_exon_mean_obs` clip to that window. The central start is 32 bp aligned relative
     to the 1 Mb window (offset 425,984 = 13,312 x 32), so 32 bp binning stays consistent.
   - The alphagenome-pytorch `compute_gene_metrics.py` already exposes this flag.

2. Per-region profile Pearson.
   - Ours (`profile_pearson_r_central`) is raw scale, pooled across tracks.
     Borzoi's `profile_pearson_r` is log1p, pooled across tracks, per region.
   - Change: add a per-region log1p track-pooled central Pearson column to
     `rna_seq_profile_metrics_per_interval.parquet` (and a matching summary in
     `compute_eval_metrics.py`). The per-track log1p value already exists per interval,
     but its mean is not the same as the pooled value.

3. 32 bp view.
   - Ours mean-pools, so the log1p profile is `log1p(mean)`. Borzoi's native view is
     sum-pooled, so it is `log1p(sum)` (= `log1p(32 * mean)`). log1p is not scale
     invariant, so the 32 bp log1p profile correlations are not comparable today.
   - Ours does not compute a 32 bp JSD (it is scale invariant, so simply add it).
   - Our 32 bp gene means (`get_exon_mean_pred_binned`) count the bins each exon touches.
     Borzoi weights each bin by its exact exon base count. Change to exact bp weighting.
   - Until this is changed, only the 1 bp metrics are comparable.

4. Region set.
   - Borzoi drops regions whose 524,288 bp input window runs off the chromosome.
     We keep any region whose central 196,608 bp fits.
   - Change (analysis side): compare on the intersection of regions, matched on
     (chrom, start, end), using Borzoi's `metrics_per_region.csv` as the reference list.

5. Gene to interval assignment tie-break.
   - Ours keeps the first qualifying interval, Borzoi keeps the one with the most exon
     overlap. With 196,608 bp windows (which tile without overlap) this only matters for
     exact 50/50 splits, so it should be a no-op once item 1 is applied.

## Cost and rollout

- The central-window gene means are not stored in current outputs, so each AlphaGenome run
  must be re-collected (about 3 to 3.5 h per run per subset on the H100).
- Write the new outputs to new directories (for example `{run}__central196k`) so existing
  results are untouched.

## Validation checklist

- `obs_mean` in the gene tables should be identical between AlphaGenome and Borzoi for the
  same genes (same bigwigs, same window, same exon mask). This is a direct test that the
  gene sets and exon masks line up.
- Number of shared regions after intersecting, and number of shared genes.
- Borzoi: predicted vs observed total ratio should be about 1. This confirms the untransform
  `scale` (a wrong scale, e.g. 0.005 vs 0.3, is a 60x error).

## Implemented on the Borzoi side (workflow 12)

`workflows/12-borzoi_evaluation/scripts/export_alphagenome_format.py` writes the AlphaGenome
RNA-seq prediction parquets from the raw Borzoi predictions, importing the helper functions from
`src/scripts/collect_predictions.py`, and `src/scripts/compute_eval_metrics.py` then runs unchanged.
Checked on 20 regions against the upstream Borzoi evaluation: gene values at 1 bp identical (1e-15),
accumulated central 1 bp Pearson identical (1e-9), same columns and dtypes as a real AlphaGenome run.

Conventions used on the Borzoi side (these already follow the AlphaGenome conventions, so items 2 and 3
above only need changes on the AlphaGenome side for the gene window and the region intersection):
- the central 196,608 bp window plays the role of the whole window (gene assignment and exon means);
  "full" scope outputs are NaN or empty;
- 1 bp = bin sum / 32 repeated; 32 bp = per-base mean in the bin (AlphaGenome's mean pooling), so the
  32 bp log1p profile is log1p(mean) on both sides;
- `interval_idx` and start/end are those of the BED interval.

Still open on the AlphaGenome side: gene assignment and exon means over the central window
(item 1), and intersecting regions (item 4).
