#!/usr/bin/env python
"""
Precompute the heavy per-figure metrics tables used by figures/paper.ipynb.

The notebook's data-prep cells (gene expression, splice site usage, splice
junctions) each re-run large groupby/merge/Pearson computations -- most
expensive being the junction duplicate-window selection (~112M rows per run).
This script runs that work once per figure and writes a compact long-format
parquet that the notebook can just load.

SSU sites/junctions covered by more than one (overlapping) test interval get
one prediction row per covering window; rather than mean-aggregating these,
we keep the single prediction from the window that best centers the
site/junction (max distance to both window edges) -- see
select_max_context_ssu_rows / select_max_context_junction_rows.

Three figures, selected with --figure:
  gene_expr  -- AlphaGenome probing vs LoRA, 1bp/32bp, three settings
                (profile per-interval / profile accumulated / gene mean exonic)
  ssu        -- AlphaGenome (probing/LoRA) + Pangolin (probing/full), common
                sites only, four settings (general / shared / WT-specific /
                K700E-specific)
  junctions  -- AlphaGenome (probing/LoRA) only, four settings (general /
                shared / WT-specific / K700E-specific)

Usage:
    python workflows/09-submission/scripts/prepare_paper_metrics.py --figure gene_expr \\
        --ag-probing-eval-dir results/bsc/evaluation/alphagenome_pytorch/full \\
        --ag-probing-run randinit__newloss__annotated__frozen__multigpu_ddp \\
        --ag-lora-eval-dir results/evaluation/alphagenome_pytorch/full \\
        --ag-lora-run randinit__newloss__annotated__lora__largegpu__nowarmup \\
        --epoch 10 --subset test \\
        --output results/paper/gene_expr_metrics.parquet
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_LABELS = {
    "SRR17111303": "WT",
    "SRR17111311": "K700E",
}
SAMPLE_ID_OF = {v: k for k, v in SAMPLE_LABELS.items()}

AG_MODEL_LABELS = {
    "probing": "AlphaGenome (probing)",
    "lora":    "AlphaGenome (LoRA)",
}
PANGOLIN_MODEL_LABELS = {
    "probing": "Pangolin (probing)",
    "full":    "Pangolin (full)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--figure", required=True, choices=["gene_expr", "ssu", "junctions"])
    p.add_argument("--output", required=True)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--subset", default="test")

    p.add_argument("--ag-probing-eval-dir")
    p.add_argument("--ag-probing-run")
    p.add_argument("--ag-lora-eval-dir")
    p.add_argument("--ag-lora-run")

    p.add_argument("--pangolin-eval-dir")
    p.add_argument("--pangolin-probing-run")
    p.add_argument("--pangolin-full-run")
    p.add_argument("--pangolin-epoch", type=int, default=5)

    p.add_argument("--test-bed", default="data/prep/finetuning/alphagenome/FOLD_1/test.bed",
                    help="Evaluation interval BED -- row order is the `interval_idx` used by "
                         "collect_predictions.py, needed to pick the best-centered duplicate "
                         "prediction per site/junction (see select_max_context_* below).")
    p.add_argument("--sequence-length", type=int, default=1_048_576,
                    help="Must match collect_predictions.py's --sequence-length for this run.")
    return p.parse_args()


def _ag_runs(args) -> dict[str, tuple[str, str]]:
    return {
        AG_MODEL_LABELS["probing"]: (args.ag_probing_eval_dir, args.ag_probing_run),
        AG_MODEL_LABELS["lora"]:    (args.ag_lora_eval_dir, args.ag_lora_run),
    }


def _pangolin_runs(args) -> dict[str, str]:
    return {
        PANGOLIN_MODEL_LABELS["probing"]: args.pangolin_probing_run,
        PANGOLIN_MODEL_LABELS["full"]:    args.pangolin_full_run,
    }


def safe_pearson(x: np.ndarray, y: np.ndarray, min_n: int = 3) -> float | None:
    """Pearson r; returns None if fewer than min_n finite pairs (too few for a stable r)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_n:
        return None
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r)


N_BOOT = 100
CI_PCTS = (2.5, 97.5)


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                          ci_pcts: tuple[float, float] = CI_PCTS, seed: int = 0) -> tuple[float, float] | None:
    """2.5/97.5 percentile (95%) CI for Pearson r, resampling (x, y) pairs with replacement."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boot_rs = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boot_rs[b] = stats.pearsonr(x[samp], y[samp])[0]
    lo, hi = np.percentile(boot_rs, ci_pcts)
    return float(lo), float(hi)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT,
                       ci_pcts: tuple[float, float] = CI_PCTS, seed: int = 0) -> tuple[float, float] | None:
    """2.5/97.5 percentile (95%) CI for the mean of `values`, resampling with replacement.

    Used when only a pre-reduced per-unit statistic survives upstream (e.g. one
    Pearson r per interval) rather than the raw paired observations.
    """
    values = np.asarray(values)
    n = len(values)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, ci_pcts)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Shared: max-context-window selection for SSU/junction duplicates
# ---------------------------------------------------------------------------
#
# SSU sites and junctions falling inside more than one (overlapping) test
# interval get one prediction row per covering window, and predictions differ
# slightly across windows (context-dependent). Rather than mean-aggregating
# these duplicates, we keep the single prediction from the window that best
# centers the site/junction (max distance to both window edges), since
# predictions near a window's edge are the most context-starved.


def pad_interval(start: int, end: int, seq_len: int) -> tuple[int, int]:
    """Mirrors collect_predictions.py's pad_interval -- must stay in sync."""
    if end - start >= seq_len:
        center = (start + end) // 2
        return max(0, center - seq_len // 2), center - seq_len // 2 + seq_len
    pad = seq_len - (end - start)
    padded_start = max(0, start - pad // 2)
    return padded_start, padded_start + seq_len


def load_test_windows(test_bed: str, sequence_length: int) -> pd.DataFrame:
    """Load the evaluation interval BED with each window's padded bounds.

    `iv_idx` is the 0-based row order in the BED file, which is the same
    order collect_predictions.py's `test_intervals.iterrows()` processes
    intervals in -- i.e. the `interval_idx` in junction_scores.parquet, and
    (implicitly, since it has no such column) the append order of duplicate
    rows for the same site in ssu_scores.parquet.
    """
    bed = pd.read_csv(test_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    starts = bed["start"].to_numpy()
    ends = bed["end"].to_numpy()
    window_start = np.empty(len(bed), dtype=np.int64)
    window_end = np.empty(len(bed), dtype=np.int64)
    for i in range(len(bed)):
        window_start[i], window_end[i] = pad_interval(int(starts[i]), int(ends[i]), sequence_length)
    bed["window_start"] = window_start
    bed["window_end"] = window_end
    bed["iv_idx"] = np.arange(len(bed))
    return bed


def select_max_context_junction_rows(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per (junction, sample), keep only the row whose window best centers both breakpoints.

    junction_scores.parquet already carries `interval_idx`, so this is a
    straightforward vectorized merge + groupby-idxmax (no reconstruction needed).
    """
    win = windows[["iv_idx", "window_start", "window_end"]].rename(columns={"iv_idx": "interval_idx"})
    merged = df.merge(win, on="interval_idx", how="left")

    donor = merged["donor_pos_1based"].to_numpy() - 1
    acceptor = merged["acceptor_pos_1based"].to_numpy() - 1
    ws = merged["window_start"].to_numpy()
    we = merged["window_end"].to_numpy()
    clearance = np.minimum.reduce([donor - ws, we - donor, acceptor - ws, we - acceptor])
    merged["_clearance"] = clearance

    key = JUNC_KEY + ["sample_id"]
    best_idx = merged.groupby(key)["_clearance"].idxmax()
    best = merged.loc[best_idx].drop(columns=["window_start", "window_end", "interval_idx", "_clearance"])
    return best.reset_index(drop=True)


def _containing_windows_long(chroms_pos: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """For each unique (chrom, exon_pos) in `chroms_pos`, the ascending-iv_idx-sorted
    list of windows containing it, exploded into a long (chrom, exon_pos, rank, iv_idx)
    table -- `rank` is the 0-based position in that ascending-iv_idx ordering.

    Vectorized per chromosome (broadcast window bounds against unique positions);
    only loops over unique positions (tens of thousands), never over raw rows
    (millions), to build the ragged per-position candidate lists.
    """
    out_chrom, out_pos, out_rank, out_iv = [], [], [], []
    for chrom, wchrom in windows.groupby("chrom", sort=False):
        positions = chroms_pos.loc[chroms_pos["chrom"] == chrom, "exon_pos"].unique()
        if len(positions) == 0:
            continue
        order = np.argsort(wchrom["iv_idx"].to_numpy())
        ws = wchrom["window_start"].to_numpy()[order]
        we = wchrom["window_end"].to_numpy()[order]
        iv_idx = wchrom["iv_idx"].to_numpy()[order]

        contains = (ws[:, None] < positions[None, :]) & (positions[None, :] <= we[:, None])
        for j, pos in enumerate(positions):
            hits = iv_idx[contains[:, j]]  # already ascending (iv_idx pre-sorted above)
            out_chrom.extend([chrom] * len(hits))
            out_pos.extend([pos] * len(hits))
            out_rank.extend(range(len(hits)))
            out_iv.extend(hits.tolist())

    return pd.DataFrame({"chrom": out_chrom, "exon_pos": out_pos, "rank": out_rank, "iv_idx": out_iv})


def select_max_context_ssu_rows(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per (site, sample), keep only the row from the window that best centers the site.

    ssu_scores.parquet has no interval_idx, so it's reconstructed: rows for a
    given (chrom, exon_pos, strand, sample_id) are appended in strictly
    ascending iv_idx order by collect_predictions.py's single serial loop, so
    the k-th row in file order is the k-th smallest-iv_idx window containing
    that position. `rank` (groupby-cumcount, vectorized) recovers k; a single
    merge against the precomputed long candidate table recovers iv_idx.
    """
    df = df.reset_index(drop=True)
    key = ["chrom", "exon_pos", "strand", "sample_id"]
    df["rank"] = df.groupby(key).cumcount()

    long = _containing_windows_long(df[["chrom", "exon_pos"]], windows)
    merged = df.merge(long, on=["chrom", "exon_pos", "rank"], how="left")
    n_unmatched = merged["iv_idx"].isna().sum()
    if n_unmatched:
        print("  WARNING: {} of {} ssu rows could not be matched to a window "
              "(rank exceeded reconstructed candidate count); these keep their "
              "own value but are never preferred as the max-context pick.".format(
                  n_unmatched, len(merged)))

    merged = merged.merge(windows[["iv_idx", "window_start", "window_end"]], on="iv_idx", how="left")
    pos0 = merged["exon_pos"].to_numpy() - 1
    ws = merged["window_start"].to_numpy(dtype="float64")
    we = merged["window_end"].to_numpy(dtype="float64")
    clearance = np.minimum(pos0 - ws, we - pos0)
    clearance = np.where(np.isnan(clearance), -np.inf, clearance)
    merged["_clearance"] = clearance

    best_idx = merged.groupby(key)["_clearance"].idxmax()
    best = merged.loc[best_idx].drop(columns=["rank", "iv_idx", "window_start", "window_end", "_clearance"])
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1 — gene expression
# ---------------------------------------------------------------------------

def per_gene_pearson_rows(rna_df: pd.DataFrame, resolution_label: str) -> list[dict]:
    rows = []
    for sample_id, sample_label in SAMPLE_LABELS.items():
        sub = rna_df[rna_df["track_name"] == sample_id].dropna(subset=["pred_log_mean", "obs_log_mean"])
        obs, pred = sub["obs_log_mean"].values, sub["pred_log_mean"].values
        r = safe_pearson(obs, pred)
        if r is None:
            continue
        ci = bootstrap_pearson_ci(obs, pred)
        row = {"sample": sample_label, "pearson_r": r, "n": len(sub), "resolution": resolution_label}
        row["ci_lo"], row["ci_hi"] = ci if ci is not None else (None, None)
        rows.append(row)
    return rows


def profile_corr_rows(pred_dir: str, scope: str, resolution_label: str) -> list[dict]:
    """Pooled accumulated Pearson r -- only sufficient statistics survive upstream
    (ProfileCorrAccumulator), so no raw pairs or per-unit r's exist to resample;
    no bootstrap CI here."""
    fpath = os.path.join(pred_dir, "rna_seq_profile_corr_{}_{}.parquet".format(scope, resolution_label))
    df = pd.read_parquet(fpath)
    rows = []
    for track_name, sub in df.groupby("track_name"):
        sample_label = SAMPLE_LABELS.get(track_name, track_name)
        rows.append({
            "sample": sample_label,
            "pearson_r": float(sub["pearson_r"].mean()),
            "n": int(sub["n_positions"].sum()),
            "resolution": resolution_label,
            "ci_lo": None,
            "ci_hi": None,
        })
    return rows


def profile_corr_rows_per_interval(pred_dir: str, resolution_label: str) -> list[dict]:
    fpath = os.path.join(pred_dir, "rna_seq_profile_corr_per_interval.parquet")
    df = pd.read_parquet(fpath)
    col = "pearson_r_central_{}".format(resolution_label)
    rows = []
    for track_name, sub in df.groupby("track_name"):
        sample_label = SAMPLE_LABELS.get(track_name, track_name)
        valid = sub[col].dropna()
        # Only per-interval r's survive upstream, not raw per-position pairs, so
        # the bootstrap resamples interval-level r's rather than raw positions.
        ci = bootstrap_mean_ci(valid.values)
        rows.append({
            "sample": sample_label,
            "pearson_r": float(valid.mean()),
            "n": int(len(valid)),
            "resolution": resolution_label,
            "ci_lo": ci[0] if ci is not None else None,
            "ci_hi": ci[1] if ci is not None else None,
        })
    return rows


def prepare_gene_expr(args) -> pd.DataFrame:
    resolutions = ["1bp", "32bp"]
    settings = {
        "profile_per_interval": "Profile: mean per interval",
        "profile_accumulated":  "Profile: accumulated",
        "gene_mean_exonic":     "Gene: mean exonic coverage",
    }

    # Independent collect_predictions.py runs can end up evaluating a slightly
    # different subset of overlapping test intervals (same test.bed, but not every
    # interval is redundant), so one run's rna_seq_per_gene.parquet can be missing
    # a handful of genes the other has. Restrict the gene-level comparison to genes
    # present in every model's run (per resolution) so n and the gene set are
    # identical across models -- profile_per_interval/profile_accumulated already
    # match exactly across models and don't need this.
    ag_runs = _ag_runs(args)
    rna_by_model_res = {}
    for model_label, (eval_dir, run_name) in ag_runs.items():
        pred_dir = os.path.join(eval_dir, run_name, "epoch{}".format(args.epoch), args.subset, "predictions")
        rna_by_model_res[model_label] = {
            "1bp":  pd.read_parquet(os.path.join(pred_dir, "rna_seq_per_gene.parquet")),
            "32bp": pd.read_parquet(os.path.join(pred_dir, "rna_seq_per_gene_32bp.parquet")),
        }

    common_genes = {
        res_label: set.intersection(*(
            set(rna_by_model_res[m][res_label]["gene_id"].unique()) for m in ag_runs
        ))
        for res_label in resolutions
    }
    for res_label in resolutions:
        for model_label in ag_runs:
            n_genes = rna_by_model_res[model_label][res_label]["gene_id"].nunique()
            n_common = len(common_genes[res_label])
            if n_genes != n_common:
                print("  {} ({}): dropping {} gene(s) not shared across all models ({} -> {})".format(
                    model_label, res_label, n_genes - n_common, n_genes, n_common))

    records = []
    for model_label, (eval_dir, run_name) in ag_runs.items():
        pred_dir = os.path.join(eval_dir, run_name, "epoch{}".format(args.epoch), args.subset, "predictions")
        for res_label in resolutions:
            for row in profile_corr_rows_per_interval(pred_dir, res_label):
                row.update({"model": model_label, "setting": "profile_per_interval"})
                records.append(row)
            for row in profile_corr_rows(pred_dir, "central", res_label):
                row.update({"model": model_label, "setting": "profile_accumulated"})
                records.append(row)
            rna_common = rna_by_model_res[model_label][res_label]
            rna_common = rna_common[rna_common["gene_id"].isin(common_genes[res_label])]
            for row in per_gene_pearson_rows(rna_common, res_label):
                row.update({"model": model_label, "setting": "gene_mean_exonic"})
                records.append(row)

    df = pd.DataFrame(records)
    df["setting_label"] = df["setting"].map(settings)
    return df


# ---------------------------------------------------------------------------
# Figure 2 — splice site usage
# ---------------------------------------------------------------------------

def load_ssu_scores(eval_dir: str, run_name: str, epoch: int, subset: str, rename_pos: str | None = None) -> pd.DataFrame:
    fpath = os.path.join(eval_dir, run_name, "epoch{}".format(epoch), subset, "predictions", "ssu_scores.parquet")
    df = pd.read_parquet(fpath)
    if rename_pos and rename_pos in df.columns:
        df = df.rename(columns={rename_pos: "exon_pos"})
    return df


def common_sites(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Intersection of (chrom, exon_pos, strand) splice sites present in every df."""
    sites = None
    for df in dfs:
        s = set(map(tuple, df[["chrom", "exon_pos", "strand"]].drop_duplicates().values.tolist()))
        sites = s if sites is None else (sites & s)
    return pd.DataFrame(list(sites), columns=["chrom", "exon_pos", "strand"])


def ssu_pearson_rows(df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for sample_id, sample_label in SAMPLE_LABELS.items():
        grp = df[df["sample_id"] == sample_id].dropna(subset=["pred_ssu", "obs_ssu"])
        pred, obs = grp["pred_ssu"].values, grp["obs_ssu"].values
        r = safe_pearson(pred, obs)
        if r is None:
            continue
        ci = bootstrap_pearson_ci(pred, obs)
        row = {"sample": sample_label, "pearson_r": r, "n": len(grp), "model": model_name}
        row["ci_lo"], row["ci_hi"] = ci if ci is not None else (None, None)
        rows.append(row)
    return rows


def site_specificity(df: pd.DataFrame, key_cols: list[str], value_col: str) -> tuple[set, set, set]:
    """Partition sites/junctions by whether observed usage/count is nonzero in WT, K700E, or both.

    Returns (wt_specific, k7_specific, shared): wt_specific/k7_specific are nonzero in only
    one sample; shared is the true intersection between conditions (nonzero in both) --
    distinct from `general`, which is simply the raw set common across all models'
    predictions, regardless of per-condition usage.
    """
    wt_id, k7_id = SAMPLE_ID_OF["WT"], SAMPLE_ID_OF["K700E"]
    pivot = df.pivot_table(index=key_cols, columns="sample_id", values=value_col, aggfunc="mean")
    wt_vals, k7_vals = pivot[wt_id].fillna(0), pivot[k7_id].fillna(0)
    wt_specific = set(map(tuple, pivot[(wt_vals > 0) & (k7_vals == 0)].index))
    k7_specific = set(map(tuple, pivot[(k7_vals > 0) & (wt_vals == 0)].index))
    shared = set(map(tuple, pivot[(wt_vals > 0) & (k7_vals > 0)].index))
    return wt_specific, k7_specific, shared


def filter_by_key(df: pd.DataFrame, key_cols: list[str], key_set: set | None) -> pd.DataFrame:
    if key_set is None:
        return df
    key_df = pd.DataFrame(list(key_set), columns=key_cols)
    return df.merge(key_df, on=key_cols, how="inner")


def prepare_ssu(args) -> pd.DataFrame:
    key = ["chrom", "exon_pos", "strand"]
    labels = {
        "general":        "General",
        "shared":         "Shared splice sites",
        "wt_specific":    "WT-specific sites",
        "k700e_specific": "K700E-specific sites",
    }

    windows = load_test_windows(args.test_bed, args.sequence_length)

    ag_dfs = {
        label: load_ssu_scores(eval_dir, run_name, args.epoch, args.subset, rename_pos="exon_pos_1based")
        for label, (eval_dir, run_name) in _ag_runs(args).items()
    }
    pg_dfs = {
        label: load_ssu_scores(args.pangolin_eval_dir, run_name, args.pangolin_epoch, args.subset)
        for label, run_name in _pangolin_runs(args).items()
    }

    ag_dedups = {label: select_max_context_ssu_rows(df, windows) for label, df in ag_dfs.items()}
    pg_dedups = {label: select_max_context_ssu_rows(df, windows) for label, df in pg_dfs.items()}

    sites = common_sites(*ag_dedups.values(), *pg_dedups.values())
    commons = {
        label: df.merge(sites, on=key, how="inner")
        for label, df in {**ag_dedups, **pg_dedups}.items()
    }

    wt_specific, k700e_specific, shared = site_specificity(next(iter(commons.values())), key, "obs_ssu")

    records = []
    for model_label, df in commons.items():
        for setting, key_set in [
            ("general", None),
            ("shared", shared),
            ("wt_specific", wt_specific),
            ("k700e_specific", k700e_specific),
        ]:
            for row in ssu_pearson_rows(filter_by_key(df, key, key_set), model_label):
                row["setting"] = setting
                records.append(row)

    df = pd.DataFrame(records)
    df["setting_label"] = df["setting"].map(labels)
    return df


# ---------------------------------------------------------------------------
# Figure 3 — splice junctions
# ---------------------------------------------------------------------------

JUNC_KEY = ["chrom", "donor_pos_1based", "acceptor_pos_1based", "strand"]


def junction_pearson_rows(df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for sample_id, sample_label in SAMPLE_LABELS.items():
        grp = df[(df["sample_id"] == sample_id) & (df["obs_count"] > 0)]
        pred, obs = np.log1p(grp["pred_count"].values), np.log1p(grp["obs_count"].values)
        r = safe_pearson(pred, obs)
        if r is None:
            continue
        ci = bootstrap_pearson_ci(pred, obs)
        row = {"sample": sample_label, "pearson_r": r, "n": len(grp), "model": model_name}
        row["ci_lo"], row["ci_hi"] = ci if ci is not None else (None, None)
        rows.append(row)
    return rows


def prepare_junctions(args) -> pd.DataFrame:
    labels = {
        "general":        "General",
        "shared":         "Shared junctions",
        "wt_specific":    "WT-specific junctions",
        "k700e_specific": "K700E-specific junctions",
    }

    windows = load_test_windows(args.test_bed, args.sequence_length)

    dedups = {}
    for model_label, (eval_dir, run_name) in _ag_runs(args).items():
        pred_dir = os.path.join(eval_dir, run_name, "epoch{}".format(args.epoch), args.subset, "predictions")
        junc_df = pd.read_parquet(os.path.join(pred_dir, "junction_scores.parquet"))
        dedups[model_label] = select_max_context_junction_rows(junc_df, windows)

    wt_specific, k700e_specific, shared = site_specificity(next(iter(dedups.values())), JUNC_KEY, "obs_count")

    records = []
    for model_label, df in dedups.items():
        for setting, key_set in [
            ("general", None),
            ("shared", shared),
            ("wt_specific", wt_specific),
            ("k700e_specific", k700e_specific),
        ]:
            for row in junction_pearson_rows(filter_by_key(df, JUNC_KEY, key_set), model_label):
                row["setting"] = setting
                records.append(row)

    df = pd.DataFrame(records)
    df["setting_label"] = df["setting"].map(labels)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.figure == "gene_expr":
        df = prepare_gene_expr(args)
    elif args.figure == "ssu":
        df = prepare_ssu(args)
    elif args.figure == "junctions":
        df = prepare_junctions(args)
    else:
        raise ValueError("Unknown figure: {}".format(args.figure))

    print(df)
    df.to_parquet(args.output, index=False, compression="zstd")
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
