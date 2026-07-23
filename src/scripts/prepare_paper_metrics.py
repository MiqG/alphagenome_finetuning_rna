#!/usr/bin/env python
"""
Precompute the heavy per-figure metrics tables used by figures/paper.ipynb.

The notebook's data-prep cells (gene expression, splice site usage, splice
junctions) each re-run large groupby/merge/Pearson computations -- most
expensive being the junction dedup (~112M rows per run). This script runs
that work once per figure and writes a compact long-format parquet that the
notebook can just load.

Three figures, selected with --figure:
  gene_expr  -- AlphaGenome probing vs LoRA, 1bp/32bp, three settings
                (profile per-interval / profile accumulated / gene mean exonic)
  ssu        -- AlphaGenome (probing/LoRA) + Pangolin (probing/full), common
                sites only, three settings (general / WT-specific / K700E-specific)
  junctions  -- AlphaGenome (probing/LoRA) only, three settings (general /
                WT-specific / K700E-specific)

Usage:
    python src/scripts/prepare_paper_metrics.py --figure gene_expr \\
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


# ---------------------------------------------------------------------------
# Figure 1 — gene expression
# ---------------------------------------------------------------------------

def per_gene_pearson_rows(rna_df: pd.DataFrame, resolution_label: str) -> list[dict]:
    rows = []
    for sample_id, sample_label in SAMPLE_LABELS.items():
        sub = rna_df[rna_df["track_name"] == sample_id].dropna(subset=["pred_log_mean", "obs_log_mean"])
        r = safe_pearson(sub["obs_log_mean"].values, sub["pred_log_mean"].values)
        if r is None:
            continue
        rows.append({"sample": sample_label, "pearson_r": r, "n": len(sub), "resolution": resolution_label})
    return rows


def profile_corr_rows(pred_dir: str, scope: str, resolution_label: str) -> list[dict]:
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
        rows.append({
            "sample": sample_label,
            "pearson_r": float(valid.mean()),
            "n": int(len(valid)),
            "resolution": resolution_label,
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


def dedup_ssu_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (site, sample) rows created by overlapping test-interval windows."""
    key = ["chrom", "exon_pos", "strand", "sample_id"]
    return df.groupby(key, as_index=False).agg(
        pred_ssu=("pred_ssu", "mean"),
        obs_ssu=("obs_ssu", "mean"),
        alpha_juncs=("alpha_juncs", "max"),
    )


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
        r = safe_pearson(grp["pred_ssu"].values, grp["obs_ssu"].values)
        if r is None:
            continue
        rows.append({"sample": sample_label, "pearson_r": r, "n": len(grp), "model": model_name})
    return rows


def site_specificity(df: pd.DataFrame, key_cols: list[str], value_col: str) -> tuple[set, set]:
    """Sites/junctions where observed usage/count is > 0 in one sample and 0 in the other."""
    wt_id, k7_id = SAMPLE_ID_OF["WT"], SAMPLE_ID_OF["K700E"]
    pivot = df.pivot_table(index=key_cols, columns="sample_id", values=value_col, aggfunc="mean")
    wt_specific = set(map(tuple, pivot[(pivot[wt_id] > 0) & (pivot[k7_id].fillna(0) == 0)].index))
    k7_specific = set(map(tuple, pivot[(pivot[k7_id] > 0) & (pivot[wt_id].fillna(0) == 0)].index))
    return wt_specific, k7_specific


def filter_by_key(df: pd.DataFrame, key_cols: list[str], key_set: set | None) -> pd.DataFrame:
    if key_set is None:
        return df
    key_df = pd.DataFrame(list(key_set), columns=key_cols)
    return df.merge(key_df, on=key_cols, how="inner")


def prepare_ssu(args) -> pd.DataFrame:
    key = ["chrom", "exon_pos", "strand"]
    labels = {
        "general":        "General",
        "wt_specific":    "WT-specific sites",
        "k700e_specific": "K700E-specific sites",
    }

    ag_dfs = {
        label: load_ssu_scores(eval_dir, run_name, args.epoch, args.subset, rename_pos="exon_pos_1based")
        for label, (eval_dir, run_name) in _ag_runs(args).items()
    }
    pg_dfs = {
        label: load_ssu_scores(args.pangolin_eval_dir, run_name, args.pangolin_epoch, args.subset)
        for label, run_name in _pangolin_runs(args).items()
    }

    ag_dedups = {label: dedup_ssu_mean(df) for label, df in ag_dfs.items()}
    pg_dedups = {label: dedup_ssu_mean(df) for label, df in pg_dfs.items()}

    sites = common_sites(*ag_dedups.values(), *pg_dedups.values())
    commons = {
        label: df.merge(sites, on=key, how="inner")
        for label, df in {**ag_dedups, **pg_dedups}.items()
    }

    wt_specific, k700e_specific = site_specificity(next(iter(commons.values())), key, "obs_ssu")

    records = []
    for model_label, df in commons.items():
        for setting, key_set in [
            ("general", None),
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


def dedup_junction_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (junction, sample) rows created by overlapping test-interval windows."""
    key = JUNC_KEY + ["sample_id"]
    return df.groupby(key, as_index=False).agg(
        pred_count=("pred_count", "mean"),
        obs_count=("obs_count", "mean"),
    )


def junction_pearson_rows(df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for sample_id, sample_label in SAMPLE_LABELS.items():
        grp = df[(df["sample_id"] == sample_id) & (df["obs_count"] > 0)]
        r = safe_pearson(np.log1p(grp["pred_count"].values), np.log1p(grp["obs_count"].values))
        if r is None:
            continue
        rows.append({"sample": sample_label, "pearson_r": r, "n": len(grp), "model": model_name})
    return rows


def prepare_junctions(args) -> pd.DataFrame:
    labels = {
        "general":        "General",
        "wt_specific":    "WT-specific junctions",
        "k700e_specific": "K700E-specific junctions",
    }

    dedups = {}
    for model_label, (eval_dir, run_name) in _ag_runs(args).items():
        pred_dir = os.path.join(eval_dir, run_name, "epoch{}".format(args.epoch), args.subset, "predictions")
        junc_df = pd.read_parquet(os.path.join(pred_dir, "junction_scores.parquet"))
        dedups[model_label] = dedup_junction_mean(junc_df)

    wt_specific, k700e_specific = site_specificity(next(iter(dedups.values())), JUNC_KEY, "obs_count")

    records = []
    for model_label, df in dedups.items():
        for setting, key_set in [
            ("general", None),
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
