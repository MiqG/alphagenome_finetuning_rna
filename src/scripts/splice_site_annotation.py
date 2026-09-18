"""GTF labels for the workflow's existing genomic-left/right site channels.

The legacy STAR/SSU schema calls the left exon flank "donor" on BOTH strands.
On the minus strand this is not the biological donor. Keep that stored schema
explicit here; changing model channels requires a separate training migration.
"""
from __future__ import annotations

import pandas as pd

LABEL_SCHEMA = "transcript_junctions_genomic_roles_v1"


def gtf_splice_sites(gtf: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    """Return 1-based exonic junction endpoints from adjacent transcript exons.

    Do not union-merge gene exons: that erases alternative splice boundaries.
    Do not label isolated transcript ends as splice sites. Unknown strands are
    excluded. Pairing never crosses transcripts, genes, chromosomes or strands.
    """
    columns = ["Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id"]
    missing = set(columns + ["Feature"]) - set(gtf.columns)
    if missing:
        raise ValueError(f"GTF splice annotation lacks columns: {sorted(missing)}")
    exons = gtf.loc[
        (gtf.Feature == "exon") & gtf.Strand.isin(["+", "-"]), columns
    ].copy()
    if exons.empty:
        return set()
    if exons[columns].isna().any().any():
        raise ValueError("GTF exon coordinates or transcript identifiers are missing")
    for col in ["Chromosome", "Strand", "gene_id", "transcript_id"]:
        exons[col] = exons[col].astype(str)
    for col in ["Start", "End"]:
        numeric = pd.to_numeric(exons[col], errors="raise")
        if not (numeric == numeric.astype("int64")).all():
            raise ValueError("GTF exon coordinates must be integers")
        exons[col] = numeric.astype("int64")
    if ((exons.Start < 0) | (exons.End <= exons.Start)).any():
        raise ValueError("GTF exons must be nonempty 0-based half-open intervals")
    group = ["Chromosome", "Strand", "gene_id", "transcript_id"]
    exons = exons.drop_duplicates().sort_values(group + ["Start", "End"]).reset_index(drop=True)
    following_start = exons.groupby(group, observed=True, sort=False).Start.shift(-1)
    pairs = exons.loc[following_start > exons.End].copy()
    pairs["right_exon_base"] = following_start.loc[pairs.index].astype("int64") + 1
    result = set()
    for chrom, end, right, strand in pairs[
        ["Chromosome", "End", "right_exon_base", "Strand"]
    ].itertuples(index=False, name=None):
        result.add((chrom, int(end), strand, "donor"))
        result.add((chrom, int(right), strand, "acceptor"))
    return result
