"""Load, validate and summarise the reproducible-start truth table.

The column maps below are the only place the hand-off format is spelled out; edit them when
the regenerated release lands. ``RELEASE_COLUMN_MAP`` follows the ribotish fork's
``release/starts`` table (v50); ``ANNOTATION_COLUMN_MAP`` its ``release/transcripts`` table.
``codon_class`` and ``region`` are always recomputed here, never read from the input.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.truth.schema import (
    CODON_CLASSES,
    REGIONS,
    SPLITS,
    START_KEY,
    STRANDS,
    TRUTH_COLUMNS,
    codon_class,
    gpos_from_locus,
    line_of,
    normalize_codon,
    split_of,
    tag_regions,
)

# source column -> truth column. ``locus`` (Ribo-TISH convention) becomes ``gpos`` on load;
# a source that already gives ``gpos`` maps to it directly. ``pause`` and ``imputed`` are
# optional; ``tx_pos`` is optional and, if present, must agree with the annotation.
RELEASE_COLUMN_MAP = {
    "line": "condition",
    "Tid": "transcript_id",
    "Symbol": "gene",
    "Chromosome": "chrom",
    "Strand": "strand",
    "Locus": "locus",
    "Start": "tx_pos",
    "StartCodon": "codon",
    "rep1_called": "rep1",
    "rep2_called": "rep2",
    "Imputed": "imputed",
}
ANNOTATION_COLUMN_MAP = {
    "Tid": "transcript_id",
    "chrom": "chrom",
    "strand": "strand",
    "exon_starts": "exon_starts",
    "exon_ends": "exon_ends",
    "cds_start": "cds_start",
    "cds_end": "cds_end",
}
_REQUIRED_INPUT = ("condition", "transcript_id", "chrom", "strand", "codon", "rep1", "rep2")
_ANNOTATION = tuple(ANNOTATION_COLUMN_MAP.values())


def _read(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)) and str(source).endswith(".parquet"):
        return pd.read_parquet(source)
    if isinstance(source, (str, Path, io.IOBase)):
        return pd.read_csv(source, sep="\t")
    raise TypeError(f"unsupported source {type(source).__name__}")


def _renamed(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    present = {k: v for k, v in column_map.items() if k in df.columns}
    return df.rename(columns=present)[list(present.values())]


def _as_boolean(values: pd.Series, name: str) -> pd.Series:
    """Nullable boolean from bools, 0/1 or ``true``/``false`` strings; raises otherwise."""
    if values.dtype == "boolean":
        return values
    # True/False also match 1/0 and 1.0/0.0 (equal hashes).
    lookup = {True: True, False: False, "true": True, "false": False}
    lookup |= {"True": True, "False": False, "1": True, "0": False}
    out = []
    for v in values:
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) or v == "":
            out.append(pd.NA)
        elif v in lookup:
            out.append(lookup[v])
        else:
            raise ValueError(f"{name}: not a boolean: {v!r}")
    return pd.Series(pd.array(out, dtype="boolean"), index=values.index, name=name)


def load_annotation(source, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Transcript annotation (exons and genomic CDS) in the schema :func:`tag_regions` reads."""
    df = _renamed(_read(source), ANNOTATION_COLUMN_MAP if column_map is None else column_map)
    missing = [c for c in _ANNOTATION if c not in df.columns]
    if missing:
        raise ValueError(f"annotation missing columns {missing}")
    df["transcript_id"] = df.transcript_id.astype(str)
    return df


def load_truth(
    source,
    annotation,
    column_map: dict[str, str] | None = None,
    annotation_column_map: dict[str, str] | None = None,
    *,
    drop_imputed: bool = True,
) -> pd.DataFrame:
    """Load the label pipeline's start table into the validated truth schema.

    Args:
        source: path (``.parquet`` or TSV), text buffer, or DataFrame; one row per condition
            and start.
        annotation: transcript annotation (same kinds of source) for ``region`` and ``tx_pos``.
        column_map: source -> truth column names (default :data:`RELEASE_COLUMN_MAP`).
        annotation_column_map: default :data:`ANNOTATION_COLUMN_MAP`.
        drop_imputed: drop rows flagged ``imputed`` (canonical starts added without a call).

    Returns:
        DataFrame with :data:`TRUTH_COLUMNS`, validated by :func:`validate_truth`.

    Raises:
        ValueError: missing columns, bad codons or strands, a ``tx_pos`` that disagrees with
            the annotation, or duplicate starts.
    """
    df = _renamed(_read(source), RELEASE_COLUMN_MAP if column_map is None else column_map)
    missing = [c for c in _REQUIRED_INPUT if c not in df.columns]
    if "gpos" not in df.columns and "locus" not in df.columns:
        missing.append("gpos or locus")
    if missing:
        raise ValueError(f"truth source missing columns {missing}")
    if drop_imputed and "imputed" in df.columns:
        df = df[~_as_boolean(df.imputed, "imputed").fillna(False).to_numpy(bool)]
    df = df.reset_index(drop=True)
    for col in ("condition", "transcript_id", "chrom", "strand"):
        df[col] = df[col].astype(str)
    bad = sorted(set(df.strand) - set(STRANDS))
    if bad:
        raise ValueError(f"invalid strands {bad}")
    if "gpos" not in df.columns:
        df["gpos"] = gpos_from_locus(df.locus, df.strand)
    df["gpos"] = df.gpos.astype(np.int64)
    df["codon"] = [normalize_codon(c) for c in df.codon]
    df["codon_class"] = [codon_class(c) for c in df.codon]
    tagged = tag_regions(df, load_annotation(annotation, annotation_column_map))
    if "tx_pos" in df.columns:
        given = df.tx_pos.astype(np.int64).to_numpy()
        off = given != tagged.tx_pos.to_numpy()
        if off.any():
            raise ValueError(f"{int(off.sum())} starts have tx_pos disagreeing with the annotation")
    df["tx_pos"] = tagged.tx_pos
    df["region"] = tagged.region
    df["gene"] = df["gene"].astype(str) if "gene" in df.columns else df.transcript_id
    df["pause"] = (
        _as_boolean(df.pause, "pause")
        if "pause" in df.columns
        else pd.array([pd.NA] * len(df), dtype="boolean")
    )
    df["rep1"] = _as_boolean(df.rep1, "rep1")
    df["rep2"] = _as_boolean(df.rep2, "rep2")
    df["reproducible"] = (df.rep1.fillna(False) & df.rep2.fillna(False)).astype(bool)
    df["line"] = [line_of(c) for c in df.condition]
    df["split"] = [split_of(c) for c in df.chrom]
    out = df[list(TRUTH_COLUMNS)]
    validate_truth(out)
    return out


def validate_truth(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` is not a valid truth table.

    Checks: every :data:`TRUTH_COLUMNS` present, no nulls in identity columns, strands in
    ``+``/``-``, non-negative integer coordinates, valid codons whose ``codon_class`` matches
    the codon, known regions and splits, boolean replicate/pause flags, ``reproducible`` equal
    to rep1 and rep2, and one row per (condition, chrom, strand, gpos).
    """
    missing = [c for c in TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"truth table missing columns {missing}")
    ident = ["condition", "line", "transcript_id", "chrom", "strand", "gpos", "tx_pos", "codon"]
    nulls = [c for c in ident if df[c].isna().any()]
    if nulls:
        raise ValueError(f"null values in {nulls}")
    if not set(df.strand) <= set(STRANDS):
        raise ValueError(f"invalid strands {sorted(set(df.strand) - set(STRANDS))}")
    for col in ("gpos", "tx_pos"):
        if not pd.api.types.is_integer_dtype(df[col]) or (df[col] < 0).any():
            raise ValueError(f"{col} must be non-negative integers")
    for value in df.codon:
        if normalize_codon(value) != value:
            raise ValueError(f"codon not normalised: {value!r}")
    expected = [codon_class(c) for c in df.codon]
    if (df.codon_class.to_numpy() != np.asarray(expected, dtype=object)).any():
        raise ValueError("codon_class disagrees with codon")
    if not set(df.codon_class) <= set(CODON_CLASSES):
        raise ValueError(f"unknown codon_class {sorted(set(df.codon_class) - set(CODON_CLASSES))}")
    if not set(df.region) <= set(REGIONS):
        raise ValueError(f"unknown region {sorted(set(df.region) - set(REGIONS))}")
    splits = {*SPLITS.values(), "train"}
    if not set(df.split) <= splits:
        raise ValueError(f"unknown split {sorted(set(df.split) - splits)}")
    for col in ("rep1", "rep2", "pause"):
        if df[col].dtype != "boolean":
            raise ValueError(f"{col} must be pandas 'boolean' dtype, got {df[col].dtype}")
    both = (df.rep1.fillna(False) & df.rep2.fillna(False)).to_numpy(bool)
    if df.reproducible.dtype != bool or (df.reproducible.to_numpy() != both).any():
        raise ValueError("reproducible must equal rep1 & rep2 (NA counts as not supported)")
    dup = df.duplicated(list(START_KEY), keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} rows share a (condition, chrom, strand, gpos) start")


def summarize_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per condition x codon_class x region x reproducible (Phase 0 "done when").

    Every codon class and region appears for every condition and reproducible value, with
    zero counts where empty, so the table reads the same from release to release.
    """
    validate_truth(df)
    index = pd.MultiIndex.from_product(
        [sorted(set(df.condition)), CODON_CLASSES, REGIONS, [True, False]],
        names=["condition", "codon_class", "region", "reproducible"],
    )
    keys = ["condition", "codon_class", "region", "reproducible"]
    counts = df.groupby(keys, observed=True).size().reindex(index, fill_value=0)
    return counts.rename("n").reset_index()
