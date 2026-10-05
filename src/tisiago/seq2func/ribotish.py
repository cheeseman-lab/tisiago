"""Jimmy's Ribo-TISH invocation and filter (aTIS_data/scripts/processing.sh), and call overlap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIS_P_MAX = 0.01
RIBO_P_MAX = 0.01
FISHER_Q_MAX = 0.05


def predict_command(
    ribotish: str,
    tis_bams: list,
    chx_bams: list,
    tis_paras: list,
    chx_paras: list,
    gtf: str | Path,
    fasta: str | Path,
    output: str | Path,
    estimate: str | Path,
) -> list[str]:
    """Jimmy's pooled ``ribotish predict``; ``estimate`` isolates the background file."""
    join = lambda paths: ",".join(str(p) for p in paths)  # noqa: E731
    return [
        str(ribotish), "predict",
        "-t", join(tis_bams), "-b", join(chx_bams),
        "--tispara", join(tis_paras), "--ribopara", join(chx_paras),
        "-g", str(gtf), "-f", str(fasta), "-o", str(output), "-e", str(estimate),
        "--minaalen", "3", "--alt", "--seq", "--aaseq", "--verbose",
    ]


def filter_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Jimmy's significance filter on a ``*_TIS_predict_all.txt`` table."""
    keep = (
        (df.TISPvalue <= TIS_P_MAX)
        & (df.RiboPvalue <= RIBO_P_MAX)
        & (df.FisherQvalue <= FISHER_Q_MAX)
    )
    return df[keep]


def call_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Identify a call by transcript and transcript-coordinate start."""
    return set(zip(df.Tid.astype(str), df.Start.astype(int), strict=True))


def compare_calls(observed: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Overlap of two filtered call tables."""
    a, b = call_keys(observed), call_keys(reference)
    union = a | b
    return {
        "n_observed": len(a),
        "n_reference": len(b),
        "n_shared": len(a & b),
        "jaccard": len(a & b) / len(union) if union else 1.0,
    }
