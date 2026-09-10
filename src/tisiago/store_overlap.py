"""Validate representation identity at sites shared by two aligned stores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.manifest import SITE_KEY
from tisiago.representation_eval import _evo_keys


def align_shared_sites(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions for one-to-one shared transcript-relative sites."""
    unique_manifests = {}
    for name, manifest in (("reference", reference), ("candidate", candidate)):
        missing = {*SITE_KEY, "codon", "label_tis"} - set(manifest)
        if missing:
            raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
        duplicated = manifest.duplicated(list(SITE_KEY), keep=False)
        if duplicated.any():
            conflicts = (
                manifest.loc[duplicated]
                .groupby(list(SITE_KEY), sort=False)[["codon", "label_tis"]]
                .nunique()
                .gt(1)
                .any(axis=1)
            )
            if conflicts.any():
                raise ValueError(
                    f"{name} manifest has conflicting duplicate transcript sites"
                )
        positions = np.flatnonzero(
            ~manifest.duplicated(list(SITE_KEY), keep="first")
        )
        unique = manifest.iloc[positions][[*SITE_KEY, "codon", "label_tis"]].copy()
        unique[f"{name}_row"] = positions
        unique_manifests[name] = unique
    reference_sites = unique_manifests["reference"]
    candidate_sites = unique_manifests["candidate"]
    shared = reference_sites.merge(
        candidate_sites,
        on=list(SITE_KEY),
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if not len(shared):
        raise ValueError("stores contain no shared transcript sites")
    identity_matches = (
        shared.codon_reference.astype(str).to_numpy()
        == shared.codon_candidate.astype(str).to_numpy()
    ) & (
        shared.label_tis_reference.to_numpy()
        == shared.label_tis_candidate.to_numpy()
    )
    if not identity_matches.all():
        raise ValueError(
            f"{int((~identity_matches).sum())} shared sites disagree on codon or label"
        )
    return (
        shared.reference_row.to_numpy(dtype=np.int64),
        shared.candidate_row.to_numpy(dtype=np.int64),
    )


def compare_overlap_feature(
    reference_path: Path,
    candidate_path: Path,
    reference_rows: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    chunk_size: int = 2_048,
) -> dict[str, float | bool | int]:
    """Compare aligned feature rows without loading either complete array."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("reference and candidate rows must be aligned")
    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError(
            f"feature widths differ: {reference.shape[1]} != {candidate.shape[1]}"
        )
    exact_count = 0
    value_count = 0
    max_abs = 0.0
    for start in range(0, len(reference_rows), chunk_size):
        end = min(start + chunk_size, len(reference_rows))
        expected = np.asarray(reference[reference_rows[start:end]])
        observed = np.asarray(candidate[candidate_rows[start:end]])
        if not (np.isfinite(expected).all() and np.isfinite(observed).all()):
            raise ValueError("overlap features contain non-finite values")
        exact_count += int(np.sum(expected == observed))
        value_count += expected.size
        if expected.size:
            difference = observed.astype(np.float64) - expected.astype(np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
    return {
        "n_rows": int(len(reference_rows)),
        "exact": exact_count == value_count,
        "exact_fraction": float(exact_count / max(1, value_count)),
        "max_abs": max_abs,
    }


def main() -> None:
    """Compare TXP features shared by curated and dense experiment stores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-store", required=True)
    parser.add_argument("--candidate-store", required=True)
    parser.add_argument("--keys", nargs="+", default=_evo_keys("TXP"))
    parser.add_argument("--chunk-size", type=int, default=2_048)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    reference_store = Path(args.reference_store)
    candidate_store = Path(args.candidate_store)
    columns = [*SITE_KEY, "codon", "label_tis"]
    reference_manifest = pd.read_parquet(
        reference_store / "manifest.parquet", columns=columns
    )
    candidate_manifest = pd.read_parquet(
        candidate_store / "manifest.parquet", columns=columns
    )
    reference_rows, candidate_rows = align_shared_sites(
        reference_manifest, candidate_manifest
    )
    print(f"shared unique transcript sites: {len(reference_rows):,}")
    failed = []
    for key in args.keys:
        metrics = compare_overlap_feature(
            reference_store / "embeddings" / key,
            candidate_store / "embeddings" / key,
            reference_rows,
            candidate_rows,
            chunk_size=args.chunk_size,
        )
        print(
            f"{key}: exact={metrics['exact']} "
            f"exact_fraction={metrics['exact_fraction']:.8f} "
            f"max_abs={metrics['max_abs']:.6g}"
        )
        if not metrics["exact"]:
            failed.append(key)
    if args.require_exact and failed:
        raise SystemExit("shared-site features differ: " + ", ".join(failed))


if __name__ == "__main__":
    main()
