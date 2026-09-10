"""Compare aligned embedding-shard artifacts across backend environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare_extraction_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, dict[str, float | bool]]:
    """Validate shard alignment and measure embedding drift in float64.

    The stored features are float16, whose reductions can overflow. Values are
    therefore promoted before subtraction, norms, and summary statistics.
    """
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        if set(reference.files) != set(candidate.files):
            missing = sorted(set(reference.files) - set(candidate.files))
            extra = sorted(set(candidate.files) - set(reference.files))
            raise AssertionError(f"artifact keys differ: missing={missing}, extra={extra}")
        if "row_idx" not in reference.files:
            raise AssertionError("artifacts do not contain row_idx")
        if not np.array_equal(reference["row_idx"], candidate["row_idx"]):
            raise AssertionError("artifact row_idx arrays are not identical")

        results: dict[str, dict[str, float | bool]] = {}
        for key in sorted(set(reference.files) - {"row_idx"}):
            expected = reference[key]
            observed = candidate[key]
            if expected.shape != observed.shape:
                raise AssertionError(
                    f"{key}: shape mismatch {expected.shape} != {observed.shape}"
                )
            if not (np.isfinite(expected).all() and np.isfinite(observed).all()):
                raise AssertionError(f"{key}: artifact contains non-finite values")

            expected64 = expected.astype(np.float64)
            observed64 = observed.astype(np.float64)
            if expected64.size == 0:
                results[key] = {
                    "allclose": True,
                    "exact_fraction": 1.0,
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "rmse": 0.0,
                    "max_row_relative_l2": 0.0,
                }
                continue
            difference = observed64 - expected64
            row_denominator = np.linalg.norm(expected64, axis=1)
            row_difference = np.linalg.norm(difference, axis=1)
            relative_l2 = np.divide(
                row_difference,
                row_denominator,
                out=np.full_like(row_difference, np.inf),
                where=row_denominator > 0,
            )
            relative_l2[(row_denominator == 0) & (row_difference == 0)] = 0.0
            results[key] = {
                "allclose": bool(np.allclose(expected64, observed64, rtol=rtol, atol=atol)),
                "exact_fraction": float(np.mean(expected == observed)),
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_row_relative_l2": float(np.max(relative_l2)),
            }
    return results


def main() -> None:
    """Run artifact comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--require-allclose",
        action="store_true",
        help="Exit unsuccessfully if any feature block exceeds the tolerances",
    )
    args = parser.parse_args()

    results = compare_extraction_artifacts(
        args.reference,
        args.candidate,
        rtol=args.rtol,
        atol=args.atol,
    )
    failed = []
    for key, metrics in results.items():
        print(
            f"{key}: allclose={metrics['allclose']} "
            f"exact={metrics['exact_fraction']:.6f} "
            f"max_abs={metrics['max_abs']:.6g} "
            f"mean_abs={metrics['mean_abs']:.6g} "
            f"rmse={metrics['rmse']:.6g} "
            f"max_row_rel_l2={metrics['max_row_relative_l2']:.6g}"
        )
        if not metrics["allclose"]:
            failed.append(key)
    if args.require_allclose and failed:
        raise SystemExit("feature blocks exceeded tolerance: " + ", ".join(failed))


if __name__ == "__main__":
    main()
