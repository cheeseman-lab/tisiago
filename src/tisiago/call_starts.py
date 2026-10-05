"""Discrete start-site calls from the frozen multi-seed ensemble head.

Scores the selected rows of a store with every seed artifact written by
``tisiago.representation_eval``, averages the calibrated probabilities (the same ensemble
the evaluation reports), and applies the validation-selected ensemble threshold from
``run.json``. Refuses to write unless the calls reproduce the logged ensemble metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.caller import evaluate_at_threshold
from tisiago.linear_head import LinearHeadArtifact

CALL_COLUMNS = (
    "row_idx",
    "transcript_id",
    "gene",
    "chrom",
    "gstart",
    "strand",
    "mrna_index",
    "codon",
    "codon_class",
    "region_class",
    "label_tis",
    "split",
)
LOGIT_MARGIN = 1e-3


def score_logits(
    embedding_root: Path,
    artifacts: list[LinearHeadArtifact],
    rows: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Return per-seed raw logits, shape ``[n_seeds, len(rows)]``, in ``rows`` order.

    Reads one feature block per chunk and accumulates each artifact's partial logit, so the
    full concatenated feature matrix is never materialized.
    """
    keys = artifacts[0].feature_keys
    if any(artifact.feature_keys != keys for artifact in artifacts):
        raise ValueError("all seed artifacts must share one feature schema")
    arrays = {key: np.load(Path(embedding_root) / key, mmap_mode="r") for key in keys}
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    logits = np.empty((len(artifacts), len(rows)), dtype=np.float64)
    for start in range(0, len(rows), chunk_size):
        chunk = sorted_rows[start : start + chunk_size]
        chunk_logits = [np.full(len(chunk), a.intercept, dtype=np.float32) for a in artifacts]
        for key in keys:
            block = np.asarray(arrays[key][chunk], dtype=np.float32)
            for logit, artifact in zip(chunk_logits, artifacts, strict=True):
                logit += artifact.partial_logit_from_blocks({key: block})
        logits[:, order[start : start + len(chunk)]] = chunk_logits
    return logits


def ensemble_probability(
    artifacts: list[LinearHeadArtifact], logits: np.ndarray, shift: float = 0.0
) -> np.ndarray:
    """Mean calibrated probability over seeds, optionally with every logit shifted."""
    return np.mean(
        [a.calibrate_logits(logit + shift) for a, logit in zip(artifacts, logits, strict=True)],
        axis=0,
    )


def score_rows(
    embedding_root: Path,
    artifacts: list[LinearHeadArtifact],
    rows: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Return the mean calibrated probability over ``artifacts`` for ``rows``."""
    logits = score_logits(embedding_root, artifacts, rows, chunk_size=chunk_size)
    return ensemble_probability(artifacts, logits)


def build_calls(
    manifest: pd.DataFrame, rows: np.ndarray, probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Join scores onto manifest rows (in ``rows`` order) and mark calls at ``threshold``."""
    columns = [column for column in CALL_COLUMNS if column in manifest.columns]
    calls = manifest.iloc[np.asarray(rows)][columns].reset_index(drop=True)
    calls["p_tis"] = np.asarray(probability, dtype=np.float64)
    calls["called"] = calls.p_tis >= threshold
    return calls


def _stratum_metrics(calls: pd.DataFrame) -> dict:
    labels = calls.label_tis.to_numpy().astype(bool)
    called = calls.called.to_numpy()
    fixed = evaluate_at_threshold(
        called.astype(np.float64), labels, calls.transcript_id.to_numpy(), 1.0
    )
    true_positives = fixed["true_positives"]
    false_positives = fixed["false_positives"]
    return {
        "n_codons": int(len(calls)),
        "n_positive": int(labels.sum()),
        "n_called": int(called.sum()),
        "recall": fixed["recall"],
        "precision": float(true_positives / max(1, true_positives + false_positives)),
        "fp_per_transcript": fixed["fp_per_transcript"],
    }


def summarize_calls(calls: pd.DataFrame) -> dict:
    """Caller metrics overall and per codon class (FP/transcript over all transcripts)."""
    n_transcripts = calls.transcript_id.nunique()
    summary = {"all": _stratum_metrics(calls)}
    for codon_class, group in calls.groupby("codon_class", observed=True):
        metrics = _stratum_metrics(group)
        metrics["fp_per_transcript"] = float(
            (group.called & ~group.label_tis.astype(bool)).sum() / max(1, n_transcripts)
        )
        summary[str(codon_class)] = metrics
    return summary


def _check_logged_ensemble(
    run_dir: Path, arm: str, calls: pd.DataFrame, summary: dict, band_rows: int
) -> None:
    """Require the logged TP/FP counts to match ours up to rows at the threshold.

    The evaluation scored with the unfolded sklearn pipeline, so logits agree only to float
    precision, and isotonic steps can turn that into a visible probability jump. Only rows
    whose call changes when every seed logit moves by ``LOGIT_MARGIN`` may differ.
    """
    metrics = pd.read_csv(run_dir / "metrics.tsv", sep="\t", dtype={"seed": str})
    logged = metrics[(metrics.arm == arm) & (metrics.seed == "ensemble")]
    if len(logged) != 1:
        raise RuntimeError(f"metrics.tsv has no unique ensemble row for arm {arm!r}")
    logged = logged.iloc[0]
    ours = summary["all"]
    counts = {
        "recall": (ours["recall"] * ours["n_positive"], ours["n_positive"]),
        "fp_per_transcript": (
            ours["fp_per_transcript"] * calls.transcript_id.nunique(),
            calls.transcript_id.nunique(),
        ),
    }
    for name, (observed, scale) in counts.items():
        if name not in logged:
            continue
        delta = abs(observed - float(logged[name]) * scale)
        if delta > band_rows + 1e-6:
            raise RuntimeError(
                f"calls do not reproduce the logged ensemble {name}: "
                f"{ours[name]:.6f} vs {float(logged[name]):.6f} "
                f"({delta:.0f} rows differ, {band_rows} within the threshold band)"
            )


def run(*, store: Path, run_dir: Path, arm: str, split: str, out: Path) -> dict:
    """Score ``split`` rows, verify against the logged ensemble, and write calls + summary."""
    store, run_dir, out = Path(store), Path(run_dir), Path(out)
    run_meta = json.loads((run_dir / "run.json").read_text())
    threshold = float(run_meta["ensemble_thresholds"][arm])
    artifacts = [
        LinearHeadArtifact.load(run_dir / f"{arm}_seed{seed}.npz") for seed in run_meta["seeds"]
    ]
    with np.load(run_dir / "row_selections.npz") as selections:
        rows = selections[split]
    manifest = pd.read_parquet(store / "manifest.parquet")
    print(f"scoring {len(rows):,} {split} rows with {len(artifacts)} {arm} seeds", flush=True)
    logits = score_logits(store / "embeddings", artifacts, rows)
    probability = ensemble_probability(artifacts, logits)
    calls = build_calls(manifest, rows, probability, threshold)
    summary = summarize_calls(calls)
    low = ensemble_probability(artifacts, logits, shift=-LOGIT_MARGIN)
    high = ensemble_probability(artifacts, logits, shift=LOGIT_MARGIN)
    band_rows = int(((low < threshold) & (high >= threshold)).sum())
    if split == "test":
        _check_logged_ensemble(run_dir, arm, calls, summary, band_rows)
    summary = {
        "arm": arm,
        "split": split,
        "threshold": threshold,
        "threshold_band_rows": band_rows,
        **summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(out, index=False)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/dense_exp_store")
    ap.add_argument("--run-dir", default="data/dense_txp_representation")
    ap.add_argument("--arm", default="ag_w8k")
    ap.add_argument("--split", default="test", help="A key of row_selections.npz.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"data/calls/{args.arm}_{args.split}_calls.parquet"
    summary = run(
        store=args.store, run_dir=args.run_dir, arm=args.arm, split=args.split, out=Path(out)
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
