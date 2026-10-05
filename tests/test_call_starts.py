"""Tests for discrete start-site calls from the frozen ensemble head."""

import json

import numpy as np
import pandas as pd
import pytest

from tisiago.call_starts import (
    build_calls,
    ensemble_probability,
    run,
    score_logits,
    score_rows,
    summarize_calls,
)
from tisiago.linear_head import LinearHeadArtifact

KEYS = ("a/off0.npy", "b/off0.npy")
DIMS = (2, 3)


def _artifact(seed: int, threshold: float = 0.5) -> LinearHeadArtifact:
    rng = np.random.default_rng(seed)
    return LinearHeadArtifact(
        feature_keys=KEYS,
        feature_dims=DIMS,
        coefficient=rng.normal(size=sum(DIMS)),
        intercept=float(rng.normal()),
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=threshold,
    )


def _store(tmp_path, n=40):
    rng = np.random.default_rng(0)
    root = tmp_path / "store" / "embeddings"
    blocks = {}
    for key, width in zip(KEYS, DIMS, strict=True):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        blocks[key] = rng.normal(size=(n, width)).astype(np.float16)
        np.save(path, blocks[key])
    manifest = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "transcript_id": [f"tx{i // 10}" for i in range(n)],
            "gene": [f"g{i // 10}" for i in range(n)],
            "chrom": "chr8",
            "gstart": np.arange(n) * 3,
            "strand": "+",
            "mrna_index": np.arange(n) * 3,
            "codon": "ATG",
            "codon_class": ["AUG" if i % 2 else "near_cognate" for i in range(n)],
            "label_tis": (np.arange(n) % 5 == 0).astype(int),
            "split": "test",
        }
    )
    manifest.to_parquet(tmp_path / "store" / "manifest.parquet")
    return tmp_path / "store", blocks, manifest


def test_score_rows_averages_calibrated_seed_predictions(tmp_path):
    store, blocks, _ = _store(tmp_path)
    artifacts = [_artifact(1), _artifact(2)]
    rows = np.array([3, 7, 8, 20, 39])

    observed = score_rows(store / "embeddings", artifacts, rows, chunk_size=2)

    selected = {key: blocks[key][rows].astype(np.float32) for key in KEYS}
    expected = np.mean([a.predict_blocks(selected) for a in artifacts], axis=0)
    np.testing.assert_allclose(observed, expected, rtol=1e-6)


def test_build_calls_thresholds_and_keeps_row_alignment(tmp_path):
    _, _, manifest = _store(tmp_path)
    rows = np.array([5, 1, 30])
    probability = np.array([0.9, 0.2, 0.5])

    calls = build_calls(manifest, rows, probability, threshold=0.5)

    assert calls.row_idx.tolist() == [5, 1, 30]
    assert calls.called.tolist() == [True, False, True]
    assert calls.p_tis.tolist() == [0.9, 0.2, 0.5]
    assert {"transcript_id", "codon_class", "label_tis", "mrna_index"} <= set(calls.columns)


def test_summarize_calls_reports_strata():
    calls = pd.DataFrame(
        {
            "transcript_id": ["t1", "t1", "t2", "t2"],
            "codon_class": ["AUG", "near_cognate", "AUG", "near_cognate"],
            "label_tis": [1, 0, 1, 1],
            "called": [True, True, False, True],
        }
    )

    summary = summarize_calls(calls)

    assert summary["all"]["recall"] == pytest.approx(2 / 3)
    assert summary["all"]["precision"] == pytest.approx(2 / 3)
    assert summary["all"]["fp_per_transcript"] == pytest.approx(0.5)
    assert summary["AUG"]["recall"] == pytest.approx(0.5)
    assert summary["near_cognate"]["recall"] == pytest.approx(1.0)


def _run_dir(tmp_path, artifacts, rows, threshold, recall):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for seed, artifact in enumerate(artifacts):
        artifact.save(run_dir / f"arm_seed{seed}.npz")
    np.savez(run_dir / "row_selections.npz", test=rows)
    (run_dir / "run.json").write_text(
        json.dumps(
            {"seeds": list(range(len(artifacts))), "ensemble_thresholds": {"arm": threshold}}
        )
    )
    pd.DataFrame(
        [{"arm": "arm", "seed": "ensemble", "threshold": threshold, "recall": recall}]
    ).to_csv(run_dir / "metrics.tsv", sep="\t", index=False)
    return run_dir


def test_run_writes_calls_and_checks_logged_ensemble(tmp_path):
    store, _, manifest = _store(tmp_path)
    artifacts = [_artifact(1), _artifact(2)]
    rows = np.arange(0, 40, 2)
    probability = score_rows(store / "embeddings", artifacts, rows)
    threshold = float(np.median(probability))
    expected = summarize_calls(build_calls(manifest, rows, probability, threshold))
    run_dir = _run_dir(tmp_path, artifacts, rows, threshold, expected["all"]["recall"])
    out = tmp_path / "calls" / "arm_test_calls.parquet"

    summary = run(store=store, run_dir=run_dir, arm="arm", split="test", out=out)

    calls = pd.read_parquet(out)
    assert len(calls) == len(rows)
    assert calls.called.sum() == (probability >= threshold).sum()
    assert summary["all"]["recall"] == pytest.approx(expected["all"]["recall"])
    assert json.loads(out.with_suffix(".summary.json").read_text())["threshold"] == threshold


def test_run_fails_when_logged_ensemble_does_not_reproduce(tmp_path):
    store, _, _ = _store(tmp_path)
    artifacts = [_artifact(1), _artifact(2)]
    rows = np.arange(0, 40, 2)
    run_dir = _run_dir(tmp_path, artifacts, rows, threshold=0.5, recall=0.123456)

    with pytest.raises(RuntimeError, match="do not reproduce"):
        run(store=store, run_dir=run_dir, arm="arm", split="test", out=tmp_path / "c.parquet")


def test_run_tolerates_flips_only_within_threshold_band(tmp_path):
    store, _, manifest = _store(tmp_path)
    artifacts = [_artifact(1), _artifact(2)]
    rows = np.arange(0, 40, 2)
    logits = score_logits(store / "embeddings", artifacts, rows)
    probability = ensemble_probability(artifacts, logits)
    border = int(np.argsort(probability)[10])
    # Threshold sits between the border row's p at its logit and at logit + margin/2.
    threshold = float(ensemble_probability(artifacts, logits[:, [border]], shift=5e-4)[0])
    ours = summarize_calls(build_calls(manifest, rows, probability, threshold))["all"]
    is_positive = bool(manifest.label_tis.iloc[rows[border]])
    n_transcripts = manifest.iloc[rows].transcript_id.nunique()
    logged_recall = ours["recall"] + (1 / ours["n_positive"] if is_positive else 0.0)
    run_dir = _run_dir(tmp_path, artifacts, rows, threshold, logged_recall)
    metrics = pd.read_csv(run_dir / "metrics.tsv", sep="\t")
    metrics["fp_per_transcript"] = ours["fp_per_transcript"] + (
        0.0 if is_positive else 1 / n_transcripts
    )
    metrics.to_csv(run_dir / "metrics.tsv", sep="\t", index=False)

    summary = run(store=store, run_dir=run_dir, arm="arm", split="test", out=tmp_path / "c.pq")

    assert summary["threshold_band_rows"] >= 1
