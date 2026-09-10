"""Tests for portable calibrated linear-head artifacts."""

import numpy as np

from tisiago.caller import fit_calibrated_head
from tisiago.linear_head import LinearHeadArtifact


def test_linear_head_artifact_roundtrip_matches_fitted_head(tmp_path):
    rng = np.random.default_rng(4)
    train = rng.normal(size=(200, 5)).astype(np.float32)
    calibration = rng.normal(size=(100, 5)).astype(np.float32)
    test = rng.normal(size=(50, 5)).astype(np.float32)
    y_train = (train[:, 0] - train[:, 3] > 0).astype(int)
    y_calibration = (calibration[:, 0] - calibration[:, 3] > 0).astype(int)
    fitted = fit_calibrated_head(train, y_train, calibration, y_calibration)
    artifact = LinearHeadArtifact.from_fitted_head(
        fitted,
        ["left", "right"],
        [2, 3],
        operating_threshold=0.25,
    )
    path = tmp_path / "head.npz"
    artifact.save(path)

    restored = LinearHeadArtifact.load(path)
    observed = restored.predict_blocks({"left": test[:, :2], "right": test[:, 2:]})

    np.testing.assert_allclose(observed, fitted["predict"](test), rtol=1e-6, atol=1e-7)
    assert restored.operating_threshold == 0.25


def test_linear_head_artifact_rejects_missing_blocks():
    artifact = LinearHeadArtifact(
        feature_keys=("one",),
        feature_dims=(2,),
        coefficient=np.ones(2),
        intercept=0.0,
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=0.5,
    )

    try:
        artifact.predict_blocks({})
        raise AssertionError("missing block should fail")
    except ValueError as error:
        assert "missing" in str(error)


def test_partial_logits_sum_to_full_logit_with_one_intercept():
    artifact = LinearHeadArtifact(
        feature_keys=("left", "right"),
        feature_dims=(2, 1),
        coefficient=np.array([2.0, -1.0, 0.5]),
        intercept=0.25,
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=0.5,
    )
    left = np.array([[1.0, 3.0], [2.0, -1.0]])
    right = np.array([[4.0], [6.0]])

    observed = artifact.partial_logit_from_blocks({"left": left})
    observed += artifact.partial_logit_from_blocks({"right": right})
    observed += artifact.intercept

    expected = artifact.logit_from_blocks({"left": left, "right": right})
    np.testing.assert_allclose(observed, expected)


def test_linear_head_rejects_invalid_calibration_curve():
    try:
        LinearHeadArtifact(
            feature_keys=("x",),
            feature_dims=(1,),
            coefficient=np.ones(1),
            intercept=0.0,
            isotonic_x=np.array([0.0, 1.0]),
            isotonic_y=np.array([0.8, 0.2]),
            operating_threshold=0.5,
        )
        raise AssertionError("decreasing isotonic probabilities should fail")
    except ValueError as error:
        assert "monotone" in str(error)
