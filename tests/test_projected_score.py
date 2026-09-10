"""Tests for scalar partial-logit assembly and completion."""

import numpy as np

from tisiago.linear_head import LinearHeadArtifact
from tisiago.projected_score import assemble_partial_logits, score_projected_blocks


def _artifact():
    return LinearHeadArtifact(
        feature_keys=("other/model.npy", "evo2/TXP/layer/off0.npy"),
        feature_dims=(2, 1),
        coefficient=np.array([2.0, -1.0, 0.5]),
        intercept=0.25,
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=0.5,
    )


def test_assemble_partial_logits_restores_global_row_order(tmp_path):
    np.savez(
        tmp_path / "part0.npz",
        row_idx=np.array([2, 0]),
        **{"partial_logit::seed0": np.array([[3.0], [1.0]], np.float32)},
    )
    np.savez(
        tmp_path / "part1.npz",
        row_idx=np.array([1]),
        **{"partial_logit::seed0": np.array([[2.0]], np.float32)},
    )

    partials = assemble_partial_logits(tmp_path, "part*.npz", 3, ("seed0",))

    np.testing.assert_array_equal(partials["seed0"], [1.0, 2.0, 3.0])


def test_projected_score_matches_full_block_prediction(tmp_path):
    artifact = _artifact()
    external = np.array([[1.0, 3.0], [2.0, -1.0], [4.0, 2.0]], np.float16)
    txp = np.array([[4.0], [6.0], [2.0]], np.float16)
    path = tmp_path / "other"
    path.mkdir()
    np.save(path / "model.npy", external)
    txp_partial = artifact.partial_logit_from_blocks(
        {"evo2/TXP/layer/off0.npy": txp}
    )

    observed = score_projected_blocks(
        {"seed0": artifact},
        {"seed0": txp_partial},
        tmp_path,
        chunk_size=2,
    )["seed0"]

    expected = artifact.predict_blocks(
        {"other/model.npy": external, "evo2/TXP/layer/off0.npy": txp}
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
