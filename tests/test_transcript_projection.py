"""Tests for coefficient-fused transcript projection."""

import numpy as np

from tisiago.linear_head import LinearHeadArtifact
from tisiago.transcript_inference import TranscriptRequest
from tisiago.transcript_projection import (
    project_transcript_result,
    transcript_feature_blocks,
    txp_feature_keys,
    validate_txp_artifacts,
)


def _request():
    return TranscriptRequest(
        transcript_id="tx",
        sequence="A" * 20,
        positions=[2, 5, 8, 11, 14, 17],
        requests=[
            (9, 0, 0),
            (9, 3, 1),
            (9, 6, 2),
            (9, 9, 3),
            (4, 0, 2),
            (4, 3, 3),
            (4, 6, 4),
            (4, 9, 5),
        ],
    )


def test_transcript_feature_blocks_align_candidates_and_offsets():
    values = np.arange(12, dtype=np.float32).reshape(6, 2)

    rows, blocks = transcript_feature_blocks(
        _request(), {"blocks.28.mlp.l3": values}
    )

    np.testing.assert_array_equal(rows, [4, 9])
    np.testing.assert_array_equal(
        blocks["evo2/TXP/blocks.28.mlp.l3/off0.npy"], values[[2, 0]]
    )
    np.testing.assert_array_equal(
        blocks["evo2/TXP/blocks.28.mlp.l3/off9.npy"], values[[5, 3]]
    )


def test_projected_txp_partial_matches_blockwise_artifact_projection():
    keys = tuple(
        f"evo2/TXP/blocks.28.mlp.l3/off{offset}.npy"
        for offset in (0, 3, 6, 9)
    )
    artifact = LinearHeadArtifact(
        feature_keys=("other/model.npy", *keys),
        feature_dims=(1, 2, 2, 2, 2),
        coefficient=np.arange(9, dtype=np.float32) / 10,
        intercept=2.0,
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=0.5,
    )
    result = {"blocks.28.mlp.l3": np.arange(12, dtype=np.float32).reshape(6, 2)}
    rows, blocks = transcript_feature_blocks(_request(), result)

    projected_rows, partials = project_transcript_result(
        _request(), result, {"seed0": artifact}
    )

    np.testing.assert_array_equal(projected_rows, rows)
    expected = artifact.partial_logit_from_blocks({key: blocks[key] for key in keys})
    np.testing.assert_allclose(partials["seed0"], expected)


def test_txp_artifact_schema_validation_rejects_wrong_layer():
    artifact = LinearHeadArtifact(
        feature_keys=("evo2/TXP/wrong/off0.npy",),
        feature_dims=(2,),
        coefficient=np.ones(2),
        intercept=0.0,
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        operating_threshold=0.5,
    )

    try:
        validate_txp_artifacts(
            {"head": artifact}, txp_feature_keys(["blocks.28.mlp.l3"])
        )
        raise AssertionError("a wrong TXP layer should fail before extraction")
    except ValueError as error:
        assert "not configured" in str(error)
