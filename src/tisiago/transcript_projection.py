"""Coefficient-fused projection of one transcript backend result."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tisiago.linear_head import LinearHeadArtifact
from tisiago.transcript_inference import TRANSCRIPT_LENGTH_TAG, TRANSCRIPT_OFFSETS


def transcript_feature_blocks(
    request, result: Mapping[str, np.ndarray], *, dtype=np.float16
):
    """Align backend rows into candidate blocks quantized like stored features."""
    member_rows = np.fromiter(
        (row for row, _, _ in request.requests), dtype=np.int64
    )
    member_offsets = np.fromiter(
        (offset for _, offset, _ in request.requests), dtype=np.int64
    )
    source_rows = np.fromiter(
        (source for _, _, source in request.requests), dtype=np.int64
    )
    output_rows, destination_rows = np.unique(member_rows, return_inverse=True)
    blocks = {}
    for layer, raw_values in result.items():
        values = np.asarray(raw_values)
        if values.ndim != 2 or len(values) != len(request.positions):
            raise ValueError(
                f"{layer}: expected ({len(request.positions)}, width), got {values.shape}"
            )
        for offset in TRANSCRIPT_OFFSETS:
            selected = member_offsets == offset
            if int(selected.sum()) != len(output_rows):
                raise ValueError(
                    f"{layer}/off{offset}: expected one value per candidate row"
                )
            block = np.empty((len(output_rows), values.shape[1]), dtype=dtype)
            block[destination_rows[selected]] = values[source_rows[selected]]
            key = f"evo2/{TRANSCRIPT_LENGTH_TAG}/{layer}/off{offset}.npy"
            blocks[key] = block
    if not blocks:
        raise ValueError("backend result contains no feature layers")
    return output_rows, blocks


def txp_feature_keys(layers) -> set[str]:
    """Return the persisted feature-key schema for configured TXP layers."""
    return {
        f"evo2/{TRANSCRIPT_LENGTH_TAG}/{layer}/off{offset}.npy"
        for layer in layers
        for offset in TRANSCRIPT_OFFSETS
    }


def validate_txp_artifacts(
    artifacts: Mapping[str, LinearHeadArtifact], configured_keys: set[str]
) -> None:
    """Validate names and model-specific schemas before loading the backend."""
    if not artifacts:
        raise ValueError("at least one linear head artifact is required")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("linear head artifact names must be unique")
    for name, artifact in artifacts.items():
        selected = {
            key
            for key in artifact.feature_keys
            if key.startswith(f"evo2/{TRANSCRIPT_LENGTH_TAG}/")
        }
        if not selected:
            raise ValueError(f"{name}: artifact contains no TXP feature blocks")
        unexpected = sorted(selected - configured_keys)
        if unexpected:
            raise ValueError(f"{name}: TXP feature blocks are not configured: {unexpected}")


def project_transcript_result(
    request,
    result: Mapping[str, np.ndarray],
    artifacts: Mapping[str, LinearHeadArtifact],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return additive TXP logits without building candidate-by-hidden-width blocks."""
    if not artifacts:
        raise ValueError("at least one linear head artifact is required")
    member_rows = np.fromiter(
        (row for row, _, _ in request.requests), dtype=np.int64
    )
    member_offsets = np.fromiter(
        (offset for _, offset, _ in request.requests), dtype=np.int64
    )
    source_rows = np.fromiter(
        (source for _, _, source in request.requests), dtype=np.int64
    )
    output_rows, destination_rows = np.unique(member_rows, return_inverse=True)
    expected = {
        name: {
            key
            for key in artifact.feature_keys
            if key.startswith(f"evo2/{TRANSCRIPT_LENGTH_TAG}/")
        }
        for name, artifact in artifacts.items()
    }
    for name, keys in expected.items():
        if not keys:
            raise ValueError(f"{name}: artifact contains no TXP feature blocks")
    seen = {name: set() for name in artifacts}
    projected = {
        name: np.zeros(len(output_rows), dtype=np.float32) for name in artifacts
    }
    slices = {name: artifact.feature_slices() for name, artifact in artifacts.items()}

    for layer, raw_values in result.items():
        raw_values = np.asarray(raw_values)
        if raw_values.ndim != 2 or len(raw_values) != len(request.positions):
            raise ValueError(
                f"{layer}: expected ({len(request.positions)}, width), got {raw_values.shape}"
            )
        if not np.isfinite(raw_values).all():
            raise ValueError(f"{layer}: backend result contains non-finite values")
        # Heads are fitted on persisted fp16 arrays. Quantize before projection so
        # production scores use exactly the same feature distribution.
        values = raw_values.astype(np.float16).astype(np.float32)
        for offset in TRANSCRIPT_OFFSETS:
            selected = member_offsets == offset
            if int(selected.sum()) != len(output_rows):
                raise ValueError(
                    f"{layer}/off{offset}: expected one value per candidate row"
                )
            key = f"evo2/{TRANSCRIPT_LENGTH_TAG}/{layer}/off{offset}.npy"
            participating = [name for name in artifacts if key in expected[name]]
            if not participating:
                continue
            coefficients = []
            for name in participating:
                coefficient = artifacts[name].coefficient[slices[name][key]]
                if coefficient.shape != (values.shape[1],):
                    raise ValueError(
                        f"{name}/{key}: coefficient width {len(coefficient)} differs "
                        f"from backend width {values.shape[1]}"
                    )
                coefficients.append(coefficient)
                seen[name].add(key)
            position_logits = values @ np.stack(coefficients, axis=1)
            source = source_rows[selected]
            destination = destination_rows[selected]
            for column, name in enumerate(participating):
                projected[name][destination] += position_logits[source, column]

    for name in artifacts:
        missing = sorted(expected[name] - seen[name])
        if missing:
            raise ValueError(f"{name}: backend result is missing feature blocks: {missing}")
    return output_rows, projected
