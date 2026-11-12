"""Executable semantic checks for frozen gruyerenome embedding backends."""

from __future__ import annotations

import argparse

import numpy as np


def _layer_map(output) -> dict[str, np.ndarray]:
    """Normalize a backend array or layer dictionary."""
    if isinstance(output, dict):
        return {str(key): np.asarray(value) for key, value in output.items()}
    return {"embedding": np.asarray(output)}


def validate_sparse_dense_parity(
    backend,
    sequences: list[str],
    positions: list[list[int]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> float:
    """Require sparse position extraction to match rows from dense extraction."""
    dense = backend.embed(sequences)
    sparse = backend.embed_positions(sequences, positions)
    if not (len(dense) == len(sparse) == len(sequences)):
        raise AssertionError("backend returned the wrong number of sequences")
    primary_layer = getattr(backend, "layer_names", [None])[0]
    maximum_error = 0.0
    for dense_output, sparse_output, selected in zip(dense, sparse, positions, strict=True):
        sparse_layers = _layer_map(sparse_output)
        key = primary_layer if primary_layer in sparse_layers else "embedding"
        expected = np.asarray(dense_output)[np.asarray(selected, dtype=np.int64)]
        observed = sparse_layers[key]
        if expected.shape != observed.shape:
            raise AssertionError(
                f"sparse/dense shape mismatch: {expected.shape} != {observed.shape}"
            )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(expected.astype(np.float64) - observed))),
        )
        if not np.allclose(expected, observed, rtol=rtol, atol=atol):
            raise AssertionError(f"sparse extraction differs from dense rows for {key}")
    return maximum_error


def validate_batch_singleton_parity(
    backend,
    sequences: list[str],
    positions: list[list[int]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> float:
    """Require a multi-sequence call to match independent singleton calls."""
    batched = backend.embed_positions(sequences, positions)
    singleton = [
        backend.embed_positions([sequence], [selected])[0]
        for sequence, selected in zip(sequences, positions, strict=True)
    ]
    maximum_error = 0.0
    for batch_output, single_output in zip(batched, singleton, strict=True):
        batch_layers = _layer_map(batch_output)
        single_layers = _layer_map(single_output)
        if batch_layers.keys() != single_layers.keys():
            raise AssertionError("batch and singleton layer keys differ")
        for layer in batch_layers:
            expected, observed = single_layers[layer], batch_layers[layer]
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(expected.astype(np.float64) - observed))),
            )
            if not np.allclose(expected, observed, rtol=rtol, atol=atol):
                raise AssertionError(f"batch extraction differs from singleton for {layer}")
    return maximum_error


def _validation_sequences(length: int) -> list[str]:
    """Create deterministic, non-repetitive equal-length DNA validation inputs."""
    if length < 32:
        raise ValueError("validation sequence length must be at least 32")
    rng = np.random.default_rng(1729)
    bases = np.asarray(list("ACGT"))
    return ["".join(rng.choice(bases, size=length)) for _ in range(2)]


def main() -> None:
    """Run gruyerenome extraction-contract checks on the configured GPU backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    from gruyerenome import Config, load_backend

    config = Config.from_yaml(args.config)
    backend = load_backend(config)
    sequences = _validation_sequences(args.length)
    positions = [[args.length // 4, args.length // 2, 3 * args.length // 4] for _ in sequences]
    sparse_error = validate_sparse_dense_parity(
        backend, sequences, positions, rtol=args.rtol, atol=args.atol
    )
    batch_error = validate_batch_singleton_parity(
        backend, sequences, positions, rtol=args.rtol, atol=args.atol
    )
    print(f"PASS sparse-vs-dense max_abs_error={sparse_error:.3g}")
    print(f"PASS batch-vs-singleton max_abs_error={batch_error:.3g}")


if __name__ == "__main__":
    main()
