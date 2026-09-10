"""Executable semantic checks for frozen gruyerenome embedding backends."""

from __future__ import annotations

import argparse

import numpy as np


def _layer_map(output) -> dict[str, np.ndarray]:
    """Normalize an AlphaGenome array or Evo2 layer dictionary."""
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
    for dense_output, sparse_output, selected in zip(
        dense, sparse, positions, strict=True
    ):
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


def validate_causal_suffix_invariance(
    backend,
    sequence: str,
    position: int,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Require an Evo2 state to be invariant to mutations strictly after it."""
    if not 0 <= position < len(sequence) - 1:
        raise ValueError("causal test position must have at least one suffix nucleotide")
    replacement = {"A": "C", "C": "G", "G": "T", "T": "A"}
    suffix_index = position + 1
    mutated = list(sequence)
    mutated[suffix_index] = replacement[mutated[suffix_index]]
    reference, changed = backend.embed_positions(
        [sequence, "".join(mutated)], [[position], [position]]
    )
    ref_layers, changed_layers = _layer_map(reference), _layer_map(changed)
    if ref_layers.keys() != changed_layers.keys():
        raise AssertionError("causal comparison layer keys differ")
    errors = {}
    for layer in ref_layers:
        expected, observed = ref_layers[layer], changed_layers[layer]
        error = float(
            np.max(np.abs(expected.astype(np.float64) - observed))
        )
        errors[layer] = error
        if not np.allclose(expected, observed, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{layer}: state at {position} changed after suffix mutation at "
                f"{suffix_index}; max_abs_error={error:.6g}, "
                f"reference_absmax={float(np.max(np.abs(expected))):.6g}"
            )
    return errors


def mutation_influence_profile(
    backend,
    sequence: str,
    position: int,
    *,
    deltas: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3, 4, 8, 16, 32),
) -> dict[str, dict[int, tuple[float, float]]]:
    """Measure how mutations around one position alter its extracted state.

    Returns ``layer -> delta -> (max_abs_effect, mean_abs_effect)``. This is a
    diagnostic rather than a pass/fail test: a causal, index-aligned state should
    respond at non-positive deltas and remain invariant at positive deltas.
    """
    mutation_positions = [position + delta for delta in deltas]
    if any(not 0 <= target < len(sequence) for target in mutation_positions):
        raise ValueError("mutation profile extends outside the sequence")
    replacement = {"A": "C", "C": "G", "G": "T", "T": "A"}
    sequences = [sequence]
    for target in mutation_positions:
        mutated = list(sequence)
        mutated[target] = replacement[mutated[target]]
        sequences.append("".join(mutated))
    outputs = backend.embed_positions(sequences, [[position]] * len(sequences))
    reference = _layer_map(outputs[0])
    profile: dict[str, dict[int, tuple[float, float]]] = {
        layer: {} for layer in reference
    }
    for delta, output in zip(deltas, outputs[1:], strict=True):
        changed = _layer_map(output)
        if reference.keys() != changed.keys():
            raise AssertionError("mutation-profile layer keys differ")
        for layer in reference:
            difference = np.abs(
                reference[layer].astype(np.float64)
                - changed[layer].astype(np.float64)
            )
            profile[layer][delta] = (
                float(np.max(difference)),
                float(np.mean(difference)),
            )
    return profile


def validate_current_base_sensitivity(
    backend,
    sequence: str,
    position: int,
    *,
    minimum_effect: float = 1e-6,
) -> dict[str, float]:
    """Require the state at an Evo2 position to encode its current nucleotide.

    Together with suffix invariance, this pins the indexing convention used by
    TIS features: state ``h_i`` includes base ``i`` but no base after ``i``.
    """
    if not 0 <= position < len(sequence):
        raise ValueError("sensitivity test position is outside the sequence")
    replacement = {"A": "C", "C": "G", "G": "T", "T": "A"}
    mutated = list(sequence)
    mutated[position] = replacement[mutated[position]]
    reference, changed = backend.embed_positions(
        [sequence, "".join(mutated)], [[position], [position]]
    )
    ref_layers, changed_layers = _layer_map(reference), _layer_map(changed)
    if ref_layers.keys() != changed_layers.keys():
        raise AssertionError("current-base comparison layer keys differ")
    effects = {
        layer: float(
            np.max(
                np.abs(
                    ref_layers[layer].astype(np.float64)
                    - changed_layers[layer].astype(np.float64)
                )
            )
        )
        for layer in ref_layers
    }
    too_small = {layer: effect for layer, effect in effects.items() if effect <= minimum_effect}
    if too_small:
        raise AssertionError(
            "state did not respond to its current nucleotide: "
            + ", ".join(f"{layer}={effect:.3g}" for layer, effect in too_small.items())
        )
    return effects


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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print mutation influence around the tested Evo2 state",
    )
    parser.add_argument(
        "--allow-suffix-sensitivity",
        action="store_true",
        help=(
            "Report suffix sensitivity as a non-fatal audit result. Use only for "
            "input designs, such as TXP, that never rely on suffix invariance."
        ),
    )
    args = parser.parse_args()

    from gruyerenome import Config, load_backend

    config = Config.from_yaml(args.config)
    backend = load_backend(config)
    sequences = _validation_sequences(args.length)
    positions = [
        [args.length // 4, args.length // 2, 3 * args.length // 4]
        for _ in sequences
    ]
    sparse_error = validate_sparse_dense_parity(
        backend, sequences, positions, rtol=args.rtol, atol=args.atol
    )
    batch_error = validate_batch_singleton_parity(
        backend, sequences, positions, rtol=args.rtol, atol=args.atol
    )
    print(f"PASS sparse-vs-dense max_abs_error={sparse_error:.3g}")
    print(f"PASS batch-vs-singleton max_abs_error={batch_error:.3g}")
    if config.model == "evo2":
        if args.profile:
            profile = mutation_influence_profile(
                backend,
                sequences[0],
                args.length // 2,
            )
            for layer, effects in profile.items():
                detail = ", ".join(
                    f"{delta:+d}:{maximum:.3g}/{mean:.3g}"
                    for delta, (maximum, mean) in effects.items()
                )
                print(
                    f"DIAGNOSTIC mutation influence {layer} "
                    f"(delta:max/mean): {detail}",
                    flush=True,
                )
        try:
            errors = validate_causal_suffix_invariance(
                backend,
                sequences[0],
                args.length // 2,
                rtol=args.rtol,
                atol=args.atol,
            )
        except AssertionError as error:
            if not args.allow_suffix_sensitivity:
                raise
            print(f"AUDIT suffix-sensitive backend (allowed by input design): {error}")
        else:
            detail = ", ".join(
                f"{layer}={error:.3g}" for layer, error in errors.items()
            )
            print(f"PASS causal suffix invariance max_abs_error: {detail}")
        effects = validate_current_base_sensitivity(
            backend,
            sequences[0],
            args.length // 2,
        )
        detail = ", ".join(f"{layer}={effect:.3g}" for layer, effect in effects.items())
        print(f"PASS current-base sensitivity max_abs_effect: {detail}")


if __name__ == "__main__":
    main()
