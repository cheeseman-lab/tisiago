import numpy as np

from tisiago.backend_contract import (
    mutation_influence_profile,
    validate_batch_singleton_parity,
    validate_causal_suffix_invariance,
    validate_current_base_sensitivity,
    validate_sparse_dense_parity,
)


class FakeCausalBackend:
    """Small deterministic backend with per-position causal states."""

    layer_names = ["layer"]

    @staticmethod
    def _embed(sequence):
        encoded = np.frombuffer(sequence.encode(), dtype=np.uint8).astype(np.float32)
        cumulative = np.cumsum(encoded)
        return np.column_stack([encoded, cumulative])

    def embed(self, sequences):
        """Return dense embeddings for each input sequence."""
        return [self._embed(sequence) for sequence in sequences]

    def embed_positions(self, sequences, positions):
        """Gather requested positions from each dense embedding."""
        return [
            {"layer": self._embed(sequence)[selected]}
            for sequence, selected in zip(sequences, positions, strict=True)
        ]


def test_backend_contract_accepts_correct_sparse_batch_and_causal_semantics():
    backend = FakeCausalBackend()
    sequences = ["ACGT" * 16, "TGCA" * 16]
    positions = [[3, 17, 40], [4, 18, 41]]
    assert validate_sparse_dense_parity(backend, sequences, positions) == 0.0
    assert validate_batch_singleton_parity(backend, sequences, positions) == 0.0
    assert validate_causal_suffix_invariance(backend, sequences[0], 20) == {"layer": 0.0}
    effects = validate_current_base_sensitivity(backend, sequences[0], 20)
    assert effects["layer"] > 0


def test_mutation_profile_distinguishes_prefix_from_suffix():
    """A causal backend responds to prefix/current mutations, never suffix mutations."""
    backend = FakeCausalBackend()
    sequence = "ACGT" * 16
    profile = mutation_influence_profile(
        backend,
        sequence,
        20,
        deltas=(-2, -1, 0, 1, 2),
    )["layer"]
    assert all(profile[delta][0] > 0 for delta in (-2, -1, 0))
    assert all(profile[delta] == (0.0, 0.0) for delta in (1, 2))
