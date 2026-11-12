import numpy as np

from tisiago.backend_contract import (
    validate_batch_singleton_parity,
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


def test_backend_contract_accepts_correct_sparse_and_batch_semantics():
    backend = FakeCausalBackend()
    sequences = ["ACGT" * 16, "TGCA" * 16]
    positions = [[3, 17, 40], [4, 18, 41]]
    assert validate_sparse_dense_parity(backend, sequences, positions) == 0.0
    assert validate_batch_singleton_parity(backend, sequences, positions) == 0.0
