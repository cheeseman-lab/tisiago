"""Open-frame features computed in JAX from one-hot input (seqconv head)."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tisiago.seq2func.augment import reverse_complement  # noqa: E402
from tisiago.seq2func.orf_features import (  # noqa: E402
    FEATURE_NAMES,
    NUM_FEATURES,
    orf_features,
    strand_features_np,
)

BASES = "ACGT"
STOPS = ("TAA", "TAG", "TGA")


def onehot(seq: str) -> np.ndarray:
    out = np.zeros((len(seq), 4), np.float32)
    for i, base in enumerate(seq):
        if base in BASES:
            out[i, BASES.index(base)] = 1.0
    return out


def rc(seq: str) -> str:
    return seq[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def scaled(distance: int, cap: int) -> float:
    return np.log1p(distance) / np.log1p(cap)


def reference_strand(seq: str, cap: int) -> np.ndarray:
    """Brute-force (S, 9) plus-strand features: stop dist k0..2, ATG dist k0..2, open k0..2."""
    n = len(seq)
    stops = {j for j in range(n - 2) if seq[j : j + 3] in STOPS}
    atgs = {j for j in range(n - 2) if seq[j : j + 3] == "ATG"}
    out = np.zeros((n, 9), np.float32)
    for i in range(n):
        for k in range(3):
            c = i - k
            in_frame = range(c, -1, -3) if c >= 0 else range(0)
            last_stop = next((j for j in in_frame if j in stops), -1)
            last_atg = next((j for j in in_frame if j in atgs), -1)
            out[i, k] = scaled(min(i - last_stop, cap) if last_stop >= 0 else cap, cap)
            out[i, 3 + k] = scaled(min(i - last_atg, cap) if last_atg >= 0 else cap, cap)
            out[i, 6 + k] = float(last_atg > last_stop)
    return out


def reference(seq: str, cap: int) -> np.ndarray:
    minus = reference_strand(rc(seq), cap)[::-1]
    return np.concatenate([reference_strand(seq, cap), minus], axis=1)


def features(seq: str, cap: int = 3000) -> np.ndarray:
    return np.asarray(orf_features(jnp.asarray(onehot(seq)[None]), cap=cap))[0]


def test_feature_layout():
    assert NUM_FEATURES == 18 == len(FEATURE_NAMES)
    assert FEATURE_NAMES[0] == "plus_stop_dist_k0"
    assert FEATURE_NAMES[8] == "plus_open_k2"
    assert FEATURE_NAMES[9] == "minus_stop_dist_k0"
    assert features("ACGTACGT").shape == (8, 18)


def test_hand_written_plus_strand():
    # A0 T1 G2 A3 A4 A5 T6 A7 G8 C9 C10 C11: ATG at 0; stops TGA at 1 and TAG at 6.
    cap = 100
    f = features("ATGAAATAGCCC", cap=cap)
    # i=5, k=2: codon at 3 (AAA); in-frame starts {3, 0}: no stop, ATG at 0 -> open ORF.
    assert f[5, 2] == pytest.approx(scaled(cap, cap))
    assert f[5, 5] == pytest.approx(scaled(5, cap))
    assert f[5, 8] == 1.0
    # i=8, k=2: codon at 6 is TAG (stop after the ATG) -> closed.
    assert f[8, 2] == pytest.approx(scaled(2, cap))
    assert f[8, 5] == pytest.approx(scaled(8, cap))
    assert f[8, 8] == 0.0
    # i=0, k=0: the ATG itself is open; distance 0.
    assert f[0, 3] == pytest.approx(0.0)
    assert f[0, 6] == 1.0
    # i=7, k=0: frame of start 7 (residue 1) has the TGA at 1, no ATG.
    assert f[7, 0] == pytest.approx(scaled(6, cap))
    assert f[7, 3] == pytest.approx(scaled(cap, cap))
    assert f[7, 6] == 0.0
    # i < k: the codon would start before the window -> capped, closed.
    assert f[1, 2] == pytest.approx(scaled(cap, cap))
    assert f[1, 8] == 0.0


def test_hand_written_minus_strand():
    # Reverse complement of the plus example: the ATG reads 3'->5' at forward positions 11..9.
    cap = 100
    f = features(rc("ATGAAATAGCCC"), cap=cap)
    # forward i=11 is minus-strand position 0 (the A of ATG), k=0.
    assert f[11, 9 + 3] == pytest.approx(0.0)
    assert f[11, 9 + 6] == 1.0
    # forward i=6 is minus position 5, k=2: open from the ATG 5 nt upstream on minus.
    assert f[6, 9 + 5] == pytest.approx(scaled(5, cap))
    assert f[6, 9 + 8] == 1.0
    # the plus strand of this sequence has no ATG at all.
    assert np.allclose(f[:, 3:6], scaled(cap, cap))
    assert (f[:, 6:9] == 0).all()


def test_distances_are_capped():
    seq = "ATG" + "C" * 40
    f = features(seq, cap=10)
    assert f[30, 3] == pytest.approx(scaled(10, 10))
    assert f[5, 5] == pytest.approx(scaled(5, 10))  # k=2: codon at 3, in frame with the ATG
    assert f[30, 6] == 1.0  # still open: no stop in frame


def test_n_bases_make_no_codons():
    f = features("ANG" + "TNA" + "NNN", cap=50)
    assert np.allclose(f[:, [0, 1, 2, 3, 4, 5]], scaled(50, 50))
    assert (f[:, 6:9] == 0).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("length", [31, 32, 33, 60])
def test_matches_brute_force_reference(seed, length):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGTTAN"), size=length))
    np.testing.assert_allclose(features(seq, cap=17), reference(seq, 17), rtol=1e-6, atol=1e-6)


def test_reverse_complement_swaps_strands_and_flips():
    rng = np.random.default_rng(3)
    seq = "".join(rng.choice(list("ACGTTAG"), size=90))
    x = onehot(seq)
    targets = np.zeros((90, 2), np.float32)
    x_rc, _, _ = reverse_complement(x, targets, targets.astype(bool), [(0, 1)])
    f = np.asarray(orf_features(jnp.asarray(x[None])))[0]
    f_rc = np.asarray(orf_features(jnp.asarray(x_rc[None])))[0]
    expected = np.concatenate([f[::-1, 9:], f[::-1, :9]], axis=1)
    np.testing.assert_array_equal(f_rc, expected)


def test_features_lie_in_unit_interval():
    rng = np.random.default_rng(4)
    f = features("".join(rng.choice(list("ACGTTAGN"), size=500)), cap=60)
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert f[:, :6].max() == pytest.approx(1.0)


def test_batched_and_jittable():
    seqs = ["ATGAAATAGCCC", "CCCATGTGAAAA"]
    x = jnp.asarray(np.stack([onehot(s) for s in seqs]))
    out = jax.jit(lambda s: orf_features(s, cap=100))(x)
    assert out.shape == (2, 12, 18) and out.dtype == jnp.float32
    for b, seq in enumerate(seqs):
        np.testing.assert_allclose(np.asarray(out[b]), reference(seq, 100), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("length", [1, 2, 31, 60, 301])
def test_numpy_strand_features_match_jax_and_reference(seed, length):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGTTAN"), size=length))
    got = strand_features_np(onehot(seq), cap=17)
    assert got.shape == (length, 9) and got.dtype == np.float32
    np.testing.assert_allclose(got, reference_strand(seq, 17), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got, features(seq, cap=17)[:, :9], rtol=1e-6, atol=1e-6)
