import sys
import types

import numpy as np
import pandas as pd
import pytest

from tisiago.truth.psite_check import (
    RIBOTISH_MINUS_SHIFT,
    PsiteConventionError,
    bigwig_reader,
    check_psite_convention,
    check_same_convention,
    psite_offsets,
)

CHROM_SIZE = 100_000
N = 30


def _starts(n: int = N) -> pd.DataFrame:
    gpos = 1_000 + 500 * np.arange(2 * n)
    return pd.DataFrame(
        {
            "chrom": ["chr1"] * n + ["chr2"] * n,
            "strand": ["+"] * n + ["-"] * n,
            "gpos": gpos,
        }
    )


def _reader(starts: pd.DataFrame, plus_offset: int, minus_genomic: int, seed: int = 0):
    """Synthetic uniform background plus a peak of 20 per start.

    The peak sits at transcript offset ``plus_offset`` on + and genomic ``gpos + minus_genomic``
    on -.
    """
    rng = np.random.default_rng(seed)
    tracks = {}
    for chrom in ("chr1", "chr2"):
        for strand in ("+", "-"):
            tracks[(chrom, strand)] = rng.uniform(0.0, 1.0, CHROM_SIZE)
    for chrom, strand, gpos in starts.itertuples(index=False):
        peak = gpos + plus_offset if strand == "+" else gpos + minus_genomic
        tracks[(chrom, strand)][peak] += 20.0

    def read(chrom, strand, positions):
        return tracks[(chrom, strand)][np.asarray(positions)]

    return read


@pytest.mark.parametrize("offset", [0, 2, -3])
def test_offsets_recovered_when_strands_agree(offset):
    starts = _starts()
    # Same convention on -: transcript offset d is genomic gpos - d.
    result = psite_offsets(_reader(starts, offset, -offset), starts)
    assert (result.plus, result.minus) == (offset, offset)
    assert (result.n_plus, result.n_minus) == (N, N)
    assert result.minus_genomic_shift == 0
    check_psite_convention(result, expected=offset)


def test_ribotish_027_minus_shift_is_caught():
    starts = _starts()
    # 0.2.7 places the minus P-site one base genomically higher than canonical.
    result = psite_offsets(_reader(starts, 0, RIBOTISH_MINUS_SHIFT), starts)
    assert result.plus == 0
    assert result.minus == -RIBOTISH_MINUS_SHIFT
    assert result.minus_genomic_shift == RIBOTISH_MINUS_SHIFT == 1
    with pytest.raises(PsiteConventionError, match="0.2.7"):
        check_psite_convention(result)


def test_other_strand_disagreement_is_caught():
    starts = _starts()
    result = psite_offsets(_reader(starts, 0, 2), starts)
    assert result.minus == -2
    with pytest.raises(PsiteConventionError, match="disagree"):
        check_psite_convention(result)


def test_expected_offset_enforced_unless_disabled():
    starts = _starts()
    result = psite_offsets(_reader(starts, 1, -1), starts)
    with pytest.raises(PsiteConventionError, match="expected"):
        check_psite_convention(result)
    check_psite_convention(result, expected=None)


def test_same_convention_compares_two_tracks():
    starts = _starts()
    fork = psite_offsets(_reader(starts, 0, 0, seed=1), starts)
    model_027 = psite_offsets(_reader(starts, 0, 1, seed=2), starts)
    check_same_convention(fork, fork)
    with pytest.raises(PsiteConventionError, match="minus"):
        check_same_convention(model_027, fork)


def test_too_few_starts_or_no_signal_raise():
    starts = _starts(5)
    with pytest.raises(ValueError, match="need"):
        psite_offsets(_reader(starts, 0, 0), starts)
    starts = _starts()
    with pytest.raises(ValueError, match="no track signal"):
        psite_offsets(lambda c, s, p: np.zeros(len(p)), starts)


def test_bigwig_reader_sums_files_and_blocks(monkeypatch):
    data = {
        "a.bw": {"chr1": np.arange(5_000, dtype=float)},
        "b.bw": {"chr1": np.ones(5_000)},
    }

    class FakeBigWig:
        def __init__(self, path):
            self.values_ = data[path]

        def chroms(self):
            return {c: len(v) for c, v in self.values_.items()}

        def values(self, chrom, a, b):
            return list(self.values_[chrom][a:b])

        def close(self):
            pass

    fake = types.ModuleType("pyBigWig")
    fake.open = FakeBigWig
    monkeypatch.setitem(sys.modules, "pyBigWig", fake)
    read = bigwig_reader({"+": ["a.bw", "b.bw"], "-": ["b.bw"]})
    positions = np.array([4_999, 10, 3_000, 6_000, -1])
    assert read("chr1", "+", positions).tolist() == [5_000.0, 11.0, 3_001.0, 0.0, 0.0]
    assert read("chr1", "-", positions).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert read("chrX", "+", positions).tolist() == [0.0] * 5
