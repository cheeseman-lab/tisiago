from types import SimpleNamespace

import pytest

pytest.importorskip("pyfaidx")
pytest.importorskip("jax")  # cache_seq imports txcache, which needs the agft env

from tisiago.tracks.cache_seq import transcript_codes  # noqa: E402


def test_transcript_codes_splice_exons_and_reverse_complement_minus():
    fasta = {"c": "AACCGGTTNA"}
    plus = SimpleNamespace(chrom="c", strand="+", exons=[(4, 6), (0, 2)])
    minus = SimpleNamespace(chrom="c", strand="-", exons=[(0, 2), (4, 6)])
    assert transcript_codes(fasta, plus).tolist() == [0, 0, 2, 2]  # AA + GG
    assert transcript_codes(fasta, minus).tolist() == [1, 1, 3, 3]  # revcomp(AAGG) = CCTT
