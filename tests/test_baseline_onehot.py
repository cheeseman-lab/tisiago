import pandas as pd
import pytest

from tisiago.baseline.onehot import codon_onehot, window_onehot


def test_codon_onehot_encodes_each_position():
    out = codon_onehot(pd.DataFrame({"codon": ["ATG", "CTG"]}))
    assert out.shape == (2, 12)
    assert out[0].reshape(3, 4).argmax(1).tolist() == [0, 3, 2]
    assert out[1].reshape(3, 4).argmax(1).tolist() == [1, 3, 2]


@pytest.mark.parametrize(
    ("gstart", "strand", "expected"),
    [
        (4, "+", [2, 2, 0, 3, 2]),  # plus: G G [A T G]
        (10, "-", [2, 2, 0, 3, 2]),  # minus: revcomp of [CAT]CC at 7..12 -> G G [A T G]
    ],
)
def test_window_onehot_centres_the_codon(tmp_path, gstart, strand, expected):
    pytest.importorskip("pyfaidx")
    from pyfaidx import Fasta

    fasta = tmp_path / "g.fa"
    fasta.write_text(">c\nGGGGATGCATCC\n")
    manifest = pd.DataFrame(
        {"chrom": ["c"], "gstart": [gstart], "strand": [strand], "codon": ["ATG"]}
    )
    out = window_onehot(manifest, Fasta(str(fasta)), half=2)
    assert out.shape == (1, 20)
    assert out[0].reshape(5, 4).argmax(1).tolist() == expected
