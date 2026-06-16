from tisiago.enumerate_codons import NEAR_COGNATES, classify_codon, spliced_positions


def test_classify_aug():
    assert classify_codon("ATG") == "AUG"


def test_classify_near_cognates_are_single_mismatch_from_atg():
    # exactly the 9 single-substitution neighbours of ATG
    expected = {"CTG", "GTG", "TTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"}
    assert NEAR_COGNATES == expected
    for c in expected:
        assert classify_codon(c) == "near_cognate"


def test_classify_non_cognate():
    for c in ["AAA", "CCC", "GGG", "TTT", "GAT", "TAG"]:
        assert classify_codon(c) == "non_cognate"


def test_classify_lowercase_and_n():
    assert classify_codon("atg") == "AUG"  # case-insensitive
    assert classify_codon("ANG") == "non_cognate"  # ambiguous base -> non_cognate


def test_spliced_positions_plus_strand_two_exons():
    # exons (0-based, half-open) [100,105) and [200,203): mRNA = 5 + 3 = 8 bases.
    # 5'->3' on + strand is ascending genomic order.
    exons = [(100, 105), (200, 203)]
    pos = spliced_positions(exons, "+")
    assert list(pos) == [100, 101, 102, 103, 104, 200, 201, 202]


def test_spliced_positions_minus_strand_reverses_and_descends():
    # same exons, - strand: mRNA 5'->3' walks high genomic -> low.
    exons = [(100, 105), (200, 203)]
    pos = spliced_positions(exons, "-")
    assert list(pos) == [202, 201, 200, 104, 103, 102, 101, 100]


def test_spliced_positions_length_matches_total_exon_length():
    exons = [(0, 10), (50, 60), (100, 130)]
    assert len(spliced_positions(exons, "+")) == 10 + 10 + 30
