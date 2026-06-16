from tisiago.enumerate_codons import NEAR_COGNATES, classify_codon


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
