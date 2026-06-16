from tisiago.enumerate_codons import (
    NEAR_COGNATES,
    classify_codon,
    parse_gtf_exons,
    spliced_positions,
)


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


def test_parse_gtf_exons(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        "#comment line\n"
        'chr1\tHAVANA\ttranscript\t101\t300\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr1\tHAVANA\texon\t101\t105\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr1\tHAVANA\texon\t201\t203\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr2\tHAVANA\texon\t51\t60\t.\t-\t.\tgene_id "G2"; transcript_id "ENST2.2";\n'
    )
    models = parse_gtf_exons(str(gtf), keep={"ENST1.1", "ENST2.2"})
    # GTF 1-based inclusive -> 0-based half-open: 101..105 -> [100,105)
    assert models["ENST1.1"] == {"chrom": "chr1", "strand": "+", "exons": [(100, 105), (200, 203)]}
    assert models["ENST2.2"] == {"chrom": "chr2", "strand": "-", "exons": [(50, 60)]}


def test_parse_gtf_exons_filters_by_keep(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'chr1\tHAVANA\texon\t1\t5\t.\t+\t.\ttranscript_id "KEEP.1";\n'
        'chr1\tHAVANA\texon\t1\t5\t.\t+\t.\ttranscript_id "DROP.1";\n'
    )
    models = parse_gtf_exons(str(gtf), keep={"KEEP.1"})
    assert set(models) == {"KEEP.1"}
