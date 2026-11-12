import numpy as np
import pandas as pd
import pytest

from tisiago.truth.schema import (
    cds_tx_bounds,
    codon_class,
    genome_to_tx,
    gpos_from_locus,
    gstart_from_gpos,
    line_of,
    region_of,
    split_of,
    tag_regions,
)

# Two exons, [100, 110) and [200, 220): 30 nt of mRNA.
STARTS, ENDS = [100, 200], [110, 220]


@pytest.mark.parametrize(
    ("codon", "expected"),
    [
        ("ATG", "AUG"),
        ("AUG", "AUG"),
        ("atg", "AUG"),
        ("AAG", "AAG"),
        ("AGG", "AGG"),
        ("CTG", "other_near_cognate"),
        ("GTG", "other_near_cognate"),
        ("ACG", "other_near_cognate"),
        ("ATT", "other_near_cognate"),
        ("TAA", "non_cognate"),
        ("GGG", "non_cognate"),
    ],
)
def test_codon_class(codon, expected):
    assert codon_class(codon) == expected


@pytest.mark.parametrize("codon", ["AT", "ATGA", "ANG", "", None])
def test_codon_class_rejects_bad_codons(codon):
    with pytest.raises(ValueError):
        codon_class(codon)


@pytest.mark.parametrize(
    ("tx_pos", "expected"),
    [(0, "utr5"), (9, "utr5"), (10, "annotated"), (13, "cds_inframe"), (37, "cds_inframe")]
    + [(14, "cds_outframe"), (15, "cds_outframe"), (40, "other"), (55, "other")],
)
def test_region_of(tx_pos, expected):
    assert region_of(tx_pos, 10, 40) == expected


def test_region_of_without_cds_is_other():
    assert region_of(5, None, None) == "other"
    assert region_of(5, np.nan, np.nan) == "other"


def test_genome_to_tx_plus():
    assert genome_to_tx(100, "+", STARTS, ENDS) == 0
    assert genome_to_tx(105, "+", STARTS, ENDS) == 5
    assert genome_to_tx(200, "+", STARTS, ENDS) == 10
    assert genome_to_tx(219, "+", STARTS, ENDS) == 29
    assert genome_to_tx(150, "+", STARTS, ENDS) is None


def test_genome_to_tx_minus_runs_down_the_genome():
    assert genome_to_tx(219, "-", STARTS, ENDS) == 0
    assert genome_to_tx(200, "-", STARTS, ENDS) == 19
    assert genome_to_tx(109, "-", STARTS, ENDS) == 20
    assert genome_to_tx(100, "-", STARTS, ENDS) == 29
    assert genome_to_tx(220, "-", STARTS, ENDS) is None


def test_genome_to_tx_accepts_strings_and_unsorted_exons():
    assert genome_to_tx(200, "-", "200,100", "220,110") == 19
    assert genome_to_tx(200, "+", "100,200,", "110,220,") == 10


def test_cds_tx_bounds_both_strands():
    # Genomic CDS [105, 215) spans the junction: 5 nt in exon 1, 15 in exon 2.
    assert cds_tx_bounds("+", STARTS, ENDS, 105, 215) == (5, 25)
    # On - the CDS begins at genomic 214 (tx 5) and ends at genomic 105 (tx 24).
    assert cds_tx_bounds("-", STARTS, ENDS, 105, 215) == (5, 25)
    assert cds_tx_bounds("+", STARTS, ENDS, None, None) is None
    with pytest.raises(ValueError):
        cds_tx_bounds("+", STARTS, ENDS, 150, 215)


def test_locus_conversion_round_trip():
    locus = np.array([105, 215])
    strand = np.array(["+", "-"])
    gpos = gpos_from_locus(locus, strand)
    assert gpos.tolist() == [105, 214]
    assert gstart_from_gpos(gpos, strand).tolist() == [105, 215]


def test_split_and_line():
    assert [split_of(c) for c in ("chr7", "chr8", "chr9", "chr1")] == [
        "val",
        "test",
        "test",
        "train",
    ]
    assert line_of("RPE1_Async") == line_of("RPE1_Sen") == "RPE1"
    assert line_of("K562") == "K562"


def _annotation():
    return pd.DataFrame(
        {
            "transcript_id": ["TP", "TM", "NC"],
            "chrom": ["chr1", "chr1", "chr2"],
            "strand": ["+", "-", "+"],
            "exon_starts": ["100,200", "100,200", "0"],
            "exon_ends": ["110,220", "110,220", "50"],
            "cds_start": [105, 105, pd.NA],
            "cds_end": [215, 215, pd.NA],
        }
    )


def test_tag_regions_plus_and_minus():
    starts = pd.DataFrame(
        {
            "transcript_id": ["TP"] * 4 + ["TM"] * 6 + ["NC"],
            "chrom": ["chr1"] * 10 + ["chr2"],
            "strand": ["+"] * 4 + ["-"] * 6 + ["+"],
            # +: utr5, annotated, in-frame across the junction (tx 11), out-of-frame
            # -: annotated (214), utr5 (219), in-frame (211 = tx 8), out-of-frame (210),
            #    in-frame in exon 1 (106 = tx 23), 3'UTR (104 = tx 25)
            "gpos": [101, 105, 201, 106, 214, 219, 211, 210, 106, 104, 10],
        }
    )
    tagged = tag_regions(starts, _annotation())
    assert tagged.tx_pos.tolist() == [1, 5, 11, 6, 5, 0, 8, 9, 23, 25, 10]
    assert tagged.region.tolist() == [
        "utr5",
        "annotated",
        "cds_inframe",
        "cds_outframe",
        "annotated",
        "utr5",
        "cds_inframe",
        "cds_outframe",
        "cds_inframe",
        "other",
        "other",
    ]


def test_tag_regions_rejects_inconsistent_starts():
    ann = _annotation()
    intronic = pd.DataFrame(
        {"transcript_id": ["TM"], "chrom": ["chr1"], "strand": ["-"], "gpos": [150]}
    )
    with pytest.raises(ValueError, match="not exonic"):
        tag_regions(intronic, ann)
    wrong_strand = intronic.assign(strand="+", gpos=105)
    with pytest.raises(ValueError, match="transcript on"):
        tag_regions(wrong_strand, ann)
    unknown = intronic.assign(transcript_id="TX")
    with pytest.raises(ValueError, match="missing from annotation"):
        tag_regions(unknown, ann)
