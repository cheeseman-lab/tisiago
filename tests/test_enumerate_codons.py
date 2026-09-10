import os
from pathlib import Path

import pandas as pd
import pytest

from tisiago.enumerate_codons import (
    NEAR_COGNATES,
    classify_codon,
    enumerate_transcript,
    parse_gtf_exons,
    spliced_positions,
)

_REF = os.environ.get("TISIAGO_REFERENCE_DIR")
GENOME = Path(_REF) / "Gencode_v49_GRCh38.primary_assembly.genome.fa" if _REF else None
GTF = Path(_REF) / "gencode.v49.primary_assembly.annotation.gtf" if _REF else None


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


@pytest.mark.skipif(
    GENOME is None or GTF is None or not GENOME.is_file() or not GTF.is_file(),
    reason="set TISIAGO_REFERENCE_DIR to run the reference integration test",
)
def test_enumerated_coords_match_curated_manifest():
    from pyfaidx import Fasta

    man = pd.read_parquet("data/store/manifest.parquet")
    sample_tx = (
        man[man.split == "test"]
        .drop_duplicates("transcript_id")
        .groupby("strand")
        .head(5)
        .transcript_id.tolist()
    )
    models = parse_gtf_exons(str(GTF), keep=set(sample_tx))
    fa = Fasta(str(GENOME), sequence_always_upper=True, rebuild=False)

    n_checked = 0
    for tx in sample_tx:
        if tx not in models:
            continue
        rows = enumerate_transcript(tx, models[tx], fa)
        by_idx = rows.set_index("mrna_index")
        cur = man[(man.transcript_id == tx)]
        for _, c in cur.iterrows():
            if c.mrna_index not in by_idx.index:
                continue
            e = by_idx.loc[c.mrna_index]
            e = e.iloc[0] if hasattr(e, "iloc") and e.ndim > 1 else e
            assert int(e.gstart) == int(c.gstart), (
                f"{tx} idx {c.mrna_index}: gstart {e.gstart} != {c.gstart}"
            )
            assert e.codon == c.codon, f"{tx} idx {c.mrna_index}: codon {e.codon} != {c.codon}"
            n_checked += 1
    assert n_checked >= 20, f"only checked {n_checked} candidates"
