import pandas as pd
import pytest

from tisiago.sequence import (
    fetch_genomic_window,
    reverse_complement,
    spliced_transcript_sequence,
)
from tisiago.transcript_inference import (
    TRANSCRIPT_LENGTH_TAG,
    build_transcript_request,
    expected_transcript_keys,
)


class FakeFasta(dict):
    """Minimal pyfaidx-like chromosome mapping for sequence tests."""


def _rows(codon="ATG"):
    return pd.DataFrame(
        [
            {
                "row_idx": 7,
                "transcript_id": "tx1",
                "mrna_index": 2,
                "chrom": "chr1",
                "strand": "+",
                "codon": codon,
            },
            {
                "row_idx": 9,
                "transcript_id": "tx1",
                "mrna_index": 8,
                "chrom": "chr1",
                "strand": "+",
                "codon": "AAC",
            },
        ]
    )


def test_shared_sequence_helpers_are_strand_aware_and_pad_windows():
    fasta = FakeFasta(chr1="AACCGGTT")
    assert reverse_complement("ACGTN") == "NACGT"
    assert fetch_genomic_window(fasta, "chr1", -2, 6, "+") == "NNAACC"
    assert fetch_genomic_window(fasta, "chr1", -2, 6, "-") == "GGTTNN"


def test_spliced_transcript_sequence_uses_exons_and_biological_orientation():
    fasta = FakeFasta(chr1="AAATGCCCTTTTAACCCGGG")
    plus = {"chrom": "chr1", "strand": "+", "exons": [(0, 8), (12, 18)]}
    minus = {"chrom": "chr1", "strand": "-", "exons": [(0, 8), (12, 18)]}
    assert spliced_transcript_sequence(fasta, plus) == "AAATGCCCAACCCG"
    assert spliced_transcript_sequence(fasta, minus) == reverse_complement("AAATGCCCAACCCG")


def test_transcript_request_deduplicates_positions_and_pads_only_on_the_right():
    fasta = FakeFasta(chr1="AAATGCCCTTTTAACCCGGG")
    model = {"chrom": "chr1", "strand": "+", "exons": [(0, 8), (12, 18)]}
    request = build_transcript_request(_rows(), model, fasta, bucket_size=8)
    assert request.transcript_id == "tx1"
    assert request.positions == [2, 5, 8, 11, 14, 17]
    assert len(request.requests) == 8
    assert len(request.sequence) == 24
    assert request.sequence.startswith("AAATGCCCAACCCG")
    assert request.sequence.endswith("N" * 10)


def test_transcript_input_is_independent_of_manifest_candidate_coverage():
    """Curated and dense candidate subsets must produce the same model sequence."""
    fasta = FakeFasta(chr1="AAATGCCCTTTTAACCCGGG")
    model = {"chrom": "chr1", "strand": "+", "exons": [(0, 20)]}
    rows = _rows()
    rows.loc[rows.index[1], "codon"] = "TTT"
    sparse = build_transcript_request(rows.iloc[[0]], model, fasta)
    dense = build_transcript_request(rows, model, fasta)
    assert sparse.sequence == dense.sequence
    assert sparse.sequence == "AAATGCCCTTTTAACCCGGG" + ("N" * 9)


def test_transcript_request_rejects_coordinate_or_annotation_mismatch():
    fasta = FakeFasta(chr1="AAATGCCCTTTTAACCCGGG")
    model = {"chrom": "chr1", "strand": "+", "exons": [(0, 8), (12, 18)]}
    with pytest.raises(ValueError, match="codon mismatch"):
        build_transcript_request(_rows(codon="CTG"), model, fasta)


def test_transcript_keys_have_a_distinct_feature_namespace():
    keys = expected_transcript_keys(["blocks.28.mlp.l3"])
    assert TRANSCRIPT_LENGTH_TAG == "TXP"
    assert keys == {
        f"evo2::TXP::blocks.28.mlp.l3::off{offset}" for offset in (0, 3, 6, 9)
    }
