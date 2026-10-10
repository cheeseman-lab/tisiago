"""Transcript-coordinate (spliced) open-frame features for the seqconv_tx head."""

import numpy as np
import pytest

from tisiago.seq2func.augment import reverse_complement
from tisiago.seq2func.data import one_hot
from tisiago.seq2func.orf_features import strand_features_np
from tisiago.seq2func.tx_features import (
    NUM_TX_CHANNELS,
    TX_CHANNEL_NAMES,
    Transcript,
    TranscriptFeatures,
    load_mane_transcripts,
)

CAP = 50


def rc(seq: str) -> str:
    return seq[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def scaled(distance: int, cap: int = CAP) -> float:
    return np.log1p(distance) / np.log1p(cap)


def random_genome(seed: int, length: int) -> str:
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), size=length))


def spliced(genome: str, exons) -> str:
    return "".join(genome[s:e] for s, e in exons)


def test_channel_layout():
    assert NUM_TX_CHANNELS == 20 == len(TX_CHANNEL_NAMES)
    assert TX_CHANNEL_NAMES[0] == "plus_tx_stop_dist_k0"
    assert TX_CHANNEL_NAMES[9] == "plus_exonic"
    assert TX_CHANNEL_NAMES[10] == "minus_tx_stop_dist_k0"
    assert TX_CHANNEL_NAMES[19] == "minus_exonic"


def test_frame_continues_across_the_junction():
    # exon 1: CC ATG AAA C | intron (stop-rich in every frame) | exon 2: CC GGG TAA ...
    # The ATG frame runs on through the junction (ATG AAA CCC GGG TAA) on the mRNA only.
    exon1, intron, exon2 = "CCATGAAAC", "TAATAATAAGTAGTAG", "CCGGGTAAGG"
    genome = "GG" + exon1 + intron + exon2 + "GG"
    e1 = (2, 2 + len(exon1))
    e2 = (e1[1] + len(intron), e1[1] + len(intron) + len(exon2))
    tx = Transcript("T1", "G1", "chr1", "+", (e1, e2))
    out = TranscriptFeatures([tx], cap=CAP).window({"chr1": genome}, "chr1", 0, len(genome))
    assert out.shape == (len(genome), 20) and out.dtype == np.float32
    # Exon-2 'G' of GGG at mRNA index 12: codon start 11 (k=1); ATG at mRNA 2, in frame.
    g = e2[0] + 3
    assert out[g, 4] == pytest.approx(scaled(12 - 2))
    assert out[g, 7] == 1.0  # open
    assert out[g, 1] == pytest.approx(scaled(CAP))  # no upstream in-frame stop on the mRNA
    # The TAA in exon 2 closes the ORF.
    taa = e2[0] + 5
    assert out[taa, 6] == 0.0 and out[taa, 0] == pytest.approx(0.0)
    # Everything equals the definition on the spliced string, placed at exonic positions.
    expected = strand_features_np(one_hot(spliced(genome, tx.exons)), CAP)
    positions = np.r_[e1[0] : e1[1], e2[0] : e2[1]]
    np.testing.assert_array_equal(out[positions, :9], expected)
    exonic = np.zeros(len(genome), bool)
    exonic[positions] = True
    np.testing.assert_array_equal(out[:, 9], exonic.astype(np.float32))
    assert not out[~exonic, :10].any()
    assert not out[:, 10:].any()


def test_window_slice_uses_the_full_transcript():
    genome = random_genome(0, 400)
    exons = ((20, 60), (120, 170), (300, 380))
    tx = Transcript("T1", "G1", "chr1", "+", exons)
    feats = TranscriptFeatures([tx], cap=CAP)
    full = feats.window({"chr1": genome}, "chr1", 0, 400)
    part = TranscriptFeatures([tx], cap=CAP).window({"chr1": genome}, "chr1", 140, 320)
    np.testing.assert_array_equal(part, full[140:320])
    assert TranscriptFeatures([tx]).overlapping("chr1", 60, 120) == []  # intronic window


def test_minus_strand_transcript():
    genome = random_genome(1, 300)
    exons = ((30, 90), (150, 200), (240, 280))
    tx = Transcript("T1", "G1", "chr1", "-", exons)
    out = TranscriptFeatures([tx], cap=CAP).window({"chr1": genome}, "chr1", 0, 300)
    mrna = rc(spliced(genome, exons))
    expected = strand_features_np(one_hot(mrna), CAP)[::-1]  # back to ascending genomic order
    positions = np.concatenate([np.arange(s, e) for s, e in exons])
    np.testing.assert_array_equal(out[positions, 10:19], expected)
    assert (out[positions, 19] == 1).all() and out[:, 19].sum() == len(positions)
    assert not out[:, :10].any()


def test_overlap_tie_break_longest_then_smallest_id():
    genome = random_genome(2, 300)
    fasta = {"chr1": genome}
    short = Transcript("A_short", "G1", "chr1", "+", ((10, 100),))
    long_ = Transcript("Z_long", "G2", "chr1", "+", ((5, 60), (80, 200)))
    out = TranscriptFeatures([short, long_], cap=CAP).window(fasta, "chr1", 0, 300)
    only_long = TranscriptFeatures([long_], cap=CAP).window(fasta, "chr1", 0, 300)
    only_short = TranscriptFeatures([short], cap=CAP).window(fasta, "chr1", 0, 300)
    np.testing.assert_array_equal(out[5:60], only_long[5:60])
    np.testing.assert_array_equal(out[80:200], only_long[80:200])
    np.testing.assert_array_equal(out[60:80], only_short[60:80])  # long one is intronic here
    # Equal lengths: the smaller id wins, whatever the input order.
    b = Transcript("B", "G3", "chr1", "+", ((50, 150),))
    c = Transcript("C", "G4", "chr1", "+", ((52, 152),))
    for order in ([b, c], [c, b]):
        tied = TranscriptFeatures(order, cap=CAP).window(fasta, "chr1", 0, 300)
        only_b = TranscriptFeatures([b], cap=CAP).window(fasta, "chr1", 0, 300)
        np.testing.assert_array_equal(tied[52:150], only_b[52:150])
    # Opposite strands do not compete.
    minus = Transcript("M", "G5", "chr1", "-", ((10, 100),))
    both = TranscriptFeatures([short, minus], cap=CAP).window(fasta, "chr1", 0, 300)
    np.testing.assert_array_equal(both[:, :10], only_short[:, :10])
    assert both[10:100, 19].all()


def test_reverse_complement_augmentation_is_consistent():
    """RC of (one-hot + features) == features of the mirrored genome and annotation."""
    n = 400
    genome = random_genome(3, n)
    txs = [
        Transcript("P", "G1", "chr1", "+", ((20, 80), (130, 210), (260, 330))),
        Transcript("M", "G2", "chr1", "-", ((50, 120), (200, 280))),
    ]
    mirrored = [
        Transcript(
            t.transcript_id,
            t.gene_id,
            "chr1",
            "+-"[t.strand == "+"],
            tuple(sorted((n - e, n - s) for s, e in t.exons)),
        )
        for t in txs
    ]
    start, end = 15, 390
    seq = np.concatenate(
        [
            one_hot(genome[start:end]),
            TranscriptFeatures(txs, cap=CAP).window({"chr1": genome}, "chr1", start, end),
        ],
        axis=1,
    )
    targets = np.zeros((end - start, 2), np.float32)
    seq_rc, _, _ = reverse_complement(seq, targets, targets > 0, [(0, 1)])
    m_start, m_end = n - end, n - start
    expected = np.concatenate(
        [
            one_hot(rc(genome)[m_start:m_end]),
            TranscriptFeatures(mirrored, cap=CAP).window(
                {"chr1": rc(genome)}, "chr1", m_start, m_end
            ),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(seq_rc, expected)


def test_cache_reuses_and_evicts():
    genome = random_genome(4, 500)
    txs = [Transcript(f"T{i}", f"G{i}", "chr1", "+", ((i * 100, i * 100 + 80),)) for i in range(5)]
    feats = TranscriptFeatures(txs, cap=CAP, cache_bytes=3 * 80 * 9 * 4)
    first = feats.mrna_features({"chr1": genome}, txs[0])
    assert feats.mrna_features({"chr1": genome}, txs[0]) is first
    feats.window({"chr1": genome}, "chr1", 0, 500)
    assert len(feats._cache) == 3 and feats._cached_bytes == 3 * 80 * 9 * 4


GTF = """\
##description: toy
chr1\tT\tgene\t11\t200\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding";
chr1\tT\ttranscript\t11\t200\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; tag "MANE_Select";
chr1\tT\texon\t101\t200\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; tag "basic"; tag "MANE_Select";
chr1\tT\texon\t11\t50\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; tag "basic"; tag "MANE_Select";
chr1\tT\tCDS\t21\t50\t.\t+\t0\tgene_id "G1"; transcript_id "T1"; tag "MANE_Select";
chr1\tT\texon\t11\t300\t.\t+\t.\tgene_id "G1"; transcript_id "T1b"; tag "basic";
chr2\tT\texon\t5\t60\t.\t-\t.\tgene_id "G2"; transcript_id "T2"; tag "MANE_Select";
chr2\tT\texon\t1\t90\t.\t+\t.\tgene_id "G3"; transcript_id "T3"; tag "Ensembl_canonical";
"""


def test_load_mane_transcripts(tmp_path):
    path = tmp_path / "toy.gtf"
    path.write_text(GTF)
    txs = {t.transcript_id: t for t in load_mane_transcripts(path)}
    assert set(txs) == {"T1", "T2"}
    assert txs["T1"] == Transcript("T1", "G1", "chr1", "+", ((10, 50), (100, 200)))
    assert txs["T1"].length == 140
    assert txs["T2"] == Transcript("T2", "G2", "chr2", "-", ((4, 60),))


def test_n_bases_and_unknown_chromosome():
    genome = "ATGAAANNNAAATAG" + "C" * 20
    tx = Transcript("T1", "G1", "chr1", "+", ((0, 15),))
    feats = TranscriptFeatures([tx], cap=CAP)
    out = feats.window({"chr1": genome}, "chr1", 0, 35)
    assert out[5, 8] == 1.0  # k=2, codon at 3 (AAA) after the ATG: open
    assert out[8, 8] == 1.0  # codon at 6 has N: absent, not a stop, so still open
    assert out[14, 8] == 0.0  # TAG at 12 closes it
    assert not feats.window({"chr1": genome}, "chrX", 0, 35).any()
