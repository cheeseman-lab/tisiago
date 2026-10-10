"""Tests for the augmented prefetching batch iterator."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

from tisiago.seq2func.data import augment, iterate, make_example, one_hot
from tisiago.seq2func.targets import channel_layout, strand_pairs
from tisiago.seq2func.tracks import track_prefix, write_bigwigs

SIZES = [("chr1", 64)]
SEQ = ("ACGT" * 16)[:64]


class FixedRng:
    """Deterministic stand-in for a numpy Generator."""

    def __init__(self, draw: float, offset: int):
        """Fix the RC draw and the shift offset."""
        self.draw, self.offset = draw, offset

    def random(self):
        """Return the fixed draw."""
        return self.draw

    def integers(self, low, high):
        """Return the fixed offset."""
        return self.offset


@pytest.fixture
def world(tmp_path):
    fasta = tmp_path / "g.fa"
    fasta.write_text(f">chr1\n{SEQ}\n")
    Fasta(str(fasta)).close()
    for library in ("tis", "chx", "rna"):
        counts = {("chr1", "+"): Counter({10: 4, 20: 1}), ("chr1", "-"): Counter({25: 2})}
        write_bigwigs(counts, SIZES, track_prefix(tmp_path / "tracks", "HeLa", library, "pooled"))
    genes = pd.DataFrame(
        {
            "condition": ["HeLa"],
            "chrom": ["chr1"],
            "start": [0],
            "end": [64],
            "strand": ["+"],
            "expressed": [True],
        }
    )
    windows = pd.DataFrame({"chrom": ["chr1"] * 3, "start": [0, 8, 16], "end": [32, 40, 48]})
    return dict(
        fasta=str(fasta),
        tracks_dir=tmp_path / "tracks",
        conditions=["HeLa"],
        genes=genes,
        windows=windows,
    )


def kwargs(w):
    return {k: w[k] for k in ("fasta", "tracks_dir", "conditions", "genes")}


def test_one_hot_rows():
    x = one_hot("ACGTN")
    assert x.dtype == np.float32 and x.shape == (5, 4)
    assert x[:4].tolist() == np.eye(4).tolist()
    assert x[4].tolist() == [0, 0, 0, 0]


def test_make_example_shapes_and_values(world):
    ex = make_example(**kwargs(world), chrom="chr1", start=0, end=32)
    assert ex["sequences"].shape == (32, 4)
    assert ex["targets"].shape == (32, 6)
    assert ex["mask"].shape == (32, 6)
    assert ex["targets"][10, 0] == 4
    assert ex["sequences"][0].tolist() == [1, 0, 0, 0]


def test_make_example_pads_outside_chromosome(world):
    ex = make_example(**kwargs(world), chrom="chr1", start=-4, end=4)
    assert ex["sequences"][:4].sum() == 0
    assert ex["sequences"][4].tolist() == [1, 0, 0, 0]


def test_augment_reverse_complement_swaps_strand(world):
    ex = make_example(**kwargs(world), chrom="chr1", start=0, end=32)
    pairs = strand_pairs(channel_layout(world["conditions"]))
    out = augment(ex, FixedRng(0.0, 0), pairs)
    assert out["targets"][31 - 10, 1] == 4
    assert out["targets"][10, 0] == 0
    out = augment(ex, FixedRng(0.0, 2), pairs)
    assert out["targets"][31 - 10 + 2, 1] == 4


def test_augment_no_rc_shift_only(world):
    ex = make_example(**kwargs(world), chrom="chr1", start=0, end=32)
    pairs = strand_pairs(channel_layout(world["conditions"]))
    out = augment(ex, FixedRng(0.9, 3), pairs)
    assert out["targets"][13, 0] == 4
    assert not out["mask"][:3].any()


def test_iterate_once_yields_each_window(world):
    batches = list(
        iterate(world["windows"], **kwargs(world), seed=0, augment_examples=False, loop=False)
    )
    assert len(batches) == 3
    assert set(batches[0]) == {"sequences", "targets", "mask"}
    assert batches[0]["sequences"].shape == (1, 32, 4)
    assert batches[0]["targets"].shape == (1, 32, 6)
    assert batches[0]["targets"][0, 10, 0] == 4


def test_iterate_loops_and_augments(world):
    it = iterate(world["windows"], **kwargs(world), seed=1, augment_examples=True)
    batches = [next(it) for _ in range(7)]
    assert all(b["sequences"].shape == (1, 32, 4) for b in batches)


def _transcripts():
    from tisiago.seq2func.tx_features import Transcript, TranscriptFeatures

    plus = Transcript("TP", "GP", "chr1", "+", ((2, 12), (20, 40)))
    minus = Transcript("TM", "GM", "chr1", "-", ((30, 60),))
    return TranscriptFeatures([plus, minus], cap=50)


def test_make_example_with_transcript_features(world):
    transcripts = _transcripts()
    base = make_example(**kwargs(world), chrom="chr1", start=0, end=32)
    ex = make_example(**kwargs(world), chrom="chr1", start=0, end=32, transcripts=transcripts)
    assert ex["sequences"].shape == (32, 24) and ex["sequences"].dtype == np.float32
    np.testing.assert_array_equal(ex["sequences"][:, :4], base["sequences"])
    np.testing.assert_array_equal(ex["targets"], base["targets"])
    seq = Fasta(world["fasta"], as_raw=True)
    np.testing.assert_array_equal(ex["sequences"][:, 4:], transcripts.window(seq, "chr1", 0, 32))
    assert ex["sequences"][2:12, 13].all() and not ex["sequences"][12:20, 13].any()


def test_augment_moves_transcript_features_with_the_sequence(world):
    from tisiago.seq2func.augment import reverse_complement, shift

    ex = make_example(**kwargs(world), chrom="chr1", start=0, end=32, transcripts=_transcripts())
    pairs = strand_pairs(channel_layout(world["conditions"]))
    out = augment(ex, FixedRng(0.0, 2), pairs)
    rc = reverse_complement(ex["sequences"], ex["targets"], ex["mask"], pairs)
    expected = shift(*rc, 2)[0]
    np.testing.assert_array_equal(out["sequences"], expected)
    # plus-strand exonic block moved to the minus block, positions mirrored and shifted
    assert out["sequences"][31 - 2 + 2, 4 + 19] == 1.0
    assert out["sequences"][31 - 2 + 2, 4 + 9] == 0.0


def test_iterate_passes_transcripts(world):
    batches = list(
        iterate(
            world["windows"],
            **kwargs(world),
            seed=0,
            augment_examples=True,
            loop=False,
            transcripts=_transcripts(),
        )
    )
    assert all(b["sequences"].shape == (1, 32, 24) for b in batches)


def test_iterate_worker_error_raised_in_consumer(world, tmp_path):
    (tmp_path / "tracks" / "HeLa" / "tis_reppooled.plus.bw").unlink()
    it = iterate(world["windows"], **kwargs(world), seed=0, augment_examples=False)
    with pytest.raises(Exception):
        next(it)


def test_track_reader_matches_load_targets(world):
    from tisiago.seq2func.data import TrackReader
    from tisiago.seq2func.targets import load_targets

    reader = TrackReader(world["tracks_dir"], world["conditions"])
    try:
        for start, end in [(0, 32), (-4, 20), (50, 70)]:
            expected = load_targets(world["tracks_dir"], world["conditions"], "chr1", start, end)
            assert np.array_equal(reader.read("chr1", start, end), expected)
    finally:
        reader.close()


def test_iterate_multi_worker_loop_and_error(world, tmp_path):
    it = iterate(world["windows"], **kwargs(world), seed=0, augment_examples=True, workers=2)
    assert all(next(it)["targets"].shape == (1, 32, 6) for _ in range(9))
    it.close()
    (tmp_path / "tracks" / "HeLa" / "chx_reppooled.minus.bw").unlink()
    it = iterate(world["windows"], **kwargs(world), seed=0, augment_examples=False, workers=2)
    with pytest.raises(Exception):
        next(it)
