"""Tests for the augmented prefetching batch iterator."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

from tisiago.tracks.data import augment, iterate, make_example, one_hot
from tisiago.tracks.targets import channel_layout, strand_pairs
from tisiago.tracks.track_io import track_prefix, write_bigwigs

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
    from tisiago.tracks.tx_features import Transcript, TranscriptFeatures

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
    from tisiago.tracks.augment import reverse_complement, shift

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
    from tisiago.tracks.data import TrackReader
    from tisiago.tracks.targets import load_targets

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


# --- rnacond: (window, condition) pairs with the condition's RNA as extra input channels ---

RNA_VALUES = {"HeLa": 5.0, "K562": 7.0}
# genome-wide pooled RNA totals: value (+) and value + 1 (-) reads
DEPTH = {c: 5e7 / (2 * v + 1) for c, v in RNA_VALUES.items()}


def _raw(rna, condition):
    """Depth-scaled RNA input back to read counts."""
    return np.rint(np.asarray(rna, np.float64) / DEPTH[condition])


def _which(value):
    """Condition and raw count of a plus-strand depth-scaled RNA value."""
    condition = min(DEPTH, key=lambda c: abs(RNA_VALUES[c] * DEPTH[c] - value))
    return RNA_VALUES[condition], condition


@pytest.fixture
def rna_world(tmp_path):
    fasta = tmp_path / "g.fa"
    fasta.write_text(f">chr1\n{SEQ}\n")
    Fasta(str(fasta)).close()
    tracks = tmp_path / "tracks"
    for condition, value in RNA_VALUES.items():
        tis = {("chr1", "+"): Counter({20: 3}), ("chr1", "-"): Counter({26: 2})}
        chx = {("chr1", "+"): Counter({21: 4}), ("chr1", "-"): Counter({30: 1})}
        # RNA: plus 5' end at 20 (the TIS+ P-site), minus 5' end at 25 (no minus shift)
        rna = {("chr1", "+"): Counter({20: value}), ("chr1", "-"): Counter({25: value + 1})}
        for library, counts in (("tis", tis), ("chx", chx), ("rna", rna)):
            write_bigwigs(counts, SIZES, track_prefix(tracks, condition, library, "pooled"))
    genes = pd.DataFrame(
        {
            "condition": ["HeLa", "K562"],
            "chrom": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [64, 64],
            "strand": ["+", "+"],
            "expressed": [True, False],
        }
    )
    windows = pd.DataFrame({"chrom": ["chr1"] * 3, "start": [0, 8, 16], "end": [32, 40, 48]})
    return dict(fasta=str(fasta), tracks_dir=tracks, genes=genes, windows=windows)


def _rna_example(w, condition, start=0, end=32):
    from tisiago.tracks.data import make_rnacond_example

    return make_rnacond_example(
        w["fasta"], w["tracks_dir"], condition, w["genes"], "chr1", start, end
    )


def test_make_rnacond_example_channels(rna_world):
    from tisiago.tracks.masking import expression_mask
    from tisiago.tracks.targets import load_targets, rnacond_layout

    base = make_example(
        rna_world["fasta"], rna_world["tracks_dir"], ["K562"], rna_world["genes"], "chr1", 0, 32
    )
    ex = _rna_example(rna_world, "K562")
    assert ex["sequences"].shape == (32, 6) and ex["sequences"].dtype == np.float32
    np.testing.assert_array_equal(ex["sequences"][:, :4], base["sequences"])
    rna = _raw(ex["sequences"][:, 4:], "K562")
    assert rna[20, 0] == 7.0 and rna[:, 0].sum() == 7.0
    assert rna[25, 1] == 8.0 and rna[:, 1].sum() == 8.0
    # targets: TIS+, TIS-, CHX+, CHX- of K562 only, canonical minus P-sites
    assert ex["targets"].shape == (32, 4) and ex["mask"].shape == (32, 4)
    np.testing.assert_array_equal(ex["targets"], base["targets"][:, :4])
    np.testing.assert_array_equal(
        ex["targets"], load_targets(rna_world["tracks_dir"], ["K562"], "chr1", 0, 32)[:, :4]
    )
    assert ex["targets"][20, 0] == 3 and ex["targets"][25, 1] == 2
    layout = rnacond_layout("K562")
    expected = expression_mask(layout, rna_world["genes"], "chr1", 0, 32)
    np.testing.assert_array_equal(ex["mask"], expected)
    # the K562 + gene is unexpressed -> its + channels are missing data; HeLa's are not
    assert not ex["mask"][:, [0, 2]].any() and ex["mask"][:, [1, 3]].all()
    assert _rna_example(rna_world, "HeLa")["mask"].all()
    assert _raw(_rna_example(rna_world, "HeLa")["sequences"][20, 4], "HeLa") == 5.0


def test_rnacond_augment_moves_rna_with_targets(rna_world):
    from tisiago.tracks.augment import reverse_complement, shift
    from tisiago.tracks.targets import rnacond_layout

    ex = _rna_example(rna_world, "HeLa")
    pairs = strand_pairs(rnacond_layout("HeLa"))
    out = augment(ex, FixedRng(0.0, 2), pairs)
    rc = reverse_complement(ex["sequences"], ex["targets"], ex["mask"], pairs)
    for key, value in zip(("sequences", "targets", "mask"), shift(*rc, 2), strict=True):
        np.testing.assert_array_equal(out[key], value)
    # RNA plus 5' end and the TIS+ P-site sat together at 20: both now on minus, at 31 - 20 + 2
    rna = _raw(out["sequences"][:, 4:], "HeLa")
    assert rna[13, 1] == 5.0 and out["targets"][13, 1] == 3
    assert rna[:, 0].sum() == 6.0 and rna[31 - 25 + 2, 0] == 6.0
    assert out["targets"][:, [0, 2]].sum() == ex["targets"][:, [1, 3]].sum()
    # the one-hot is still reverse-complemented
    np.testing.assert_array_equal(out["sequences"][2:, :4], ex["sequences"][::-1, 3::-1][:-2])
    # shift only: RNA and targets move together
    out = augment(ex, FixedRng(0.9, 3), pairs)
    assert _raw(out["sequences"][23, 4], "HeLa") == 5.0 and out["targets"][23, 0] == 3
    assert not out["mask"][:3].any() and not out["sequences"][:3].any()


def test_iterate_rnacond_pairs_in_order(rna_world):
    batches = list(
        iterate(
            rna_world["windows"],
            fasta=rna_world["fasta"],
            tracks_dir=rna_world["tracks_dir"],
            conditions=["HeLa", "K562"],
            genes=rna_world["genes"],
            seed=0,
            augment_examples=False,
            loop=False,
            rna_conditioned=True,
        )
    )
    assert len(batches) == 6
    assert set(batches[0]) == {"sequences", "targets", "mask"}
    assert all(b["sequences"].shape == (1, 32, 6) for b in batches)
    assert all(b["targets"].shape == (1, 32, 4) for b in batches)
    seen = [
        (_which(b["sequences"][0, :, 4].max())[0], int(b["sequences"][0, :, 4].argmax()))
        for b in batches
    ]
    # condition-major, windows in order: RNA plus read at 20 -> 20, 12, 4 in windows 0, 8, 16
    assert seen == [(5.0, 20), (5.0, 12), (5.0, 4), (7.0, 20), (7.0, 12), (7.0, 4)]
    assert batches[3]["mask"][0, :, 0].sum() == 0  # K562 + gene unexpressed


def test_iterate_rnacond_loop_shuffles_every_pair_once_per_epoch(rna_world):
    it = iterate(
        rna_world["windows"],
        fasta=rna_world["fasta"],
        tracks_dir=rna_world["tracks_dir"],
        conditions=["HeLa", "K562"],
        genes=rna_world["genes"],
        seed=3,
        augment_examples=False,
        workers=1,
        rna_conditioned=True,
    )
    pairs = []
    for _ in range(12):
        rna = next(it)["sequences"][0, :, 4]
        pairs.append((_which(rna.max())[0], int(rna.argmax())))
    it.close()
    every = {(v, p) for v in RNA_VALUES.values() for p in (20, 12, 4)}
    assert set(pairs[:6]) == every and set(pairs[6:]) == every
    in_order = [(5.0, 20), (5.0, 12), (5.0, 4), (7.0, 20), (7.0, 12), (7.0, 4)]
    assert pairs[:6] != in_order or pairs[6:] != in_order
    assert pairs[:6] != pairs[6:]  # reshuffled per epoch


def test_iterate_rnacond_augments_and_rejects_transcripts(rna_world):
    kw = dict(
        fasta=rna_world["fasta"],
        tracks_dir=rna_world["tracks_dir"],
        conditions=["HeLa"],
        genes=rna_world["genes"],
        seed=0,
        rna_conditioned=True,
    )
    it = iterate(rna_world["windows"], augment_examples=True, **kw)
    assert all(next(it)["sequences"].shape == (1, 32, 6) for _ in range(5))
    it.close()
    with pytest.raises(ValueError, match="transcripts"):
        next(
            iterate(rna_world["windows"], augment_examples=False, transcripts=_transcripts(), **kw)
        )


def test_iterate_rnacond_held_out_condition_never_read(rna_world, tmp_path):
    """Only the listed conditions are paired: a held-out condition's tracks are never opened."""
    for path in (tmp_path / "tracks" / "K562").iterdir():
        path.unlink()
    batches = list(
        iterate(
            rna_world["windows"],
            fasta=rna_world["fasta"],
            tracks_dir=rna_world["tracks_dir"],
            conditions=["HeLa"],
            genes=rna_world["genes"],
            seed=0,
            augment_examples=False,
            loop=False,
            rna_conditioned=True,
        )
    )
    assert len(batches) == 3
    assert all(_raw(b["sequences"][0, :, 4].max(), "HeLa") == 5.0 for b in batches)


def test_rna_depth_scale_is_genome_wide_pooled_total(rna_world, tmp_path):
    from tisiago.tracks.data import RNA_REFERENCE_DEPTH, rna_depth_scale, rna_depth_scales

    assert RNA_REFERENCE_DEPTH == 5e7  # a 50M-read library keeps its read units

    # HeLa pooled RNA: 5 (+) and 6 (-) reads on chr1
    assert rna_depth_scale(rna_world["tracks_dir"], "HeLa") == pytest.approx(5e7 / 11)
    assert rna_depth_scales(rna_world["tracks_dir"], ["HeLa", "K562"]) == pytest.approx(
        {"HeLa": 5e7 / 11, "K562": 5e7 / 15}
    )
    empty = tmp_path / "empty"
    write_bigwigs({}, SIZES, track_prefix(empty, "HeLa", "rna", "pooled"))
    with pytest.raises(ValueError, match="RNA"):
        rna_depth_scale(empty, "HeLa")


def test_rnacond_rna_input_is_depth_scaled_and_depth_invariant(tmp_path):
    """Two conditions that differ only by RNA depth give identical head inputs."""
    from tisiago.tracks.data import make_rnacond_example

    fasta = tmp_path / "g.fa"
    fasta.write_text(f">chr1\n{SEQ}\n>chr2\n{SEQ}\n")
    Fasta(str(fasta)).close()
    sizes = [("chr1", 64), ("chr2", 64)]
    tracks = tmp_path / "tracks"
    rna = {
        ("chr1", "+"): Counter({20: 2, 40: 1}),
        ("chr1", "-"): Counter({25: 3}),
        ("chr2", "+"): Counter({5: 4}),  # outside every window: still in the genome-wide total
    }
    for condition, depth in (("HeLa", 1), ("K562", 7)):
        scaled = {k: Counter({p: v * depth for p, v in c.items()}) for k, c in rna.items()}
        write_bigwigs(scaled, sizes, track_prefix(tracks, condition, "rna", "pooled"))
        for library in ("tis", "chx"):
            counts = {("chr1", "+"): Counter({20: depth}), ("chr1", "-"): Counter({30: 1})}
            write_bigwigs(counts, sizes, track_prefix(tracks, condition, library, "pooled"))
    genes = pd.DataFrame(
        {
            "condition": ["HeLa", "K562"],
            "chrom": "chr1",
            "start": 0,
            "end": 64,
            "strand": "+",
            "expressed": True,
        }
    )
    a = make_rnacond_example(str(fasta), tracks, "HeLa", genes, "chr1", 0, 32)
    b = make_rnacond_example(str(fasta), tracks, "K562", genes, "chr1", 0, 32)
    np.testing.assert_allclose(b["sequences"], a["sequences"], rtol=1e-6)
    # a 10-read library scaled to 5e7 reads
    assert a["sequences"][20, 4] == pytest.approx(2 * 5e7 / 10)
    assert a["sequences"][25, 5] == pytest.approx(3 * 5e7 / 10)
    assert b["targets"][20, 0] == 7 * a["targets"][20, 0]  # targets stay raw counts
