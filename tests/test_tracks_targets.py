"""Tests for canonical stranded target windows."""

from collections import Counter

import pytest

from tisiago.tracks.targets import (
    RNACOND_LIBRARIES,
    channel_layout,
    library_channels,
    load_targets,
    modulated_channels,
    read_window,
    rnacond_layout,
    strand_pairs,
)
from tisiago.tracks.track_io import track_prefix, write_bigwigs

SIZES = [("chr1", 100)]


def test_channel_layout_and_pairs():
    layout = channel_layout(["HeLa", "K562"])
    assert layout[:3] == [("HeLa", "tis", "+"), ("HeLa", "tis", "-"), ("HeLa", "chx", "+")]
    assert len(layout) == 12
    assert strand_pairs(layout)[:2] == [(0, 1), (2, 3)]


def test_rnacond_layout_is_tis_chx_of_one_condition():
    assert RNACOND_LIBRARIES == ("tis", "chx")
    assert rnacond_layout("K562") == [
        ("K562", "tis", "+"),
        ("K562", "tis", "-"),
        ("K562", "chx", "+"),
        ("K562", "chx", "-"),
    ]
    assert rnacond_layout("K562") == channel_layout(["K562"], RNACOND_LIBRARIES)
    assert strand_pairs(rnacond_layout("HeLa")) == [(0, 1), (2, 3)]
    assert library_channels(rnacond_layout("HeLa"), "chx") == (2, 3)
    assert channel_layout(["HeLa"], ("rna",)) == [("HeLa", "rna", "+"), ("HeLa", "rna", "-")]


def test_read_window_pads_outside_chromosome(tmp_path):
    write_bigwigs({("chr1", "+"): Counter({0: 2, 99: 5})}, SIZES, tmp_path / "t")
    values = read_window(tmp_path / "t.plus.bw", "chr1", -2, 3)
    assert values.tolist() == [0, 0, 2, 0, 0]
    assert read_window(tmp_path / "t.plus.bw", "chr1", 98, 102).tolist() == [0, 5, 0, 0]


def test_load_targets_canonicalizes_minus_rpf_only(tmp_path):
    for library in ("tis", "chx", "rna"):
        counts = {("chr1", "+"): Counter({10: 1}), ("chr1", "-"): Counter({21: 3})}
        write_bigwigs(counts, SIZES, track_prefix(tmp_path, "HeLa", library, "pooled"))
    y = load_targets(tmp_path, ["HeLa"], "chr1", 0, 32)
    assert y.shape == (32, 6)
    assert y[10, 0] == 1 and y[20, 1] == 3  # tis minus: Ribo-TISH 21 -> canonical 20
    assert y[20, 3] == 3  # chx minus shifted
    assert y[21, 5] == 3 and y[20, 5] == 0  # rna minus not shifted


def test_library_channels_selects_chx_of_every_condition():
    layout = channel_layout(["HeLa", "K562"])
    assert library_channels(layout, "chx") == (2, 3, 8, 9)
    assert all(layout[i][1] == "chx" for i in library_channels(layout, "chx"))


def test_modulated_channels_union_of_libraries_both_strands():
    layout = channel_layout(["HeLa", "K562"])
    assert modulated_channels(layout, ["chx"]) == library_channels(layout, "chx")
    got = modulated_channels(layout, ["chx", "tis"])
    assert got == (0, 1, 2, 3, 6, 7, 8, 9)
    assert {(c, lib, s) for c, lib, s in (layout[i] for i in got)} == {
        (c, lib, s) for c in ("HeLa", "K562") for lib in ("tis", "chx") for s in ("+", "-")
    }
    assert modulated_channels(layout, ["tis", "chx"]) == got


@pytest.mark.parametrize("bad", [[], ["chx", "chx"], ["chx", "harr"], ["CHX"]])
def test_modulated_channels_rejects_bad_libraries(bad):
    with pytest.raises(ValueError, match="mod_libraries"):
        modulated_channels(channel_layout(["HeLa"]), bad)
