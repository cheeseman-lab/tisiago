"""Tests for chromosome-split window tiling."""

import pandas as pd
import pytest

from tisiago.tracks.windows import make_windows, merge_spans, tile


def test_merge_spans():
    assert merge_spans([(5, 10), (0, 3), (8, 12), (12, 13)]) == [(0, 3), (5, 13)]


def test_tile_covers_and_clamps():
    starts = tile([(0, 10), (250, 260)], chrom_size=300, length=100, stride=50)
    assert starts == [0, 200]
    assert all(0 <= s <= 200 for s in starts)


def test_tile_long_span_uses_stride():
    assert tile([(0, 260)], chrom_size=1000, length=100, stride=50) == [0, 50, 100, 150, 200]


def test_make_windows_respects_split():
    expr = pd.DataFrame(
        [
            ("HeLa", "chr1", 0, 50, True),
            ("HeLa", "chr7", 0, 50, True),
            ("HeLa", "chr8", 0, 50, True),
            ("K562", "chr2", 0, 50, False),
        ],
        columns=["condition", "chrom", "start", "end", "expressed"],
    )
    sizes = {"chr1": 1000, "chr2": 1000, "chr7": 1000, "chr8": 1000}
    train = make_windows(expr, sizes, "train", length=100, stride=50)
    assert set(train.chrom) == {"chr1"}
    assert set(make_windows(expr, sizes, "val", length=100, stride=50).chrom) == {"chr7"}
    assert set(make_windows(expr, sizes, "test", length=100, stride=50).chrom) == {"chr8"}
    with pytest.raises(ValueError):
        make_windows(expr, sizes, "other")
