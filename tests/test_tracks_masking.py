"""Tests for the per-condition expression loss mask."""

import numpy as np
import pandas as pd
import pytest

from tisiago.tracks.masking import expression_mask
from tisiago.tracks.targets import channel_layout


def test_masks_unexpressed_genes_on_their_strand_for_rpf_only():
    layout = channel_layout(["HeLa"])
    genes = pd.DataFrame(
        [
            ("HeLa", "chr1", 10, 20, "+", False),
            ("HeLa", "chr1", 15, 25, "+", True),  # overlaps: expressed wins
            ("HeLa", "chr1", 30, 60, "-", False),  # clipped at window end
            ("HeLa", "chr2", 0, 50, "+", False),  # other chromosome
        ],
        columns=["condition", "chrom", "start", "end", "strand", "expressed"],
    )
    mask = expression_mask(layout, genes, "chr1", 0, 40)
    assert mask.shape == (40, 6)
    assert not mask[10:15, 0].any() and mask[15:20, 0].all()  # tis +
    assert not mask[10:15, 2].any()  # chx +
    assert mask[10:15, 4].all()  # rna + never masked
    assert not mask[30:40, 1].any() and mask[30:40, 0].all()  # tis -, not tis +


def test_nullable_boolean_with_na_raises():
    layout = channel_layout(["HeLa"])
    genes = pd.DataFrame(
        [
            ("HeLa", "chr1", 10, 20, "+", pd.NA),
            ("HeLa", "chr1", 15, 25, "+", True),
        ],
        columns=["condition", "chrom", "start", "end", "strand", "expressed"],
    )
    genes["expressed"] = genes["expressed"].astype("boolean")
    with pytest.raises(ValueError, match="expressed has missing values"):
        expression_mask(layout, genes, "chr1", 0, 40)


def test_nullable_boolean_and_object_bool_match_plain_bool():
    layout = channel_layout(["HeLa"])
    data = [
        ("HeLa", "chr1", 10, 20, "+", True),
        ("HeLa", "chr1", 25, 30, "+", False),
    ]

    genes_plain = pd.DataFrame(
        data, columns=["condition", "chrom", "start", "end", "strand", "expressed"]
    )
    mask_plain = expression_mask(layout, genes_plain, "chr1", 0, 40)

    genes_nullable = pd.DataFrame(
        data, columns=["condition", "chrom", "start", "end", "strand", "expressed"]
    )
    genes_nullable["expressed"] = genes_nullable["expressed"].astype("boolean")
    mask_nullable = expression_mask(layout, genes_nullable, "chr1", 0, 40)

    genes_object = pd.DataFrame(
        [
            ("HeLa", "chr1", 10, 20, "+", bool(True)),
            ("HeLa", "chr1", 25, 30, "+", bool(False)),
        ],
        columns=["condition", "chrom", "start", "end", "strand", "expressed"],
    )
    mask_object = expression_mask(layout, genes_object, "chr1", 0, 40)

    np.testing.assert_array_equal(mask_plain, mask_nullable)
    np.testing.assert_array_equal(mask_plain, mask_object)
