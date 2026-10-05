"""Tests for the per-condition expression loss mask."""

import pandas as pd

from tisiago.seq2func.masking import expression_mask
from tisiago.seq2func.targets import channel_layout


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
