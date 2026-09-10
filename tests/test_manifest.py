import pandas as pd
import pytest

from tisiago.manifest import (
    split_grouped_indices,
    transcript_shard_mask,
    unique_site_indices,
)


def _manifest(labels=(1, 1, 0)):
    return pd.DataFrame(
        {
            "transcript_id": ["tx1", "tx1", "tx2"],
            "mrna_index": [10, 10, 20],
            "label_tis": labels,
            "split": ["train", "train", "test"],
            "chrom": ["chr1", "chr1", "chr2"],
            "gstart": [100, 100, 200],
            "strand": ["+", "+", "-"],
            "codon": ["ATG", "ATG", "CTG"],
        }
    )


def test_unique_site_indices_removes_consistent_duplicate():
    assert unique_site_indices(_manifest()).tolist() == [0, 2]


def test_unique_site_indices_rejects_conflicting_duplicate():
    with pytest.raises(ValueError, match="conflicting"):
        unique_site_indices(_manifest(labels=(1, 0, 0)))


def test_split_grouped_indices_is_disjoint_and_keeps_groups_together():
    indices = list(range(200))
    groups = [f"tx{i // 2}" for i in indices]
    left, right = split_grouped_indices(indices, groups, seed=3)
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(indices)
    for group in set(groups):
        rows = {i for i, value in enumerate(groups) if value == group}
        assert rows <= set(left) or rows <= set(right)


def test_transcript_shard_mask_is_deterministic_and_keeps_groups_together():
    transcript_id = pd.Series(["tx1", "tx2", "tx1", "tx3", "tx2"])
    masks = [transcript_shard_mask(transcript_id, shard, 3) for shard in range(3)]
    assert all(sum(mask[row] for mask in masks) == 1 for row in range(len(transcript_id)))
    for tx in transcript_id.unique():
        rows = transcript_id == tx
        assert sum(bool(mask[rows].all()) for mask in masks) == 1
