"""Chromosome-split 131 kb windows tiled over expressed genes."""

from __future__ import annotations

import pandas as pd

LENGTH = 131_072
STRIDE = 65_536
VAL_CHROMS = ("chr7",)
TEST_CHROMS = ("chr8", "chr9")
TRAIN_CHROMS = tuple(
    f"chr{c}" for c in [*range(1, 23), "X"] if f"chr{c}" not in VAL_CHROMS + TEST_CHROMS
)
SPLITS = {"train": TRAIN_CHROMS, "val": VAL_CHROMS, "test": TEST_CHROMS}


def merge_spans(spans):
    """Merge overlapping or touching half-open spans."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def tile(spans, chrom_size: int, length: int = LENGTH, stride: int = STRIDE) -> list[int]:
    """Window starts covering every span, clamped inside the chromosome."""
    last = max(chrom_size - length, 0)
    starts = set()
    for start, end in merge_spans(spans):
        position = start
        while True:
            starts.add(min(max(position, 0), last))
            if position + length >= end:
                break
            position += stride
    return sorted(starts)


def make_windows(
    expression: pd.DataFrame,
    chrom_sizes: dict[str, int],
    split: str,
    length: int = LENGTH,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """Windows for one split from genes expressed in at least one condition."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    genes = expression[expression.expressed.astype(bool)]
    rows = []
    for chrom in SPLITS[split]:
        spans = genes[genes.chrom == chrom][["start", "end"]].itertuples(index=False)
        if chrom not in chrom_sizes:
            continue
        for start in tile(list(spans), chrom_sizes[chrom], length, stride):
            rows.append((chrom, start, start + length))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])
