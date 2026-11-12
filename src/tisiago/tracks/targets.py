"""Canonical stranded target windows for fine-tuning (minus-strand RPF P-sites un-shifted)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyBigWig

from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
from tisiago.tracks.samples import LIBRARIES
from tisiago.tracks.track_io import track_prefix

STRAND_FILES = {"+": "plus", "-": "minus"}
RNACOND_LIBRARIES = ("tis", "chx")


def channel_layout(
    conditions: Sequence[str], libraries: Sequence[str] = LIBRARIES
) -> list[tuple[str, str, str]]:
    """Channels in model order: condition, then library, then (plus, minus)."""
    return [(c, lib, s) for c in conditions for lib in libraries for s in ("+", "-")]


def rnacond_layout(condition: str) -> list[tuple[str, str, str]]:
    """The 4 output channels of an ``rnacond`` example: TIS+, TIS-, CHX+, CHX- of ``condition``."""
    return channel_layout([condition], RNACOND_LIBRARIES)


def library_channels(layout: list[tuple[str, str, str]], library: str) -> tuple[int, ...]:
    """Indices of ``library``'s channels in ``layout``."""
    return tuple(i for i, (_, lib, _) in enumerate(layout) if lib == library)


def modulated_channels(
    layout: list[tuple[str, str, str]], libraries: Sequence[str]
) -> tuple[int, ...]:
    """Indices of every channel of ``libraries`` in ``layout`` order (both strands)."""
    known = {lib for _, lib, _ in layout}
    libraries = list(libraries)
    if not libraries or len(set(libraries)) != len(libraries) or not set(libraries) <= known:
        raise ValueError(
            f"mod_libraries must be unique names from {sorted(known)}, got {libraries!r}"
        )
    return tuple(i for i, (_, lib, _) in enumerate(layout) if lib in libraries)


def strand_pairs(layout: list[tuple[str, str, str]]) -> list[tuple[int, int]]:
    """Index pairs (plus, minus) for each condition/library."""
    index = {channel: i for i, channel in enumerate(layout)}
    return [(i, index[(c, lib, "-")]) for i, (c, lib, s) in enumerate(layout) if s == "+"]


def read_window(bw_path: str | Path, chrom: str, start: int, end: int) -> np.ndarray:
    """Per-base values for [start, end), zero-padded outside the chromosome."""
    out = np.zeros(end - start, dtype=np.float32)
    bw = pyBigWig.open(str(bw_path))
    try:
        size = bw.chroms().get(chrom, 0)
        lo, hi = max(start, 0), min(end, size)
        if hi > lo:
            values = np.asarray(bw.values(chrom, lo, hi), dtype=np.float32)
            out[lo - start : hi - start] = np.nan_to_num(values)
    finally:
        bw.close()
    return out


def load_targets(
    tracks_dir: str | Path,
    conditions: Sequence[str],
    chrom: str,
    start: int,
    end: int,
    rep: int | str = "pooled",
) -> np.ndarray:
    """Targets ``(end - start, C)`` in :func:`channel_layout` order, canonical P-sites."""
    columns = []
    for condition, library, strand in channel_layout(conditions):
        prefix = track_prefix(tracks_dir, condition, library, rep)
        shift = RIBOTISH_MINUS_SHIFT if strand == "-" and library != "rna" else 0
        path = f"{prefix}.{STRAND_FILES[strand]}.bw"
        columns.append(read_window(path, chrom, start + shift, end + shift))
    return np.stack(columns, axis=1)
