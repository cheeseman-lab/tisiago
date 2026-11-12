"""Assert the P-site convention of a TIS track from its peak at annotated AUG starts.

Harringtonine arrests the initiating ribosome with the start codon in the P site, so a track in
a correct convention peaks at the codon's first base (offset 0) on both strands. Offsets are in
transcript orientation (positive = 3'): on - an offset ``d`` is genomic ``gpos - d``.

Ribo-TISH 0.2.7 places minus-strand P-sites one base genomically higher than the canonical base
(``RIBOTISH_MINUS_SHIFT``; tisiago's own tracks keep it), which reads here as a minus-strand
offset one less than the plus-strand offset. The ribotish fork's ``make_tracks.psite_coord``
subtracts it, so its release bigWigs should not show it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT

MAX_OFFSET = 5
MIN_STARTS = 20
BLOCK_GAP = 1_000

Reader = Callable[[str, str, np.ndarray], np.ndarray]


class PsiteConventionError(AssertionError):
    """A track's P-site placement differs between strands or from the expected base."""


@dataclass(frozen=True)
class PsiteOffsets:
    """Peak offset per strand and the summed profiles behind it."""

    plus: int
    minus: int
    n_plus: int
    n_minus: int
    profile_plus: np.ndarray
    profile_minus: np.ndarray

    @property
    def minus_genomic_shift(self) -> int:
        """Genomic shift of the minus-strand peak relative to the plus-strand one."""
        return self.plus - self.minus


def offset_profile(
    reader: Reader, starts: pd.DataFrame, strand: str, max_offset: int = MAX_OFFSET
) -> tuple[np.ndarray, int]:
    """Track summed over ``starts`` on ``strand`` at offsets ``-max_offset..max_offset``.

    Args:
        reader: ``reader(chrom, strand, positions) -> values`` at 0-based genomic bases of
            that strand's track.
        starts: rows with ``chrom``, ``strand``, ``gpos`` (codon first base, see
            :mod:`tisiago.truth.schema`).
        strand: ``+`` or ``-``.
        max_offset: half-width of the profile.

    Returns:
        ``(profile, n_starts)``.
    """
    offsets = np.arange(-max_offset, max_offset + 1)
    sign = 1 if strand == "+" else -1
    sub = starts[starts.strand.astype(str) == strand]
    profile = np.zeros(len(offsets))
    for chrom, group in sub.groupby(sub.chrom.astype(str), sort=True):
        gpos = group.gpos.to_numpy(np.int64)
        positions = (gpos[:, None] + sign * offsets[None, :]).ravel()
        values = np.asarray(reader(chrom, strand, positions), dtype=np.float64)
        profile += np.nan_to_num(values.reshape(len(gpos), len(offsets))).sum(axis=0)
    return profile, len(sub)


def _peak(profile: np.ndarray, strand: str) -> int:
    if not np.any(profile > 0):
        raise ValueError(f"no track signal around {strand} starts")
    return int(np.argmax(profile)) - (len(profile) - 1) // 2


def psite_offsets(
    reader: Reader,
    starts: pd.DataFrame,
    max_offset: int = MAX_OFFSET,
    min_starts: int = MIN_STARTS,
) -> PsiteOffsets:
    """Peak offset of a TIS track around annotated starts, per strand.

    Ties go to the most 5' offset. Pass annotated AUG starts only (``codon_class == "AUG"``,
    ``region == "annotated"``) so the peak is initiation, not a pause.

    Raises:
        ValueError: fewer than ``min_starts`` starts on a strand, or no signal.
    """
    found = {}
    for strand in ("+", "-"):
        profile, n = offset_profile(reader, starts, strand, max_offset)
        if n < min_starts:
            raise ValueError(f"{n} starts on {strand} strand, need {min_starts}")
        found[strand] = (profile, n, _peak(profile, strand))
    (pp, n_p, op), (pm, n_m, om) = found["+"], found["-"]
    return PsiteOffsets(op, om, n_p, n_m, pp, pm)


def check_psite_convention(offsets: PsiteOffsets, expected: int | None = 0) -> None:
    """Raise :class:`PsiteConventionError` unless both strands peak at the same offset.

    Args:
        offsets: from :func:`psite_offsets`.
        expected: required offset of both peaks (0 = the codon's first base); None skips it.
    """
    shift = offsets.minus_genomic_shift
    if shift == RIBOTISH_MINUS_SHIFT:
        raise PsiteConventionError(
            f"minus strand is {shift:+d} nt genomically from plus: the Ribo-TISH 0.2.7 "
            "minus-strand shift (RIBOTISH_MINUS_SHIFT) is present"
        )
    if shift != 0:
        raise PsiteConventionError(
            f"strands disagree: plus peaks at {offsets.plus:+d}, minus at {offsets.minus:+d}"
        )
    if expected is not None and offsets.plus != expected:
        raise PsiteConventionError(f"peaks at {offsets.plus:+d}, expected {expected:+d}")


def check_same_convention(a: PsiteOffsets, b: PsiteOffsets) -> None:
    """Raise :class:`PsiteConventionError` unless two tracks place P-sites identically.

    Use on a model track and a release bigWig: a 1-nt minus-strand difference is the
    0.2.7 / fork mismatch of spec Phase 0 item 4.
    """
    if (a.plus, a.minus) != (b.plus, b.minus):
        raise PsiteConventionError(
            f"conventions differ: plus {a.plus:+d} vs {b.plus:+d}, "
            f"minus {a.minus:+d} vs {b.minus:+d}"
        )


def bigwig_reader(paths: Mapping[str, Sequence[str | Path]]) -> Reader:
    """Reader summing 1-bp bigWigs per strand (``{"+": [...], "-": [...]}``); 0 off-chrom."""
    import pyBigWig

    def read(chrom: str, strand: str, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64)
        out = np.zeros(len(positions))
        if not len(positions):
            return out
        order = np.argsort(positions, kind="stable")
        breaks = np.flatnonzero(np.diff(positions[order]) > BLOCK_GAP) + 1
        for path in paths[strand]:
            bw = pyBigWig.open(str(path))
            if bw is None:
                raise FileNotFoundError(path)
            try:
                size = bw.chroms().get(chrom) or 0
                for block in np.split(order, breaks):
                    p = positions[block]
                    a, b = max(int(p.min()), 0), min(int(p.max()) + 1, size)
                    if b <= a:
                        continue
                    dense = np.nan_to_num(np.asarray(bw.values(chrom, a, b), dtype=np.float64))
                    ok = (p >= a) & (p < b)
                    out[block[ok]] += dense[p[ok] - a]
            finally:
                bw.close()
        return out

    return read
