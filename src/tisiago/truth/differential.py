"""Reproducibly differential start usage between two conditions (spec Phase 0 item 5, M2).

A start's usage share in one replicate is its TIS count over its gene's total in that
replicate (:func:`usage_shares`). Between conditions A and B a start is differential when, in
every replicate pairing (default rep1-rep1 and rep2-rep2, each replicate used once), the share
difference A - B has the same sign and magnitude at least ``min_delta``. Starts are joined
across conditions on the genomic site (``chrom``, ``strand``, ``gpos``), since the chosen
transcript can differ between lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from tisiago.truth.schema import SITE_KEY

MIN_DELTA = 0.1
PAIRINGS = ((1, 1), (2, 2))
_SHARE_COLUMNS = ("condition", "rep", *SITE_KEY, "share")


def usage_shares(counts: pd.DataFrame, group: str = "gene") -> pd.DataFrame:
    """Share of each start in its ``group``'s TIS count, per condition and replicate.

    Args:
        counts: rows with ``condition``, ``rep``, ``group``, the site key and ``count``.
        group: column the shares sum to 1 within.

    Returns:
        ``counts`` with a ``share`` column; NaN where the group has no reads.
    """
    out = counts.copy()
    total = out.groupby(["condition", "rep", group])["count"].transform("sum")
    out["share"] = (out["count"] / total.where(total > 0)).astype(np.float64)
    return out


def _wide(shares: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _SHARE_COLUMNS if c not in shares.columns]
    if missing:
        raise ValueError(f"shares missing columns {missing}")
    dup = shares.duplicated(["condition", "rep", *SITE_KEY])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (condition, rep, site) share rows")
    return shares.pivot(index=list(SITE_KEY), columns=["condition", "rep"], values="share")


def _pair(
    wide: pd.DataFrame,
    a: str,
    b: str,
    min_delta: float,
    pairings: Sequence[tuple[int, int]],
) -> tuple[int, pd.DataFrame]:
    deltas = {}
    for ra, rb in pairings:
        if (a, ra) not in wide.columns or (b, rb) not in wide.columns:
            raise ValueError(f"no shares for {a} rep{ra} or {b} rep{rb}")
        deltas[f"delta_{ra}_{rb}"] = wide[(a, ra)] - wide[(b, rb)]
    d = pd.DataFrame(deltas)
    tested = d.notna().all(axis=1)
    values = d.to_numpy()
    up = tested & (values >= min_delta).all(axis=1)
    down = tested & (values <= -min_delta).all(axis=1)
    hit = d[up | down].copy()
    hit["delta"] = hit.mean(axis=1)
    hit["direction"] = np.where(up[up | down], 1, -1)
    hit.insert(0, "condition_b", b)
    hit.insert(0, "condition_a", a)
    return int(tested.sum()), hit.reset_index()


def differential_starts(
    shares: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    min_delta: float = MIN_DELTA,
    pairings: Sequence[tuple[int, int]] = PAIRINGS,
) -> pd.DataFrame:
    """Starts whose usage share differs reproducibly between two conditions.

    Args:
        shares: long table with ``condition``, ``rep``, ``chrom``, ``strand``, ``gpos``,
            ``share`` (e.g. from :func:`usage_shares`).
        condition_a: first condition; ``direction`` +1 means its share is higher.
        condition_b: second condition.
        min_delta: minimum absolute share difference in every pairing.
        pairings: ``(rep in A, rep in B)`` pairs that must all agree.

    Returns:
        One row per differential start: ``condition_a``, ``condition_b``, the site key, one
        ``delta_<ra>_<rb>`` per pairing, the mean ``delta`` and ``direction`` (+1 / -1).
        A start missing a share in any pairing is not tested.
    """
    return _pair(_wide(shares), condition_a, condition_b, min_delta, pairings)[1]


def _pairs(shares: pd.DataFrame, conditions: Sequence[str] | None) -> list[tuple[str, str]]:
    names = list(dict.fromkeys(shares.condition.astype(str))) if conditions is None else conditions
    return list(combinations(names, 2))


def all_differential_starts(
    shares: pd.DataFrame,
    conditions: Sequence[str] | None = None,
    min_delta: float = MIN_DELTA,
    pairings: Sequence[tuple[int, int]] = PAIRINGS,
) -> pd.DataFrame:
    """:func:`differential_starts` for every condition pair, concatenated.

    Pairs are ``(a, b)`` with ``a`` before ``b`` in ``conditions`` (default: first
    appearance order in ``shares``).
    """
    wide = _wide(shares)
    frames = [_pair(wide, a, b, min_delta, pairings)[1] for a, b in _pairs(shares, conditions)]
    columns = ["condition_a", "condition_b", *SITE_KEY, "delta", "direction"]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def count_differential(
    shares: pd.DataFrame,
    conditions: Sequence[str] | None = None,
    min_delta: float = MIN_DELTA,
    pairings: Sequence[tuple[int, int]] = PAIRINGS,
) -> pd.DataFrame:
    """Differential-start counts per condition pair (Phase 0 "done when").

    Returns:
        One row per pair: ``condition_a``, ``condition_b``, ``n_tested`` (starts with a share
        in every pairing), ``n_a_higher``, ``n_b_higher``, ``n_differential``.
    """
    wide = _wide(shares)
    rows = []
    for a, b in _pairs(shares, conditions):
        tested, hit = _pair(wide, a, b, min_delta, pairings)
        up = int((hit.direction == 1).sum())
        rows.append(
            {
                "condition_a": a,
                "condition_b": b,
                "n_tested": tested,
                "n_a_higher": up,
                "n_b_higher": len(hit) - up,
                "n_differential": len(hit),
            }
        )
    return pd.DataFrame(rows)
