"""Start-calling metrics against the reproducible-start truth set (re-scope spec §4).

M1: precision and recall of the top ``k`` scored codons per transcript, overall and by codon
class and in-CDS vs outside. M2: direction accuracy on reproducibly differential starts.
Starts are matched on the genomic site within a condition (``condition``, ``chrom``,
``strand``, ``gpos``; see :mod:`tisiago.truth.schema`).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from tisiago.truth.schema import CODON_CLASSES, IN_CDS, REGIONS, SITE_KEY, START_KEY

ALL = "all"
CANDIDATE_COLUMNS = (*START_KEY, "transcript_id", "codon_class", "region", "score")
M1_COLUMNS = (
    "k",
    "condition",
    "stratum",
    "n_calls",
    "n_tp",
    "precision",
    "n_truth",
    "n_recalled",
    "recall",
    "n_truth_unscored",
)
M2_COLUMNS = ("condition_a", "condition_b", "n_starts", "n_differential", "n_correct", "accuracy")


def _check_candidates(candidates: pd.DataFrame) -> None:
    missing = [c for c in CANDIDATE_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidates missing columns {missing}")
    if not set(candidates.codon_class) <= set(CODON_CLASSES):
        raise ValueError("candidates have unknown codon_class values")
    if not set(candidates.region) <= set(REGIONS):
        raise ValueError("candidates have unknown region values")
    if candidates.score.isna().any():
        raise ValueError("candidates have NaN scores")


def top_k_calls(candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top ``k`` codons per (condition, transcript) by score, one row per called site.

    Ties are broken by the lower ``gpos``. A site reached through several transcripts is
    called once, with its first (highest-scoring) row.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    order = candidates.sort_values(
        ["condition", "transcript_id", "score", "gpos"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    top = order.groupby(["condition", "transcript_id"], sort=False).head(k)
    top = top.sort_values("score", ascending=False, kind="stable")
    return top.drop_duplicates(list(START_KEY)).reset_index(drop=True)


def _strata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    codon = frame.codon_class.to_numpy()
    in_cds = frame.region.isin(IN_CDS).to_numpy()
    out = {ALL: np.ones(len(frame), dtype=bool)}
    out |= {f"codon_class:{c}": codon == c for c in CODON_CLASSES}
    out |= {"cds:in": in_cds, "cds:out": ~in_cds}
    return out


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[list(START_KEY)].astype({"gpos": np.int64}))


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _m1_rows(k: int, condition: str, calls, truth, scored) -> list[dict]:
    call_hit = _keys(calls).isin(_keys(truth)) if len(calls) else np.zeros(0, bool)
    recalled = _keys(truth).isin(_keys(calls)) if len(truth) else np.zeros(0, bool)
    covered = _keys(truth).isin(scored) if len(truth) else np.zeros(0, bool)
    call_strata, truth_strata = _strata(calls), _strata(truth)
    rows = []
    for name, cmask in call_strata.items():
        tmask = truth_strata[name]
        n_calls, n_tp = int(cmask.sum()), int((cmask & call_hit).sum())
        n_truth = int((tmask & covered).sum())
        n_rec = int((tmask & covered & recalled).sum())
        rows.append(
            {
                "k": k,
                "condition": condition,
                "stratum": name,
                "n_calls": n_calls,
                "n_tp": n_tp,
                "precision": _ratio(n_tp, n_calls),
                "n_truth": n_truth,
                "n_recalled": n_rec,
                "recall": _ratio(n_rec, n_truth),
                "n_truth_unscored": int((tmask & ~covered).sum()),
            }
        )
    return rows


def m1_at_budget(
    candidates: pd.DataFrame, truth: pd.DataFrame, k: int | Iterable[int] = 1
) -> pd.DataFrame:
    """M1: precision / recall of the top ``k`` codons per transcript against reproducible truth.

    Args:
        candidates: one row per scored codon with ``condition``, ``chrom``, ``strand``,
            ``gpos``, ``transcript_id``, ``codon_class``, ``region``, ``score``. Tag
            candidates with the same rules as the truth (:mod:`tisiago.truth.schema`).
        truth: truth table (:func:`tisiago.truth.load_truth`); only ``reproducible`` rows of
            the candidates' conditions count.
        k: call budget per transcript, or several.

    Returns:
        Tidy DataFrame (:data:`M1_COLUMNS`), one row per ``k`` x condition (plus ``all``,
        pooled) x stratum: ``all``, ``codon_class:<class>``, ``cds:in`` (cds_inframe,
        cds_outframe) and ``cds:out``. Precision strata use each call's own tags; recall
        strata the truth's. Recall counts only truth starts present among the candidates;
        the rest are ``n_truth_unscored``. NaN where a denominator is 0.
    """
    _check_candidates(candidates)
    ks = [k] if isinstance(k, (int, np.integer)) else list(k)
    conditions = sorted(set(candidates.condition.astype(str)))
    truth = truth[truth.reproducible.astype(bool) & truth.condition.isin(conditions)]
    scored = _keys(candidates)
    rows = []
    for kk in ks:
        calls = top_k_calls(candidates, int(kk))
        rows += _m1_rows(int(kk), ALL, calls, truth, scored)
        for condition in conditions:
            rows += _m1_rows(
                int(kk),
                condition,
                calls[calls.condition == condition],
                truth[truth.condition == condition],
                scored,
            )
    return pd.DataFrame(rows, columns=list(M1_COLUMNS))


def m2_direction(differential: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """M2: fraction of reproducibly differential starts whose predicted change has the right sign.

    Args:
        differential: from :func:`tisiago.truth.differential.all_differential_starts`
            (``condition_a``, ``condition_b``, site key, ``direction``).
        predictions: one row per condition and site with ``condition``, ``chrom``,
            ``strand``, ``gpos``, ``score``.

    Returns:
        :data:`M2_COLUMNS`, one row per condition pair plus ``all``/``all``: ``n_starts``
        (differential starts with a prediction in both conditions; read this first),
        ``n_differential``, ``n_correct`` and ``accuracy`` (NaN, not 0, when ``n_starts`` is
        0). A predicted difference of exactly 0 counts as wrong.
    """
    missing = [c for c in (*START_KEY, "score") if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    if predictions.duplicated(list(START_KEY)).any():
        raise ValueError("predictions have duplicate (condition, site) rows")
    score = predictions.set_index(list(START_KEY))["score"]
    rows = []
    total = {"n_starts": 0, "n_differential": 0, "n_correct": 0}
    pairs = differential.groupby(["condition_a", "condition_b"], sort=False)
    for (a, b), group in pairs:
        site = [group[c].to_numpy() for c in SITE_KEY]
        sa = score.reindex(pd.MultiIndex.from_arrays([np.full(len(group), a), *site])).to_numpy()
        sb = score.reindex(pd.MultiIndex.from_arrays([np.full(len(group), b), *site])).to_numpy()
        ok = ~(np.isnan(sa) | np.isnan(sb))
        correct = np.sign(sa[ok] - sb[ok]) == group.direction.to_numpy()[ok]
        row = {
            "condition_a": a,
            "condition_b": b,
            "n_starts": int(ok.sum()),
            "n_differential": len(group),
            "n_correct": int(correct.sum()),
        }
        row["accuracy"] = _ratio(row["n_correct"], row["n_starts"])
        rows.append(row)
        for key in total:
            total[key] += row[key]
    rows.insert(0, {"condition_a": ALL, "condition_b": ALL, **total})
    rows[0]["accuracy"] = _ratio(total["n_correct"], total["n_starts"])
    return pd.DataFrame(rows, columns=list(M2_COLUMNS))
