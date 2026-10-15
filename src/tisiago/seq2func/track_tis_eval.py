"""Score candidate start codons by TIS-track signal on the frozen-embedding baseline's protocol.

Usage::

    python -m tisiago.seq2func.track_tis_eval --split val \
        --predicted K562=data/p6/runs/<run>/predicted_{split}_float ...
    python -m tisiago.seq2func.track_tis_eval --split test --i-am-final ...

Rows are the dense representation run's (``--baseline-dir``: ``row_selections.npz`` over
``--store``): calibration and operating halves of the validation transcripts and the test rows
(AUG + near-cognate on the held-out chromosomes). Each codon's A is mapped to the TIS-track
position ``offset`` nt 3' of it in transcript orientation, in the tracks' Ribo-TISH convention
(minus strand at canonical + ``RIBOTISH_MINUS_SHIFT``). The offset is the argmax of the summed
real-track TIS profile (rep1 + rep2, all conditions) around covered calibration positives and is
used for both sources. Per condition ``peak`` sums the track over offset +-1 and ``enrich``
divides it by the mean over offset +-30 plus each ``--eps``; variants take the max over conditions
(``*_max``, the labels are a union over lines) or K562 alone (``*_K562``, reference only).
``peak_norm_max`` scales each condition's peak per million of that source's TIS total over the
VAL windows, the same constant for operating and test rows (no test totals are read).

A source is ``predicted`` (``--mode float`` pooled TIS bigWigs from
:mod:`tisiago.seq2func.predict_tracks`) or ``real`` (rep1 + rep2 real tracks). Rows whose A lies
outside the union of the split's windows score 0 under both sources. The variant is selected on
the operating rows (recall at ``--budget`` FP/transcript, tie: AUPRC), its threshold is chosen
there and applied once to test. ``report.json`` and ``report.md`` hold the offset, coverage,
operating selection, test metrics by stratum with transcript-bootstrap CIs, and the baseline's
numbers read from ``--baseline-dir``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from sklearn.metrics import average_precision_score, roc_auc_score

from tisiago.caller import evaluate_at_threshold, recall_at_fp_budget
from tisiago.representation_eval import _bootstrap, near_neighbor_pairs
from tisiago.seq2func.predict_tracks import (
    MANIFEST,
    REPS,
    SPLIT_FILES,
    in_region,
    pooled_prefix,
    region_of,
)
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT
from tisiago.seq2func.targets import STRAND_FILES
from tisiago.seq2func.tracks import track_prefix

CONDITIONS = ("K562", "HeLa", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")
REFERENCE = "K562"
SOURCES = ("predicted", "real")
PEAK_HALF = 1
CONTEXT_HALF = 30
PROFILE_HALF = 30
EPS = 1e-3
EPS_GRID = (1e-3, 1 / 61, 0.1, 1.0, 3.0, 10.0, 30.0)
PER_MILLION = 1e6
CHUNK = 10_000_000
BOOTSTRAP_SEED = 20260909
AGREEMENT_TOLERANCE = 0.005
PAIRED_METRICS = ("recall", "fp_per_transcript", "auprc")
BLOCK_GAP = 100_000
BASELINE_ARM = "ag_w8k"
COLUMNS = [
    "row_idx",
    "transcript_id",
    "chrom",
    "strand",
    "mrna_index",
    "gstart",
    "codon_class",
    "label_tis",
]

PathsFn = Callable[[str, str], list[Path]]


def a_plus(gstart, strand) -> np.ndarray:
    """Vectorised :func:`tisiago.tiling.a_plus_of`: plus-strand position of each codon's A."""
    gstart = np.asarray(gstart, dtype=np.int64)
    return np.where(np.asarray(strand) == "-", gstart - 1, gstart)


def candidate_centre(gstart, strand, offset: int) -> np.ndarray:
    """TIS-track file position ``offset`` nt 3' (transcript orientation) of each codon's A.

    Minus-strand positions run down the genome and add the Ribo-TISH shift.
    """
    minus = np.asarray(strand) == "-"
    a = a_plus(gstart, strand)
    return np.where(minus, a - offset + RIBOTISH_MINUS_SHIFT, a + offset)


def coverage(rows: pd.DataFrame, region: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    """Whether each codon's A (canonical plus-strand position) lies in ``region``."""
    a = a_plus(rows.gstart, rows.strand.astype(str).to_numpy())
    out = np.zeros(len(rows), dtype=bool)
    chroms = rows.chrom.astype(str).to_numpy()
    for chrom, spans in region.items():
        mask = chroms == chrom
        out[mask] = in_region(a[mask], list(spans))
    return out


def coverage_table(labels, covered) -> dict:
    """Covered counts and fractions of positives and negatives."""
    labels = np.asarray(labels).astype(bool)
    covered = np.asarray(covered, dtype=bool)
    out = {}
    for name, mask in (("positives", labels), ("negatives", ~labels)):
        n = int(mask.sum())
        hit = int((covered & mask).sum())
        out[name] = {"n": n, "covered": hit, "fraction": hit / n if n else float("nan")}
    return out


def real_paths(tracks_dir: str | Path) -> PathsFn:
    """Real TIS replicate tracks of a condition and strand (summed when read)."""

    def paths(condition: str, strand: str) -> list[Path]:
        name = STRAND_FILES[strand]
        return [
            Path(f"{track_prefix(tracks_dir, condition, 'tis', rep)}.{name}.bw") for rep in REPS
        ]

    return paths


def predicted_paths(dirs: Mapping[str, str | Path]) -> PathsFn:
    """``--mode float`` pooled predicted TIS track of a condition and strand."""

    def paths(condition: str, strand: str) -> list[Path]:
        prefix = pooled_prefix(dirs[condition], condition, "tis")
        return [prefix.with_name(f"{prefix.name}.{STRAND_FILES[strand]}.bw")]

    return paths


def read_dense(paths: Sequence[Path], chrom: str, lo: int, hi: int) -> np.ndarray:
    """Sum of 1-bp bigWigs over file positions ``[lo, hi)``; 0 where absent or off-chromosome."""
    out = np.zeros(max(hi - lo, 0), dtype=np.float64)
    for path in paths:
        bw = pyBigWig.open(str(path))
        if bw is None:
            raise FileNotFoundError(path)
        try:
            size = bw.chroms().get(chrom)
            a, b = max(lo, 0), min(hi, size or 0)
            if b > a:
                values = bw.values(chrom, a, b, numpy=True)
                out[a - lo : b - lo] += np.nan_to_num(values, nan=0.0)
        finally:
            bw.close()
    return out


def _groups(rows: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Row positions per (chrom, strand)."""
    keys = [rows.chrom.astype(str).to_numpy(), rows.strand.astype(str).to_numpy()]
    return pd.Series(np.arange(len(rows))).groupby(keys, sort=True).indices


def _blocks(centres: np.ndarray, reach: int, gap: int | None = None):
    """Yield ``(row positions, lo, hi)`` per block of rows read together as ``[lo, hi)``.

    Rows whose ``centre +- reach`` spans lie within ``gap`` (default ``BLOCK_GAP``) share a block.
    """
    gap = BLOCK_GAP if gap is None else gap
    order = np.argsort(centres, kind="stable")
    ordered = centres[order]
    breaks = np.flatnonzero(np.diff(ordered) > 2 * reach + gap) + 1
    for block in np.split(order, breaks):
        yield block, int(centres[block].min()) - reach, int(centres[block].max()) + reach + 1


def condition_scores(
    rows: pd.DataFrame, paths: PathsFn, condition: str, offset: int, eps: float = EPS
) -> dict[str, np.ndarray]:
    """Per row ``peak`` (sum over offset +-1), ``context`` (mean over +-30), ``enrich``.

    ``enrich`` = peak / (context + eps).
    """
    centres = candidate_centre(rows.gstart, rows.strand.astype(str).to_numpy(), offset)
    peak = np.zeros(len(rows))
    context = np.zeros(len(rows))
    for (chrom, strand), index in _groups(rows).items():
        for block, lo, hi in _blocks(centres[index], CONTEXT_HALF):
            centre = centres[index[block]]
            dense = read_dense(paths(condition, strand), chrom, lo, hi)
            cumulative = np.concatenate([[0.0], np.cumsum(dense)])
            for half, target in ((PEAK_HALF, peak), (CONTEXT_HALF, context)):
                target[index[block]] = (
                    cumulative[centre + half + 1 - lo] - cumulative[centre - half - lo]
                )
    context = context / (2 * CONTEXT_HALF + 1)
    return {"peak": peak, "context": context, "enrich": peak / (context + eps)}


def offset_profile(
    rows: pd.DataFrame, paths: PathsFn, conditions: Sequence[str], half: int = PROFILE_HALF
) -> np.ndarray:
    """Summed track at offsets ``-half..half`` around the rows: ``[condition, strand(+,-), d]``."""
    centres = candidate_centre(rows.gstart, rows.strand.astype(str).to_numpy(), 0)
    offsets = np.arange(-half, half + 1)
    out = np.zeros((len(conditions), 2, len(offsets)))
    for (chrom, strand), index in _groups(rows).items():
        s = 0 if strand == "+" else 1
        for block, lo, hi in _blocks(centres[index], half):
            positions = centres[index[block]][:, None]
            positions = positions + (offsets if strand == "+" else -offsets)[None, :]
            for c, condition in enumerate(conditions):
                dense = read_dense(paths(condition, strand), chrom, lo, hi)
                out[c, s] += dense[positions - lo].sum(axis=0)
    return out


def estimate_offset(profile: np.ndarray) -> int:
    """Offset of the summed profile's maximum (ties: the most 5' offset)."""
    total = profile.sum(axis=(0, 1))
    return int(np.argmax(total)) - (len(total) - 1) // 2


def enrich_name(eps: float, scope: str = "max") -> str:
    """Variant name of ``enrich`` at ``eps`` (``scope`` is ``max`` or a condition)."""
    return f"enrich_{scope}_eps{eps:g}"


def selectable(eps_grid: Sequence[float] = EPS_GRID, *, normalised: bool = True) -> tuple:
    """Selectable variants in tie-break order: peak, depth-normalised peak, enrich per eps."""
    return ("peak_max", *(("peak_norm_max",) if normalised else ()), *map(enrich_name, eps_grid))


def aggregate(
    per_condition: Mapping[str, Mapping[str, np.ndarray]],
    *,
    eps_grid: Sequence[float] = EPS_GRID,
    totals: Mapping[str, float] | None = None,
    reference: str = REFERENCE,
) -> dict[str, np.ndarray]:
    """Score variants from per-condition ``peak`` and ``context``.

    ``*_max`` take the max over conditions; ``peak_norm_max`` first divides each condition's peak
    by its track total (``totals``) per million; ``enrich`` is computed at every ``eps_grid``
    value. The reference condition alone is kept as ``peak_<ref>`` / ``enrich_<ref>_eps*``.
    """
    peaks = {c: np.asarray(s["peak"], dtype=np.float64) for c, s in per_condition.items()}
    contexts = {c: np.asarray(s["context"], dtype=np.float64) for c, s in per_condition.items()}
    out = {"peak_max": np.max(list(peaks.values()), axis=0)}
    if totals is not None:
        bad = [c for c in peaks if not totals.get(c, 0) > 0]
        if bad:
            raise ValueError(f"no positive track total for {bad}")
        out["peak_norm_max"] = np.max([peaks[c] / totals[c] * PER_MILLION for c in peaks], axis=0)
    for eps in eps_grid:
        out[enrich_name(eps)] = np.max([peaks[c] / (contexts[c] + eps) for c in peaks], axis=0)
    if reference in peaks:
        out[f"peak_{reference}"] = peaks[reference]
        for eps in eps_grid:
            out[enrich_name(eps, reference)] = peaks[reference] / (contexts[reference] + eps)
    return out


def track_total(
    paths: PathsFn,
    condition: str,
    region: Mapping[str, Sequence[tuple[int, int]]],
    chunk: int = CHUNK,
) -> float:
    """Sum of a condition's track over canonical ``region`` on both strands."""
    total = 0.0
    for strand in STRAND_FILES:
        shift = RIBOTISH_MINUS_SHIFT if strand == "-" else 0
        for chrom, spans in region.items():
            for start, end in spans:
                for lo in range(int(start), int(end), chunk):
                    hi = min(lo + chunk, int(end))
                    total += float(
                        read_dense(paths(condition, strand), chrom, lo + shift, hi + shift).sum()
                    )
    return total


def score_rows(
    rows: pd.DataFrame,
    covered: np.ndarray,
    paths: PathsFn,
    conditions: Sequence[str],
    offset: int,
    eps_grid: Sequence[float] = EPS_GRID,
    totals: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """All variants for ``rows``; rows outside the windows score 0."""
    per_condition = {}
    for condition in conditions:
        print(f"  scoring {condition} rows={len(rows):,}", flush=True)
        per_condition[condition] = condition_scores(rows, paths, condition, offset)
    variants = aggregate(per_condition, eps_grid=eps_grid, totals=totals)
    return {name: np.where(covered, score, 0.0) for name, score in variants.items()}


def _auprc(score, labels) -> float:
    return float(average_precision_score(labels, score)) if np.any(labels) else float("nan")


def operating_metrics(score, labels, transcripts, budget: float = 1.0) -> dict:
    """Recall at the FP/transcript budget, its threshold, and AUPRC."""
    selected = recall_at_fp_budget(score, labels, transcripts, budget=budget)
    return {**selected, "auprc": _auprc(score, np.asarray(labels).astype(bool))}


def select_variant(
    scores: Mapping[str, np.ndarray],
    labels,
    transcripts,
    *,
    candidates: Sequence[str],
    budget: float = 1.0,
) -> tuple[str, dict[str, dict]]:
    """Pick the candidate with the highest recall at the budget (tie: AUPRC, then order)."""
    table = {name: operating_metrics(s, labels, transcripts, budget) for name, s in scores.items()}
    best = max(candidates, key=lambda name: (table[name]["recall"], table[name]["auprc"]))
    return best, table


def fixed_metrics(score, labels, transcripts, threshold: float, pairs=None) -> dict:
    """Metrics at a fixed threshold, ranking metrics and optional near-neighbour win rate."""
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    fixed = evaluate_at_threshold(score, labels, transcripts, threshold)
    admitted = fixed["true_positives"] + fixed["false_positives"]
    out = {
        **fixed,
        "precision": fixed["true_positives"] / max(1, admitted),
        "auprc": _auprc(score, labels),
        "auroc": float(roc_auc_score(labels, score))
        if len(np.unique(labels)) == 2
        else float("nan"),
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "transcripts": int(len(np.unique(transcripts))),
    }
    if pairs is not None:
        positive, negative = pairs
        out["winrate64"] = (
            float(np.mean(score[positive] > score[negative])) if len(positive) else float("nan")
        )
    return out


def run_protocol(
    operating: dict, test: dict | None, *, candidates: Sequence[str], budget: float = 1.0
) -> dict:
    """Select the variant and threshold on ``operating``; apply each variant's once to ``test``.

    Blocks hold ``scores`` (variant -> array), ``labels`` and ``transcripts``; a test block may
    add ``strata`` (name -> row mask, default ``{"all": all rows}``) and ``pairs`` (positive and
    negative near-neighbour row indices).
    """
    selected, table = select_variant(
        operating["scores"],
        operating["labels"],
        operating["transcripts"],
        candidates=candidates,
        budget=budget,
    )
    result = {"selected": selected, "threshold": table[selected]["threshold"], "operating": table}
    if test is not None:
        result["test"] = _apply(test, {name: m["threshold"] for name, m in table.items()})
    return result


def _apply(test: dict, thresholds: Mapping[str, float]) -> dict:
    """Each score at its fixed threshold, per stratum (see :func:`run_protocol`)."""
    n = len(test["labels"])
    strata = test.get("strata") or {"all": np.ones(n, dtype=bool)}
    pairs = test.get("pairs")
    out = {}
    for name, score in test["scores"].items():
        threshold = thresholds[name]
        per_stratum = {}
        for stratum, mask in strata.items():
            stratum_pairs = None
            if pairs is not None:
                keep = mask[pairs[0]] & mask[pairs[1]]
                remap = np.cumsum(mask) - 1
                stratum_pairs = (remap[pairs[0][keep]], remap[pairs[1][keep]])
            per_stratum[stratum] = fixed_metrics(
                score[mask],
                test["labels"][mask],
                test["transcripts"][mask],
                threshold,
                stratum_pairs,
            )
        out[name] = per_stratum
    return out


def run_protocol_fixed(score, frame: pd.DataFrame, strata, pairs, threshold: float) -> dict:
    """Metrics of a fixed-threshold score per stratum (the baseline's recomputation)."""
    block = {
        "scores": {"score": np.asarray(score, dtype=np.float64)},
        "labels": frame.label_tis.to_numpy().astype(bool),
        "transcripts": frame.transcript_id.to_numpy(),
        "strata": strata,
        "pairs": pairs,
    }
    return _apply(block, {"score": threshold})["score"]


def bootstrap_cis(
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    frame: pd.DataFrame,
    strata: Mapping[str, np.ndarray],
    *,
    n_bootstrap: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Transcript-bootstrap 95% CIs per stratum and arm (``representation_eval._bootstrap``)."""
    out: dict = {}
    for stratum, mask in strata.items():
        sub = frame.loc[mask].reset_index(drop=True)
        _, _, pair_groups = near_neighbor_pairs(sub, np.arange(len(sub)))
        intervals, _ = _bootstrap(
            {arm: np.asarray(s)[mask] for arm, s in scores.items()},
            dict(thresholds),
            sub.label_tis.to_numpy().astype(bool),
            sub.transcript_id.to_numpy(),
            pair_groups,
            n_bootstrap=n_bootstrap,
        )
        for row in intervals.itertuples():
            out.setdefault(row.arm, {}).setdefault(stratum, {})[row.metric] = {
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
            }
    return out


def _transcript_groups(transcripts) -> list[np.ndarray]:
    """Row indices per transcript in first-appearance order (as ``_bootstrap`` builds them)."""
    return [
        np.asarray(indices, dtype=np.int64)
        for indices in pd.Series(np.arange(len(transcripts)))
        .groupby(np.asarray(transcripts), sort=False)
        .apply(list)
    ]


def _paired_values(score, labels, threshold: float, n_transcripts: int) -> dict[str, float]:
    admitted = score >= threshold
    return {
        "recall": float(np.sum(admitted & labels) / max(1, np.sum(labels))),
        "fp_per_transcript": float(np.sum(admitted & ~labels) / max(1, n_transcripts)),
        "auprc": _auprc(score, labels),
    }


def paired_bootstrap(
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    labels,
    transcripts,
    *,
    reference: str,
    n_bootstrap: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired transcript bootstrap of every arm minus ``reference`` (one resample set for all).

    Each arm keeps its own threshold. The resamples equal ``representation_eval._bootstrap``'s
    for the same seed and rows. Returns ``arms`` (marginal 95% CIs) and ``<arm>-<reference>``
    (point difference and 95% CI) for recall, FP/transcript and AUPRC.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    labels = np.asarray(labels).astype(bool)
    groups = _transcript_groups(transcripts)
    arms = list(scores)
    draws = {arm: {m: [] for m in PAIRED_METRICS} for arm in arms}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[i] for i in sampled])
        for arm in arms:
            values = _paired_values(
                np.asarray(scores[arm])[rows], labels[rows], thresholds[arm], len(sampled)
            )
            for metric, value in values.items():
                draws[arm][metric].append(value)

    def interval(values) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "ci_low": float(np.nanquantile(values, 0.025)),
            "ci_high": float(np.nanquantile(values, 0.975)),
        }

    points = {
        arm: _paired_values(np.asarray(scores[arm]), labels, thresholds[arm], len(groups))
        for arm in arms
    }
    out: dict = {
        "arms": {arm: {m: interval(draws[arm][m]) for m in PAIRED_METRICS} for arm in arms}
    }
    for arm in arms:
        if arm == reference:
            continue
        out[f"{arm}-{reference}"] = {
            m: {
                "point": points[arm][m] - points[reference][m],
                **interval(np.subtract(draws[arm][m], draws[reference][m])),
            }
            for m in PAIRED_METRICS
        }
    return out


def load_baseline_calls(path: str | Path, row_idx, labels) -> np.ndarray:
    """Baseline ``p_tis`` aligned to ``row_idx`` (the test rows), labels checked."""
    calls = pd.read_parquet(path, columns=["row_idx", "label_tis", "p_tis"])
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if len(calls) != len(row_idx) or set(calls.row_idx) != set(row_idx.tolist()):
        raise ValueError(f"{path}: row_idx do not match the test rows")
    calls = calls.set_index("row_idx").loc[row_idx]
    if not np.array_equal(calls.label_tis.to_numpy().astype(bool), np.asarray(labels).astype(bool)):
        raise ValueError(f"{path}: label_tis differs from the manifest")
    return calls.p_tis.to_numpy(dtype=np.float64)


def baseline_agreement(stored: dict, recomputed: Mapping[str, dict], tol: float) -> dict:
    """Largest stored-vs-recomputed difference over recall, FP/transcript and AUPRC."""
    diffs = {}
    for stratum, metrics in recomputed.items():
        reference = stored["all"] if stratum == "all" else stored["strata"].get(stratum, {})
        for metric in PAIRED_METRICS:
            value = reference.get(metric)
            if value is not None and np.isfinite(value) and np.isfinite(metrics[metric]):
                diffs[f"{stratum}/{metric}"] = float(metrics[metric] - value)
    worst = max((abs(v) for v in diffs.values()), default=0.0)
    return {"ok": worst <= tol, "tolerance": tol, "max_abs_diff": worst, "diffs": diffs}


def baseline_numbers(baseline_dir: str | Path, arm: str = BASELINE_ARM) -> dict:
    """The baseline's ensemble test metrics, strata and bootstrap CIs from its output TSVs."""
    base = Path(baseline_dir)
    files = {
        "metrics": base / "metrics.tsv",
        "stratified": base / "stratified_metrics.tsv",
        "bootstrap": base / "bootstrap_ci.tsv",
    }
    metrics = pd.read_csv(files["metrics"], sep="\t", dtype={"seed": str})
    row = metrics[(metrics.arm == arm) & (metrics.seed == "ensemble")]
    if len(row) != 1:
        raise ValueError(f"{files['metrics']}: no single ensemble row for arm {arm}")
    stratified = pd.read_csv(files["stratified"], sep="\t")
    stratified = stratified[stratified.arm == arm]
    intervals = pd.read_csv(files["bootstrap"], sep="\t")
    intervals = intervals[intervals.arm == arm]
    out = {
        "arm": arm,
        "files": {k: str(v) for k, v in files.items()},
        "all": row.drop(columns=["arm", "seed"]).iloc[0].to_dict(),
        "strata": {
            r["stratum"]: {k: v for k, v in r.items() if k not in ("arm", "stratum")}
            for r in stratified.to_dict("records")
        },
        "bootstrap_ci": {
            r.metric: {"ci_low": r.ci_low, "ci_high": r.ci_high} for r in intervals.itertuples()
        },
    }
    run = base / "run.json"
    if run.is_file():
        meta = json.loads(run.read_text())
        out["files"]["run"] = str(run)
        out["run"] = {k: meta.get(k) for k in ("store", "split_counts", "test_codon_classes")}
    return out


def _parse_predicted(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        condition, sep, template = item.partition("=")
        if not sep or not template:
            raise ValueError(f"--predicted takes CONDITION=DIR_TEMPLATE, not {item!r}")
        out[condition] = template
    return out


def predicted_dirs(
    templates: Mapping[str, str], split: str, conditions, windows_dir: str | Path
) -> dict[str, Path]:
    """Resolve ``{split}`` templates and check each is a float prediction of ``split``.

    The prediction must use ``windows_dir`` and no real-data rescaling or masking
    (``match_depth`` / ``mask_unexpressed`` off).
    """
    out = {}
    for condition in conditions:
        if condition not in templates:
            raise ValueError(f"no --predicted directory for {condition}")
        path = Path(templates[condition].format(split=split))
        manifest = path / condition / MANIFEST
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        meta = json.loads(manifest.read_text())
        if meta.get("mode") != "float" or meta.get("split") != split:
            raise ValueError(
                f"{manifest}: need --mode float on split {split}, "
                f"got mode={meta.get('mode')} split={meta.get('split')}"
            )
        if meta.get("match_depth") or meta.get("mask_unexpressed"):
            raise ValueError(f"{manifest}: --match-depth / --mask-unexpressed use real data")
        used = meta.get("windows_dir")
        if used is None or Path(used).resolve() != Path(windows_dir).resolve():
            raise ValueError(f"{manifest}: predicted on windows {used}, not {windows_dir}")
        out[condition] = path
    return out


def _fmt(value) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{value:.3f}"


def _ci(ci: Mapping | None) -> str:
    return f"[{_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}]" if ci else ""


def render_markdown(report: dict) -> str:
    """Human-readable summary of ``report``."""
    offset = report["offset"]
    lines = [
        f"# P6 track TIS score — split {report['split']}",
        "",
        f"Offset (real-track TIS profile argmax over {offset['positives']} covered calibration "
        f"positives): **{offset['offset']}** nt 3' of the codon A "
        f"(plus {offset['by_strand']['+']}, minus {offset['by_strand']['-']}).",
        "",
        "## Coverage (codon A inside the split's windows)",
        "",
        "| rows | positives covered | negatives covered |",
        "|---|---|---|",
    ]
    for rows, table in report["coverage"].items():
        pos, neg = table["positives"], table["negatives"]
        lines.append(
            f"| {rows} | {pos['covered']}/{pos['n']} ({_fmt(pos['fraction'])}) "
            f"| {neg['covered']}/{neg['n']} ({_fmt(neg['fraction'])}) |"
        )
    lines += ["", "## Operating selection (recall @ budget, tie AUPRC)", ""]
    lines += ["| source | variant | recall | FP/tx | threshold | AUPRC | selected |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for source, result in report["sources"].items():
        for name, m in result["operating"].items():
            mark = "yes" if name == result["selected"] else ""
            lines.append(
                f"| {source} | {name} | {_fmt(m['recall'])} | {_fmt(m['fp_per_transcript'])} "
                f"| {m['threshold']:.4g} | {_fmt(m['auprc'])} | {mark} |"
            )
    if all("test" in result for result in report["sources"].values()) and report["sources"]:
        baseline = report["baseline"]
        lines += ["", "## Test (threshold from operating, applied once)", ""]
        lines += ["| arm | stratum | recall | FP/tx | precision | AUPRC | recall CI | AUPRC CI |"]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for source, result in report["sources"].items():
            name = result["selected"]
            for stratum, m in result["test"][name].items():
                ci = result["bootstrap_ci"].get(stratum, {})
                lines.append(
                    f"| {source} ({name}) | {stratum} | {_fmt(m['recall'])} "
                    f"| {_fmt(m['fp_per_transcript'])} | {_fmt(m['precision'])} "
                    f"| {_fmt(m['auprc'])} | {_ci(ci.get('recall'))} | {_ci(ci.get('auprc'))} |"
                )
        arm = f"baseline {baseline['arm']}"
        a, ci = baseline["all"], baseline["bootstrap_ci"]
        lines.append(
            f"| {arm} | all | {_fmt(a.get('recall'))} | {_fmt(a.get('fp_per_transcript'))} "
            f"| {_fmt(a.get('precision'))} | {_fmt(a.get('auprc'))} | {_ci(ci.get('recall'))} "
            f"| {_ci(ci.get('auprc'))} |"
        )
        for stratum, m in baseline["strata"].items():
            if stratum.startswith("codon_class:"):
                lines.append(
                    f"| {arm} | {stratum} | {_fmt(m.get('recall'))} "
                    f"| {_fmt(m.get('fp_per_transcript'))} | {_fmt(m.get('precision'))} "
                    f"| {_fmt(m.get('auprc'))} | | |"
                )
        recomputed = baseline.get("recomputed", {})
        for stratum, m in recomputed.items():
            if stratum == "all" or stratum.startswith("codon_class:"):
                lines.append(
                    f"| {arm} (recomputed from calls) | {stratum} | {_fmt(m['recall'])} "
                    f"| {_fmt(m['fp_per_transcript'])} | {_fmt(m['precision'])} "
                    f"| {_fmt(m['auprc'])} | | |"
                )
        if "agreement" in baseline:
            agreement = baseline["agreement"]
            verdict = "agree" if agreement["ok"] else "**DISAGREE**"
            lines += [
                "",
                f"Stored vs recomputed baseline: {verdict} (max |diff| "
                f"{agreement['max_abs_diff']:.4f}, tolerance {agreement['tolerance']}).",
            ]
        lines += ["", f"Baseline files: {', '.join(baseline['files'].values())}"]
        lines += [
            "",
            "Stratum CIs resample only the transcripts present in the stratum; the stored "
            "baseline has no stratum CIs, so compare strata through the paired table below.",
        ]
        if report.get("paired"):
            lines += ["", "## Paired transcript bootstrap (same resamples; arm minus baseline)", ""]
            lines += [
                "| stratum | comparison | metric | difference | 95% CI |",
                "|---|---|---|---|---|",
            ]
            for stratum, result in report["paired"].items():
                for comparison, metrics in result.items():
                    if comparison == "arms":
                        continue
                    for metric, d in metrics.items():
                        lines.append(
                            f"| {stratum} | {comparison} | {metric} | {d['point']:+.3f} "
                            f"| {_ci(d)} |"
                        )
        lines += ["", "All variants on test (each at its own operating threshold):", ""]
        lines += ["| source | variant | recall | FP/tx | AUPRC |", "|---|---|---|---|---|"]
        for source, result in report["sources"].items():
            for name, per_stratum in result["test"].items():
                m = per_stratum["all"]
                lines.append(
                    f"| {source} | {name} | {_fmt(m['recall'])} "
                    f"| {_fmt(m['fp_per_transcript'])} | {_fmt(m['auprc'])} |"
                )
    lines += [
        "",
        "Caveat: the P6 checkpoints were chosen on val (chr7) profile KL over 100 windows, so the "
        "operating half is not fully untouched by P6 model selection (the baseline's was).",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--store", type=Path, default=Path("data/dense_exp_store"))
    ap.add_argument("--baseline-dir", type=Path, default=Path("data/dense_txp_representation"))
    ap.add_argument("--baseline-arm", default=BASELINE_ARM)
    ap.add_argument("--selections", type=Path, help="default: <baseline-dir>/row_selections.npz")
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows_new"))
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/p6/tracks"))
    ap.add_argument(
        "--predicted",
        nargs="+",
        default=[],
        metavar="C=DIR",
        help="per condition: predict_tracks --mode float output dir; '{split}' is substituted",
    )
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=list(EPS_GRID),
        help="enrich eps candidates, selected on the operating rows with the variant",
    )
    ap.add_argument(
        "--baseline-calls",
        type=Path,
        default=Path("data/calls/ag_w8k_test_calls.parquet"),
        help="baseline per-row test scores (p_tis) for the paired bootstrap (--split test)",
    )
    ap.add_argument("--profile-half", type=int, default=PROFILE_HALF)
    ap.add_argument("--bootstrap", type=int, default=1_000)
    ap.add_argument("--out-dir", type=Path, help="default: data/p6/track_tis_eval/<split>")
    return ap


def main() -> None:
    """CLI: offset, coverage, operating selection, and (``--split test``) the test report."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    final = args.split == "test"
    try:
        templates = _parse_predicted(args.predicted)
    except ValueError as exc:
        ap.error(str(exc))
    if "predicted" in args.sources and not templates:
        ap.error("--sources predicted needs --predicted C=DIR for every condition")
    out_dir = args.out_dir or Path("data/p6/track_tis_eval") / args.split

    manifest = pd.read_parquet(args.store / "manifest.parquet", columns=COLUMNS)
    selections = np.load(args.selections or args.baseline_dir / "row_selections.npz")
    keys = ["calibration", "operating", *(["test"] if final else [])]
    frames = {key: manifest.iloc[selections[key]].reset_index(drop=True) for key in keys}
    regions = {
        split: region_of(pd.read_parquet(args.windows_dir / SPLIT_FILES[split]))
        for split in (["val", "test"] if final else ["val"])
    }
    split_of = {"calibration": "val", "operating": "val", "test": "test"}
    covered = {key: coverage(frame, regions[split_of[key]]) for key, frame in frames.items()}
    report: dict = {
        "split": args.split,
        "params": {
            "store": str(args.store),
            "selections": str(args.selections or args.baseline_dir / "row_selections.npz"),
            "windows_dir": str(args.windows_dir),
            "tracks_dir": str(args.tracks_dir),
            "conditions": list(args.conditions),
            "budget": args.budget,
            "eps_grid": list(args.eps),
            "peak_half": PEAK_HALF,
            "context_half": CONTEXT_HALF,
            "bootstrap": args.bootstrap,
            "selectable": list(selectable(args.eps)),
        },
        "coverage": {
            key: coverage_table(frame.label_tis.to_numpy(), covered[key])
            for key, frame in frames.items()
        },
    }

    baseline_run = args.baseline_dir / "run.json"
    if baseline_run.is_file():
        counts = json.loads(baseline_run.read_text()).get("split_counts", {})
        for key in keys:
            if key in counts and counts[key] != len(frames[key]):
                ap.error(f"{key}: {len(frames[key])} rows but the baseline used {counts[key]}")
    calibration = frames["calibration"]
    positives = calibration[(calibration.label_tis.to_numpy() == 1) & covered["calibration"]]
    real = real_paths(args.tracks_dir)
    profile = offset_profile(positives, real, args.conditions, args.profile_half)
    offset = estimate_offset(profile)
    half = args.profile_half
    report["offset"] = {
        "offset": offset,
        "source": "real",
        "rows": "calibration",
        "positives": int(len(positives)),
        "offsets": list(range(-half, half + 1)),
        "profile": profile.sum(axis=(0, 1)).tolist(),
        "by_strand": {s: estimate_offset(profile[:, [i]]) for i, s in enumerate("+-")},
        "by_condition": {c: estimate_offset(profile[[i]]) for i, c in enumerate(args.conditions)},
    }
    print(f"offset {offset} from {len(positives)} calibration positives", flush=True)

    paths_by_source: dict[str, Callable[[str], PathsFn]] = {"real": lambda split: real}
    if "predicted" in args.sources:
        try:
            dirs = {
                s: predicted_dirs(templates, s, args.conditions, args.windows_dir) for s in regions
            }
        except (ValueError, FileNotFoundError) as exc:
            ap.error(str(exc))
        paths_by_source["predicted"] = lambda split: predicted_paths(dirs[split])
        report["params"]["predicted"] = {
            s: {c: str(p) for c, p in d.items()} for s, d in dirs.items()
        }
        predicted_profile = offset_profile(
            positives, predicted_paths(dirs["val"]), args.conditions, half
        )
        report["offset"]["predicted_profile_argmax"] = estimate_offset(predicted_profile)

    # depth: each source's own TIS total over the VAL windows (predicted: predicted totals); the
    # same constant scales operating and test, so the operating threshold transfers to test
    totals = {
        source: {
            c: track_total(paths_by_source[source]("val"), c, regions["val"])
            for c in args.conditions
        }
        for source in args.sources
    }
    factors: dict[str, dict[str, dict[str, float]]] = {source: {} for source in args.sources}

    def block(key: str, source: str) -> dict:
        frame = frames[key]
        print(f"{source} {key}", flush=True)
        split = split_of[key]
        paths = paths_by_source[source](split)
        scores = score_rows(
            frame, covered[key], paths, args.conditions, offset, args.eps, totals[source]
        )
        factors[source][key] = {c: PER_MILLION / t for c, t in totals[source].items()}
        return {
            "scores": scores,
            "labels": frame.label_tis.to_numpy().astype(bool),
            "transcripts": frame.transcript_id.to_numpy(),
        }

    test_frame = frames.get("test")
    strata = pairs = None
    if final:
        classes = test_frame.codon_class.astype(str).to_numpy()
        strands = test_frame.strand.astype(str).to_numpy()
        strata = {"all": np.ones(len(test_frame), dtype=bool)}
        strata |= {f"codon_class:{c}": classes == c for c in sorted(set(classes))}
        strata |= {f"strand:{s}": strands == s for s in sorted(set(strands))}
        positive, negative, _ = near_neighbor_pairs(test_frame, np.arange(len(test_frame)))
        pairs = (positive, negative)
    report["sources"] = {}
    selected_scores, selected_thresholds = {}, {}
    for source in args.sources:
        test_block = None
        if final:
            test_block = block("test", source) | {"strata": strata, "pairs": pairs}
        result = run_protocol(
            block("operating", source),
            test_block,
            candidates=selectable(args.eps),
            budget=args.budget,
        )
        result["depth_normalisation"] = {
            "region": "val",
            "totals": totals[source],
            "factor": factors[source],
        }
        if final:
            selected_scores[source] = test_block["scores"][result["selected"]]
            selected_thresholds[source] = result["threshold"]
        report["sources"][source] = result
    if final:
        ci_strata = {k: v for k, v in strata.items() if not k.startswith("strand:")}
        cis = bootstrap_cis(
            selected_scores, selected_thresholds, test_frame, ci_strata, n_bootstrap=args.bootstrap
        )
        for source in args.sources:
            report["sources"][source]["bootstrap_ci"] = cis[source]
        baseline = baseline_numbers(args.baseline_dir, args.baseline_arm)
        baseline_score = load_baseline_calls(
            args.baseline_calls,
            test_frame.row_idx.to_numpy(),
            test_frame.label_tis.to_numpy(),
        )
        threshold = float(baseline["all"]["threshold"])
        baseline["files"]["calls"] = str(args.baseline_calls)
        baseline["recomputed"] = run_protocol_fixed(
            baseline_score, test_frame, strata, pairs, threshold
        )
        baseline["agreement"] = baseline_agreement(
            baseline, baseline["recomputed"], AGREEMENT_TOLERANCE
        )
        if not baseline["agreement"]["ok"]:
            print(f"WARNING baseline recomputation disagrees: {baseline['agreement']}", flush=True)
        report["baseline"] = baseline
        arms = dict(selected_scores) | {"baseline": baseline_score}
        arm_thresholds = dict(selected_thresholds) | {"baseline": threshold}
        report["paired"] = {
            stratum: paired_bootstrap(
                {arm: score[mask] for arm, score in arms.items()},
                arm_thresholds,
                test_frame.label_tis.to_numpy()[mask],
                test_frame.transcript_id.to_numpy()[mask],
                reference="baseline",
                n_bootstrap=args.bootstrap,
            )
            for stratum, mask in ci_strata.items()
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (out_dir / "report.md").write_text(render_markdown(report))
    print(render_markdown(report), flush=True)


if __name__ == "__main__":
    main()
