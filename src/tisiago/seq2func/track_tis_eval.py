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
divides it by the mean over offset +-30 plus ``--eps``; variants take the max over conditions
(``*_max``, the labels are a union over lines) or K562 alone (``*_K562``, reference only).

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
KINDS = ("peak", "enrich")
SELECTABLE = ("peak_max", "enrich_max")
PEAK_HALF = 1
CONTEXT_HALF = 30
PROFILE_HALF = 30
EPS = 1e-3
BLOCK_GAP = 100_000
BASELINE_ARM = "ag_w8k"
COLUMNS = ["transcript_id", "chrom", "strand", "mrna_index", "gstart", "codon_class", "label_tis"]

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
    """``peak`` (sum over offset +-1) and ``enrich`` (peak / (mean over +-30 + eps)) per row."""
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
    return {"peak": peak, "enrich": peak / (context / (2 * CONTEXT_HALF + 1) + eps)}


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


def aggregate(
    per_condition: Mapping[str, Mapping[str, np.ndarray]], reference: str = REFERENCE
) -> dict[str, np.ndarray]:
    """Score variants: max over conditions per kind and the reference condition alone."""
    out = {}
    for kind in KINDS:
        out[f"{kind}_max"] = np.max([scores[kind] for scores in per_condition.values()], axis=0)
    if reference in per_condition:
        for kind in KINDS:
            out[f"{kind}_{reference}"] = np.asarray(per_condition[reference][kind])
    return out


def score_rows(
    rows: pd.DataFrame,
    covered: np.ndarray,
    paths: PathsFn,
    conditions: Sequence[str],
    offset: int,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """All variants for ``rows``; rows outside the windows score 0."""
    per_condition = {}
    for condition in conditions:
        print(f"  scoring {condition} rows={len(rows):,}", flush=True)
        per_condition[condition] = condition_scores(rows, paths, condition, offset, eps)
    return {name: np.where(covered, score, 0.0) for name, score in aggregate(per_condition).items()}


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
    candidates: Sequence[str] = SELECTABLE,
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


def run_protocol(operating: dict, test: dict | None, *, budget: float = 1.0) -> dict:
    """Select the variant and threshold on ``operating``; apply each variant's once to ``test``.

    Blocks hold ``scores`` (variant -> array), ``labels`` and ``transcripts``; a test block may
    add ``strata`` (name -> row mask, default ``{"all": all rows}``) and ``pairs`` (positive and
    negative near-neighbour row indices).
    """
    selected, table = select_variant(
        operating["scores"], operating["labels"], operating["transcripts"], budget=budget
    )
    result = {"selected": selected, "threshold": table[selected]["threshold"], "operating": table}
    if test is None:
        return result
    n = len(test["labels"])
    strata = test.get("strata") or {"all": np.ones(n, dtype=bool)}
    pairs = test.get("pairs")
    result["test"] = {}
    for name, score in test["scores"].items():
        threshold = table[name]["threshold"]
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
        result["test"][name] = per_stratum
    return result


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


def predicted_dirs(templates: Mapping[str, str], split: str, conditions) -> dict[str, Path]:
    """Resolve ``{split}`` templates and check each is a float prediction of ``split``."""
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
        lines += ["", f"Baseline files: {', '.join(baseline['files'].values())}"]
        lines += ["", "All variants on test (each at its own operating threshold):", ""]
        lines += ["| source | variant | recall | FP/tx | AUPRC |", "|---|---|---|---|---|"]
        for source, result in report["sources"].items():
            for name, per_stratum in result["test"].items():
                m = per_stratum["all"]
                lines.append(
                    f"| {source} | {name} | {_fmt(m['recall'])} "
                    f"| {_fmt(m['fp_per_transcript'])} | {_fmt(m['auprc'])} |"
                )
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
    ap.add_argument("--eps", type=float, default=EPS)
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
            "eps": args.eps,
            "peak_half": PEAK_HALF,
            "context_half": CONTEXT_HALF,
            "bootstrap": args.bootstrap,
            "selectable": list(SELECTABLE),
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
            dirs = {s: predicted_dirs(templates, s, args.conditions) for s in regions}
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

    def block(key: str, source: str) -> dict:
        frame = frames[key]
        print(f"{source} {key}", flush=True)
        paths = paths_by_source[source](split_of[key])
        return {
            "scores": score_rows(frame, covered[key], paths, args.conditions, offset, args.eps),
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
        result = run_protocol(block("operating", source), test_block, budget=args.budget)
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
        report["baseline"] = baseline_numbers(args.baseline_dir, args.baseline_arm)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (out_dir / "report.md").write_text(render_markdown(report))
    print(render_markdown(report), flush=True)


if __name__ == "__main__":
    main()
