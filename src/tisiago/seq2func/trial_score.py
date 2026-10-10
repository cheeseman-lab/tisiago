"""Score one P6 autoresearch trial and append it to the results table.

Usage::

    python -m tisiago.seq2func.trial_score --run data/p6/runs/ar_seqconv_w50 \
        --ceiling data/p6/eval/ceiling_val.tsv \
        --reference-periodicity \
            data/p6/periodicity/reference_condition_k562/regions_val_max100_replicate.tsv \
        --base data/p6/runs/k562_base/trial_score.json --out autoresearch/p6/results.tsv

Reads the run's ``run.json`` (config, best step), its validation evaluate TSV
(``<run>/eval/<arm>_val.tsv`` from ``evaluate --out-dir <run>/eval``) and its periodicity
regions TSV (``<run>/periodicity/regions_val*.tsv``); both can be passed explicitly. Writes
``<run>/trial_score.json`` (the ``--base`` input of later trials) and appends one row to
``--out``. Validation (chr7) only: there is no test-split input.

Fit (means over the run's channels; the ceiling rows are the same condition/library/strand):

- ``val_profile_kl`` = mean ``profile_kl``; ``profile_kl_rel`` = ceiling KL / model KL.
- ``gene_log_pearson`` = mean ``log_pearson``; ``log_pearson_rel`` = model r / ceiling r.

Both relative values are 1 at the replicate ceiling and fall below 1 as the model falls short of
it. The ceiling scores replicate 1 against smoothed replicate 2 while the model is scored
against the pooled replicates, so a value above 1 is possible and is not "better than data".

Periodicity (CHX, pooled CDS ``all`` segments as ``periodicity.summarize``): ``dom_agree``,
``switch_pred_within`` vs ``switch_obs_within``, ``annot_frame0_pred`` vs ``annot_frame0_obs``,
``p3_pred_median`` / ``p3_obs_median``, and 5'UTR ``utr5_dom_agree``. ``periodicity_score`` is the
mean of ``dom_agree / ref_dom_agree`` (the replicate reference), ``1 - |switch_pred_within -
switch_obs_within|`` and ``annot_frame0_pred / annot_frame0_obs``, each clipped to [0, 1]; NaN
when any term is NaN.

``gate_fit`` (needs ``--base``): ``val_profile_kl <= base + 0.02`` and ``gene_log_pearson >=
base - 0.005``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.overfit import finite_or_none

KL_TOLERANCE = 0.02
PEARSON_TOLERANCE = 0.005
CHANNEL_KEYS = ["condition", "library", "strand"]
IO_ARGS = frozenset(
    {
        "out",
        "windows_dir",
        "expression",
        "tracks_dir",
        "workers",
        "prefetch",
        "checkpoint",
        "genome",
        "overwrite",
        "eval_every",
        "eval_windows",
    }
)
COLUMNS = [
    "run",
    "config",
    "best_step",
    "val_profile_kl",
    "ceiling_profile_kl",
    "profile_kl_rel",
    "gene_log_pearson",
    "ceiling_log_pearson",
    "log_pearson_rel",
    "gate_fit",
    "periodicity_score",
    "dom_agree",
    "ref_dom_agree",
    "switch_pred_within",
    "switch_obs_within",
    "annot_frame0_pred",
    "annot_frame0_obs",
    "p3_pred_median",
    "p3_obs_median",
    "utr5_dom_agree",
]


def run_config(args: dict) -> str:
    """Compact JSON of the run's training arguments, without paths and I/O settings."""
    return json.dumps({k: v for k, v in sorted(args.items()) if k not in IO_ARGS})


def fit_metrics(model: pd.DataFrame, ceiling: pd.DataFrame) -> dict:
    """Channel-mean KL and gene log-Pearson of the model, the ceiling, and their ratios."""
    keys = model[CHANNEL_KEYS].drop_duplicates()
    ceil = keys.merge(ceiling, on=CHANNEL_KEYS, how="left")
    if len(ceil) != len(keys) or ceil.profile_kl.isna().all():
        raise ValueError("ceiling TSV lacks the run's channels")
    kl, ceil_kl = float(model.profile_kl.mean()), float(ceil.profile_kl.mean())
    r, ceil_r = float(model.log_pearson.mean()), float(ceil.log_pearson.mean())
    return {
        "val_profile_kl": kl,
        "ceiling_profile_kl": ceil_kl,
        "profile_kl_rel": ceil_kl / kl if kl > 0 else math.nan,
        "gene_log_pearson": r,
        "ceiling_log_pearson": ceil_r,
        "log_pearson_rel": r / ceil_r if ceil_r > 0 else math.nan,
    }


def pooled(regions: pd.DataFrame, region: str) -> dict:
    """``periodicity._pooled`` of the ``all`` segments of ``region``; empty if none."""
    from tisiago.seq2func.periodicity import _pooled

    rows = regions[(regions.region == region) & (regions.segment == "all")]
    return _pooled(rows) if len(rows) else {}


def read_markdown_summary(path: Path) -> pd.DataFrame:
    """The pipe table of a ``periodicity`` ``summary_*.md`` as a frame."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"{path} has no summary table")
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    table = pd.DataFrame(rows, columns=header)
    for col in table.columns:
        if col not in ("condition", "region", "stratum"):
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def reference_dom_agree(path: Path) -> float:
    """Pooled CDS ``dom_agree`` of the replicate reference (regions TSV or summary markdown)."""
    path = Path(path)
    if path.suffix == ".md":
        table = read_markdown_summary(path)
        rows = table[(table.region == "cds") & (table.stratum == "all")]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one cds/all row, found {len(rows)}")
        return float(rows.dom_agree.iloc[0])
    return float(pooled(pd.read_csv(path, sep="\t"), "cds").get("dom_agree", math.nan))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) if math.isfinite(x) else math.nan


def _ratio(num: float, den: float) -> float:
    return num / den if math.isfinite(num) and math.isfinite(den) and den > 0 else math.nan


def periodicity_metrics(regions: pd.DataFrame, ref_dom_agree: float) -> dict:
    """CDS ``all`` periodicity metrics, 5'UTR ``dom_agree`` and ``periodicity_score``."""
    cds = pooled(regions, "cds")
    utr = pooled(regions, "utr5")
    get = lambda d, k: float(d.get(k, math.nan))  # noqa: E731
    out = {
        "dom_agree": get(cds, "dom_agree"),
        "ref_dom_agree": float(ref_dom_agree),
        "switch_pred_within": get(cds, "switch_pred_within"),
        "switch_obs_within": get(cds, "switch_obs_within"),
        "annot_frame0_pred": get(cds, "annot_frame0_pred"),
        "annot_frame0_obs": get(cds, "annot_frame0_obs"),
        "p3_pred_median": get(cds, "p3_pred_median"),
        "p3_obs_median": get(cds, "p3_obs_median"),
        "utr5_dom_agree": get(utr, "dom_agree"),
    }
    terms = [
        _clip01(_ratio(out["dom_agree"], out["ref_dom_agree"])),
        _clip01(1.0 - abs(out["switch_pred_within"] - out["switch_obs_within"])),
        _clip01(_ratio(out["annot_frame0_pred"], out["annot_frame0_obs"])),
    ]
    out["periodicity_score"] = float(np.mean(terms)) if all(map(math.isfinite, terms)) else math.nan
    return out


def gate_fit(score: dict, base: dict | None) -> bool | None:
    """Fit gate against the base run's score; ``None`` without a base."""
    if base is None:
        return None
    return bool(
        score["val_profile_kl"] <= base["val_profile_kl"] + KL_TOLERANCE
        and score["gene_log_pearson"] >= base["gene_log_pearson"] - PEARSON_TOLERANCE
    )


def default_eval_tsv(run: Path, arm: str) -> Path:
    """``<run>/eval/<arm>_val.tsv`` as ``evaluate --out-dir <run>/eval`` writes it."""
    from tisiago.seq2func.evaluate import output_paths

    return output_paths(Path(run) / "eval", arm, "val")["model"]


def default_periodicity_tsv(run: Path) -> Path:
    """The single model ``regions_val*.tsv`` under ``<run>/periodicity``."""
    found = sorted(
        p
        for p in (Path(run) / "periodicity").glob("regions_val*.tsv")
        if not p.stem.endswith("_replicate")
    )
    if len(found) != 1:
        raise ValueError(f"expected one regions_val*.tsv in {run}/periodicity, found {found}")
    return found[0]


def score_run(
    run: Path,
    *,
    ceiling: Path,
    reference_periodicity: Path,
    base: dict | None = None,
    eval_tsv: Path | None = None,
    periodicity_tsv: Path | None = None,
) -> dict:
    """One results row (see the module docstring) for the run directory ``run``."""
    run = Path(run)
    info = json.loads((run / "run.json").read_text())
    args = info.get("args", {})
    model = pd.read_csv(eval_tsv or default_eval_tsv(run, args["arm"]), sep="\t")
    if set(model.get("split", ["val"])) != {"val"}:
        raise ValueError("trial scores read validation (chr7) metrics only")
    regions = pd.read_csv(periodicity_tsv or default_periodicity_tsv(run), sep="\t")
    score = {
        "run": run.name,
        "config": run_config(args),
        "best_step": (info.get("best") or {}).get("step"),
        **fit_metrics(model, pd.read_csv(ceiling, sep="\t")),
        **periodicity_metrics(regions, reference_dom_agree(reference_periodicity)),
    }
    score["gate_fit"] = gate_fit(score, base)
    return {k: score[k] for k in COLUMNS}


def append_row(row: dict, path: Path) -> None:
    """Append ``row`` to the TSV at ``path`` (header on creation), under an exclusive lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        header = fh.readline().rstrip("\n")
        if header and header.split("\t") != COLUMNS:
            raise ValueError(f"{path} has another header; move it aside or use a new --out")
        fh.seek(0, 2)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            fh, sep="\t", index=False, header=not header, na_rep="NA"
        )


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory (run.json)")
    ap.add_argument("--ceiling", type=Path, required=True, help="evaluate ceiling_val.tsv")
    ap.add_argument(
        "--reference-periodicity",
        type=Path,
        required=True,
        help="replicate reference regions TSV or summary markdown",
    )
    ap.add_argument("--base", type=Path, default=None, help="base run's trial_score.json")
    ap.add_argument("--eval-tsv", type=Path, default=None, help="default <run>/eval/<arm>_val.tsv")
    ap.add_argument(
        "--periodicity", type=Path, default=None, help="default <run>/periodicity/regions_val*.tsv"
    )
    ap.add_argument("--out", type=Path, default=Path("autoresearch/p6/results.tsv"))
    return ap


def main() -> None:
    """CLI: score one run, write ``<run>/trial_score.json``, append to ``--out``."""
    args = build_parser().parse_args()
    base = json.loads(args.base.read_text()) if args.base else None
    row = score_run(
        args.run,
        ceiling=args.ceiling,
        reference_periodicity=args.reference_periodicity,
        base=base,
        eval_tsv=args.eval_tsv,
        periodicity_tsv=args.periodicity,
    )
    clean = {k: finite_or_none(v) if isinstance(v, float) else v for k, v in row.items()}
    (args.run / "trial_score.json").write_text(json.dumps(clean, indent=2) + "\n")
    append_row(row, args.out)
    print(json.dumps(clean, indent=2), flush=True)


if __name__ == "__main__":
    main()
