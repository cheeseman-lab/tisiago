"""Val scores (incl. the TIS sharpness columns) of step-2 runs from their exported float bigWigs.

Reads ``<export>/<run>/<C>/{tis,chx}_pooled.{plus,minus}.bw`` back into transcript space for
the cache's val transcripts and scores them with the unchanged ``fastloop.score_predictions``
(held-out condition only; other conditions zero and not scored). One row per run is appended
to ``--results`` (``fastloop.result_columns``; default ``sharpness_<condition>.tsv`` next to
this script). Positions shared by two transcripts on one strand hold the first transcript's
prediction (export convention), so values can differ slightly from the run's ``record.json``.
CPU only.

``--replicate`` instead writes replicate-referenced sharpness (:func:`replicate_sharpness`) to
``repsharp_<condition>.tsv``: real starts selected on replicate 2 and measured on replicate 1,
against the model resampled at replicate 1's per-transcript read count.

Usage:
    python autoresearch/p6sharp/sharpness.py --cache data/p6/txcache/fc2d_k562 --condition K562 \
        --runs sbflux_k562_s0 sbflux_k562_s1   # -> autoresearch/p6sharp/sharpness_k562.tsv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.predict_tracks import pooled_prefix, strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.targets import STRAND_FILES
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of


def export_predictions(cache: TxCache, export: Path, condition: str) -> dict[str, np.ndarray]:
    """``{transcript_id: (len(cache.conditions), L, 2)}`` val predictions from the bigWigs."""
    k = cache.condition_index(condition)
    reader = BigWigs()
    out = {}
    try:
        for row in cache.rows("val").itertuples():
            unit = unit_of(row)
            pred = np.zeros((len(cache.conditions), int(row.length), 2), np.float32)
            for c, lib in enumerate(fastloop.LIBRARIES):
                prefix = pooled_prefix(export, condition, lib)
                path = prefix.with_name(f"{prefix.name}.{STRAND_FILES[unit.strand]}.bw")
                values = spliced_values(reader, path, unit, strand_shift(unit.strand))
                pred[k, :, c] = values[: int(row.length)]
            out[row.transcript_id] = pred
    finally:
        reader.close()
    return out


def _window_stats(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per site: window mass, share at the start nt and argmax at the start (+-SHARP_FLANK)."""
    f = fastloop.SHARP_FLANK
    win = profiles[:, fastloop.FLANK - f : fastloop.FLANK + f + 1]
    total = win.sum(1)
    share = np.where(total > 0, win[:, f] / np.maximum(total, 1e-12), np.nan)
    return total, share, (win.argmax(1) == f).astype(float)


def replicate_sharpness(
    cache: TxCache, preds: dict, condition: str, seed: int = 0, min_reads: float = 50.0
) -> tuple[pd.DataFrame, dict]:
    """Sharpness with a replicate reference, free of the start-selection bias.

    Sites: Ribo-TISH starts of ``condition`` on expressed val transcripts, either all
    (``selection=all``) or those whose replicate-2 TIS argmax (+-30 nt) is within 1 nt of the
    start (``rep2hit``). Measured on replicate 1 (``rep1``), on the model sampled multinomially
    at each transcript's replicate-1 TIS read count (``model``, fixed seed) and on the float
    model (``model_float``). Share / argmax at the start within +-15 nt, over sites where both
    rep1 and the resampled model have reads in the window; per class (atg / nc / utr5) x exon
    region (all / first / later). Also top-1% concentration per transcript with replicate depths
    ``>= min_reads``: rep1 and rep2 thinned (hypergeometric) and the model sampled, all to
    ``n = min(rep1, rep2)`` reads.
    """
    rng = np.random.default_rng(seed)
    k = cache.condition_index(condition)
    val = cache.rows("val")
    units = val[val[f"expressed_{condition}"].to_numpy(bool)]
    vec, top = {}, {"rep1": [], "rep2": [], "model": []}
    for row in units.itertuples():
        lo, hi = int(row.offset), int(row.offset) + int(row.length)
        arr = np.rint(np.asarray(cache.array(row.shard, "tis")[k, lo:hi], np.float64))
        r1, r2 = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
        m = np.clip(np.asarray(preds[row.transcript_id][k, :, 0], np.float64), 0.0, None)
        p = m / m.sum() if m.sum() > 0 else None
        ms = rng.multinomial(int(r1.sum()), p) if p is not None else np.zeros_like(r1)
        vec[row.transcript_id] = (r1, r2, m, ms)
        n = int(min(r1.sum(), r2.sum()))
        if n >= min_reads and p is not None:
            for name, rep in (("rep1", r1), ("rep2", r2)):
                thin = rng.multivariate_hypergeometric(rep, n) if rep.sum() > n else rep
                top[name].append(fastloop.top_share(thin))
            top["model"].append(fastloop.top_share(rng.multinomial(n, p)))
    sites = cache.sites[cache.sites.condition == condition]
    sites = sites[sites.transcript_id.isin(set(vec))]
    junction = {
        r.transcript_id: (int(r.junctions[0]) if len(r.junctions) else None)
        for r in units.itertuples()
    }
    prof = {name: [] for name in ("rep1", "rep2", "model", "model_float")}
    cls, utr5, region = [], [], []
    for tid, group in sites.groupby("transcript_id", sort=False):
        pos = group.tx_pos.to_numpy(int)
        r1, r2, m, ms = vec[tid]
        for name, v in (("rep1", r1), ("rep2", r2), ("model", ms), ("model_float", m)):
            prof[name].append(fastloop.site_profiles(v, pos))
        cls.append(fastloop.start_classes(group.codon))
        utr5.append(group.tistype.astype(str).str.startswith(fastloop.UTR5_TYPE).to_numpy())
        region.append(fastloop.exon_regions(pos, junction[tid]))
    prof = {name: np.concatenate(parts) for name, parts in prof.items()}
    cls, utr5, region = np.concatenate(cls), np.concatenate(utr5), np.concatenate(region)
    stats = {name: _window_stats(p) for name, p in prof.items()}
    both = (stats["rep1"][0] > 0) & (stats["model"][0] > 0)
    rows = []
    selections = {"all": np.ones(len(cls), bool), "rep2hit": fastloop.site_hits(prof["rep2"])}
    for selection, chosen in selections.items():
        for c in fastloop.SHARP_CLASSES:
            in_class = utr5 if c == "utr5" else cls == c
            for reg in ("all", "first", "later"):
                sel = chosen & in_class & both & (True if reg == "all" else region == reg)
                for name in ("rep1", "model", "model_float"):
                    _, share, argmax = stats[name]
                    n = int(sel.sum())
                    rows.append(
                        {
                            "selection": selection,
                            "cls": c,
                            "region": reg,
                            "source": name,
                            "n": n,
                            "share15": float(np.mean(share[sel])) if n else float("nan"),
                            "argmax15": float(np.mean(argmax[sel])) if n else float("nan"),
                        }
                    )
    tops = {f"top1_{k}": float(np.mean(v)) if v else float("nan") for k, v in top.items()}
    tops["top1_n"] = len(top["model"])
    return pd.DataFrame(rows), tops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--export", type=Path, default=Path("data/p6/txtrain/export"))
    ap.add_argument("--results", type=Path, default=None, help="default sharpness_<c>.tsv here")
    ap.add_argument("--replicate", action="store_true", help="replicate-referenced sharpness")
    args = ap.parse_args()
    if args.replicate:
        return replicate_main(args)
    if args.results is None:
        args.results = Path(__file__).parent / f"sharpness_{args.condition.lower()}.tsv"
    cache = TxCache(args.cache)
    if cache.config.get("split") == "test":
        raise ValueError("validation only")
    columns = fastloop.result_columns([args.condition])
    for run in args.runs:
        t0 = time.time()
        preds = export_predictions(cache, args.export / run, args.condition)
        table, peaks = fastloop.score_predictions(
            cache, {"model": preds}, [args.condition], [args.condition]
        )
        meta = {
            "name": run,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": fastloop._git_commit(),
            "cache": str(args.cache),
            "heldout": args.condition,
            "heldout_list": [args.condition],
            "eval_seconds": time.time() - t0,
        }
        row = fastloop.results_row(meta, table, peaks, [args.condition])
        fastloop.append_result(args.results, row, columns)
        c = args.condition
        print(
            f"{run}: TIS nt {row[f'{c}_tis_nt']:.1f} share15 atg/nc/utr5 "
            + " / ".join(f"{row[f'{c}_tis_sharp_share15_{x}']:.3f}" for x in ("atg", "nc", "utr5"))
            + f" top1 {row[f'{c}_tis_top1_resampled']:.3f} (real {row[f'{c}_tis_top1_real']:.3f})",
            flush=True,
        )


def replicate_main(args) -> None:
    """Append :func:`replicate_sharpness` rows of every run to ``repsharp_<condition>.tsv``."""
    out = args.results or Path(__file__).parent / f"repsharp_{args.condition.lower()}.tsv"
    cache = TxCache(args.cache)
    if cache.config.get("split") == "test":
        raise ValueError("validation only")
    for run in args.runs:
        preds = export_predictions(cache, args.export / run, args.condition)
        table, tops = replicate_sharpness(cache, preds, args.condition)
        table.insert(0, "run", run)
        for key, value in tops.items():
            table[key] = value
        table.to_csv(out, sep="\t", index=False, mode="a", header=not out.exists())
        print(run, {k: round(v, 3) for k, v in tops.items()}, flush=True)


if __name__ == "__main__":
    main()
