"""Val scores (incl. the TIS sharpness columns) of step-2 runs from their exported float bigWigs.

Reads ``<export>/<run>/<C>/{tis,chx}_pooled.{plus,minus}.bw`` back into transcript space for
the cache's val transcripts and scores them with the unchanged ``fastloop.score_predictions``
(held-out condition only; other conditions zero and not scored). One row per run is appended
to ``--results`` (``fastloop.result_columns``). Positions shared by two transcripts on one strand
hold the first transcript's prediction (export convention), so values can differ slightly from
the run's own ``record.json``. CPU only.

Usage:
    python autoresearch/p6sharp/sharpness.py --cache data/p6/txcache/fc2d_k562 --condition K562 \
        --runs sbflux_k562_s0 sbflux_k562_s1 --results autoresearch/p6sharp/sharpness.tsv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--export", type=Path, default=Path("data/p6/txtrain/export"))
    ap.add_argument("--results", type=Path, default=Path("autoresearch/p6sharp/sharpness.tsv"))
    args = ap.parse_args()
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


if __name__ == "__main__":
    main()
