"""New 5'UTR columns (fastloop.utr5_summary) for exported val predictions (step 2 / step 1).

Usage:
    python autoresearch/p6utr5/score_exports.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.predict_tracks import strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of

EXPORTS = {
    "step2_s0": "data/p6/txtrain/export/sb_k562_s0/K562",
    "step1_s0": "data/p6/fastloop/export_twostage/ts_k562_s0/K562",
}


def main() -> None:
    cache = TxCache("data/p6/txcache/fc2d_k562")
    k = cache.condition_index("K562")
    rows = cache.rows("val")
    reader = BigWigs()
    sources = {}
    for name, path in EXPORTS.items():
        preds = {}
        for r in rows.itertuples():
            unit = unit_of(r)
            arr = np.zeros((len(cache.conditions), int(r.length), 2), np.float32)
            for c, lib in enumerate(("tis", "chx")):
                f = Path(path) / f"{lib}_pooled.{'plus' if unit.strand == '+' else 'minus'}.bw"
                arr[k, :, c] = spliced_values(reader, f, unit, strand_shift(unit.strand))
            preds[r.transcript_id] = arr
        sources[name] = preds
    ckpt = fastloop.ckpt_predictions(cache)
    sources["genomic"] = ckpt
    _, peaks = fastloop.score_predictions(cache, sources, ["K562"], heldout=["K562"])
    cols = ["source"] + [c for c in peaks.columns if c.startswith("utr5_")] + ["rep_ratio_nc"]
    print(peaks[cols].round(3).to_string())


if __name__ == "__main__":
    main()
