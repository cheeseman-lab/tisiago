"""Did the flux head's learned stop hazard find stop codons? Hazard at stop vs other codons.

Usage:
    python autoresearch/p6flux/stop_hazard.py data/p6/fastloop/export_flux/<name> [...]
"""

from __future__ import annotations

import sys

import numpy as np

from tisiago.seq2func import fastloop
from tisiago.seq2func.txcache import TxCache

N_TX = 60


def main() -> None:
    import jax

    for export in sys.argv[1:]:
        config, params = fastloop.load_export(export)
        cache = TxCache(config["cache"])
        model = fastloop.rebuild_model(config, cache, params, "val")
        seen = []
        orig = fastloop.flux_chx_output

        def spy(p_chx, stop_logit, *args, _orig=orig):
            seen.append(np.asarray(jax.nn.sigmoid(stop_logit + fastloop.FLUX_STOP_BIAS)))
            return _orig(p_chx, stop_logit, *args)

        fastloop.flux_chx_output = spy
        rows = cache.rows("val")
        rows = rows[rows.expressed_K562].head(N_TX)
        stop_h, other_h = [], []
        for row in rows.itertuples():
            size = fastloop.bucket_length(min(int(row.length), 32768), 4096, 32768)
            batch = fastloop.stack_examples(
                [fastloop.crop_example(cache, row, "K562", 0, size)], [row]
            )
            seen.clear()
            model.apply(params, fastloop._device_batch(batch))
            hz = seen[0][0, : int(row.length)]
            seq = batch["seq"][0, : int(row.length)]
            is_stop = np.zeros(len(hz), bool)
            for i in range(len(seq) - 2):
                is_stop[i] = tuple(int(x) for x in seq[i : i + 3]) in fastloop.STOP_CODES
            stop_h.append(hz[is_stop])
            other_h.append(hz[~is_stop])
        fastloop.flux_chx_output = orig
        s, o = np.concatenate(stop_h), np.concatenate(other_h)
        print(
            f"{config['name']}: hazard at stop codons mean {s.mean():.4f} (median "
            f"{np.median(s):.4f}, n={len(s)}); elsewhere mean {o.mean():.4f} (median "
            f"{np.median(o):.4f}); ratio {s.mean() / o.mean():.2f}"
        )


if __name__ == "__main__":
    main()
