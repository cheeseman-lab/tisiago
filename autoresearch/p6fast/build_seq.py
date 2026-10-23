"""Write ``shards/<shard>/seq.npy`` (mRNA base codes A C G T = 0..3, N = 4) into transcript caches.

Usage: python autoresearch/p6fast/build_seq.py --genome $TISIAGO_GENOME CACHE [CACHE ...]
Rows follow the cache's shard layout (index order, ``offset``/``length``), 5'->3' sense (minus
strand reverse-complemented), so ``seq[offset + i]`` is the base at mRNA position ``i``. Caches
with identical index rows reuse the first one's arrays. Sequence only: no annotation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

from tisiago.seq2func.txcache import TxCache, shard_dir, unit_of

CODE = np.full(256, 4, np.int8)
for i, b in enumerate("ACGT"):
    CODE[ord(b)] = CODE[ord(b.lower())] = i
COMPLEMENT = np.array([3, 2, 1, 0, 4], np.int8)


def transcript_codes(fasta, unit) -> np.ndarray:
    parts = [str(fasta[unit.chrom][int(s) : int(e)]) for s, e in sorted(unit.exons)]
    codes = CODE[np.frombuffer("".join(parts).encode(), np.uint8)]
    return codes if unit.strand == "+" else COMPLEMENT[codes[::-1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caches", nargs="+", type=Path)
    ap.add_argument("--genome", required=True)
    args = ap.parse_args()
    fasta = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    done: dict[str, np.ndarray] = {}
    first_index = None
    for path in args.caches:
        cache = TxCache(path)
        cols = ["transcript_id", "shard", "offset", "length"]
        same = first_index is not None and cache.index[cols].equals(first_index[cols])
        first_index = cache.index if first_index is None else first_index
        for shard, rows in cache.index.groupby("shard", sort=False):
            if same and shard in done:
                out = done[shard]
            else:
                rows = rows.sort_values("offset")
                out = np.full(int(rows.length.sum()), 4, np.int8)
                for row in rows.itertuples():
                    unit = unit_of(row)
                    codes = transcript_codes(fasta, unit)
                    assert len(codes) == int(row.length), (row.transcript_id, len(codes))
                    out[int(row.offset) : int(row.offset) + int(row.length)] = codes
                done[shard] = out
            dest = shard_dir(path, shard)
            dest.mkdir(parents=True, exist_ok=True)
            tmp = dest / ".seq.npy.tmp.npy"
            np.save(tmp, out)
            tmp.rename(dest / "seq.npy")
            print(path, shard, len(out), f"N={float((out == 4).mean()):.4f}", flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
