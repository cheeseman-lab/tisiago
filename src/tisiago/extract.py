"""Extract per-candidate frozen embeddings for one model spec + manifest shard.

GPU driver. Reuses the gruyerenome backends (``embed_positions``) and the tiling
module: it tiles a shard of the manifest, fetches genomic windows, runs one
forward pass per tile (batched), slices the candidate positions, and writes a
per-shard ``.npz`` partial keyed by ``row_idx``. ``assemble_store.py`` later
scatters all shards into the row-aligned vector store.

Sharding is by transcript (stable crc32 hash) so a transcript's candidates stay
together and tile sharing is preserved within a shard.

Run in the backend's env (``alphagenome`` for ag*, ``gruyerenome``/evo2 env for
evo2_8k). Needs ``pyfaidx`` for genomic fetch::

    python scripts/tis/extract_candidates.py \
        --manifest data/tis/manifest.parquet --tile-spec ag16k \
        --config configs/tis_alphagenome_16k.yaml \
        --genome /lab/.../genome.fa --out-dir data/tis/parts \
        --shard-id 0 --n-shards 20
"""

from __future__ import annotations

import argparse
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

from gruyerenome import Config, load_backend

from tisiago.tiling import MODEL_SPECS, group_into_tiles

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _fetch_window(fa: Fasta, chrom: str, start: int, window: int, strand: str) -> str:
    """Fetch a genomic window, N-padding chromosome overruns, revcomp on minus."""
    end = start + window
    chrom_len = len(fa[chrom])
    s, e = max(0, start), min(chrom_len, end)
    core = str(fa[chrom][s:e]).upper()
    seq = ("N" * (s - start)) + core + ("N" * (end - e))
    return _revcomp(seq) if strand == "-" else seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tile-spec", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--config", required=True, help="gruyerenome backend YAML.")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--tile-batch", type=int, default=0, help="Tiles per forward pass (default: config.batch_size).")
    args = ap.parse_args()

    import pandas as pd

    spec = MODEL_SPECS[args.tile_spec]
    cfg = Config.from_yaml(args.config)
    assert cfg.model == spec["backend"], f"config.model={cfg.model} != spec backend {spec['backend']}"
    tile_batch = args.tile_batch or cfg.batch_size
    window = spec["window"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_parquet(args.manifest)
    # shard by transcript so tiles stay intact
    shard_mask = m.transcript_id.map(lambda t: zlib.crc32(str(t).encode()) % args.n_shards == args.shard_id)
    shard = m[shard_mask]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(shard)} candidates / {shard.transcript_id.nunique()} transcripts", flush=True)
    if len(shard) == 0:
        np.savez_compressed(out_dir / f"{args.tile_spec}_shard{args.shard_id:05d}.npz", row_idx=np.zeros(0, np.int64))
        return

    tiles = group_into_tiles(list(shard.itertuples(index=False)), spec)
    print(f"[shard {args.shard_id}] {len(tiles)} tiles (share={len(shard)/max(1,len(tiles)):.2f})", flush=True)

    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    backend = load_backend(cfg)
    is_ag = spec["backend"].startswith("alphagenome")
    offsets = spec["offsets"]
    ltag = spec["length_tag"]

    # output accumulators: key -> {row_idx: vector}
    acc: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    for b in range(0, len(tiles), tile_batch):
        batch = tiles[b : b + tile_batch]
        seqs, positions, members = [], [], []
        for t in batch:
            seqs.append(_fetch_window(fa, t.chrom, t.start, window, t.strand))
            pos_list, mem_list = [], []
            for mem in t.members:
                for k in offsets:
                    p = min(window - 1, mem.offset + k)
                    pos_list.append(p)
                    mem_list.append((mem.row_idx, k))
            positions.append(pos_list)
            members.append(mem_list)

        outs = backend.embed_positions(seqs, positions)

        for i, mem_list in enumerate(members):
            out = outs[i]
            for j, (row_idx, k) in enumerate(mem_list):
                if is_ag:
                    acc[f"{spec['backend']}::{ltag}::decoder_1bp::off{k}"][row_idx] = out[j]
                else:
                    for layer, arr in out.items():
                        acc[f"{spec['backend']}::{ltag}::{layer}::off{k}"][row_idx] = arr[j]
        if (b // tile_batch) % 50 == 0:
            print(f"[shard {args.shard_id}] tiles {b+len(batch)}/{len(tiles)}", flush=True)

    # serialize: row_idx vector + one stacked array per key (row order = sorted row_idx)
    all_rows = sorted({ri for d in acc.values() for ri in d})
    save = {"row_idx": np.asarray(all_rows, dtype=np.int64)}
    for key, d in acc.items():
        dim = next(iter(d.values())).shape[0]
        mat = np.zeros((len(all_rows), dim), dtype=np.float16)
        for r, ri in enumerate(all_rows):
            if ri in d:
                mat[r] = d[ri].astype(np.float16)
        save[key] = mat
    out_path = out_dir / f"{args.tile_spec}_shard{args.shard_id:05d}.npz"
    np.savez_compressed(out_path, **save)
    print(f"[shard {args.shard_id}] wrote {out_path} ({len(all_rows)} candidates, {len(acc)} keys)", flush=True)


if __name__ == "__main__":
    main()
