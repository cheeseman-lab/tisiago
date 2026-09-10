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
        --genome /path/to/genome.fa --out-dir data/tis/parts \
        --shard-id 0 --n-shards 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tisiago.extraction_io import PreallocatedShard, shard_is_complete
from tisiago.manifest import transcript_shard_mask
from tisiago.sequence import fetch_genomic_window
from tisiago.tiling import MODEL_SPECS, group_into_tiles, tile_position_requests


def _expected_keys(spec: dict, cfg) -> set[str]:
    """Return the exact archive members produced by one model configuration."""
    ltag = spec["length_tag"]
    if spec["backend"].startswith("alphagenome"):
        layers = ["decoder_1bp"]
    else:
        layers = list(getattr(cfg, "evo2_layers", None) or ["blocks.28.mlp.l3"])
    return {
        f"{spec['backend']}::{ltag}::{layer}::off{offset}"
        for layer in layers
        for offset in spec["offsets"]
    }


def main() -> None:
    """Extract one manifest shard for one frozen-model tile specification."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tile-spec", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--config", required=True, help="gruyerenome backend YAML.")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--tile-batch",
        type=int,
        default=0,
        help="Tiles per forward pass (default: config.batch_size).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip an existing shard after validating its rows and feature keys.",
    )
    ap.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress NPZ output (default: enabled; --no-compress favors write speed).",
    )
    args = ap.parse_args()

    import pandas as pd
    from gruyerenome import Config, load_backend
    from pyfaidx import Fasta

    spec = MODEL_SPECS[args.tile_spec]
    cfg = Config.from_yaml(args.config)
    if cfg.model != spec["backend"]:
        raise ValueError(f"config.model={cfg.model} != spec backend {spec['backend']}")
    tile_batch = args.tile_batch or cfg.batch_size
    if tile_batch <= 0:
        raise ValueError("tile batch size must be positive")
    if not 0 <= args.shard_id < args.n_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < n-shards")
    window = spec["window"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tile_spec}_shard{args.shard_id:05d}.npz"
    expected_keys = _expected_keys(spec, cfg)

    started = time.perf_counter()
    m = pd.read_parquet(args.manifest)
    required = {"row_idx", "transcript_id", "chrom", "gstart", "strand", "codon"}
    missing_columns = required - set(m.columns)
    if missing_columns:
        raise ValueError(f"manifest is missing columns: {sorted(missing_columns)}")
    # shard by transcript so tiles stay intact
    shard_mask = transcript_shard_mask(m.transcript_id, args.shard_id, args.n_shards)
    shard = m[shard_mask]
    row_idx = np.sort(shard.row_idx.to_numpy(dtype=np.int64))
    if len(np.unique(row_idx)) != len(row_idx):
        raise ValueError("manifest row_idx values must be unique")
    print(
        f"[shard {args.shard_id}/{args.n_shards}] {len(shard)} candidates / "
        f"{shard.transcript_id.nunique()} transcripts",
        flush=True,
    )
    resume_keys = expected_keys if len(row_idx) else set()
    if args.resume and shard_is_complete(out_path, row_idx, resume_keys):
        print(f"[shard {args.shard_id}] complete output exists; skipping {out_path}", flush=True)
        return
    if len(shard) == 0:
        PreallocatedShard(row_idx).save(out_path, compressed=args.compress)
        return

    tiles = group_into_tiles(shard.itertuples(index=False), spec)
    # Stable genomic order improves FASTA locality and makes reruns reproducible.
    tiles.sort(key=lambda tile: (tile.chrom, tile.start, tile.strand))
    print(
        f"[shard {args.shard_id}] {len(tiles)} tiles "
        f"(share={len(shard) / max(1, len(tiles)):.2f})",
        flush=True,
    )

    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    backend = load_backend(cfg)
    is_ag = spec["backend"].startswith("alphagenome")
    offsets = spec["offsets"]
    ltag = spec["length_tag"]

    output = PreallocatedShard(row_idx)
    fetch_seconds = inference_seconds = collect_seconds = 0.0

    for b in range(0, len(tiles), tile_batch):
        batch = tiles[b : b + tile_batch]
        stage_started = time.perf_counter()
        seqs, positions, members = [], [], []
        for t in batch:
            seqs.append(fetch_genomic_window(fa, t.chrom, t.start, window, t.strand))
            pos_list, mem_list = tile_position_requests(t, offsets)
            positions.append(pos_list)
            members.append(mem_list)
        fetch_seconds += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        outs = backend.embed_positions(seqs, positions)
        inference_seconds += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        for i, mem_list in enumerate(members):
            out = outs[i]
            member_rows = np.fromiter((row for row, _, _ in mem_list), dtype=np.int64)
            member_offsets = np.fromiter((offset for _, offset, _ in mem_list), dtype=np.int64)
            source_rows = np.fromiter((source for _, _, source in mem_list), dtype=np.int64)
            if is_ag:
                for k in offsets:
                    select = member_offsets == k
                    key = f"{spec['backend']}::{ltag}::decoder_1bp::off{k}"
                    output.add(key, member_rows[select], out[source_rows[select]])
            else:
                for layer, arr in out.items():
                    for k in offsets:
                        select = member_offsets == k
                        key = f"{spec['backend']}::{ltag}::{layer}::off{k}"
                        output.add(key, member_rows[select], arr[source_rows[select]])
        collect_seconds += time.perf_counter() - stage_started
        if (b // tile_batch) % 50 == 0:
            print(f"[shard {args.shard_id}] tiles {b+len(batch)}/{len(tiles)}", flush=True)

    write_started = time.perf_counter()
    output.save(
        out_path,
        expected_keys=expected_keys,
        compressed=args.compress,
    )
    write_seconds = time.perf_counter() - write_started
    elapsed = time.perf_counter() - started
    print(
        f"[shard {args.shard_id}] wrote {out_path} "
        f"({len(row_idx)} candidates, {len(output.arrays)} keys; "
        f"compression={'on' if args.compress else 'off'})",
        flush=True,
    )
    print(
        f"[shard {args.shard_id}] timing: total={elapsed:.1f}s "
        f"fetch={fetch_seconds:.1f}s inference={inference_seconds:.1f}s "
        f"collect={collect_seconds:.1f}s write={write_seconds:.1f}s "
        f"throughput={len(tiles) / max(inference_seconds, 1e-9):.2f} tiles/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
