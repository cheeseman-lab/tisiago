"""Full 1-bp embeddings (1536-d) at every codon of ``codons.parquet``: base and fine-tuned backbone.

Usage (GPU, agft env):
    python autoresearch/p6nc/embed_codons.py --codons data/p6nc/codons.parquet \
        --checkpoint data/p6/runs/ar_rnacond_k562_fc2d_l2w300/best --out data/p6nc

Runs each val window once (``data/p6/windows_new/val.parquet``; a position takes the window whose
centre is nearest, as the txcache) through (1) the base AlphaGenome backbone and (2) the final
model's fine-tuned backbone, keeping the embedding at the codon's first nt. Writes
``<out>/emb_base.npy`` and ``<out>/emb_final.npy`` (rows of ``codons.parquet``, float16).
val chr7 only.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--codons", type=Path, default=Path("data/p6nc/codons.parquet"))
    ap.add_argument("--cache", type=Path, default=Path("data/p6/txcache/ar_rnacond_k562_v2"))
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("data/p6/runs/ar_rnacond_k562_fc2d_l2w300/best")
    )
    ap.add_argument("--windows", type=Path, default=Path("data/p6/windows_new/val.parquet"))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--out", type=Path, default=Path("data/p6nc"))
    args = ap.parse_args()

    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import dna_model
    from pyfaidx import Fasta

    from tisiago.seq2func.evaluate import head_kind_of, run_args
    from tisiago.seq2func.model import (
        ORGANISM,
        _alphagenome_backbone,
        load_finetuned,
        load_model,
        register_track_head,
    )
    from tisiago.seq2func.train import parse_rnacond_arm
    from tisiago.seq2func.txcache import TxCache, _window_onehot, unit_of, window_rows

    t0 = time.time()
    codons = pd.read_parquet(args.codons, columns=["transcript_id", "tx_pos"])
    cache = TxCache(args.cache)
    units = {r.transcript_id: unit_of(r) for r in cache.rows("val").itertuples()}
    windows = pd.read_parquet(args.windows)
    if set(windows.chrom) != {"chr7"}:
        raise SystemExit("val windows must be chr7 only")
    genomic = np.zeros(len(codons), np.int64)
    for tid, group in codons.groupby("transcript_id", sort=False):
        genomic[group.index.to_numpy()] = units[tid].positions[group.tx_pos.to_numpy()]
    plan = window_rows(windows, "chr7", genomic)
    print(f"{len(codons)} codons, {len(plan)} windows", flush=True)

    arm = run_args(args.checkpoint).get("arm")
    kind = head_kind_of(args.checkpoint)
    head, _, _ = parse_rnacond_arm(arm)
    register_track_head(head, 4, kind=kind)
    base = load_model(args.base_checkpoint, head)
    is_head = lambda k: k.startswith("head/")  # noqa: E731
    base_p = {k: v for k, v in base._params.items() if not is_head(k)}
    base_s = {k: v for k, v in base._state.items() if not is_head(k)}
    final = load_finetuned(str(args.checkpoint), str(args.base_checkpoint), head, 4, kind)
    fin_p = {k: v for k, v in final._params.items() if not is_head(k)}
    fin_s = {k: v for k, v in final._state.items() if not is_head(k)}
    backbone = _alphagenome_backbone()
    org = dna_model.convert_to_organism_index(ORGANISM)

    @hk.transform_with_state
    def embed_forward(onehot):
        organism = jnp.full((onehot.shape[0],), org, jnp.int32)
        return backbone(onehot, organism).get_sequence_embeddings(resolution=1)

    @jax.jit
    def embed(params, state, onehot):
        out, _ = embed_forward.apply(params, state, None, onehot[None])
        return out[0].astype(jnp.float16)

    print(f"models loaded {time.time() - t0:.0f}s on {jax.devices()}", flush=True)
    fasta = Fasta(str(args.genome), as_raw=True)
    out_base = np.zeros((len(codons), 1536), np.float16)
    out_final = np.zeros((len(codons), 1536), np.float16)
    for k, (start, end, rows) in enumerate(plan):
        onehot = jnp.asarray(_window_onehot(fasta, "chr7", start, end))
        sel = genomic[rows] - start
        out_base[rows] = np.asarray(embed(base_p, base_s, onehot))[sel]
        out_final[rows] = np.asarray(embed(fin_p, fin_s, onehot))[sel]
        if k % 20 == 0:
            print(f"window {k}/{len(plan)} {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "emb_base.npy", out_base)
    np.save(args.out / "emb_final.npy", out_final)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
