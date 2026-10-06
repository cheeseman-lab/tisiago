"""One-window overfit of the heads-only track head (P6.1 training-core check).

Usage::

    python -m tisiago.seq2func.overfit --condition HeLa --chrom chr12 --center 6536000 \
        --steps 300 --out data/p6/overfit/HeLa.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HEAD = "p6_overfit"
WINDOW = 131_072
TIMING_SKIP = 2


def fetch_onehot(genome: str, chrom: str, start: int, end: int) -> np.ndarray:
    """One-hot ``(end - start, 4)`` for ``[start, end)``, N-padded past the chromosome ends."""
    from alphagenome_research.model.one_hot_encoder import DNAOneHotEncoder
    from pyfaidx import Fasta

    fa = Fasta(genome, as_raw=True, sequence_always_upper=True)
    try:
        size = len(fa[chrom])
        lo, hi = max(start, 0), min(end, size)
        seq = "N" * (lo - start) + fa[chrom][lo:hi] + "N" * (end - hi)
    finally:
        fa.close()
    return DNAOneHotEncoder(dtype=np.float32).encode(seq)


def write_json_atomic(payload: dict, out: Path) -> None:
    """Write ``payload`` to ``out`` via a temp file in the same directory and ``os.replace``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> None:
    """CLI: overfit one window and exit non-zero unless the loss falls by >= 90%."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--chrom", default="chr12")
    ap.add_argument("--center", type=int, default=6_536_000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tracks-dir", default="data/p6/tracks")
    ap.add_argument("--checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("--checkpoint is required (or set ALPHAGENOME_WEIGHTS_PATH)")
    if not args.genome:
        ap.error("--genome is required (or set TISIAGO_GENOME)")
    if args.steps <= TIMING_SKIP:
        ap.error(f"--steps must be > {TIMING_SKIP}")
    out = args.out or Path(f"data/p6/overfit/{args.condition}.json")

    import jax
    import jax.numpy as jnp

    from tisiago.seq2func.model import load_model, make_train_step, register_track_head
    from tisiago.seq2func.targets import channel_layout, load_targets

    start, end = args.center - WINDOW // 2, args.center + WINDOW // 2
    onehot = fetch_onehot(args.genome, args.chrom, start, end)
    targets = load_targets(args.tracks_dir, [args.condition], args.chrom, start, end)
    num_tracks = len(channel_layout([args.condition]))
    batch = {
        "sequences": jnp.asarray(onehot[None]),
        "targets": jnp.asarray(targets[None]),
        "mask": jnp.ones((1, *targets.shape), bool),
    }
    print(f"window {args.chrom}:{start}-{end} targets {targets.shape} sums {targets.sum(0)}")

    register_track_head(HEAD, num_tracks)
    t0 = time.time()
    model = load_model(args.checkpoint, HEAD, detach_backbone=True)
    print(f"model load {time.time() - t0:.1f}s", flush=True)
    step, opt_state = make_train_step(
        model, HEAD, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    params, state = model._params, model._state

    losses, times = [], []
    for i in range(args.steps):
        t = time.time()
        params, opt_state, aux = step(params, state, opt_state, batch)
        losses.append(float(aux["loss"]))
        times.append(time.time() - t)
        if i < 3 or i % 25 == 0 or i == args.steps - 1:
            print(
                f"step {i:4d} loss {losses[-1]:.5f} poisson {float(aux['poisson']):.5f} "
                f"multinomial {float(aux['multinomial']):.5f} dt {times[-1]:.3f}s",
                flush=True,
            )

    stats = jax.local_devices()[0].memory_stats() or {}
    initial, final = losses[0], losses[-1]
    reduction = 1.0 - final / initial
    payload = {
        "condition": args.condition,
        "chrom": args.chrom,
        "start": start,
        "end": end,
        "num_tracks": num_tracks,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "losses": losses,
        "compile_seconds": times[:TIMING_SKIP],
        "step_seconds": float(np.mean(times[TIMING_SKIP:])),
        "peak_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
        "initial": initial,
        "final": final,
        "reduction": reduction,
    }
    write_json_atomic(payload, out)
    print(
        f"initial {initial:.4f} final {final:.4f} reduction {reduction:.4f} "
        f"step {payload['step_seconds']:.3f}s peak {payload['peak_gib']:.2f} GiB -> {out}"
    )
    if reduction < 0.9:
        sys.exit(1)


if __name__ == "__main__":
    main()
