"""One-window overfit of the track head, optionally with the backbone (P6.1 training-core check).

Usage::

    python -m tisiago.seq2func.overfit --condition HeLa --chrom chr12 --center 6536000 \
        --steps 300 --out data/p6/overfit/HeLa.json
    python -m tisiago.seq2func.overfit --condition HeLa --head conv --train-backbone \
        --backbone-learning-rate 1e-5 --steps 3000 --out data/p6/overfit/HeLa_conv_ft.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HEAD = "p6_overfit"
WINDOW = 131_072
TIMING_SKIP = 2
PROFILE_KL_MAX = 1.0
HEAD_CHOICES = ("mlp", "conv", "seqconv", "seqconv_tx", "resid_tx", "resid_chx")
TRANSCRIPT_KINDS = frozenset({"seqconv_tx", "resid_tx", "resid_chx"})
MOD_KINDS = frozenset({"resid_tx", "resid_chx"})
CHX_KINDS = frozenset({"resid_chx"})


def fetch_onehot(genome: str, chrom: str, start: int, end: int) -> np.ndarray:
    """One-hot ``(end - start, 4)`` for ``[start, end)``, N-padded past the chromosome ends."""
    from alphagenome_research.model.one_hot_encoder import DNAOneHotEncoder
    from pyfaidx import Fasta

    fa = Fasta(genome, as_raw=True, sequence_always_upper=True)
    try:
        size = len(fa[chrom])
        lo = min(max(start, 0), end)
        hi = max(min(end, size), lo)
        body = fa[chrom][lo:hi] if hi > lo else ""
        seq = "N" * (lo - start) + body + "N" * (end - hi)
    finally:
        fa.close()
    return DNAOneHotEncoder(dtype=np.float32).encode(seq)


def passes_gate(initial: float, final: float, threshold: float = 0.9) -> bool:
    """True only for finite losses, ``initial > 0`` and ``1 - final / initial >= threshold``."""
    if not (math.isfinite(initial) and math.isfinite(final) and initial > 0):
        return False
    return 1.0 - final / initial >= threshold


def passes_profile(final_multinomial: float, threshold: float = PROFILE_KL_MAX) -> bool:
    """True only for a finite final multinomial (profile KL) ``<= threshold``."""
    return math.isfinite(final_multinomial) and final_multinomial <= threshold


def finite_or_none(value: float) -> float | None:
    """``value`` if finite, else ``None`` (serialised as JSON ``null``)."""
    return value if math.isfinite(value) else None


def write_json_atomic(payload: dict, out: Path) -> None:
    """Write ``payload`` to ``out`` via a temp file in the same directory and ``os.replace``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> None:
    """CLI: overfit one window; exit non-zero unless loss falls >= 90% and profile KL <= 1."""
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
    ap.add_argument("--head", choices=HEAD_CHOICES, default="mlp")
    ap.add_argument("--train-backbone", action="store_true")
    ap.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="backbone AdamW lr with --train-backbone (default: --learning-rate / 100)",
    )
    ap.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--clip-norm", type=float, default=None)
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("--checkpoint is required (or set ALPHAGENOME_WEIGHTS_PATH)")
    if not args.genome:
        ap.error("--genome is required (or set TISIAGO_GENOME)")
    if args.head in TRANSCRIPT_KINDS:
        ap.error(f"--head {args.head} needs transcript features; use train.py")
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

    register_track_head(HEAD, num_tracks, kind=args.head)
    t0 = time.time()
    model = load_model(args.checkpoint, HEAD, detach_backbone=not args.train_backbone)
    print(f"model load {time.time() - t0:.1f}s", flush=True)
    backbone_lr = None
    if args.train_backbone:
        backbone_lr = args.backbone_learning_rate
        if backbone_lr is None:
            backbone_lr = args.learning_rate / 100
    step, opt_state = make_train_step(
        model,
        HEAD,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        heads_only=not args.train_backbone,
        backbone_learning_rate=backbone_lr,
        total_steps=args.steps if args.schedule == "cosine" else None,
        warmup_steps=args.warmup_steps,
        clip_norm=args.clip_norm,
    )
    params, state = model._params, model._state

    losses, multinomials, times = [], [], []
    for i in range(args.steps):
        t = time.time()
        params, opt_state, aux = step(params, state, opt_state, batch)
        losses.append(float(aux["loss"]))
        multinomials.append(float(aux["multinomial"]))
        times.append(time.time() - t)
        if i < 3 or i % 25 == 0 or i == args.steps - 1:
            print(
                f"step {i:4d} loss {losses[-1]:.5f} poisson {float(aux['poisson']):.5f} "
                f"multinomial {multinomials[-1]:.5f} dt {times[-1]:.3f}s",
                flush=True,
            )

    stats = jax.local_devices()[0].memory_stats() or {}
    initial, final = losses[0], losses[-1]
    passed = passes_gate(initial, final)
    final_multinomial = multinomials[-1]
    finite_multinomials = [v for v in multinomials if math.isfinite(v)]
    best_multinomial = min(finite_multinomials) if finite_multinomials else math.nan
    passed_profile = passes_profile(final_multinomial)
    reduction = 1.0 - final / initial if initial != 0 else math.nan
    payload = {
        "condition": args.condition,
        "chrom": args.chrom,
        "start": start,
        "end": end,
        "num_tracks": num_tracks,
        "head": args.head,
        "train_backbone": args.train_backbone,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "backbone_learning_rate": backbone_lr,
        "losses": [finite_or_none(v) for v in losses],
        "non_finite_steps": [i for i, v in enumerate(losses) if not math.isfinite(v)],
        "compile_seconds": times[:TIMING_SKIP],
        "step_seconds": float(np.mean(times[TIMING_SKIP:])),
        "peak_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
        "initial": finite_or_none(initial),
        "final": finite_or_none(final),
        "reduction": finite_or_none(reduction),
        "passed": passed,
        "schedule": args.schedule,
        "warmup_steps": args.warmup_steps,
        "clip_norm": args.clip_norm,
        "final_multinomial": finite_or_none(final_multinomial),
        "best_multinomial": finite_or_none(best_multinomial),
        "passed_profile": passed_profile,
    }
    write_json_atomic(payload, out)
    print(
        f"initial {initial:.4f} final {final:.4f} reduction {reduction:.4f} "
        f"profile KL {final_multinomial:.4f} (best {best_multinomial:.4f}) "
        f"step {payload['step_seconds']:.3f}s peak {payload['peak_gib']:.2f} GiB -> {out}"
    )
    if not (passed and passed_profile):
        sys.exit(1)


if __name__ == "__main__":
    main()
