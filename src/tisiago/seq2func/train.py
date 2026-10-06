"""Fine-tune AlphaGenome on Ribo-seq tracks with chr7 model selection (P6.2).

Usage::

    python -m tisiago.seq2func.train --arm condition:HeLa --steps 600 --eval-every 200 \
        --eval-windows 20 --out data/p6/runs/hela
    python -m tisiago.seq2func.train --arm shared --heads-only --out data/p6/runs/shared_frozen

Model selection reads only ``val.parquet`` (chr7); the held-out test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from tisiago.seq2func.metrics import profile_kl
from tisiago.seq2func.overfit import finite_or_none, write_json_atomic
from tisiago.seq2func.samples import CONDITIONS

TRAIN_WINDOWS = "train.parquet"
VAL_WINDOWS = "val.parquet"
SELECTION_KEY = "val_profile_kl"


def parse_arm(arm: str) -> tuple[str, tuple[str, ...]]:
    """``"shared"`` or ``"condition:<C>"`` -> ``(head name, conditions)``."""
    if arm == "shared":
        return "p6_shared", tuple(CONDITIONS)
    kind, _, condition = arm.partition(":")
    if kind != "condition" or condition not in CONDITIONS:
        raise ValueError(f"arm must be 'shared' or 'condition:<one of {CONDITIONS}>', not {arm!r}")
    return f"p6_{condition.lower()}", (condition,)


def window_profile_kl(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean over channels of the masked ``profile_kl`` of one window; NaN if no channel counts."""
    kl = profile_kl(pred, target, mask)
    return float(np.nanmean(kl)) if np.isfinite(kl).any() else math.nan


def mean_window_kl(values: Iterable[float]) -> float:
    """NaN-skipping mean over windows; NaN when no window is finite."""
    arr = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(arr).any():
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.where(np.isfinite(arr), arr, np.nan)))


def save_dir_atomic(write: Callable[[Path], None], dest: Path) -> None:
    """Call ``write(tmp)`` on a sibling temp dir, then swap it into ``dest``.

    On failure the previous ``dest`` is left untouched and the temp dir is removed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"))
    old = dest.parent / f".{dest.name}.old"
    try:
        write(tmp)
        if old.exists():
            shutil.rmtree(old)
        if dest.exists():
            os.replace(dest, old)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def train_loop(
    step_fn,
    params,
    state,
    opt_state,
    batches: Iterable,
    *,
    steps: int,
    eval_every: int,
    eval_fn: Callable,
    on_improve: Callable,
    on_checkpoint: Callable,
    log_path: Path,
) -> dict | None:
    """Run ``steps`` updates, evaluating every ``eval_every`` steps and after the last step.

    Each eval appends one flushed JSON line (step, mean train loss since the previous eval,
    ``eval_fn(params)`` metrics, ``improved``, elapsed seconds) to ``log_path``, then calls
    ``on_checkpoint(params, opt_state, step)`` and, when the finite ``val_profile_kl`` beats
    the best so far, ``on_improve(params, step)``. Exceptions propagate after the completed
    lines are flushed.

    Returns:
        The best eval record, or ``None`` if no eval produced a finite ``val_profile_kl``.
    """
    if steps < 1 or eval_every < 1:
        raise ValueError(f"steps and eval_every must be >= 1, got {steps}, {eval_every}")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = None
    best_kl = math.inf
    losses = []
    t0 = time.time()
    iterator = iter(batches)
    with open(log_path, "a") as fh:
        for step in range(1, steps + 1):
            params, opt_state, aux = step_fn(params, state, opt_state, next(iterator))
            losses.append(aux["loss"])
            if step % eval_every and step != steps:
                continue
            metrics = eval_fn(params)
            kl = float(metrics[SELECTION_KEY])
            improved = math.isfinite(kl) and kl < best_kl
            record = {
                "step": step,
                "train_loss": finite_or_none(float(np.mean([float(v) for v in losses]))),
                **{k: finite_or_none(float(v)) for k, v in metrics.items()},
                "improved": improved,
                "elapsed_seconds": time.time() - t0,
            }
            losses = []
            fh.write(json.dumps(record, allow_nan=False) + "\n")
            fh.flush()
            on_checkpoint(params, opt_state, step)
            if improved:
                best_kl, best = kl, record
                on_improve(params, step)
    return best


def git_commit() -> str | None:
    """Current ``HEAD`` of the repository containing this file, or ``None``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    """CLI parser; defaults are the P6.2 primary configuration (backbone fine-tuning)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", required=True, help="'shared' or 'condition:<C>'")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="default data/p6/runs/<arm>")
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/p6/expression/expression.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/p6/tracks"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--head", choices=("mlp", "conv"), default="mlp")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument(
        "--heads-only",
        dest="train_backbone",
        action="store_false",
        help="frozen-backbone baseline (default trains the backbone)",
    )
    return ap


def main() -> None:
    """CLI: train one arm, select on chr7 profile KL, keep ``best/`` and ``last/``."""
    ap = build_parser()
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("--checkpoint is required (or set ALPHAGENOME_WEIGHTS_PATH)")
    if not args.genome:
        ap.error("--genome is required (or set TISIAGO_GENOME)")
    try:
        head, conditions = parse_arm(args.arm)
    except ValueError as exc:
        ap.error(str(exc))
    out = args.out or Path("data/p6/runs") / args.arm.replace(":", "_").lower()
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import pandas as pd
    from alphagenome_research.model import dna_model

    from tisiago.seq2func.data import iterate
    from tisiago.seq2func.model import (
        ORGANISM,
        load_model,
        make_train_step,
        register_track_head,
    )
    from tisiago.seq2func.targets import channel_layout

    train_windows = pd.read_parquet(args.windows_dir / TRAIN_WINDOWS)
    val_windows = pd.read_parquet(args.windows_dir / VAL_WINDOWS).iloc[: args.eval_windows]
    expression = pd.read_parquet(args.expression)
    genes = expression[expression.condition.isin(conditions)].reset_index(drop=True)
    num_tracks = len(channel_layout(conditions))
    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "head_name": head,
        "conditions": list(conditions),
        "num_tracks": num_tracks,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    write_json_atomic(run, out / "run.json")
    print(json.dumps(run, indent=2), flush=True)

    register_track_head(head, num_tracks, kind=args.head)
    t0 = time.time()
    model = load_model(args.checkpoint, head, detach_backbone=not args.train_backbone)
    print(f"model load {time.time() - t0:.1f}s", flush=True)
    step_fn, opt_state = make_train_step(
        model,
        head,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        heads_only=not args.train_backbone,
        backbone_learning_rate=args.backbone_learning_rate if args.train_backbone else None,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        clip_norm=args.clip_norm,
    )
    params, state = model._params, model._state
    strand_reindexing = model._metadata[ORGANISM].strand_reindexing
    org_index = dna_model.convert_to_organism_index(ORGANISM)

    @jax.jit
    def predict(params, state, sequences):
        batch_size = sequences.shape[0]
        return model._predict(
            params,
            state,
            sequences,
            jnp.full((batch_size,), org_index, jnp.int32),
            requested_outputs=(),
            negative_strand_mask=jnp.zeros((batch_size,), bool),
            strand_reindexing=strand_reindexing,
        )[head]

    data_kwargs = dict(
        fasta=args.genome, tracks_dir=args.tracks_dir, conditions=conditions, genes=genes
    )

    def eval_fn(params) -> dict:
        t = time.time()
        kls = []
        for batch in iterate(
            val_windows, seed=args.seed, augment_examples=False, loop=False, **data_kwargs
        ):
            pred = np.asarray(predict(params, state, jnp.asarray(batch["sequences"])))
            kls.append(window_profile_kl(pred[0], batch["targets"][0], batch["mask"][0]))
        return {
            SELECTION_KEY: mean_window_kl(kls),
            "val_finite_windows": int(np.isfinite(kls).sum()),
            "eval_seconds": time.time() - t,
        }

    def save(dest: Path, params) -> None:
        model._params = params
        save_dir_atomic(lambda tmp: model.save_checkpoint(tmp, save_full_model=True), dest)

    def on_checkpoint(params, opt_state, step) -> None:
        save(out / "last", params)

    def on_improve(params, step) -> None:
        save(out / "best", params)
        print(f"step {step}: new best -> {out / 'best'}", flush=True)

    log_path = out / "log.jsonl"
    log_path.unlink(missing_ok=True)
    batches = iterate(
        train_windows,
        seed=args.seed,
        augment_examples=True,
        loop=True,
        prefetch=args.prefetch,
        workers=args.workers,
        **data_kwargs,
    )
    try:
        best = train_loop(
            step_fn,
            params,
            state,
            opt_state,
            batches,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_fn=eval_fn,
            on_improve=on_improve,
            on_checkpoint=on_checkpoint,
            log_path=log_path,
        )
    finally:
        batches.close()
    run["best"] = best
    stats = jax.local_devices()[0].memory_stats() or {}
    run["peak_gib"] = stats.get("peak_bytes_in_use", 0) / 2**30
    write_json_atomic(run, out / "run.json")
    print(f"best {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
