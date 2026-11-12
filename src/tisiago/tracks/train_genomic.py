"""Stage 1 (``genomic``): fine-tune AlphaGenome on Ribo-seq tracks, chr7 model selection.

Usage (reference recipe, one fold; arguments in ``configs/tracks/genomic_fc2d.args``)::

    python -m tisiago.tracks.train_genomic --arm rnacond:HeLa \
        $(cat configs/tracks/genomic_fc2d.args) \
        --out data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0

``--arm rnacond:<group>`` trains one model on every condition outside the held-out cell-line
group (``samples.CELL_LINE_GROUPS``): examples are (window, condition) pairs, the condition's
pooled RNA (scaled to a 5e7-read library by its genome-wide pooled RNA reads) is an input and its
TIS/CHX tracks the 4 targets.

Model selection reads only ``val.parquet`` (chr7): ``--eval-windows`` (default 20) evenly spaced
val windows of each training condition; the held-out test split is never loaded. ``best/`` is
selected on ``val_profile_kl``; ``val_frame`` (the codon-phase block KL on CHX channels) and the
per-component train losses are logged, not selected on. ``--init-from`` starts from a finished
run's parameters (``init_from`` in ``run.json``).
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

from tisiago.tracks.checkpoint import finite_or_none, write_json_atomic
from tisiago.tracks.head_kinds import DEFAULT_HEAD, HEAD_CHOICES
from tisiago.tracks.metrics import profile_kl
from tisiago.tracks.samples import CELL_LINE_GROUPS, CONDITIONS

TRAIN_WINDOWS = "train.parquet"
VAL_WINDOWS = "val.parquet"
SELECTION_KEY = "val_profile_kl"
LOSS_COMPONENTS = ("poisson", "multinomial", "frame", "local", "local2")
RUN_RECORDS = ("log.jsonl", "run.json", "best", "last")


RNACOND = "rnacond"
EVAL_WINDOWS = 20


def parse_rnacond_arm(arm: str) -> tuple[str, str, tuple[str, ...]]:
    """``"rnacond:<G>"`` -> ``(head name, held-out group, training conditions)``."""
    kind, _, group = arm.partition(":")
    if kind != RNACOND or group not in CELL_LINE_GROUPS:
        raise ValueError(f"arm must be 'rnacond:<one of {tuple(CELL_LINE_GROUPS)}>', not {arm!r}")
    held = CELL_LINE_GROUPS[group]
    return f"p6_rnacond_{group.lower()}", group, tuple(c for c in CONDITIONS if c not in held)


def local_channels_of(arm: str, libraries=("tis",)) -> tuple[int, ...]:
    """Channel indices of ``libraries`` in ``arm``'s output layout (local-multinomial channels)."""
    from tisiago.tracks.targets import modulated_channels, rnacond_layout

    return modulated_channels(rnacond_layout(parse_rnacond_arm(arm)[2][0]), libraries)


def arm_fields(arm: str) -> dict:
    """run.json fields of ``arm``.

    Head name, conditions, ``num_tracks``, CHX ``frame_channels``, the held-out group and
    conditions, the training conditions, the output ``channel_layout`` (library, strand) and the
    ``rna_input`` description.
    """
    from tisiago.tracks.rna_features import RNA_FEATURE_NAMES, RNA_WIDTHS
    from tisiago.tracks.targets import library_channels, rnacond_layout

    head, group, conditions = parse_rnacond_arm(arm)
    layout = rnacond_layout(conditions[0])
    return {
        "head_name": head,
        "conditions": list(conditions),
        "num_tracks": len(layout),
        "frame_channels": list(library_channels(layout, "chx")),
        "heldout_group": group,
        "heldout_conditions": list(CELL_LINE_GROUPS[group]),
        "train_conditions": list(conditions),
        "channel_layout": [[library, strand] for _, library, strand in layout],
        "rna_input": {
            "library": "rna",
            "rep": "pooled",
            "strands": ["+", "-"],
            "features": list(RNA_FEATURE_NAMES),
            "box_widths": list(RNA_WIDTHS),
        },
    }


def selection_window_indices(n: int, k: int) -> list[int]:
    """``k`` evenly spaced row positions of ``n`` val windows (all of them if ``k >= n``)."""
    if k >= n:
        return list(range(n))
    return sorted({int(i) for i in np.linspace(0, n - 1, k).round()})


def rnacond_data_fields(arm: str, tracks_dir: str | Path) -> dict:
    """run.json RNA normalisation of an rnacond arm: genome-wide totals and depth scales.

    Recorded for the training and the held-out conditions (in :data:`CONDITIONS` order).
    """
    from tisiago.tracks.data import RNA_REFERENCE_DEPTH, rna_total

    _, group, conditions = parse_rnacond_arm(arm)
    every = [c for c in CONDITIONS if c in conditions or c in CELL_LINE_GROUPS[group]]
    totals = {c: rna_total(tracks_dir, c) for c in every}
    return {
        "rna_normalisation": "depth: pooled RNA x rna_reference_depth / genome-wide pooled RNA "
        "reads (pooled rna bigWig header sumData, both strands)",
        "rna_reference_depth": RNA_REFERENCE_DEPTH,
        "rna_totals": totals,
        "rna_depth_scales": {c: RNA_REFERENCE_DEPTH / t for c, t in totals.items()},
    }


def check_rnacond_run(run: dict, arm: str) -> None:
    """Raise ``ValueError`` if a run.json's recorded rnacond fields disagree with ``arm``."""
    expected = arm_fields(arm)
    for key in ("heldout_group", "train_conditions", "channel_layout", "head_name"):
        if key in run and run[key] != expected[key]:
            raise ValueError(f"run.json records {key}={run[key]}, but {arm} gives {expected[key]}")


def window_profile_kl(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean over channels of the masked ``profile_kl`` of one window; NaN if no channel counts."""
    kl = profile_kl(pred, target, mask)
    return float(np.nanmean(kl)) if np.isfinite(kl).any() else math.nan


def window_frame_kl(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, channels, block: int
) -> tuple[float, float]:
    """``(sum n * KL, sum n)`` of one ``(S, C)`` window's block phase composition (as the loss).

    Same definition as ``loss.codon_phase_multinomial``: blocks of ``block`` codons on the
    window's grid, skipped when any position is masked or nothing is observed.
    """
    channels = list(channels)
    size = 3 * block
    n = (len(target) // size) * size
    if not channels or n == 0:
        return 0.0, 0.0
    shape = (n // size, block, 3, len(channels))
    p = np.asarray(pred, np.float64)[:n][:, channels].reshape(shape).sum(axis=1)
    t = np.maximum(np.asarray(target, np.float64)[:n][:, channels], 0).reshape(shape).sum(axis=1)
    keep = np.asarray(mask)[:n][:, channels].reshape(shape).all(axis=(1, 2))
    total = t.sum(axis=1)
    keep &= total > 0
    if not keep.any():
        return 0.0, 0.0
    qt = t / np.maximum(total[:, None, :], 1e-7)
    qp = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-7)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(qt > 0, qt * (np.log(qt) - np.log(np.maximum(qp, 1e-7))), 0.0)
    kl = terms.sum(axis=1)
    return float((total * kl)[keep].sum()), float(total[keep].sum())


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

    On failure the previous ``dest`` is restored and the temp dir is removed.
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
        if old.exists() and not dest.exists():
            os.replace(old, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def default_out(arm: str, train_backbone: bool) -> Path:
    """``data/tracks/runs/<arm>``, with a ``_frozen`` suffix for the heads-only baseline."""
    name = arm.replace(":", "_").lower()
    return Path("data/tracks/runs") / (name if train_backbone else f"{name}_frozen")


def existing_run_records(out: Path) -> list[str]:
    """Names in ``RUN_RECORDS`` already present in ``out``."""
    return [name for name in RUN_RECORDS if (Path(out) / name).exists()]


def check_out_dir(out: Path, overwrite: bool) -> None:
    """Raise ``ValueError`` if ``out`` holds an earlier run's records and not ``overwrite``."""
    found = existing_run_records(out)
    if found and not overwrite:
        raise ValueError(
            f"{out} already holds run records {found}; choose another --out or pass --overwrite"
        )


def init_source(init_from: Path, head: str, kind: str) -> dict:
    """Describe a two-stage init from a finished run's ``best/`` (or ``last/``) directory.

    The source must be a finished run: its ``run.json`` exists and records a non-null ``best``
    (``best/`` is rewritten at every improving eval while a run is still training). The source
    head is reused (``reuse_head`` ``"all"``) only when that ``run.json`` records the same head
    name (arm) and head kind; otherwise (``"none"``) the new head starts fresh and only the
    backbone is copied.
    """
    init_from = Path(init_from)
    if not (init_from / "checkpoint").is_dir():
        raise ValueError(f"{init_from} has no checkpoint/; pass a run's best/ or last/ directory")
    run_json = init_from.parent / "run.json"
    if not run_json.exists():
        raise ValueError(f"{init_from}: no run.json in {init_from.parent}; cannot verify the run")
    source = json.loads(run_json.read_text())
    if not source.get("best"):
        raise ValueError(
            f"{init_from.parent} is not a finished run (run.json has no best); its best/ is an "
            "intermediate checkpoint — wait for the run to finish"
        )
    source_kind = source.get("args", {}).get("head")
    source_head = source.get("head_name")
    reuse_head = "all" if source_head == head and source_kind == kind else "none"
    return {
        "path": str(init_from),
        "source_head": source_head,
        "source_kind": source_kind,
        "reuse_head": reuse_head,
    }


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

    Each eval appends one flushed JSON line (step, mean train loss and ``train_<component>``
    for each :data:`LOSS_COMPONENTS` entry in the step's aux since the previous eval,
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
    components: dict[str, list] = {}
    t0 = time.time()
    iterator = iter(batches)
    with open(log_path, "a") as fh:
        for step in range(1, steps + 1):
            params, opt_state, aux = step_fn(params, state, opt_state, next(iterator))
            losses.append(aux["loss"])
            for key in LOSS_COMPONENTS:
                if key in aux:
                    components.setdefault(key, []).append(aux[key])
            if step % eval_every and step != steps:
                continue
            metrics = eval_fn(params)
            kl = float(metrics[SELECTION_KEY])
            improved = math.isfinite(kl) and kl < best_kl
            record = {
                "step": step,
                "train_loss": finite_or_none(float(np.mean([float(v) for v in losses]))),
                **{
                    f"train_{k}": finite_or_none(float(np.mean([float(v) for v in vals])))
                    for k, vals in components.items()
                },
                **{k: finite_or_none(float(v)) for k, v in metrics.items()},
                "improved": improved,
                "elapsed_seconds": time.time() - t0,
            }
            losses, components = [], {}
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
    """CLI parser; defaults are the original genomic-stage configuration (backbone fine-tuning)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", required=True, help="'rnacond:<held-out cell-line group>'")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument(
        "--eval-windows",
        type=int,
        default=EVAL_WINDOWS,
        help="evenly spaced val windows per training condition for selection",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default data/tracks/runs/<arm>[_frozen]"
    )
    ap.add_argument("--windows-dir", type=Path, default=Path("data/tracks/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/tracks/expression/expression.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/tracks/tracks"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--head", choices=HEAD_CHOICES, default=DEFAULT_HEAD)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument(
        "--frame-weight",
        type=float,
        default=0.0,
        help="weight of the annotation-free codon-phase loss on CHX channels (0 = off)",
    )
    ap.add_argument(
        "--frame-block", type=int, default=10, help="codons per phase-composition block"
    )
    ap.add_argument(
        "--local-weight",
        type=float,
        default=0.0,
        help="weight of the per-block multinomial loss on TIS channels (0 = off)",
    )
    ap.add_argument(
        "--local-block", type=int, default=1024, help="positions per local-multinomial block"
    )
    ap.add_argument(
        "--local-libraries",
        default="tis",
        help="comma list of libraries the local multinomial applies to (default: tis)",
    )
    ap.add_argument(
        "--local2-weight",
        type=float,
        default=0.0,
        help="weight of a second per-block multinomial loss (0 = off)",
    )
    ap.add_argument("--local2-block", type=int, default=32, help="positions per second-local block")
    ap.add_argument(
        "--local2-libraries",
        default="tis",
        help="comma list of libraries the second local multinomial applies to (default: tis)",
    )
    ap.add_argument(
        "--heads-only",
        dest="train_backbone",
        action="store_false",
        help="frozen-backbone baseline (default trains the backbone)",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="two-stage init: a finished run's best/ (backbone; head too if arm and kind match)",
    )
    ap.add_argument(
        "--freeze-backbone-steps",
        type=int,
        default=0,
        help="backbone learning rate 0 for the first N steps (head warm-up)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing log.jsonl/run.json/best/last of an earlier run in --out",
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
        head, _, conditions = parse_rnacond_arm(args.arm)
    except ValueError as exc:
        ap.error(str(exc))
    if args.frame_weight < 0 or args.frame_block < 1:
        ap.error("--frame-weight must be >= 0 and --frame-block >= 1")
    if args.local_weight < 0 or args.local_block < 1:
        ap.error("--local-weight must be >= 0 and --local-block >= 1")
    if args.local2_weight < 0 or args.local2_block < 1:
        ap.error("--local2-weight must be >= 0 and --local2-block >= 1")
    try:
        local_libraries = tuple(name.strip() for name in args.local_libraries.split(","))
        local_channels_of(args.arm, local_libraries)
    except ValueError as exc:
        ap.error(f"--local-libraries: {exc}")
    try:
        local2_libraries = tuple(name.strip() for name in args.local2_libraries.split(","))
        local_channels_of(args.arm, local2_libraries)
    except ValueError as exc:
        ap.error(f"--local2-libraries: {exc}")
    if not 0 <= args.freeze_backbone_steps < args.steps:
        ap.error("--freeze-backbone-steps must be in [0, --steps)")
    if args.freeze_backbone_steps and not args.train_backbone:
        ap.error("--freeze-backbone-steps needs backbone training (drop --heads-only)")
    init = None
    if args.init_from is not None:
        try:
            init = init_source(args.init_from, head, args.head)
        except ValueError as exc:
            ap.error(str(exc))
    out = args.out or default_out(args.arm, args.train_backbone)
    try:
        check_out_dir(out, args.overwrite)
    except ValueError as exc:
        ap.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import pandas as pd

    from tisiago.tracks.data import iterate
    from tisiago.tracks.model import (
        load_model,
        make_predict,
        make_train_step,
        register_track_head,
    )

    train_windows = pd.read_parquet(args.windows_dir / TRAIN_WINDOWS)
    val_all = pd.read_parquet(args.windows_dir / VAL_WINDOWS)
    val_index = selection_window_indices(len(val_all), args.eval_windows)
    val_windows = val_all.iloc[val_index].reset_index(drop=True)
    expression = pd.read_parquet(args.expression)
    genes = expression[expression.condition.isin(conditions)].reset_index(drop=True)
    fields = arm_fields(args.arm)
    num_tracks = fields["num_tracks"]
    frame_channels = tuple(fields["frame_channels"])
    local_channels = local_channels_of(args.arm, local_libraries) if args.local_weight else ()
    local2_channels = local_channels_of(args.arm, local2_libraries) if args.local2_weight else ()
    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit": git_commit(),
        **fields,
        **rnacond_data_fields(args.arm, args.tracks_dir),
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "val_examples": len(val_windows) * len(conditions),
        "val_window_indices": val_index,
        **({"local_channels": list(local_channels)} if args.local_weight else {}),
        **({"local2_channels": list(local2_channels)} if args.local2_weight else {}),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    if init is not None:
        run["init_from"] = init
    write_json_atomic(run, out / "run.json")
    print(json.dumps(run, indent=2), flush=True)

    register_track_head(head, num_tracks, kind=args.head)
    t0 = time.time()
    model = load_model(args.checkpoint, head, detach_backbone=not args.train_backbone)
    print(f"model load {time.time() - t0:.1f}s", flush=True)
    if init is not None:
        from tisiago.tracks.model import restore_init

        init["leaves"] = restore_init(model, init["path"], head, reuse_head=init["reuse_head"])
        write_json_atomic(run, out / "run.json")
        print(f"init from {init}", flush=True)
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
        frame_weight=args.frame_weight,
        frame_channels=frame_channels,
        frame_block=args.frame_block,
        freeze_backbone_steps=args.freeze_backbone_steps,
        local_weight=args.local_weight,
        local_channels=local_channels,
        local_block=args.local_block,
        local2_weight=args.local2_weight,
        local2_channels=local2_channels,
        local2_block=args.local2_block,
    )
    params, state = model._params, model._state
    predict = make_predict(model, head)

    data_kwargs = dict(
        fasta=args.genome,
        tracks_dir=args.tracks_dir,
        conditions=conditions,
        genes=genes,
    )

    def eval_fn(params) -> dict:
        t = time.time()
        kls = []
        frame_num = frame_den = 0.0
        for batch in iterate(
            val_windows, seed=args.seed, augment_examples=False, loop=False, **data_kwargs
        ):
            pred = np.asarray(predict(params, state, jnp.asarray(batch["sequences"])))
            target, mask = batch["targets"][0], batch["mask"][0]
            kls.append(window_profile_kl(pred[0], target, mask))
            num, den = window_frame_kl(pred[0], target, mask, frame_channels, args.frame_block)
            frame_num, frame_den = frame_num + num, frame_den + den
        return {
            SELECTION_KEY: mean_window_kl(kls),
            "val_frame": frame_num / frame_den if frame_den else math.nan,
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
    if args.overwrite:
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
