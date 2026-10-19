"""Fine-tune AlphaGenome on Ribo-seq tracks with chr7 model selection (P6.2).

Usage::

    python -m tisiago.seq2func.train --arm condition:HeLa --steps 600 --eval-every 200 \
        --eval-windows 20 --out data/p6/runs/hela
    python -m tisiago.seq2func.train --arm shared --heads-only --out data/p6/runs/shared_frozen
    python -m tisiago.seq2func.train --arm condition:K562 --init-from data/p6/runs/k562_base/best \
        --head seqconv --freeze-backbone-steps 1000 --out data/p6/runs/k562_seqconv_2stage
    python -m tisiago.seq2func.train --arm rnacond:HeLa --head rnacond --steps 30000 \
        --out data/p6/runs/rnacond_hela

Model selection reads only ``val.parquet`` (chr7); the held-out test split is never loaded.
``--head seqconv_tx`` adds spliced MANE transcript features (``--gtf``) to the input.
``best/`` is selected on ``val_profile_kl``; ``val_frame`` (the codon-phase block KL on CHX
channels) and the per-component train losses are logged, not selected on. ``--init-from`` starts
from a finished run's parameters (``init_from`` in ``run.json``).

``--arm rnacond:<group>`` (with ``--head rnacond``) trains one model on every condition outside
the held-out cell-line group (``samples.CELL_LINE_GROUPS``): examples are (window, condition)
pairs, the condition's pooled RNA (in counts per million of its genome-wide pooled RNA reads) is
an input and its TIS/CHX tracks the 4 targets. Model selection uses the first ``--eval-windows``
(default 20) val windows of each training condition only.
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
from tisiago.seq2func.overfit import (
    CHX_KINDS,
    HEAD_CHOICES,
    MOD_KINDS,
    RNA_KINDS,
    TRANSCRIPT_KINDS,
    finite_or_none,
    write_json_atomic,
)
from tisiago.seq2func.samples import CELL_LINE_GROUPS, CONDITIONS, LIBRARIES

TRAIN_WINDOWS = "train.parquet"
VAL_WINDOWS = "val.parquet"
SELECTION_KEY = "val_profile_kl"
LOSS_COMPONENTS = ("poisson", "multinomial", "frame")
RUN_RECORDS = ("log.jsonl", "run.json", "best", "last")


RNACOND = "rnacond"
EVAL_WINDOWS = 100
RNACOND_EVAL_WINDOWS = 20


def is_rnacond_arm(arm: str) -> bool:
    """True for ``"rnacond:<group>"`` arms (valid or not)."""
    return arm.partition(":")[0] == RNACOND


def parse_rnacond_arm(arm: str) -> tuple[str, str, tuple[str, ...]]:
    """``"rnacond:<G>"`` -> ``(head name, held-out group, training conditions)``."""
    kind, _, group = arm.partition(":")
    if kind != RNACOND or group not in CELL_LINE_GROUPS:
        raise ValueError(f"arm must be 'rnacond:<one of {tuple(CELL_LINE_GROUPS)}>', not {arm!r}")
    held = CELL_LINE_GROUPS[group]
    return f"p6_rnacond_{group.lower()}", group, tuple(c for c in CONDITIONS if c not in held)


def parse_arm(arm: str) -> tuple[str, tuple[str, ...]]:
    """``"shared"`` or ``"condition:<C>"`` -> ``(head name, conditions)``.

    ``rnacond`` arms are refused: they have no per-condition channel layout (see
    :func:`parse_rnacond_arm`).
    """
    if arm == "shared":
        return "p6_shared", tuple(CONDITIONS)
    if is_rnacond_arm(arm):
        raise ValueError(f"{arm!r} is an rnacond arm; this command does not support it")
    kind, _, condition = arm.partition(":")
    if kind != "condition" or condition not in CONDITIONS:
        raise ValueError(f"arm must be 'shared' or 'condition:<one of {CONDITIONS}>', not {arm!r}")
    return f"p6_{condition.lower()}", (condition,)


def mod_libraries_of(value: str | None) -> tuple[str, ...]:
    """``--mod-libraries`` comma list -> library names (``None``: ``("chx",)``)."""
    if value is None:
        return ("chx",)
    libraries = tuple(name.strip() for name in value.split(","))
    if not all(libraries) or len(set(libraries)) != len(libraries):
        raise ValueError(f"--mod-libraries must be a comma list of unique names, not {value!r}")
    if not set(libraries) <= set(LIBRARIES):
        raise ValueError(f"--mod-libraries must name libraries in {LIBRARIES}, not {value!r}")
    return libraries


def channel_config(head_kind: str, layout, mod_libraries, mod_scale_tis: float) -> dict:
    """run.json channel fields: CHX ``frame_channels``; for ``resid_chx`` the modulation too."""
    from tisiago.seq2func.targets import library_channels

    out = {"frame_channels": list(library_channels(layout, "chx"))}
    if head_kind in CHX_KINDS:
        from tisiago.seq2func.model import resid_chx_spec

        channels, scales = resid_chx_spec(layout, mod_libraries, mod_scale_tis)
        out |= {
            "mod_libraries": list(mod_libraries),
            "mod_scale_tis": float(mod_scale_tis),
            "mod_channels": list(channels),
            "mod_scales": list(scales),
        }
    return out


def arm_fields(
    arm: str, head_kind: str, mod_libraries=("chx",), mod_scale_tis: float | None = None
) -> dict:
    """run.json fields of ``arm``: head name, conditions, ``num_tracks``, channel config.

    ``mod_scale_tis`` (``resid_chx``) defaults to ``MOD_SCALE``. ``rnacond`` arms add the
    held-out group and conditions, the training conditions, the output ``channel_layout``
    (library, strand) and the ``rna_input`` description.
    """
    from tisiago.seq2func.targets import channel_layout, rnacond_layout

    if mod_scale_tis is None and head_kind in CHX_KINDS:
        from tisiago.seq2func.model import MOD_SCALE

        mod_scale_tis = MOD_SCALE

    if not is_rnacond_arm(arm):
        head, conditions = parse_arm(arm)
        layout = channel_layout(conditions)
        return {
            "head_name": head,
            "conditions": list(conditions),
            "num_tracks": len(layout),
            **channel_config(head_kind, layout, mod_libraries, mod_scale_tis),
        }
    from tisiago.seq2func.rna_features import RNA_FEATURE_NAMES, RNA_WIDTHS

    head, group, conditions = parse_rnacond_arm(arm)
    layout = rnacond_layout(conditions[0])
    return {
        "head_name": head,
        "conditions": list(conditions),
        "num_tracks": len(layout),
        **channel_config(head_kind, layout, mod_libraries, mod_scale_tis),
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


def eval_windows_of(value: int | None, rnacond: bool) -> int:
    """``--eval-windows``: per condition for rnacond arms (default 20), else 100 windows."""
    if value is not None:
        return value
    return RNACOND_EVAL_WINDOWS if rnacond else EVAL_WINDOWS


def rnacond_data_fields(arm: str, tracks_dir: str | Path) -> dict:
    """run.json RNA normalisation of an rnacond arm: genome-wide totals and CPM scales.

    Recorded for the training and the held-out conditions (in :data:`CONDITIONS` order).
    """
    from tisiago.seq2func.data import rna_total

    _, group, conditions = parse_rnacond_arm(arm)
    every = [c for c in CONDITIONS if c in conditions or c in CELL_LINE_GROUPS[group]]
    totals = {c: rna_total(tracks_dir, c) for c in every}
    return {
        "rna_normalisation": "cpm: pooled RNA x 1e6 / genome-wide pooled RNA reads "
        "(pooled rna bigWig header sumData, both strands)",
        "rna_totals": totals,
        "rna_cpm_scales": {c: 1e6 / t for c, t in totals.items()},
    }


def check_rnacond_run(run: dict, arm: str) -> None:
    """Raise ``ValueError`` if a run.json's recorded rnacond fields disagree with ``arm``."""
    expected = arm_fields(arm, RNACOND)
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
    """``data/p6/runs/<arm>``, with a ``_frozen`` suffix for the heads-only baseline."""
    name = arm.replace(":", "_").lower()
    return Path("data/p6/runs") / (name if train_backbone else f"{name}_frozen")


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
    name (arm) and head kind. An ``mlp`` source into ``resid_tx`` / ``resid_chx`` of the same
    arm reuses ``"matching"`` leaves (the shared base branch; ``mod_*`` stay fresh). Otherwise
    (``"none"``) the new head starts fresh and only the backbone is copied.
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
    source_kind = source.get("args", {}).get("head") or "mlp"
    source_head = source.get("head_name")
    reuse_head = "none"
    if source_head == head:
        if source_kind == kind:
            reuse_head = "all"
        elif source_kind == "mlp" and kind in MOD_KINDS:
            reuse_head = "matching"
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
    """CLI parser; defaults are the P6.2 primary configuration (backbone fine-tuning)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--arm", required=True, help="'shared', 'condition:<C>' or 'rnacond:<held-out group>'"
    )
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument(
        "--eval-windows",
        type=int,
        default=None,
        help=f"val windows for selection (default {EVAL_WINDOWS}; rnacond arms: per training "
        f"condition, default {RNACOND_EVAL_WINDOWS})",
    )
    ap.add_argument("--out", type=Path, default=None, help="default data/p6/runs/<arm>[_frozen]")
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
    ap.add_argument(
        "--gtf",
        default=os.environ.get("TISIAGO_GTF"),
        help="MANE_Select GTF (transcript features of --head seqconv_tx)",
    )
    ap.add_argument("--head", choices=HEAD_CHOICES, default="mlp")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    ap.add_argument(
        "--mod-learning-rate",
        type=float,
        default=None,
        help="learning rate of the resid_tx / resid_chx mod_* branch (default: --learning-rate)",
    )
    ap.add_argument(
        "--mod-libraries",
        default=None,
        help="resid_chx: comma list of libraries the modulation applies to (default: chx)",
    )
    ap.add_argument(
        "--mod-scale-tis",
        type=float,
        default=None,
        help="resid_chx: modulation bound of the TIS channels (needs tis in --mod-libraries; "
        "default 1.5 = MOD_SCALE)",
    )
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
    rnacond = is_rnacond_arm(args.arm)
    try:
        if rnacond:
            head, _, conditions = parse_rnacond_arm(args.arm)
        else:
            head, conditions = parse_arm(args.arm)
    except ValueError as exc:
        ap.error(str(exc))
    if rnacond != (args.head in RNA_KINDS):
        ap.error("--arm rnacond:<group> and --head rnacond go together")
    args.eval_windows = eval_windows_of(args.eval_windows, rnacond)
    if args.head in TRANSCRIPT_KINDS and not args.gtf:
        ap.error(f"--head {args.head} needs --gtf (or set TISIAGO_GTF)")
    if args.mod_learning_rate is not None and args.head not in MOD_KINDS:
        ap.error(f"--mod-learning-rate needs --head in {sorted(MOD_KINDS)}")
    if args.mod_learning_rate is not None and args.mod_learning_rate < 0:
        ap.error("--mod-learning-rate must be >= 0")
    explicit_mod = args.mod_libraries is not None or args.mod_scale_tis is not None
    if explicit_mod and args.head not in CHX_KINDS:
        ap.error(f"--mod-libraries / --mod-scale-tis need --head in {sorted(CHX_KINDS)}")
    try:
        mod_libraries = mod_libraries_of(args.mod_libraries)
    except ValueError as exc:
        ap.error(str(exc))
    if args.mod_scale_tis is not None and "tis" not in mod_libraries:
        ap.error("--mod-scale-tis needs tis in --mod-libraries")
    if args.mod_scale_tis is not None and not 0 < args.mod_scale_tis < math.inf:
        ap.error("--mod-scale-tis must be positive and finite")
    if args.frame_weight < 0 or args.frame_block < 1:
        ap.error("--frame-weight must be >= 0 and --frame-block >= 1")
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

    from tisiago.seq2func.data import iterate
    from tisiago.seq2func.model import (
        MOD_SCALE,
        load_model,
        make_predict,
        make_train_step,
        register_track_head,
    )

    train_windows = pd.read_parquet(args.windows_dir / TRAIN_WINDOWS)
    val_windows = pd.read_parquet(args.windows_dir / VAL_WINDOWS).iloc[: args.eval_windows]
    expression = pd.read_parquet(args.expression)
    genes = expression[expression.condition.isin(conditions)].reset_index(drop=True)
    mod_scale_tis = MOD_SCALE if args.mod_scale_tis is None else args.mod_scale_tis
    fields = arm_fields(args.arm, args.head, mod_libraries, mod_scale_tis)
    num_tracks = fields["num_tracks"]
    channels = {k: fields[k] for k in ("mod_channels", "mod_scales") if k in fields}
    frame_channels = tuple(fields["frame_channels"])
    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "mod_learning_rate": args.mod_learning_rate,
        **fields,
        **(rnacond_data_fields(args.arm, args.tracks_dir) if rnacond else {}),
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "val_examples": len(val_windows) * (len(conditions) if rnacond else 1),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    if init is not None:
        run["init_from"] = init
    write_json_atomic(run, out / "run.json")
    print(json.dumps(run, indent=2), flush=True)

    register_track_head(
        head,
        num_tracks,
        kind=args.head,
        mod_channels=channels.get("mod_channels"),
        mod_scales=channels.get("mod_scales"),
    )
    t0 = time.time()
    model = load_model(args.checkpoint, head, detach_backbone=not args.train_backbone)
    print(f"model load {time.time() - t0:.1f}s", flush=True)
    if init is not None:
        from tisiago.seq2func.model import restore_init

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
        mod_learning_rate=args.mod_learning_rate,
    )
    params, state = model._params, model._state
    predict = make_predict(model, head)

    data_kwargs = dict(
        fasta=args.genome,
        tracks_dir=args.tracks_dir,
        conditions=conditions,
        genes=genes,
        rna_conditioned=rnacond,
    )
    if args.head in TRANSCRIPT_KINDS:
        from tisiago.seq2func.tx_features import TranscriptFeatures

        t0 = time.time()
        data_kwargs["transcripts"] = TranscriptFeatures.from_gtf(args.gtf)
        print(f"transcript models {time.time() - t0:.1f}s", flush=True)

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
