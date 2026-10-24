"""AlphaGenome fine-tuned on spliced-transcript input + the transcript-space head (P6 step 2).

Usage::

    python -m tisiago.seq2func.txtrain --cache data/p6/txcache/fc2d_k562 \
        --init data/p6/runs/ar_rnacond_k562_fc2d_l2w300/best --heldout K562 \
        --out data/p6/txtrain/k562 --export-dir data/p6/txtrain/export
    # smoke: --steps 50 --eval-every 50; resume: rerun the same command (``<out>/last``)
    python -m tisiago.seq2func.txtrain --predict data/p6/txtrain/k562 \
        --cache data/p6/txcache/fc2d_k562_test --split test --i-am-final \
        --export-dir data/p6/txtrain/export_test      # final: score best/ on test, once

Spec ``docs/superpowers/specs/2026-10-23-p6-spliced-backbone-design.md``. One example is one
MANE transcript of a :class:`txcache.TxCache`: its spliced, sense mRNA (``seq.npy`` codes ->
one-hot, N = zeros), N-padded to a length bucket (``min_bucket`` doubling up to
``max_bucket``; longer transcripts are randomly cropped to ``max_bucket`` in training), with the
cache's RNA input ``(K, L, 2)``, TIS/CHX targets ``(K, L, 2)`` and expression flags ``(K,)`` of
the training conditions (every cached condition not in ``heldout``; held-out conditions are
never read for training). Transcripts are drawn length-weighted among those expressed in a
training condition; draw ``i`` is seeded by ``(seed, i)``, so a resumed run sees the same data.

Model: the AlphaGenome backbone initialised from a fold's genomic fine-tune (``--init``, its
``best/``; trainable), its 1-bp embeddings (1536-d) -> a learned ``Linear(256)`` (initialised
from the cache's PCA) in place of the cache PCA -> the fast-loop head of ``--head-config``
(:func:`fastloop.make_model`, default the step-1 recipe ``p3r02_onehot41_l3``), applied once per
condition. Loss: the head config's fast-loop losses per condition (each condition is its own
batch of one), averaged over the conditions where the transcript is expressed. AdamW with
separate backbone / head learning rates, warmup-cosine, global-norm clipping.

Every ``eval_every`` steps the val transcripts are predicted for every cached condition
(transcripts longer than ``max_bucket`` tiled by ``max_bucket`` windows at half-window stride,
each position from the window whose centre is nearest, as :func:`predict_tracks.owned_spans`)
and scored by :func:`fastloop.score_predictions` on the first held-out condition; ``best/`` keeps
the parameters with the highest ``primary3``. ``last/`` (parameters + optimizer state) is
written atomically at every eval and at ``--stop-after``; an existing ``last/`` is resumed.
At the end the best parameters are scored on every condition (the cache's ``ckpt.npy`` as
``ckpt``), one row is appended to ``--results`` (:func:`fastloop.result_columns`), the record
is written to ``<out>/record.json`` and, with ``--export-dir``, the val predictions of the
held-out conditions to ``<export-dir>/<name>/`` (:func:`fastloop.export_bigwigs`). Test caches
are refused for training; only the train and val rows of a cache are read.

``--predict <out>`` loads ``<out>/best`` and the run's ``run.json`` (head config, settings; the
run hash must match), rebuilds the backbone from the run's ``init`` and predicts every transcript
of ``--split`` of ``--cache`` with the val-eval tiling. ``test`` needs a test cache
(:mod:`txcache` ``build --test --pca-from <train cache>``, whose ``pca.npz`` must equal the
training cache's) and ``--i-am-final``; ``val`` on the training cache reproduces the run's own
val record. The row goes to ``--results`` (default ``data/p6/txtrain/results_test.tsv``; never
the main ``results.tsv``), the record to ``<out>/record_<split>.json`` and, with
``--export-dir``, the split's genomic bigWigs to ``<export-dir>/<name>/``. Training flags are
refused with ``--predict``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.predict_tracks import owned_spans
from tisiago.seq2func.txcache import TEST_SPLIT, TxCache

MIN_BUCKET = 4096
MAX_BUCKET = 32768
PROJ_DIM = 256
DEFAULT_HEAD_CONFIG = Path("autoresearch/p6fast/configs/p3r02/p3r02_onehot41_l3.yaml")
DEFAULT_RESULTS = Path("data/p6/txtrain/results.tsv")
RECORD = "record.json"
STATE = "state.json"
PARAMS = "params.npz"
OPT_STATE = "opt_state.npz"


@dataclass
class Settings:
    """Training settings outside the head config (the identity fields enter the run hash)."""

    steps: int = 30000
    min_bucket: int = MIN_BUCKET
    max_bucket: int = MAX_BUCKET
    eval_every: int = 1000
    log_every: int = 50
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    warmup: int = 300
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 0
    proj_dim: int = PROJ_DIM
    proj_init: str = "pca"
    init: str | None = None
    preload: bool = False
    stop_after: int | None = None
    results: str = str(DEFAULT_RESULTS)
    export_dir: str | None = None
    export_conditions: list[str] | None = field(default=None)

    def identity(self) -> dict:
        """Fields that change what is trained (hashed with the head config)."""
        keep = (
            "steps", "min_bucket", "max_bucket", "backbone_lr", "head_lr", "warmup",
            "weight_decay", "clip", "seed", "proj_dim", "proj_init", "init",
        )  # fmt: skip
        return {k: getattr(self, k) for k in keep}


def check_cache(cache: TxCache) -> None:
    """Refuse a test cache (txcache ``--test``) or any cache with rows outside train / val."""
    if cache.config.get("split") == TEST_SPLIT:
        raise ValueError(f"{cache.path} is a test cache; txtrain never reads test")
    splits = set(cache.index.split)
    bad = sorted(splits - {"train", "val"})
    if bad:
        raise ValueError(f"{cache.path} has {bad} rows; txtrain reads train / val caches only")


def bucket_of(length: int, min_bucket: int = MIN_BUCKET, max_bucket: int = MAX_BUCKET) -> int:
    """Input length of a transcript of ``length``: the doubling bucket, at most ``max_bucket``."""
    return fastloop.bucket_length(min(int(length), int(max_bucket)), min_bucket, max_bucket)


def tiles(length: int, size: int) -> list[tuple[int, int, int]]:
    """``(start, lo, hi)`` windows of ``size`` covering ``[0, length)``; each owns ``[lo, hi)``.

    One window at 0 when ``length <= size``; else starts every ``size // 2`` plus a last window
    ending at ``length``, positions owned by the nearest window centre (ties to the earlier
    window; :func:`predict_tracks.owned_spans`).
    """
    length, size = int(length), int(size)
    if length <= size:
        return [(0, 0, length)]
    starts = list(range(0, length - size, max(1, size // 2))) + [length - size]
    frame = pd.DataFrame({"chrom": "tx", "start": starts, "end": [s + size for s in starts]})
    spans = owned_spans(frame)
    return [(s, lo, hi) for s, (lo, hi) in zip(starts, spans, strict=True) if hi > lo]


def transcript_example(
    cache: TxCache,
    row,
    conditions: Sequence[str],
    start: int,
    size: int,
    tis_target: str = "pooled",
) -> dict[str, np.ndarray]:
    """Transcript ``row`` at ``[start, start + size)``, padded to ``size``, for ``conditions``.

    ``seq`` (S,) int8 codes (N = 4 past the end), ``rna`` (K, S, 2), ``target`` (K, S, 2) TIS
    (``tis_target`` as :func:`fastloop.crop_example`) and pooled CHX, ``expressed`` (K,),
    ``rna_tx`` (K, 2) (whole-transcript sense RNA head / mean, as the fast loop), ``valid``,
    ``pos`` (S,), ``exon`` (S, 5), ``strand`` (0 +, 1 -), ``length``.
    """
    k = [cache.condition_index(c) for c in conditions]
    length, offset = int(row.length), int(row.offset)
    n = max(0, min(int(size), length - int(start)))
    lo = offset + int(start)
    seq_all = fastloop.optional_array(cache, row.shard, "seq")
    if seq_all is None:
        raise FileNotFoundError(f"{cache.path}: shard {row.shard} has no seq.npy (build_seq.py)")
    out = {
        "seq": np.full(size, fastloop.SEQ_N, np.int8),
        "rna": np.zeros((len(k), size, 2), np.float32),
        "target": np.zeros((len(k), size, 2), np.float32),
        "valid": np.zeros(size, bool),
        "pos": np.zeros(size, np.int32),
        "exon": np.zeros((size, fastloop.EXON_FEATURES), np.float32),
    }
    out["seq"][:n] = seq_all[lo : lo + n]
    rna = cache.array(row.shard, "rna")
    out["rna"][:, :n] = rna[k, lo : lo + n]
    tis = np.asarray(cache.array(row.shard, "tis")[k, lo : lo + n])
    if tis_target == "pooled":
        out["target"][:, :n, 0] = tis[..., 2]
    elif tis_target == "min":
        out["target"][:, :n, 0] = 2.0 * np.minimum(tis[..., 0], tis[..., 1])
    elif tis_target == "geo":
        out["target"][:, :n, 0] = 2.0 * np.sqrt(np.maximum(tis[..., 0] * tis[..., 1], 0.0))
    else:
        raise ValueError(f"tis_target must be pooled, min or geo, not {tis_target!r}")
    out["target"][:, :n, 1] = cache.array(row.shard, "chx")[k, lo : lo + n, 2]
    out["valid"][:n] = True
    out["pos"][:n] = np.arange(start, start + n)
    out["exon"][:n] = fastloop.exon_features(
        getattr(row, "junctions", ()), np.arange(start, start + n), length
    )
    sense = np.asarray(rna[k, offset : offset + length, 0], np.float32)
    head = sense[:, : fastloop.RNA_HEAD].mean(1)
    out["rna_tx"] = np.stack([head, sense.mean(1)], axis=1).astype(np.float32)
    out["expressed"] = np.array([bool(getattr(row, f"expressed_{c}")) for c in conditions])
    out["strand"] = np.int32(0 if row.strand == "+" else 1)
    out["length"] = np.int32(length)
    return out


def head_batch(example: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The fast-loop head batch of an example: one row per condition (``emb`` added later)."""
    k, size = example["rna"].shape[:2]

    def rows(x):
        return np.ascontiguousarray(np.broadcast_to(x, (k, *np.shape(x))))

    return {
        "rna": example["rna"],
        "target": example["target"],
        "rna_tx": example["rna_tx"],
        "valid": rows(example["valid"]),
        "pos": rows(example["pos"]),
        "exon": rows(example["exon"]),
        "seq": rows(example["seq"]),
        "strand": np.full(k, example["strand"], np.int32),
        "length": np.full(k, example["length"], np.int32),
    }


class TrainSampler:
    """Training transcripts of a cache, drawn length-weighted; draw ``i`` seeded by (seed, i)."""

    def __init__(
        self,
        cache: TxCache,
        conditions: Sequence[str],
        seed: int = 0,
        min_bucket: int = MIN_BUCKET,
        max_bucket: int = MAX_BUCKET,
        tis_target: str = "pooled",
    ):
        """Transcripts of the train split expressed in at least one of ``conditions``."""
        self.cache, self.conditions = cache, tuple(conditions)
        self.seed, self.tis_target = int(seed), tis_target
        self.min_bucket, self.max_bucket = int(min_bucket), int(max_bucket)
        rows = cache.rows("train").reset_index(drop=True)
        expressed = np.zeros(len(rows), bool)
        for c in self.conditions:
            expressed |= rows[f"expressed_{c}"].to_numpy(bool)
        self.rows = rows[expressed].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError("no training transcripts expressed in the training conditions")
        self.records = list(self.rows.itertuples(index=False))
        w = self.rows.length.to_numpy(float)
        self.p = w / w.sum()

    def draw(self, i: int) -> tuple:
        """``(row, start, size)`` of draw ``i``."""
        rng = np.random.default_rng([self.seed, int(i)])
        row = self.records[int(rng.choice(len(self.records), p=self.p))]
        size = bucket_of(row.length, self.min_bucket, self.max_bucket)
        extra = int(row.length) - size
        start = int(rng.integers(0, extra + 1)) if extra > 0 else 0
        return row, start, size

    def example(self, i: int) -> dict[str, np.ndarray]:
        """Example of draw ``i``."""
        row, start, size = self.draw(i)
        return transcript_example(self.cache, row, self.conditions, start, size, self.tis_target)


@dataclass
class Backbone:
    """``apply(params, state, onehot (1, S, 4)) -> (1, S, dim)`` 1-bp embeddings, trainable."""

    params: dict
    state: dict
    apply: Callable
    dim: int
    source: str = ""


def stub_backbone(dim: int = 16, width: int = 5, seed: int = 0) -> Backbone:
    """A tiny Haiku conv backbone with the AlphaGenome interface (tests)."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    forward = hk.transform_with_state(lambda x: hk.Conv1D(dim, width, name="stub")(x))
    params, state = forward.init(jax.random.PRNGKey(seed), jnp.zeros((1, 64, 4), jnp.float32))
    return Backbone(
        params, state, lambda p, s, x: forward.apply(p, s, None, x)[0], dim, f"stub:{dim}"
    )


def alphagenome_backbone(
    init: str | Path,
    base_checkpoint: str,
    arm: str | None = None,
    head_kind: str | None = None,
) -> Backbone:
    """The backbone parameters of a fine-tuned checkpoint (``best/``) as a trainable forward.

    The checkpoint is restored as :func:`txcache.load_backbone` (``load_finetuned``); its head
    leaves are dropped.
    """
    import haiku as hk
    import jax.numpy as jnp
    from alphagenome_research.model import dna_model

    from tisiago.seq2func.evaluate import head_kind_of, run_args
    from tisiago.seq2func.model import ORGANISM, _alphagenome_backbone, load_finetuned
    from tisiago.seq2func.train import parse_rnacond_arm

    init = Path(init)
    arm = arm or run_args(init).get("arm")
    head_kind = head_kind or head_kind_of(init)
    head, _, _ = parse_rnacond_arm(arm)
    model = load_finetuned(str(init), str(base_checkpoint), head, 4, head_kind)
    params = {k: v for k, v in model._params.items() if not k.startswith("head/")}
    state = {k: v for k, v in model._state.items() if not k.startswith("head/")}
    del model
    backbone = _alphagenome_backbone()
    org = dna_model.convert_to_organism_index(ORGANISM)

    @hk.transform_with_state
    def embed_forward(onehot):
        organism = jnp.full((onehot.shape[0],), org, jnp.int32)
        return backbone(onehot, organism).get_sequence_embeddings(resolution=1)

    def apply(p, s, onehot):
        return embed_forward.apply(p, s, None, onehot)[0]

    return Backbone(params, state, apply, 1536, f"{init} ({arm}, {head_kind})")


def check_init_heldout(init: str | Path | None, heldout: Sequence[str]) -> None:
    """Raise if ``init``'s run (``run.json``) held out different conditions than ``heldout``."""
    if init is None:
        return
    run = Path(init).parent / "run.json"
    if not run.exists():
        return
    held = json.loads(run.read_text()).get("heldout_conditions")
    if held is not None and sorted(held) != sorted(heldout):
        raise ValueError(f"{init} held out {held}, not {list(heldout)}: init from that fold")


def _onehot(seq):
    import jax
    import jax.numpy as jnp

    seq = seq.astype(jnp.int32)
    return jax.nn.one_hot(seq, 4) * (seq < 4)[..., None]


class Net:
    """Backbone + projection + fast-loop head; ``params = {"backbone": ..., "head": ...}``."""

    def __init__(self, config: Mapping, backbone: Backbone, proj_dim: int = PROJ_DIM):
        """Build the head of ``config`` with a ``Linear(proj_dim)`` embedding projection."""
        import jax.numpy as jnp

        self.config, self.backbone, self.proj_dim = config, backbone, int(proj_dim)
        ctx = {"emb_std": jnp.ones(self.proj_dim, jnp.float32)}
        self.head = fastloop.make_model(config, ctx, embed_proj=self.proj_dim)

    def embed(self, backbone_params, state, seq):
        """``(S,)`` codes -> ``(1, S, dim)`` float32 embeddings."""
        import jax.numpy as jnp

        x = self.backbone.apply(backbone_params, state, _onehot(seq)[None])
        return x.astype(jnp.float32)

    def apply(self, params, state, batch, seq):
        """``(pred (K, S, 2), aux)`` for a :func:`head_batch` and its ``seq``."""
        emb = self.embed(params["backbone"], state, seq)
        return self.head.apply(params["head"], {**batch, "emb": emb})

    def forward(self) -> Callable:
        """Jitted ``(params, state, batch, seq) -> pred (K, S, 2)`` (training eval and predict)."""
        import jax

        @jax.jit
        def forward(params, state, batch, seq):
            return self.apply(params, state, batch, seq)[0]

        return forward

    def init_head(self, key, example: Mapping[str, np.ndarray]):
        """Fresh head parameters (shapes from ``example``)."""
        import jax.numpy as jnp

        batch = {k: jnp.asarray(v) for k, v in head_batch(example).items()}
        size = example["seq"].shape[0]
        return self.head.init(key, {**batch, "emb": jnp.zeros((1, size, self.backbone.dim))})


def pca_projection(cache: TxCache, dim: int, proj_dim: int) -> dict | None:
    """``emb_proj`` parameters equal to the cache PCA (None if absent or of another shape)."""
    path = cache.path / "pca.npz"
    if not path.exists():
        return None
    with np.load(path) as pca:
        mean, comps = pca["mean"].astype(np.float32), pca["components"].astype(np.float32)
    if comps.shape != (proj_dim, dim):
        return None
    return {"w": comps.T.copy(), "b": -(mean @ comps.T)}


def example_loss(pred, batch, aux, losses: Sequence[Mapping], expressed) -> tuple:
    """Fast-loop loss of each condition row (one-sample batch), averaged over expressed rows."""
    import jax.numpy as jnp

    w = expressed.astype(jnp.float32)
    denom = jnp.maximum(w.sum(), 1.0)
    total, parts = jnp.zeros((), jnp.float32), {}
    for i in range(pred.shape[0]):
        one = {"target": batch["target"][i : i + 1], "valid": batch["valid"][i : i + 1]}
        term, terms = fastloop.total_loss(pred[i : i + 1], one, aux, losses)
        total = total + w[i] * term / denom
        for key, value in terms.items():
            parts[key] = parts.get(key, 0.0) + w[i] * value / denom
    return total, parts


def make_optimizer(params, settings: Settings):
    """Clip, then AdamW with warmup-cosine backbone and head learning rates."""
    import jax
    import optax

    steps, warmup = int(settings.steps), int(settings.warmup)

    def schedule(lr):
        return optax.warmup_cosine_decay_schedule(
            0.0, float(lr), warmup, max(steps, warmup + 1), float(lr) * 0.01
        )

    labels = {
        part: jax.tree_util.tree_map(lambda _, p=part: p, params[part])
        for part in ("backbone", "head")
    }
    wd = float(settings.weight_decay)
    return optax.chain(
        optax.clip_by_global_norm(float(settings.clip)),
        optax.multi_transform(
            {
                "backbone": optax.adamw(schedule(settings.backbone_lr), weight_decay=wd),
                "head": optax.adamw(schedule(settings.head_lr), weight_decay=wd),
            },
            labels,
        ),
    )


def _saveable(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind == "V" or arr.dtype.type.__module__.startswith("ml_dtypes"):
        return arr.view(np.dtype(f"u{arr.dtype.itemsize}"))
    return arr


def save_tree(tree, path: str | Path) -> None:
    """Leaves of ``tree`` (flatten order) in one ``.npz``; custom dtypes as raw bits."""
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    np.savez(path, **{f"{i:06d}": _saveable(x) for i, x in enumerate(leaves)})


def load_tree(template, path: str | Path):
    """Inverse of :func:`save_tree` onto ``template``'s structure, shapes and dtypes."""
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(template)
    out = []
    with np.load(path) as data:
        if len(data.files) != len(leaves):
            raise ValueError(f"{path}: {len(data.files)} leaves, expected {len(leaves)}")
        for i, ref in enumerate(leaves):
            arr = data[f"{i:06d}"]
            dtype = np.dtype(ref.dtype)
            if arr.dtype != dtype:
                arr = arr.view(dtype)
            if arr.shape != tuple(np.shape(ref)):
                raise ValueError(f"{path}: leaf {i} shape {arr.shape} != {np.shape(ref)}")
            out.append(jnp.asarray(arr))
    return jax.tree_util.tree_unflatten(treedef, out)


def save_checkpoint(dest: Path, params, opt_state, state: Mapping) -> None:
    """Atomically write ``params`` (+ ``opt_state`` unless None) and ``state.json`` to ``dest``."""
    from tisiago.seq2func.train import save_dir_atomic

    def write(tmp: Path) -> None:
        save_tree(params, tmp / PARAMS)
        if opt_state is not None:
            save_tree(opt_state, tmp / OPT_STATE)
        (tmp / STATE).write_text(json.dumps(state, indent=2, default=_json_default) + "\n")

    save_dir_atomic(write, Path(dest))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def predictor(forward: Callable, params, state) -> Callable[[Mapping], np.ndarray]:
    """``predict(example) -> (K, S, 2)`` of ``forward`` (:meth:`Net.forward`) at ``params``."""
    import jax.numpy as jnp

    def predict(example):
        batch = {k: jnp.asarray(v) for k, v in head_batch(example).items()}
        return forward(params, state, batch, jnp.asarray(example["seq"]))

    return predict


def predict_transcripts(
    predict: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    cache: TxCache,
    rows: pd.DataFrame,
    conditions: Sequence[str],
    min_bucket: int = MIN_BUCKET,
    max_bucket: int = MAX_BUCKET,
) -> dict[str, np.ndarray]:
    """Per transcript ``(len(conditions), L, 2)``: ``predict(example) -> (K, S, 2)`` per tile."""
    out = {}
    for row in rows.itertuples():
        length = int(row.length)
        pred = np.zeros((len(conditions), length, 2), np.float32)
        size = bucket_of(length, min_bucket, max_bucket)
        for start, lo, hi in tiles(length, size):
            example = transcript_example(cache, row, conditions, start, size)
            values = np.asarray(predict(example), np.float32)
            pred[:, lo:hi] = values[:, lo - start : hi - start]
        out[row.transcript_id] = pred
    return out


def export_val(
    cache: TxCache,
    preds: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    dest: str | Path,
    meta: Mapping | None = None,
) -> dict:
    """Val predictions as genomic float bigWigs: exactly :func:`fastloop.export_bigwigs`."""
    return fastloop.export_bigwigs(cache, preds, list(conditions), dest, meta=meta)


def _peak_bytes() -> int | None:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - backends without memory stats
        return None
    return stats.get("peak_bytes_in_use")


def _append_log(path: Path, record: Mapping) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def _score_row(cache, sources, conditions, heldout, meta, source="model") -> tuple:
    table, peaks = fastloop.score_predictions(cache, sources, conditions, heldout)
    row = fastloop.results_row(meta, table, peaks, cache.conditions, source)
    return row, table, peaks


def run_hash(config: Mapping, settings: Settings) -> str:
    """12-hex digest of the head config (without name) and the identity settings."""
    import hashlib

    payload = {
        "head": {k: v for k, v in config.items() if k != "name"},
        "txtrain": settings.identity(),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def train_run(
    cache: TxCache,
    config: Mapping,
    backbone: Backbone,
    out: str | Path,
    settings: Settings,
    log: Callable[[str], None] = print,
) -> dict | None:
    """Train, select on val, write ``best/``, ``last/``, the record, results row and export.

    Resumes from ``<out>/last`` when present (same run hash). Returns the record, or None when
    stopped early by ``settings.stop_after``.
    """
    import jax
    import jax.numpy as jnp
    import optax

    t0 = time.time()
    check_cache(cache)
    out = Path(out)
    if (out / RECORD).exists():
        raise FileExistsError(f"{out / RECORD} exists: this run is finished")
    out.mkdir(parents=True, exist_ok=True)
    conditions = tuple(cache.conditions)
    heldout = list(config["heldout"])
    train_conditions = fastloop.train_conditions_of(config, conditions)
    digest = run_hash(config, settings)
    name = config["name"]
    tis_target = config["data"].get("tis_target", "pooled")
    if settings.preload:
        cache.preload("train", ("rna", "tis", "chx", "seq"))
        log(f"preloaded the train split ({time.time() - t0:.0f}s)")
    sampler = TrainSampler(
        cache, train_conditions, settings.seed, settings.min_bucket, settings.max_bucket, tis_target
    )
    net = Net(config, backbone, settings.proj_dim)
    head_params = net.init_head(jax.random.PRNGKey(settings.seed), sampler.example(0))
    if settings.proj_init == "pca":
        proj = pca_projection(cache, backbone.dim, settings.proj_dim)
        if proj is not None:
            head_params = {**head_params, "emb_proj": {k: jnp.asarray(v) for k, v in proj.items()}}
    elif settings.proj_init != "random":
        raise ValueError(f"proj_init must be pca or random, not {settings.proj_init!r}")
    params = {"backbone": backbone.params, "head": head_params}
    opt = make_optimizer(params, settings)
    opt_state = opt.init(params)
    losses = config["losses"]
    bstate = backbone.state

    def objective(params, state, batch, seq, expressed):
        pred, aux = net.apply(params, state, batch, seq)
        return example_loss(pred, batch, aux, losses, expressed)

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, state, batch, seq, expressed):
        (loss, parts), grads = jax.value_and_grad(objective, has_aux=True)(
            params, state, batch, seq, expressed
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, parts

    forward = net.forward()

    def predict_with(p):
        return predictor(forward, p, bstate)

    status = {
        "step": 0,
        "best_step": None,
        "best_primary3": None,
        "history": [],
        "evals": [],
        "digest": digest,
    }
    if (out / "last" / STATE).exists():
        saved = json.loads((out / "last" / STATE).read_text())
        if saved.get("digest") != digest:
            raise ValueError(f"{out}/last is run {saved.get('digest')}, not {digest}")
        params = load_tree(params, out / "last" / PARAMS)
        opt_state = load_tree(opt_state, out / "last" / OPT_STATE)
        status = saved
        log(f"resumed {out}/last at step {status['step']}")
    else:
        run = {
            "name": name,
            "digest": digest,
            "settings": asdict(settings),
            "config": config,
            "cache": str(cache.path),
            "backbone": backbone.source,
            "train_conditions": list(train_conditions),
            "heldout": heldout,
            "git_commit": fastloop._git_commit(),
        }
        fastloop._write_json(out / "run.json", run)
    val = cache.rows("val")
    first = heldout[0]
    best_preds = None

    def evaluate(p) -> tuple[float, dict, float]:
        t1 = time.time()
        preds = predict_transcripts(
            predict_with(p), cache, val, conditions, settings.min_bucket, settings.max_bucket
        )
        meta = {"heldout_list": heldout}
        row, _, _ = _score_row(cache, {"model": preds}, [first], heldout, meta)
        return float(row["primary3"]), preds, time.time() - t1

    def checkpoint_last():
        save_checkpoint(out / "last", params, opt_state, status)

    start = int(status["step"])
    end = int(settings.steps)
    if settings.stop_after is not None:
        end = min(end, int(settings.stop_after))
    remaining = max(0, end - start)
    counter = iter(range(start, end))
    examples = fastloop.prefetched(lambda: sampler.example(next(counter)), remaining)
    window, compile_seconds = [], {}
    log_path = out / "log.jsonl"
    for i in range(start, end):
        example = next(examples)
        size = int(example["seq"].shape[0])
        t1 = time.time()
        batch = {k: jnp.asarray(v) for k, v in head_batch(example).items()}
        params, opt_state, loss, parts = step(
            params,
            opt_state,
            bstate,
            batch,
            jnp.asarray(example["seq"]),
            jnp.asarray(example["expressed"]),
        )
        loss = float(loss)
        seconds = time.time() - t1
        if size not in compile_seconds:
            compile_seconds[size] = seconds
        else:
            window.append((size, seconds))
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {i + 1}")
        status["step"] = i + 1
        record = {"step": i + 1, "loss": loss, **{k: float(v) for k, v in parts.items()}}
        if (i + 1) % max(1, settings.log_every) == 0 or i + 1 == end:
            sizes = {}
            for s, sec in window:
                sizes.setdefault(s, []).append(sec)
            record |= {
                "seconds": time.time() - t0,
                "step_seconds": {str(s): float(np.mean(v)) for s, v in sorted(sizes.items())},
                "step_counts": {str(s): len(v) for s, v in sorted(sizes.items())},
                "compile_seconds": {str(s): v for s, v in sorted(compile_seconds.items())},
                "peak_bytes": _peak_bytes(),
            }
            window = []
            status["history"].append(record)
            _append_log(log_path, record)
            log(json.dumps(record, default=_json_default))
        if (i + 1) % max(1, settings.eval_every) == 0 or i + 1 == settings.steps:
            primary3, preds, eval_seconds = evaluate(params)
            best = status["best_primary3"]
            better = status["best_step"] is None or (
                np.isfinite(primary3) and (best is None or not np.isfinite(best) or primary3 > best)
            )
            entry = {
                "step": i + 1,
                "primary3": primary3,
                "eval_seconds": eval_seconds,
                "best": bool(better),
                "peak_bytes": _peak_bytes(),
            }
            status["evals"].append(entry)
            _append_log(log_path, {"eval": entry})
            log(json.dumps({"eval": entry}, default=_json_default))
            if better:
                status["best_step"], status["best_primary3"] = i + 1, primary3
                best_preds = preds
                save_checkpoint(out / "best", params, None, status)
            checkpoint_last()
    if status["step"] < settings.steps:
        checkpoint_last()
        log(f"stopped at step {status['step']} (stop_after); rerun to resume")
        return None
    if status["best_step"] is None:
        raise ValueError("no evaluation was run")
    t2 = time.time()
    if best_preds is None:
        best_params = load_tree(params, out / "best" / PARAMS)
        best_preds = predict_transcripts(
            predict_with(best_params),
            cache,
            val,
            conditions,
            settings.min_bucket,
            settings.max_bucket,
        )
    sources = {"model": best_preds}
    ckpt = fastloop.ckpt_predictions(cache)
    if ckpt is not None:
        sources["ckpt"] = ckpt
    table, peaks = fastloop.score_predictions(cache, sources, conditions, heldout)
    history = status["history"]
    meta = {
        "config_hash": digest,
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": fastloop._git_commit(),
        "cache": str(cache.path),
        "heldout": ",".join(heldout),
        "heldout_list": heldout,
        "steps": int(settings.steps),
        "best_step": status["best_step"],
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "train_seconds": t2 - t0,
        "eval_seconds": time.time() - t2,
        "runtime_seconds": time.time() - t0,
        "init": settings.init,
        "backbone": backbone.source,
    }
    columns = fastloop.result_columns(conditions)
    row = fastloop.results_row(meta, table, peaks, conditions)
    fastloop.append_result(settings.results, row, columns)
    record = {
        "meta": meta,
        "config": config,
        "settings": asdict(settings),
        "history": history,
        "evals": status["evals"],
        "row": row,
        "reference_row": fastloop.results_row(meta, table, peaks, conditions, "ckpt")
        if ckpt is not None
        else None,
        "scorecard": table.to_dict(orient="records"),
        "peaks": peaks.to_dict(orient="records"),
    }
    if settings.export_dir is not None:
        dest = Path(settings.export_dir) / name
        export_val(
            cache,
            best_preds,
            settings.export_conditions or heldout,
            dest,
            meta={
                "name": name,
                "config_hash": digest,
                "primary3": row["primary3"],
                "params": str(out / "best"),
            },
        )
        record["export"] = str(dest)
        log(f"exported {name} val tracks to {dest}")
    fastloop._write_json(out / RECORD, record)
    log(
        f"{name} ({digest}): best step {status['best_step']}, primary3 {row['primary3']:.1f}, "
        f"total {time.time() - t0:.0f}s"
    )
    return record


TEST_RESULTS = Path("data/p6/txtrain/results_test.tsv")


def load_run(run: str | Path) -> tuple[dict, Settings]:
    """``(run.json, settings)`` of a finished training run; the run hash must match ``best/``."""
    run = Path(run)
    for need in ("run.json", f"best/{PARAMS}", f"best/{STATE}"):
        if not (run / need).exists():
            raise FileNotFoundError(f"{run / need} missing: not a txtrain run with a best/")
    payload = json.loads((run / "run.json").read_text())
    settings = Settings(**payload["settings"])
    digest = run_hash(payload["config"], settings)
    if digest != payload["digest"]:
        raise ValueError(f"{run}/run.json: run hash {digest} != saved {payload['digest']}")
    best = json.loads((run / "best" / STATE).read_text())
    if best.get("digest") != digest:
        raise ValueError(f"{run}/best is run {best.get('digest')}, not {digest}")
    return payload, settings


def predict_run(
    run: str | Path,
    cache: TxCache,
    backbone: Backbone,
    split: str = "test",
    results: str | Path = TEST_RESULTS,
    export_dir: str | Path | None = None,
    export_conditions: Sequence[str] | None = None,
    i_am_final: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Score a finished run's ``best/`` on ``split`` of ``cache`` without training.

    ``backbone`` must be the run's (same parameter tree; its weights are replaced by
    ``best/``). The head config, projection and buckets are the run's own; predictions use the
    val-eval tiling, scoring the unchanged :func:`fastloop.score_predictions` with ``split``'s
    rows in place of val (:class:`fastloop.SplitView`). Writes ``<run>/record_<split>.json``,
    one row to ``results`` (never a main ``results.tsv``) and, with ``export_dir``, the split's
    genomic bigWigs to ``<export_dir>/<name>/``.
    """
    import jax

    if split not in ("val", "test"):
        raise ValueError(f"split must be val or test, not {split!r}")
    if split == TEST_SPLIT:
        if not i_am_final:
            raise PermissionError("split test reads the held-out chromosomes; pass i_am_final")
        if cache.config.get("split") != TEST_SPLIT:
            raise ValueError(f"{cache.path} is not a test cache (txcache --test)")
    else:
        check_cache(cache)
    results = Path(results)
    if results.name == DEFAULT_RESULTS.name or results.resolve() == DEFAULT_RESULTS.resolve():
        raise ValueError(f"{results}: predictions never go to the main results.tsv")
    t0 = time.time()
    run = Path(run)
    payload, settings = load_run(run)
    config, digest, name = payload["config"], payload["digest"], payload["config"]["name"]
    heldout = list(config["heldout"])
    conditions = tuple(cache.conditions)
    missing = sorted((set(heldout) | set(payload["train_conditions"])) - set(conditions))
    if missing:
        raise ValueError(f"{cache.path} lacks the run's conditions {missing}")
    fastloop.check_pca(cache, payload.get("cache"))
    rows = cache.rows(split)
    if rows.empty:
        raise ValueError(f"{cache.path} has no {split} transcripts")
    first = next(rows.itertuples())
    size = bucket_of(first.length, settings.min_bucket, settings.max_bucket)
    net = Net(config, backbone, settings.proj_dim)
    example = transcript_example(cache, first, conditions, 0, size)
    template = {
        "backbone": backbone.params,
        "head": net.init_head(jax.random.PRNGKey(settings.seed), example),
    }
    params = load_tree(template, run / "best" / PARAMS)
    best = json.loads((run / "best" / STATE).read_text())
    preds = predict_transcripts(
        predictor(net.forward(), params, backbone.state),
        cache,
        rows,
        conditions,
        settings.min_bucket,
        settings.max_bucket,
    )
    view = fastloop.SplitView(cache, split)
    sources = {"model": preds}
    ckpt = fastloop.ckpt_predictions(view)
    if ckpt is not None:
        sources["ckpt"] = ckpt
    table, peaks = fastloop.score_predictions(view, sources, conditions, heldout)
    meta = {
        "config_hash": digest,
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": fastloop._git_commit(),
        "cache": str(cache.path),
        "heldout": ",".join(heldout),
        "heldout_list": heldout,
        "steps": int(settings.steps),
        "best_step": best.get("best_step"),
        "eval_seconds": time.time() - t0,
        "runtime_seconds": time.time() - t0,
        "split": split,
        "run": str(run),
        "train_cache": payload.get("cache"),
        "init": settings.init,
        "backbone": backbone.source,
    }
    row = fastloop.results_row(meta, table, peaks, conditions)
    fastloop.append_result(results, row, fastloop.result_columns(conditions))
    record = {
        "meta": meta,
        "config": config,
        "settings": asdict(settings),
        "row": row,
        "reference_row": fastloop.results_row(meta, table, peaks, conditions, "ckpt")
        if ckpt is not None
        else None,
        "scorecard": table.to_dict(orient="records"),
        "peaks": peaks.to_dict(orient="records"),
    }
    if export_dir is not None:
        dest = Path(export_dir) / name
        fastloop.export_bigwigs(
            cache,
            preds,
            list(export_conditions or heldout),
            dest,
            split=split,
            meta={
                "name": name,
                "config_hash": digest,
                "primary3": row["primary3"],
                "params": str(run / "best"),
            },
        )
        record["export"] = str(dest)
        log(f"exported {name} {split} tracks to {dest}")
    fastloop._write_json(run / f"record_{split}.json", record)
    log(f"{name} ({digest}) on {split}: primary3 {row['primary3']:.1f}")
    return record


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--cache", type=Path, required=True, help="train/val txcache (test only with --predict)"
    )
    ap.add_argument("--init", type=Path, default=None, help="fold's fine-tuned run best/")
    ap.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    ap.add_argument("--heldout", nargs="+", default=None, help="default: the head config's")
    ap.add_argument("--out", type=Path, default=None, help="run directory (training)")
    ap.add_argument("--name", default=None, help="default: the --out directory name")
    ap.add_argument("--steps", type=int, default=Settings.steps)
    ap.add_argument("--min-bucket", type=int, default=MIN_BUCKET)
    ap.add_argument("--max-bucket", type=int, default=MAX_BUCKET)
    ap.add_argument("--eval-every", type=int, default=Settings.eval_every)
    ap.add_argument("--log-every", type=int, default=Settings.log_every)
    ap.add_argument("--backbone-lr", type=float, default=Settings.backbone_lr)
    ap.add_argument("--head-lr", type=float, default=Settings.head_lr)
    ap.add_argument("--warmup", type=int, default=Settings.warmup)
    ap.add_argument("--weight-decay", type=float, default=Settings.weight_decay)
    ap.add_argument("--clip", type=float, default=Settings.clip)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-init", choices=("pca", "random"), default="pca")
    ap.add_argument("--preload", action="store_true", help="read the train split into memory")
    ap.add_argument("--stop-after", type=int, default=None, help="stop at this step (resumable)")
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help=f"default {DEFAULT_RESULTS} ({TEST_RESULTS} with --predict)",
    )
    ap.add_argument("--export-dir", type=Path, default=None)
    ap.add_argument("--export-conditions", nargs="+", default=None)
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="finished run directory (<out>): score its best/ on --split of --cache, no training",
    )
    ap.add_argument("--split", choices=("val", "test"), default="test", help="with --predict")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    return ap


PREDICT_FLAGS = frozenset(
    {"cache", "results", "export_dir", "export_conditions", "base_checkpoint", "predict", "split",
     "i_am_final", "help"}
)  # fmt: skip


def training_flags_given(ap: argparse.ArgumentParser, argv: Sequence[str]) -> list[str]:
    """Training-only option strings present in ``argv`` (refused with ``--predict``)."""
    given = []
    for action in ap._actions:
        if action.dest in PREDICT_FLAGS:
            continue
        for opt in action.option_strings:
            if any(a == opt or a.startswith(opt + "=") for a in argv):
                given.append(opt)
    return given


def predict_main(ap, args, argv: Sequence[str], backbone_factory: Callable | None) -> None:
    """``--predict``: rebuild the run's backbone, score its ``best/`` on ``--split``."""
    flags = training_flags_given(ap, argv)
    if flags:
        ap.error(f"--predict takes no training flags: {' '.join(flags)}")
    if args.split == TEST_SPLIT and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    cache = TxCache(args.cache)
    if args.split == TEST_SPLIT and cache.config.get("split") != TEST_SPLIT:
        ap.error(f"{args.cache} is not a test cache (txcache build --test)")
    if args.split != TEST_SPLIT:
        try:
            check_cache(cache)
        except ValueError as exc:
            ap.error(str(exc))
    try:
        _, settings = load_run(args.predict)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    t0 = time.time()
    if backbone_factory is not None:
        backbone = backbone_factory(args)
    else:
        import jax

        if not settings.init:
            ap.error(f"{args.predict}: the run has no init to rebuild its backbone from")
        if not args.base_checkpoint:
            ap.error("--base-checkpoint is required (or set ALPHAGENOME_WEIGHTS_PATH in .env)")
        backbone = alphagenome_backbone(settings.init, args.base_checkpoint)
        print(f"backbone {backbone.source} built in {time.time() - t0:.0f}s on {jax.devices()}")
    predict_run(
        args.predict,
        cache,
        backbone,
        args.split,
        args.results or TEST_RESULTS,
        args.export_dir,
        args.export_conditions,
        i_am_final=args.i_am_final,
        log=lambda s: print(s, flush=True),
    )


def main(argv: Sequence[str] | None = None, backbone_factory: Callable | None = None) -> None:
    """CLI; ``backbone_factory(args) -> Backbone`` replaces AlphaGenome (tests)."""
    import sys

    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    if args.predict is not None:
        predict_main(ap, args, argv, backbone_factory)
        return
    if args.out is None:
        ap.error("--out is required for training")
    if args.i_am_final or "--split" in argv or any(a.startswith("--split=") for a in argv):
        ap.error("--split / --i-am-final go with --predict")
    args.results = args.results or DEFAULT_RESULTS
    cache = TxCache(args.cache)
    try:
        check_cache(cache)
    except ValueError as exc:
        ap.error(str(exc))
    config = fastloop.load_config(args.head_config)
    if args.heldout:
        config["heldout"] = list(args.heldout)
    config["cache"] = str(args.cache)
    config["name"] = args.name or Path(args.out).name
    if backbone_factory is None:
        if args.init is None:
            ap.error("--init <run>/best is required")
        if not args.base_checkpoint:
            ap.error("--base-checkpoint is required (or set ALPHAGENOME_WEIGHTS_PATH in .env)")
        check_init_heldout(args.init, config["heldout"])
    settings = Settings(
        steps=args.steps,
        min_bucket=args.min_bucket,
        max_bucket=args.max_bucket,
        eval_every=args.eval_every,
        log_every=args.log_every,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        clip=args.clip,
        seed=args.seed,
        proj_init=args.proj_init,
        init=str(args.init) if args.init else None,
        preload=args.preload,
        stop_after=args.stop_after,
        results=str(args.results),
        export_dir=str(args.export_dir) if args.export_dir else None,
        export_conditions=args.export_conditions,
    )

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    if backbone_factory is not None:
        backbone = backbone_factory(args)
    else:
        import jax

        backbone = alphagenome_backbone(args.init, args.base_checkpoint)
        log(f"backbone {backbone.source} loaded in {time.time() - t0:.0f}s on {jax.devices()}")
    train_run(cache, config, backbone, args.out, settings, log)


if __name__ == "__main__":
    main()
