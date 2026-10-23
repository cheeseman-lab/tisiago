"""Fast-loop trainer and scorer over a transcript cache (P6 fast loss/structure search).

Usage::

    python -m tisiago.seq2func.fastloop configs/p6fast/ref_rnacond.yaml [more.yaml ...]
    python -m tisiago.seq2func.fastloop --reference      # score the checkpoint's own predictions
    python -m tisiago.seq2func.fastloop --predict data/p6/fastloop/export/<name> \
        --cache data/p6/txcache/<x>_test --split test --i-am-final \
        --export-dir data/p6/fastloop/export_test     # final: score a saved head on test, once

A config (YAML or JSON) is merged into :data:`DEFAULTS` and composes registered components:

- ``inputs``: names from :data:`INPUTS` (``embedding``, ``embedding_white``, ``rna``,
  ``rna_sense``, ``rna_onesided``, ``rna_onset``, ``rna_relative``, ``position``, ``exon``
  (spliced exon structure), ``onehot`` (mRNA base; needs ``seq.npy`` in the cache)),
  concatenated per position.
- ``model.trunk``: :data:`TRUNKS` (``mlp``, ``conv``, ``dilated``, ``attention``).
- ``model.tis`` / ``model.chx``: output parameterisations from :data:`OUTPUTS`: ``softplus``
  rate; ``factorised`` amount x softmax within blocks (``block`` int, two grids averaged, or
  ``"crop"``; ``temperature``); ``hazard`` scanning: ``P(start at i) = h_i prod_{j<i}(1 - h_j)``
  times the crop's summed amount. ``model.strand_heads`` gives each genomic strand its own
  output layer (the genomic head predicts both strands from one embedding); ``model.
  split_trunk`` gives CHX its own trunk (TIS keeps the first; ``model.chx_trunk`` sets
  its config); ``model.tis_context`` stacks a trunk on the shared one for TIS only (e.g. a
  dilated stack for 5'UTR-scale context; ``causal: up`` sees only the 5' side, ``down`` only
  the 3' side); ``model.tis_onehot`` (``width``, ``channels``, ``layers``) adds a one-hot
  sequence conv branch to the TIS head input (``into: context``: to the TIS context input);
  ``model.tis_exon`` feeds the ``exon`` features to the TIS branch only; ``model.chx_contrast``
  feeds log ratios of the gradient-stopped CHX prediction to the TIS output.
- ``losses``: list of ``{kind, weight, ...}`` from :data:`LOSSES`: ``poisson`` (deviance of
  totals; ``scope`` sample / global), ``multinomial`` (``scope`` global = the batch, one
  condition, like a genomic window; sample = the transcript crop; local = blocks of ``block``,
  optional ``count_power`` block weights n^p, ``target_power`` tempered observed profile and
  ``offset`` grid shift),
  ``frame`` (codon-phase KL on CHX), ``poisson_nt`` / ``nb`` (per-nucleotide likelihoods;
  NB with a learned per-library dispersion), ``coupling`` (predicted TIS profile vs the rise of
  the gradient-stopped predicted CHX). ``libraries`` selects ``tis`` / ``chx``.

Training: batches of ``data.batch`` (transcript, condition) pairs, one condition per batch
(``data.batch_condition: single``) drawn from the training conditions (default every cached
condition not in ``heldout``) with the transcript expressed there, cropped/padded to
``data.crop`` (random crop, or the 5' end with ``crop_mode: start``), transcripts weighted by
length. AdamW, warmup-cosine, global-norm clipping; numpy and JAX seeded by ``seed``.

Scoring (val transcripts, every cached condition, held-out ones marked): full-length
predictions, then exactly the scorecard's metrics (:func:`scorecard.library_rows`: gene,
gene_mrna, bin30, codon, nt as percent of replicate 1 vs 2; CHX frame) and the TIS peak share
at real Ribo-TISH starts (meta share at the start nt, per-site argmax at the start; sites with
real pooled signal within +-30 nt), also per start class (``atg`` / ``nc`` = any other codon at
the real call) with the real data's values (``_real``). ``primary2`` (phase 2) is the mean of the
held-out TIS nt percent and its near-cognate argmax-at-start as percent of real (cap 100),
appended after the phase-1 columns. Phase 3 appends reproducible-start columns: at real starts
whose replicate 2 argmax (+-30 nt) is within 1 nt of the start, ``rep_hit`` = P(prediction
argmax within 1 nt) and ``rep_ref`` = P(replicate 1 argmax within 1 nt), per start class, overall
and split by first vs later exon (``_first`` / ``_later``); ``rep_both`` = P(prediction hit |
both replicates hit); ``primary3`` = mean of held-out TIS nt percent and
``100 x min(1, rep_hit_nc / rep_ref_nc)``. The checkpoint's own predictions (``ckpt.npy``) are
scored alongside as ``ckpt``. ``--export-dir`` writes each config's val predictions as genomic
float bigWigs (:func:`export_bigwigs`) plus its parameters. One row per config is appended to
``--results`` (:data:`RESULT_COLUMNS`) and the full record written to
``--runs-dir/<name>-<hash>.json``.

``--predict <export_dir>/<name>`` rebuilds a saved head (``params.npz`` + ``config.json``; the
parameter tree must match the config's model exactly), predicts every transcript of ``--split``
of ``--cache`` (``test`` needs a test cache, :mod:`txcache` ``--test``, and ``--i-am-final``),
scores it with the same scorer (the split's rows in place of val), appends the row to
``--results`` (default ``data/p6/fastloop/results_test.tsv``; never the main ``results.tsv``),
writes ``<runs-dir>/<name>-<hash>-<split>.json`` and, with ``--export-dir``, the split's genomic
bigWigs to ``<export-dir>/<name>/``. No training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.scorecard import (
    FRAME,
    MIN_FRAME_READS,
    MIN_READS,
    SCALES,
    library_rows,
)
from tisiago.seq2func.txcache import TxCache

LIBRARIES = ("tis", "chx")
CHANNEL = {"tis": 0, "chx": 1}
FLANK = 30
EPS = 1e-7
PRIMARY_SCALES = ("codon", "nt")
START_CLASSES = ("atg", "nc")
EXON_REGIONS = ("", "_first", "_later")
REP_STATS = ("hit", "ref", "ratio", "n")
RNA_HEAD = 129
DEFAULTS: dict = {
    "name": None,
    "seed": 0,
    "cache": "data/p6/txcache/ar_rnacond_k562_v2",
    "heldout": ["K562"],
    "train_conditions": None,
    "data": {
        "crop": 4096,
        "crop_mode": "random",
        "batch": 16,
        "batch_condition": "single",
        "sample_weight": "length",
        "preload": True,
        "tis_target": "pooled",
    },
    "inputs": ["embedding", "rna"],
    "model": {
        "trunk": {"kind": "mlp", "hidden": 256, "layers": 1},
        "strand_heads": True,
        "chx_contrast": False,
        "tis": {"kind": "softplus"},
        "chx": {"kind": "softplus"},
    },
    "losses": [
        {"kind": "poisson", "scope": "global", "weight": 1.0},
        {"kind": "multinomial", "scope": "global", "weight": 5.0},
    ],
    "train": {
        "steps": 10000,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup": 100,
        "clip": 1.0,
        "log_every": 100,
    },
    "eval": {"max_length": 32768, "min_bucket": 4096},
}


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """``base`` updated recursively with ``override`` (lists and scalars are replaced)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """A YAML / JSON config merged into :data:`DEFAULTS`; ``name`` defaults to the file stem."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text) or {}
    config = resolve_config(raw)
    config["name"] = config["name"] or path.stem
    return config


def resolve_config(raw: Mapping) -> dict:
    """``raw`` merged into :data:`DEFAULTS` and checked against the registries."""
    config = deep_merge(DEFAULTS, raw)
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config keys {unknown}")
    for name in config["inputs"]:
        if name not in INPUTS:
            raise ValueError(f"unknown input {name!r}; choose from {sorted(INPUTS)}")
    for key in ("trunk", "chx_trunk", "tis_context"):
        kind = (config["model"].get(key) or {"kind": "mlp"})["kind"]
        if kind not in TRUNKS:
            raise ValueError(f"unknown {key} {kind!r}")
    for library in LIBRARIES:
        kind = config["model"][library]["kind"]
        if kind not in OUTPUTS:
            raise ValueError(f"unknown {library} output {kind!r}; choose from {sorted(OUTPUTS)}")
    for loss in config["losses"]:
        if loss.get("kind") not in LOSSES:
            raise ValueError(f"unknown loss {loss.get('kind')!r}; choose from {sorted(LOSSES)}")
    return config


def config_hash(config: Mapping) -> str:
    """12-hex digest of the resolved config without its ``name``."""
    payload = {k: v for k, v in config.items() if k != "name"}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def train_conditions_of(config: Mapping, conditions: Sequence[str]) -> tuple[str, ...]:
    """``train_conditions`` or every cached condition not held out; never a held-out one."""
    heldout = set(config["heldout"])
    chosen = config["train_conditions"] or [c for c in conditions if c not in heldout]
    leaked = sorted(set(chosen) & heldout)
    if leaked:
        raise ValueError(f"training conditions {leaked} are held out")
    missing = sorted(set(chosen) - set(conditions))
    if missing:
        raise ValueError(f"conditions {missing} are not in the cache")
    return tuple(chosen)


SEQ_N = 4  # ``seq.npy`` code of N / no sequence (A, C, G, T = 0..3)
EXON_FEATURES = 5
JUNCTION_CAP = 10000


def optional_array(cache: TxCache, shard: str, key: str) -> np.ndarray | None:
    """``cache.array(shard, key)`` or None when the file is absent (absence memoised)."""
    missing = cache.__dict__.setdefault("_fastloop_missing", set())
    if (shard, key) in missing:
        return None
    try:
        return cache.array(shard, key)
    except FileNotFoundError:
        missing.add((shard, key))
        return None


def exon_features(junctions: Sequence[int], positions: np.ndarray, length: int) -> np.ndarray:
    """``(n, 5)`` spliced exon structure at mRNA ``positions``.

    Columns: log1p(exon index) / 3, first exon (0/1), log1p(distance from the exon's 5' end, the
    cap for the first exon) / 10, log1p(distance to the next exon start or the mRNA end) / 10,
    log1p(distance to the nearest exon junction, capped at ``JUNCTION_CAP``) / 10.
    """
    j = np.asarray(junctions, np.int64).reshape(-1)
    pos = np.asarray(positions, np.int64)
    idx = np.searchsorted(j, pos, side="right")
    if len(j):
        exon_start = np.where(idx > 0, j[np.maximum(idx - 1, 0)], 0)
        exon_next = np.where(idx < len(j), j[np.minimum(idx, len(j) - 1)], int(length))
    else:
        exon_start, exon_next = np.zeros_like(pos), np.full_like(pos, int(length))
    up, down = pos - exon_start, exon_next - pos
    near = np.minimum(
        np.where(idx > 0, up, JUNCTION_CAP), np.where(idx < len(j), down, JUNCTION_CAP)
    )
    return np.stack(
        [
            np.log1p(idx) / 3.0,
            (idx == 0).astype(float),
            np.log1p(np.maximum(up, 0)) / 10.0,
            np.log1p(np.maximum(down, 0)) / 10.0,
            np.log1p(np.minimum(near, JUNCTION_CAP)) / 10.0,
        ],
        axis=-1,
    ).astype(np.float32)


def crop_example(
    cache: TxCache, row, condition: str, start: int, crop: int, tis_target: str = "pooled"
) -> dict[str, np.ndarray]:
    """One transcript ``[start, start + crop)`` of ``condition``, zero-padded to ``crop``.

    Also ``exon`` (:func:`exon_features`) and ``seq`` (``seq.npy`` codes when the cache has
    them, else ``SEQ_N``). ``tis_target``: ``pooled`` (rep1 + rep2), ``min`` (2 x min(rep1,
    rep2): only replicate-reproducible signal) or ``geo`` (2 x sqrt(rep1 x rep2)).
    """
    k = cache.condition_index(condition)
    n = max(0, min(crop, int(row.length) - start))
    lo = int(row.offset) + start
    out = {
        "emb": np.zeros((crop, cache.array(row.shard, "emb").shape[1]), np.float16),
        "rna": np.zeros((crop, 2), np.float32),
        "target": np.zeros((crop, 2), np.float32),
        "valid": np.zeros(crop, bool),
        "pos": np.zeros(crop, np.int32),
        "exon": np.zeros((crop, EXON_FEATURES), np.float32),
        "seq": np.full(crop, SEQ_N, np.int8),
    }
    out["exon"][:n] = exon_features(
        getattr(row, "junctions", ()), np.arange(start, start + n), int(row.length)
    )
    seq = optional_array(cache, row.shard, "seq")
    if seq is not None:
        out["seq"][:n] = seq[lo : lo + n]
    tx = cache.array(row.shard, "rna")[k, int(row.offset) : int(row.offset) + int(row.length), 0]
    out["rna_tx"] = np.array(
        [tx[:RNA_HEAD].mean() if len(tx) else 0.0, tx.mean() if len(tx) else 0.0], np.float32
    )
    out["emb"][:n] = cache.array(row.shard, "emb")[lo : lo + n]
    out["rna"][:n] = cache.array(row.shard, "rna")[k, lo : lo + n]
    tis = cache.array(row.shard, "tis")[k, lo : lo + n]
    if tis_target == "pooled":
        out["target"][:n, 0] = tis[:, 2]
    elif tis_target == "min":
        out["target"][:n, 0] = 2.0 * np.minimum(tis[:, 0], tis[:, 1])
    elif tis_target == "geo":
        out["target"][:n, 0] = 2.0 * np.sqrt(np.maximum(tis[:, 0] * tis[:, 1], 0.0))
    else:
        raise ValueError(f"tis_target must be pooled, min or geo, not {tis_target!r}")
    out["target"][:n, 1] = cache.array(row.shard, "chx")[k, lo : lo + n, 2]
    out["valid"][:n] = True
    out["pos"][:n] = np.arange(start, start + n)
    return out


def stack_examples(examples: Sequence[dict], rows: Sequence) -> dict[str, np.ndarray]:
    """Batch of :func:`crop_example` dicts plus per-sample ``strand`` (0 +, 1 -), ``length``."""
    batch = {k: np.stack([e[k] for e in examples]) for k in examples[0]}
    batch["strand"] = np.array([0 if r.strand == "+" else 1 for r in rows], np.int32)
    batch["length"] = np.array([int(r.length) for r in rows], np.int32)
    return batch


class Sampler:
    """Seeded training batches of (transcript, condition) crops from a cache's train split."""

    def __init__(self, cache: TxCache, config: Mapping):
        """Candidate transcripts per training condition (expressed there)."""
        self.cache = cache
        data = config["data"]
        self.crop, self.size = int(data["crop"]), int(data["batch"])
        self.tis_target = data.get("tis_target", "pooled")
        self.mode, self.single = data["crop_mode"], data["batch_condition"] == "single"
        if self.mode not in ("random", "start"):
            raise ValueError(f"crop_mode must be random or start, not {self.mode!r}")
        self.rng = np.random.default_rng(int(config["seed"]))
        self.rows = cache.rows("train").reset_index(drop=True)
        self.records = list(self.rows.itertuples(index=False))
        self.conditions = train_conditions_of(config, cache.conditions)
        self.pools = {}
        for c in self.conditions:
            idx = np.flatnonzero(self.rows[f"expressed_{c}"].to_numpy(bool))
            if not len(idx):
                continue
            w = self.rows.length.to_numpy(float)[idx]
            if data["sample_weight"] == "uniform":
                w = np.ones_like(w)
            self.pools[c] = (idx, w / w.sum())
        if not self.pools:
            raise ValueError("no training transcripts expressed in the training conditions")
        sizes = np.array([len(self.pools[c][0]) for c in self.pools], float)
        self.names = list(self.pools)
        self.condition_p = sizes / sizes.sum()

    def draw(self) -> tuple[list, list[str], list[int]]:
        """Rows, conditions and crop starts of one batch."""
        if self.single:
            conditions = [self.names[self.rng.choice(len(self.names), p=self.condition_p)]]
            conditions *= self.size
        else:
            picks = self.rng.choice(len(self.names), size=self.size, p=self.condition_p)
            conditions = [self.names[i] for i in picks]
        rows, starts = [], []
        for c in conditions:
            idx, p = self.pools[c]
            row = self.records[int(idx[self.rng.choice(len(idx), p=p)])]
            extra = int(row.length) - self.crop
            start = (
                int(self.rng.integers(0, extra + 1)) if extra > 0 and self.mode == "random" else 0
            )
            rows.append(row)
            starts.append(start)
        return rows, conditions, starts

    def batch(self) -> dict[str, np.ndarray]:
        """One training batch."""
        rows, conditions, starts = self.draw()
        examples = [
            crop_example(self.cache, r, c, s, self.crop, self.tis_target)
            for r, c, s in zip(rows, conditions, starts, strict=True)
        ]
        return stack_examples(examples, rows)


def prefetched(make: Callable[[], dict], n: int, depth: int = 4):
    """Yield ``n`` results of ``make()`` produced (in order) by one daemon thread."""
    out: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def work():
        try:
            for _ in range(n):
                if stop.is_set():
                    return
                item = make()
                while not stop.is_set():
                    try:
                        out.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            out.put(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        for _ in range(n):
            item = out.get()
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


INPUTS: dict[str, Callable] = {}
TRUNKS: dict[str, Callable] = {}
OUTPUTS: dict[str, tuple[int, Callable]] = {}
LOSSES: dict[str, Callable] = {}


def register(registry: dict, name: str, *extra):
    """Decorator: ``registry[name] = fn`` (or ``(*extra, fn)``)."""

    def wrap(fn):
        registry[name] = (*extra, fn) if extra else fn
        return fn

    return wrap


@register(INPUTS, "embedding")
def _input_embedding(batch, ctx):
    import jax.numpy as jnp

    return batch["emb"].astype(jnp.float32)


@register(INPUTS, "embedding_white")
def _input_embedding_white(batch, ctx):
    import jax.numpy as jnp

    return batch["emb"].astype(jnp.float32) / ctx["emb_std"]


@register(INPUTS, "rna")
def _input_rna(batch, ctx):
    from tisiago.seq2func.rna_features import rna_features

    return rna_features(batch["rna"])


@register(INPUTS, "rna_sense")
def _input_rna_sense(batch, ctx):
    import jax.numpy as jnp

    from tisiago.seq2func.rna_features import rna_features

    sense = batch["rna"][..., :1]
    return rna_features(jnp.concatenate([sense, sense], axis=-1))[..., :5]


@register(INPUTS, "rna_onesided")
def _input_rna_onesided(batch, ctx):
    from tisiago.seq2func.rna_features import onesided_features

    return onesided_features(batch["rna"])


@register(INPUTS, "rna_onset")
def _input_rna_onset(batch, ctx):
    """Sharpness of RNA 5' ends per strand: ``log1p(x) - log1p(box_w(x))`` (w 3, 9, 33)."""
    import jax.numpy as jnp

    from tisiago.seq2func.rna_features import box_mean

    rna = jnp.maximum(batch["rna"].astype(jnp.float32), 0.0)
    feats = [jnp.log1p(rna) - jnp.log1p(box_mean(rna, w)) for w in (3, 9, 33)]
    return jnp.concatenate(feats, axis=-1)


@register(INPUTS, "rna_relative")
def _input_rna_relative(batch, ctx):
    """Sense RNA relative to the transcript's own 5' end (first ``RNA_HEAD`` nt) and mean."""
    import jax.numpy as jnp

    from tisiago.seq2func.rna_features import box_mean

    sense = jnp.maximum(batch["rna"][..., :1].astype(jnp.float32), 0.0)
    head = jnp.log1p(batch["rna_tx"][:, None, :1])
    mean = jnp.log1p(batch["rna_tx"][:, None, 1:])
    b33, b129 = jnp.log1p(box_mean(sense, 33)), jnp.log1p(box_mean(sense, 129))
    return jnp.concatenate([b33 - head, b129 - head, b33 - mean, b129 - mean], axis=-1)


@register(INPUTS, "exon")
def _input_exon(batch, ctx):
    """Spliced exon structure (:func:`exon_features`): index, first, distances."""
    import jax.numpy as jnp

    return batch["exon"].astype(jnp.float32)


def _onehot(batch):
    import jax
    import jax.numpy as jnp

    seq = batch["seq"].astype(jnp.int32)
    return jax.nn.one_hot(seq, 4) * (seq < 4)[..., None]


@register(INPUTS, "onehot")
def _input_onehot(batch, ctx):
    """One-hot mRNA base (A, C, G, T; N = 0)."""
    return _onehot(batch)


@register(INPUTS, "position")
def _input_position(batch, ctx):
    import jax.numpy as jnp

    pos = batch["pos"].astype(jnp.float32)
    to_end = jnp.maximum(batch["length"][:, None].astype(jnp.float32) - 1 - pos, 0.0)
    return jnp.stack([jnp.log1p(pos), jnp.log1p(to_end)], axis=-1) / 10.0


@register(TRUNKS, "mlp")
def _trunk_mlp(x, valid, cfg):
    import haiku as hk
    import jax

    h = x
    for i in range(int(cfg.get("layers", 1))):
        h = jax.nn.relu(hk.Linear(int(cfg.get("hidden", 256)), name=f"hidden{i}")(h))
    return h


@register(TRUNKS, "conv")
def _trunk_conv(x, valid, cfg):
    import haiku as hk
    import jax

    width = int(cfg.get("width", 31))
    h = x
    for i in range(int(cfg.get("layers", 1))):
        conv = hk.Conv1D(int(cfg.get("hidden", 256)), width, padding="SAME", name=f"conv{i}")
        h = jax.nn.relu(conv(h)) * valid[..., None]
    return h


@register(TRUNKS, "dilated")
def _trunk_dilated(x, valid, cfg):
    import haiku as hk
    import jax

    zeros = hk.initializers.Constant(0.0)
    h = jax.nn.relu(hk.Linear(int(cfg.get("hidden", 256)), name="hidden0")(x)) * valid[..., None]
    causal = cfg.get("causal", False)  # "up": 5' context only; "down": 3' only
    for i, rate in enumerate(cfg.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))):
        rate = int(rate)
        pad = {"up": ((2 * rate, 0),), "down": ((0, 2 * rate),)}.get(causal, "SAME")
        y = hk.Conv1D(int(cfg.get("channels", 128)), 3, rate=rate, padding=pad, name=f"dil{i}")(h)
        y = hk.Conv1D(h.shape[-1], 1, w_init=zeros, b_init=zeros, name=f"dil{i}_out")(
            jax.nn.relu(y)
        )
        h = (h + y) * valid[..., None]
    return h


def _block_attention(h, valid, offset: int, block: int, heads: int, dim: int, name: str):
    import haiku as hk
    import jax
    import jax.numpy as jnp

    batch, length, width = h.shape
    per_head = dim // heads
    padded = -(-(length + offset) // block) * block
    pad = [(0, 0), (offset, padded - length - offset), (0, 0)]
    x = jnp.pad(h, pad)
    keys = jnp.pad(valid, pad[:2]).reshape(batch, padded // block, 1, 1, block)

    def project(tag):
        y = hk.Linear(dim, name=f"{name}_{tag}")(x)
        return y.reshape(batch, padded // block, block, heads, per_head)

    q, k, v = project("q"), project("k"), project("v")
    scores = jnp.einsum("bnqhd,bnkhd->bnhqk", q, k) / math.sqrt(per_head)
    table = hk.get_parameter(
        f"{name}_rel_bias", (heads, 2 * block - 1), init=hk.initializers.Constant(0.0)
    )
    rel = jnp.arange(block)[:, None] - jnp.arange(block)[None, :] + block - 1
    scores = jnp.where(keys, scores + table[:, rel][None, None], -1e30)
    att = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bnhqk,bnkhd->bnqhd", att, v).reshape(batch, padded, dim)
    zeros = hk.initializers.Constant(0.0)
    out = hk.Linear(width, w_init=zeros, b_init=zeros, name=f"{name}_out")(out)
    return out[:, offset : offset + length]


@register(TRUNKS, "attention")
def _trunk_attention(x, valid, cfg):
    import haiku as hk
    import jax

    block = int(cfg.get("block", 1024))
    h = jax.nn.relu(hk.Linear(int(cfg.get("hidden", 256)), name="hidden0")(x)) * valid[..., None]
    for layer in range(int(cfg.get("layers", 2))):
        norm = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"attn{layer}_norm")
        y = _block_attention(
            norm(h),
            valid,
            (block // 2) * (layer % 2),
            block,
            int(cfg.get("heads", 4)),
            int(cfg.get("dim", 64)),
            f"attn{layer}",
        )
        h = (h + y) * valid[..., None]
    return h


def block_softmax(amount, logits, valid, block: int, offset: int = 0):
    """``(B, L)`` amount summed per block (grid at ``-offset``), spread by a masked softmax."""
    import jax
    import jax.numpy as jnp

    batch, length = amount.shape
    padded = -(-(length + offset) // block) * block
    pad = [(0, 0), (offset, padded - length - offset)]
    a = jnp.pad(jnp.where(valid, amount, 0.0), pad).reshape(batch, -1, block)
    lg = jnp.where(jnp.pad(valid, pad), jnp.pad(logits, pad), -1e30).reshape(batch, -1, block)
    out = a.sum(-1, keepdims=True) * jax.nn.softmax(lg, axis=-1)
    return out.reshape(batch, padded)[:, offset : offset + length]


@register(OUTPUTS, "softplus", 1)
def _output_softplus(p, valid, cfg):
    import jax
    import jax.numpy as jnp

    return jnp.where(valid, jax.nn.softplus(p[..., 0]), 0.0)


@register(OUTPUTS, "factorised", 2)
def _output_factorised(p, valid, cfg):
    import jax
    import jax.numpy as jnp

    amount = jax.nn.softplus(p[..., 0])
    logits = p[..., 1] / float(cfg.get("temperature", 1.0))
    block = cfg.get("block", 1024)
    if block == "crop":
        total = jnp.where(valid, amount, 0.0).sum(-1, keepdims=True)
        weights = jax.nn.softmax(jnp.where(valid, logits, -1e30), axis=-1)
        return jnp.where(valid, total * weights, 0.0)
    block = int(block)
    out = block_softmax(amount, logits, valid, block, 0)
    if cfg.get("two_grids", True):
        out = 0.5 * (out + block_softmax(amount, logits, valid, block, block // 2))
    return jnp.where(valid, out, 0.0)


def scan_probabilities(hazard_logits, valid):
    """``P(start at i) = h_i prod_{j<i} (1 - h_j)`` with ``h = sigmoid(logits)``, 5'->3'."""
    import jax
    import jax.numpy as jnp

    log_h = jnp.where(valid, jax.nn.log_sigmoid(hazard_logits), -jnp.inf)
    log_stay = jnp.where(valid, jax.nn.log_sigmoid(-hazard_logits), 0.0)
    before = jnp.cumsum(log_stay, axis=-1) - log_stay
    return jnp.where(valid, jnp.exp(log_h + before), 0.0)


@register(OUTPUTS, "hazard", 2)
def _output_hazard(p, valid, cfg):
    import jax
    import jax.numpy as jnp

    total = jnp.where(valid, jax.nn.softplus(p[..., 0]), 0.0).sum(-1, keepdims=True)
    return total * scan_probabilities(p[..., 1], valid)


def chx_log_ratios(chx, widths=(15, 45, 150), eps: float = 0.01):
    """``(B, L)`` sense CHX -> ``(B, L, len(widths))`` downstream/upstream log mean ratios."""
    import jax.numpy as jnp

    from tisiago.seq2func.rna_features import strand_log_ratios

    both = jnp.stack([chx, chx], axis=-1)
    return strand_log_ratios(both, tuple(widths), eps)[..., : len(widths)]


def make_model(config: Mapping, ctx: Mapping, embed_proj: int | None = None):
    """Haiku ``forward(batch) -> (rate (B, L, 2) [TIS, CHX], aux)`` of ``config``.

    ``embed_proj``: ``batch["emb"]`` holds raw embeddings ``(B or 1, L, D)``, projected by a
    learned ``Linear(embed_proj)`` (module ``emb_proj``) and broadcast over the batch before the
    inputs are built (:mod:`txtrain`; in place of the cache PCA).
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp

    model = config["model"]
    trunk = TRUNKS[model["trunk"]["kind"]]
    strands = 2 if model.get("strand_heads", True) else 1

    def head(h, strand, n, name):
        out = hk.Linear(n * strands, name=name)(h).astype(jnp.float32)
        out = out.reshape(*out.shape[:-1], strands, n)
        if strands == 1:
            return out[..., 0, :]
        return jnp.where((strand == 0)[:, None, None], out[..., 0, :], out[..., 1, :])

    def forward(batch):
        valid = batch["valid"]
        if embed_proj is not None:
            emb = hk.Linear(int(embed_proj), name="emb_proj")(batch["emb"].astype(jnp.float32))
            batch = {**batch, "emb": jnp.broadcast_to(emb, (*valid.shape, emb.shape[-1]))}
        x = jnp.concatenate([INPUTS[n](batch, ctx) for n in config["inputs"]], axis=-1)
        x = jnp.where(valid[..., None], x.astype(jnp.float32), 0.0)
        h = trunk(x, valid, model["trunk"])
        h_chx = h
        if model.get("split_trunk") or model.get("chx_trunk"):
            chx_cfg = model.get("chx_trunk") or model["trunk"]
            with hk.name_scope("chx_trunk"):
                h_chx = TRUNKS[chx_cfg["kind"]](x, valid, chx_cfg)
        n_chx, chx_fn = OUTPUTS[model["chx"]["kind"]]
        chx = chx_fn(head(h_chx, batch["strand"], n_chx, "chx_out"), valid, model["chx"])
        tis_in = h
        seq_h = None
        if model.get("tis_onehot"):
            oh_cfg = model["tis_onehot"]
            seq_h = _onehot(batch)
            for i in range(int(oh_cfg.get("layers", 1))):
                conv = hk.Conv1D(
                    int(oh_cfg.get("channels", 64)),
                    int(oh_cfg.get("width", 41)) if i == 0 else 3,
                    padding="SAME",
                    name=f"tis_onehot{i}",
                )
                seq_h = jax.nn.relu(conv(seq_h)) * valid[..., None]
            if oh_cfg.get("into", "head") == "context":
                tis_in, seq_h = jnp.concatenate([tis_in, seq_h], axis=-1), None
        if model.get("tis_exon"):
            tis_in = jnp.concatenate([tis_in, batch["exon"].astype(jnp.float32)], axis=-1)
        if model.get("tis_context"):
            with hk.name_scope("tis_context"):
                ctx_cfg = model["tis_context"]
                tis_in = TRUNKS[ctx_cfg["kind"]](tis_in, valid, ctx_cfg)
        if seq_h is not None:
            tis_in = jnp.concatenate([tis_in, seq_h], axis=-1)
        if model.get("chx_contrast"):
            ratios = chx_log_ratios(jax.lax.stop_gradient(chx))
            tis_in = jax.nn.relu(
                hk.Linear(128, name="tis_hidden")(jnp.concatenate([h, ratios], -1))
            )
        n_tis, tis_fn = OUTPUTS[model["tis"]["kind"]]
        tis = tis_fn(head(tis_in, batch["strand"], n_tis, "tis_out"), valid, model["tis"])
        log_r = hk.get_parameter("nb_log_r", (2,), init=hk.initializers.Constant(math.log(10.0)))
        return jnp.stack([tis, chx], axis=-1), {"nb_log_r": log_r}

    return hk.without_apply_rng(hk.transform(forward))


def _xlogy(x, y):
    import jax.numpy as jnp

    return jnp.where(x > 0, x * jnp.log(jnp.maximum(y, EPS)), 0.0)


def _channels(loss: Mapping, default=LIBRARIES) -> tuple[int, ...]:
    return tuple(CHANNEL[lib] for lib in loss.get("libraries", default))


def _grouped(pred, target, mask, scope: str):
    """``(G, N, C)`` groups: the whole batch (``global``) or each sample (``sample``)."""
    if scope == "global":
        shape = (1, -1, pred.shape[-1])
        return pred.reshape(shape), target.reshape(shape), mask.reshape(shape)
    if scope in ("sample", "transcript"):
        return pred, target, mask
    raise ValueError(f"scope must be global or sample, not {scope!r}")


def _masked(pred, target, valid, channels):
    import jax.numpy as jnp

    idx = jnp.asarray(channels)
    mask = jnp.broadcast_to(valid[..., None], (*valid.shape, len(channels)))
    pred = jnp.where(mask, pred[..., idx].astype(jnp.float32), 0.0)
    target = jnp.where(mask, jnp.maximum(target[..., idx].astype(jnp.float32), 0.0), 0.0)
    return pred, target, mask


@register(LOSSES, "poisson")
def loss_poisson(pred, target, valid, aux, loss):
    """Poisson deviance of totals per group and channel, averaged (genomic ``poisson``)."""
    pred, target, mask = _masked(pred, target, valid, _channels(loss))
    pred, target, _ = _grouped(pred, target, mask, loss.get("scope", "global"))
    tp, tt = pred.sum(axis=1), target.sum(axis=1)
    return (2.0 * (tp - tt + _xlogy(tt, tt) - _xlogy(tt, tp))).mean()


@register(LOSSES, "multinomial")
def loss_multinomial(pred, target, valid, aux, loss):
    """Profile KL per group and channel, averaged; ``local``: counts-weighted per block."""
    scope = loss.get("scope", "global")
    channels = _channels(loss)
    if scope == "local":
        from tisiago.seq2func.loss import local_multinomial

        count_power = float(loss.get("count_power", 1.0))
        target_power = float(loss.get("target_power", 1.0))
        offset = int(loss.get("offset", 0))  # block grid starts at crop position ``offset``
        pred, target, valid = pred[:, offset:], target[:, offset:], valid[:, offset:]
        if count_power != 1.0 or target_power != 1.0:
            return weighted_local_multinomial(
                pred,
                target,
                _full_mask(pred, valid),
                channels,
                int(loss.get("block", 1024)),
                count_power,
                target_power,
            )
        return local_multinomial(
            pred, target, _full_mask(pred, valid), channels, int(loss.get("block", 1024))
        )
    pred, target, mask = _masked(pred, target, valid, channels)
    pred, target, _ = _grouped(pred, target, mask, scope)
    tp, tt = pred.sum(axis=1), target.sum(axis=1)
    p_t = target / _floor(tt)[:, None, :]
    p_p = pred / _floor(tp)[:, None, :]
    return (_xlogy(p_t, p_t) - _xlogy(p_t, p_p)).sum(axis=1).mean()


def weighted_local_multinomial(
    pred,
    target,
    mask,
    channels: Sequence[int],
    block: int = 1024,
    count_power: float = 1.0,
    target_power: float = 1.0,
):
    """Local multinomial with block weights ``n ** count_power``, tempered observed profile.

    The observed profile is ``target ** target_power`` normalised; both powers 1 give exactly
    :func:`loss.local_multinomial` (counts-weighted).
    """
    import jax.numpy as jnp

    channels = tuple(int(c) for c in channels)
    length = (pred.shape[1] // block) * block
    if not channels or length == 0:
        return jnp.zeros((), jnp.float32)
    idx = jnp.asarray(channels)
    shape = (pred.shape[0], length // block, block, len(channels))
    keep = jnp.asarray(mask)[:, :length][..., idx]
    pred = jnp.where(keep, pred[:, :length][..., idx].astype(jnp.float32), 0.0).reshape(shape)
    target = target[:, :length][..., idx].astype(jnp.float32)
    target = jnp.where(keep, jnp.maximum(target, 0.0), 0.0).reshape(shape)
    n_target = target.sum(axis=2)
    tempered = jnp.where(target > 0, jnp.power(jnp.maximum(target, EPS), target_power), 0.0)
    p_target = tempered / jnp.maximum(tempered.sum(axis=2)[:, :, None, :], EPS)
    p_pred = pred / jnp.maximum(pred.sum(axis=2)[:, :, None, :], EPS)
    kl = (_xlogy(p_target, p_target) - _xlogy(p_target, p_pred)).sum(axis=2)
    has = n_target > 0
    w = jnp.where(has, jnp.power(jnp.maximum(n_target, EPS), count_power), 0.0)
    total = w.sum()
    return jnp.where(total > 0, (w * jnp.where(has, kl, 0.0)).sum() / jnp.maximum(total, EPS), 0.0)


def _full_mask(pred, valid):
    import jax.numpy as jnp

    return jnp.broadcast_to(valid[..., None], pred.shape)


def _floor(x):
    import jax.numpy as jnp

    return jnp.maximum(x, EPS)


@register(LOSSES, "frame")
def loss_frame(pred, target, valid, aux, loss):
    """Codon-phase KL on CHX (:func:`loss.codon_phase_multinomial`, the crop's own grid)."""
    from tisiago.seq2func.loss import codon_phase_multinomial

    channels = _channels(loss, ("chx",))
    return codon_phase_multinomial(
        pred, target, _full_mask(pred, valid), channels, int(loss.get("block", 10))
    )


@register(LOSSES, "poisson_nt")
def loss_poisson_nt(pred, target, valid, aux, loss):
    """Per-nucleotide Poisson deviance summed over the batch / observed counts."""
    pred, target, _ = _masked(pred, target, valid, _channels(loss))
    dev = 2.0 * (pred - target + _xlogy(target, target) - _xlogy(target, pred))
    return dev.sum() / _floor(target.sum())


@register(LOSSES, "nb")
def loss_nb(pred, target, valid, aux, loss):
    """Per-nucleotide negative-binomial NLL (learned per-library dispersion) / observed counts."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    channels = _channels(loss)
    pred, target, mask = _masked(pred, target, valid, channels)
    r = jnp.exp(aux["nb_log_r"][jnp.asarray(channels)])
    mu = jnp.maximum(pred, EPS)
    nll = -(
        gammaln(target + r)
        - gammaln(r)
        - gammaln(target + 1.0)
        + r * (jnp.log(r) - jnp.log(r + mu))
        + target * (jnp.log(mu) - jnp.log(r + mu))
    )
    return jnp.where(mask, nll, 0.0).sum() / _floor(target.sum())


@register(LOSSES, "coupling")
def loss_coupling(pred, target, valid, aux, loss):
    """KL(normalised CHX rise || predicted TIS profile) per sample; CHX gradient-stopped.

    The rise is ``relu(log ratio)`` of the predicted CHX downstream vs upstream mean at
    ``width`` (default 45); samples without a rise contribute 0.
    """
    import jax
    import jax.numpy as jnp

    chx = jax.lax.stop_gradient(jnp.where(valid, pred[..., 1], 0.0))
    width = int(loss.get("width", 45))
    rise = jnp.where(valid, jax.nn.relu(chx_log_ratios(chx, (width,))[..., 0]), 0.0)
    tis = jnp.where(valid, pred[..., 0], 0.0)
    q = rise / _floor(rise.sum(-1, keepdims=True))
    p = tis / _floor(tis.sum(-1, keepdims=True))
    kl = (_xlogy(q, q) - _xlogy(q, p)).sum(-1)
    has = rise.sum(-1) > 0
    return jnp.where(has, kl, 0.0).sum() / jnp.maximum(has.sum(), 1)


def total_loss(pred, batch, aux, losses: Sequence[Mapping]) -> tuple:
    """``(sum of weight x term, {kind[_scope]_i: term})`` over the configured losses."""
    import jax.numpy as jnp

    target, valid = batch["target"], batch["valid"]
    total = jnp.zeros((), jnp.float32)
    parts = {}
    for i, loss in enumerate(losses):
        term = LOSSES[loss["kind"]](pred, target, valid, aux, loss)
        parts[f"{i}_{loss['kind']}"] = term
        total = total + float(loss.get("weight", 1.0)) * term
    return total, parts


def context_of(cache: TxCache) -> dict:
    """Static inputs of the components (PCA component scales)."""
    import jax.numpy as jnp

    pca = cache.path / "pca.npz"
    if pca.exists():
        std = np.sqrt(np.maximum(np.load(pca)["explained_variance"], 1e-12)).astype(np.float32)
    else:
        std = np.ones(cache.array(cache.index.shard.iloc[0], "emb").shape[1], np.float32)
    return {"emb_std": jnp.asarray(std)}


def _device_batch(batch: Mapping[str, np.ndarray]) -> dict:
    import jax.numpy as jnp

    return {k: jnp.asarray(v) for k, v in batch.items()}


def train(config: Mapping, cache: TxCache, log: Callable[[str], None] = print) -> tuple:
    """Train ``config`` on ``cache``; return ``(model, params, history, seconds)``."""
    import jax
    import optax

    t0 = time.time()
    ctx = context_of(cache)
    model = make_model(config, ctx)
    if config["data"].get("preload"):
        keys = ("emb", "rna", "tis", "chx")
        shards = cache.rows("train").shard.unique()
        if len(shards) and all(cache.has(sh, "seq") for sh in shards):
            keys += ("seq",)
        cache.preload("train", keys)
        log(f"preloaded the train split ({time.time() - t0:.0f}s)")
    sampler = Sampler(cache, config)
    tcfg = config["train"]
    steps = int(tcfg["steps"])
    first = sampler.batch()
    params = model.init(jax.random.PRNGKey(int(config["seed"])), _device_batch(first))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        float(tcfg["lr"]),
        int(tcfg["warmup"]),
        max(steps, int(tcfg["warmup"]) + 1),
        float(tcfg["lr"]) * 0.01,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(float(tcfg["clip"])),
        optax.adamw(schedule, weight_decay=float(tcfg["weight_decay"])),
    )
    opt_state = opt.init(params)
    losses = config["losses"]

    def objective(params, batch):
        pred, aux = model.apply(params, batch)
        return total_loss(pred, batch, aux, losses)

    @jax.jit
    def step(params, opt_state, batch):
        (loss, parts), grads = jax.value_and_grad(objective, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, parts

    history = []
    every = max(1, int(tcfg.get("log_every", 100)))
    running = []
    batches = prefetched(sampler.batch, max(steps - 1, 0))
    for i in range(steps):
        batch = first if i == 0 else next(batches)
        params, opt_state, loss, parts = step(params, opt_state, _device_batch(batch))
        running.append(loss)
        if (i + 1) % every == 0 or i + 1 == steps:
            record = {
                "step": i + 1,
                "loss": float(np.mean([float(v) for v in running])),
                **{k: float(v) for k, v in parts.items()},
                "seconds": time.time() - t0,
            }
            if not np.isfinite(record["loss"]):
                raise FloatingPointError(f"non-finite loss at step {i + 1}: {record}")
            history.append(record)
            running = []
            log(json.dumps(record))
    return model, params, history, time.time() - t0


def bucket_length(length: int, min_bucket: int, max_length: int) -> int:
    """Smallest power-of-two multiple of ``min_bucket`` >= ``length``, capped at ``max_length``."""
    size = int(min_bucket)
    while size < length and size < max_length:
        size *= 2
    return min(size, int(max_length))


def predict_rows(
    apply: Callable,
    cache: TxCache,
    rows: pd.DataFrame,
    conditions: Sequence[str],
    min_bucket: int = 4096,
    max_length: int = 32768,
) -> dict[str, np.ndarray]:
    """Per transcript ``(len(conditions), L, 2)`` predictions, full length (chunks past max)."""
    out = {}
    for row in rows.itertuples():
        length = int(row.length)
        pred = np.zeros((len(conditions), length, 2), np.float32)
        for start in range(0, length, max_length):
            n = min(max_length, length - start)
            size = bucket_length(n, min_bucket, max_length)
            examples = [crop_example(cache, row, c, start, size) for c in conditions]
            batch = stack_examples(examples, [row] * len(conditions))
            pred[:, start : start + n] = np.asarray(apply(_device_batch(batch)))[:, :n]
        out[row.transcript_id] = pred
    return out


class _Span:
    """The part of :class:`scorecard.Unit` that :func:`scorecard.library_rows` reads."""

    def __init__(self, cds_start: int, cds_end: int):
        self.cds_range = (int(cds_start), int(cds_end))


def site_profiles(vector: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """``(n, 2 * FLANK + 1)`` values around each mRNA position (zero past the ends)."""
    padded = np.concatenate([np.zeros(FLANK), np.asarray(vector, float), np.zeros(FLANK)])
    return (
        np.stack([padded[p : p + 2 * FLANK + 1] for p in positions])
        if len(positions)
        else (np.zeros((0, 2 * FLANK + 1)))
    )


def peak_summary(profiles: np.ndarray) -> dict[str, float]:
    """Meta share at the start nt and per-site argmax at the start, over positive profiles."""
    keep = profiles.sum(1) > 0
    if not keep.any():
        return {"meta_share": float("nan"), "argmax0": float("nan"), "sites": 0}
    prof = profiles[keep]
    norm = prof / prof.sum(1, keepdims=True)
    return {
        "meta_share": float(norm.mean(0)[FLANK]),
        "argmax0": float((prof.argmax(1) == FLANK).mean()),
        "sites": int(keep.sum()),
    }


def start_classes(codons: Sequence[str]) -> np.ndarray:
    """Start class of each real call by its codon: ``atg`` or ``nc`` (any other codon)."""
    return np.array(["atg" if str(c).upper() == "ATG" else "nc" for c in codons], dtype=object)


def class_peak_summary(profiles: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """:func:`peak_summary` per start class, keys ``meta_share_<cls>`` / ``argmax0_<cls>``."""
    out = {}
    for cls in START_CLASSES:
        summary = peak_summary(profiles[classes == cls])
        out[f"meta_share_{cls}"] = summary["meta_share"]
        out[f"argmax0_{cls}"] = summary["argmax0"]
        out[f"sites_{cls}"] = summary["sites"]
    return out


def site_hits(profiles: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """Per site: profile positive and its argmax within ``tolerance`` nt of the centre."""
    if not len(profiles):
        return np.zeros(0, bool)
    return (profiles.sum(1) > 0) & (np.abs(profiles.argmax(1) - FLANK) <= tolerance)


def exon_regions(positions: Sequence[int], first_junction: int | None) -> np.ndarray:
    """``first`` for mRNA positions in the 5'-most exon (before the first junction), else later."""
    pos = np.asarray(positions, int)
    if first_junction is None:
        return np.full(len(pos), "first", dtype=object)
    return np.where(pos < int(first_junction), "first", "later").astype(object)


def rep_summary(
    pred: np.ndarray,
    rep1: np.ndarray,
    rep2: np.ndarray,
    classes: np.ndarray,
    regions: np.ndarray,
) -> dict[str, float]:
    """Reproducible-start hit rates of ``pred`` vs replicate 1, conditioned on replicate 2.

    Keys ``rep_{hit,ref,ratio,n}_<cls><region>`` (region ``""``, ``_first``, ``_later``) and
    ``rep_both_<cls>`` (P(pred hit | both replicates hit)).
    """
    hit_p, hit_1, hit_2 = site_hits(pred), site_hits(rep1), site_hits(rep2)
    out = {}
    for cls in START_CLASSES:
        for region in EXON_REGIONS:
            sel = (classes == cls) & hit_2
            if region:
                sel &= regions == region[1:]
            n = int(sel.sum())
            hit = float(hit_p[sel].mean()) if n else float("nan")
            ref = float(hit_1[sel].mean()) if n else float("nan")
            out[f"rep_hit_{cls}{region}"] = hit
            out[f"rep_ref_{cls}{region}"] = ref
            out[f"rep_ratio_{cls}{region}"] = hit / ref if n and ref > 0 else float("nan")
            out[f"rep_n_{cls}{region}"] = n
        both = (classes == cls) & hit_1 & hit_2
        out[f"rep_both_{cls}"] = float(hit_p[both].mean()) if both.any() else float("nan")
    return out


def score_predictions(
    cache: TxCache,
    sources: Mapping[str, Mapping[str, np.ndarray]],
    conditions: Sequence[str],
    heldout: Sequence[str] = (),
    min_reads: float = MIN_READS,
    min_frame_reads: float = MIN_FRAME_READS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scorecard rows and TIS peak rows of each source on the val transcripts.

    ``sources[name][transcript_id]`` is ``(len(cache.conditions), L, 2)`` TIS, CHX in cache
    condition order. Returns ``(scorecard long table, peak table)``.
    """
    val = cache.rows("val")
    rows, peaks = [], []
    for condition in conditions:
        k = cache.condition_index(condition)
        units = val[val[f"expressed_{condition}"].to_numpy(bool)]
        spans = [_Span(r.cds_start, r.cds_end) for r in units.itertuples()]
        reps = {lib: {} for lib in LIBRARIES}
        for row in units.itertuples():
            lo, hi = int(row.offset), int(row.offset) + int(row.length)
            for lib in LIBRARIES:
                arr = np.asarray(cache.array(row.shard, lib)[k, lo:hi], np.float64)
                reps[lib].setdefault("rep1", []).append(arr[:, 0])
                reps[lib].setdefault("rep2", []).append(arr[:, 1])
                reps[lib].setdefault("pooled", []).append(arr[:, 2])
        for c, lib in enumerate(LIBRARIES):
            vectors = {"rep1": reps[lib]["rep1"], "rep2": reps[lib]["rep2"]}
            for name, preds in sources.items():
                vectors[name] = [
                    np.asarray(preds[t][k, :, c], np.float64) for t in units.transcript_id
                ]
            for item in library_rows(spans, vectors, lib, min_reads, min_frame_reads):
                rows.append({"condition": condition, "heldout": condition in heldout, **item})
        if cache.sites is None:
            continue
        sites = cache.sites[cache.sites.condition == condition]
        sites = sites[sites.transcript_id.isin(set(units.transcript_id))]
        index = {t: i for i, t in enumerate(units.transcript_id)}
        first_junction = {
            r.transcript_id: (int(r.junctions[0]) if len(r.junctions) else None)
            for r in units.itertuples()
        }
        real, preds, classes = [], {name: [] for name in sources}, []
        rep1, rep2, regions = [], [], []
        for tid, group in sites.groupby("transcript_id", sort=False):
            pos = group.tx_pos.to_numpy(int)
            real.append(site_profiles(reps["tis"]["pooled"][index[tid]], pos))
            rep1.append(site_profiles(reps["tis"]["rep1"][index[tid]], pos))
            rep2.append(site_profiles(reps["tis"]["rep2"][index[tid]], pos))
            regions.append(exon_regions(pos, first_junction[tid]))
            classes.append(start_classes(group.codon))
            for name, src in sources.items():
                preds[name].append(site_profiles(src[tid][k, :, 0], pos))
        if not real:
            continue
        real_p = np.concatenate(real)
        keep = real_p.sum(1) > 0
        cls = np.concatenate(classes)[keep]
        r1, r2 = np.concatenate(rep1)[keep], np.concatenate(rep2)[keep]
        region = np.concatenate(regions)[keep]
        peaks.append(
            {
                "condition": condition,
                "source": "real",
                **peak_summary(real_p[keep]),
                **class_peak_summary(real_p[keep], cls),
                **rep_summary(r1, r1, r2, cls, region),
            }
        )
        for name, parts in preds.items():
            prof = np.concatenate(parts)[keep]
            peaks.append(
                {
                    "condition": condition,
                    "source": name,
                    **peak_summary(prof),
                    **class_peak_summary(prof, cls),
                    **rep_summary(prof, r1, r2, cls, region),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(peaks)


def ckpt_predictions(cache: TxCache) -> dict[str, np.ndarray] | None:
    """The checkpoint's own val predictions from ``ckpt.npy`` (None when not cached)."""
    val = cache.rows("val")
    if val.empty or not all(cache.has(s, "ckpt") for s in val.shard.unique()):
        return None
    out = {}
    for row in val.itertuples():
        lo = int(row.offset)
        out[row.transcript_id] = np.asarray(
            cache.array(row.shard, "ckpt")[:, lo : lo + int(row.length)], np.float32
        )
    return out


def result_columns(conditions: Sequence[str]) -> list[str]:
    """Wide results row: identity, timing, primary, then per condition x library x scale."""
    cols = [
        "config_hash",
        "name",
        "timestamp",
        "git_commit",
        "cache",
        "heldout",
        "steps",
        "final_loss",
        "train_seconds",
        "eval_seconds",
        "runtime_seconds",
        "primary",
    ]
    for c in conditions:
        for lib in LIBRARIES:
            cols += [f"{c}_{lib}_{scale}" for scale in SCALES]
            if lib == "chx":
                cols.append(f"{c}_chx_{FRAME}")
        cols += [f"{c}_tis_peak_meta", f"{c}_tis_peak_argmax0"]
    cols.append("primary2")
    for c in conditions:
        for suffix in ("", "_real"):
            cols += [
                f"{c}_tis_peak_{stat}_{cls}{suffix}"
                for stat in ("meta", "argmax0")
                for cls in START_CLASSES
            ]
    cols.append("primary3")
    for c in conditions:
        for cls in START_CLASSES:
            cols += [
                f"{c}_tis_rep_{stat}_{cls}{region}" for region in EXON_REGIONS for stat in REP_STATS
            ]
            cols.append(f"{c}_tis_rep_both_{cls}")
    return cols


def results_row(
    meta: Mapping,
    table: pd.DataFrame,
    peaks: pd.DataFrame,
    conditions: Sequence[str],
    source: str = "model",
) -> dict:
    """One wide results row (percent of replicate; ``frame`` and peaks as values) for ``source``.

    ``primary`` is the mean of the first held-out condition's TIS codon and nt percent.
    """
    row = {col: float("nan") for col in result_columns(conditions)}
    row.update({k: v for k, v in meta.items() if k in row})
    part = table[table.source == source] if len(table) else table
    for r in part.itertuples():
        key = f"{r.condition}_{r.library}_{r.scale}"
        if key in row:
            row[key] = float(r.value if r.scale == FRAME else r.percent)
    if len(peaks):
        for r in peaks[peaks.source == source].itertuples():
            row[f"{r.condition}_tis_peak_meta"] = r.meta_share
            row[f"{r.condition}_tis_peak_argmax0"] = r.argmax0
        for src, suffix in ((source, ""), ("real", "_real")):
            for r in peaks[peaks.source == src].to_dict(orient="records"):
                for cls in START_CLASSES:
                    for stat, key in (("meta", "meta_share"), ("argmax0", "argmax0")):
                        col = f"{r['condition']}_tis_peak_{stat}_{cls}{suffix}"
                        if col in row:
                            row[col] = float(r.get(f"{key}_{cls}", float("nan")))
        for r in peaks[peaks.source == source].to_dict(orient="records"):
            for key, value in r.items():
                col = f"{r['condition']}_tis_{key}"
                if key.startswith("rep_") and col in row:
                    row[col] = float(value)
    heldout = list(meta.get("heldout_list") or [])
    if heldout:
        values = [row.get(f"{heldout[0]}_tis_{s}", float("nan")) for s in PRIMARY_SCALES]
        row["primary"] = float(np.mean(values))
        row["primary2"] = primary2(row, heldout[0])
        row["primary3"] = primary3(row, heldout[0])
    return row


def primary2(row: Mapping, condition: str) -> float:
    """Phase-2 primary: mean of TIS nt % and near-cognate argmax-at-start % of real (cap 100)."""
    real = row.get(f"{condition}_tis_peak_argmax0_nc_real", float("nan"))
    pred = row.get(f"{condition}_tis_peak_argmax0_nc", float("nan"))
    nc = min(100.0, 100.0 * pred / real) if real and np.isfinite(real) else float("nan")
    return float(np.mean([row.get(f"{condition}_tis_nt", float("nan")), nc]))


def primary3(row: Mapping, condition: str) -> float:
    """Phase-3 primary: mean of TIS nt % and ``100 x min(1, rep_hit_nc / rep_ref_nc)``."""
    hit = row.get(f"{condition}_tis_rep_hit_nc", float("nan"))
    ref = row.get(f"{condition}_tis_rep_ref_nc", float("nan"))
    nc = 100.0 * min(1.0, hit / ref) if ref and np.isfinite(ref) else float("nan")
    return float(np.mean([row.get(f"{condition}_tis_nt", float("nan")), nc]))


def append_result(path: str | Path, row: Mapping, columns: Sequence[str]) -> None:
    """Append ``row`` to a TSV with header ``columns`` (created if absent; must match)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        if header != list(columns):
            raise ValueError(f"{path} has a different header; use a new results file")
        prefix = ""
    else:
        prefix = "\t".join(columns) + "\n"

    def fmt(value):
        if isinstance(value, float):
            return "nan" if not np.isfinite(value) else f"{value:.6g}"
        return str(value)

    with open(path, "a") as handle:
        handle.write(prefix + "\t".join(fmt(row.get(c, float("nan"))) for c in columns) + "\n")


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


EXPORT_MANIFEST = "predict_tracks.json"


def export_bigwigs(
    cache: TxCache,
    preds: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    out_dir: str | Path,
    tracks_dir: str | Path | None = None,
    split: str = "val",
    meta: Mapping | None = None,
) -> dict[str, dict]:
    """Write transcript-space predictions as genomic float bigWigs (predict_tracks layout).

    ``preds[transcript_id]`` is ``(len(cache.conditions), L, 2)`` sense TIS, CHX (as
    :func:`predict_rows`). Writes ``<out_dir>/<C>/{tis,chx}_pooled.{plus,minus}.bw`` for each
    of ``conditions``: the sense prediction at every exonic position of the ``split``'s MANE
    transcripts on that transcript's strand, minus strand in the Ribo-TISH convention
    (canonical + ``RIBOTISH_MINUS_SHIFT``); no entry (= 0) elsewhere. A genomic position shared
    by two transcripts on one strand keeps the first in index order. Chromosome sizes come from
    the real pooled tracks; ``<out_dir>/<C>/predict_tracks.json`` records provenance and totals.
    """
    import pyBigWig

    from tisiago.seq2func.predict_tracks import chrom_sizes_of, pooled_prefix, strand_shift
    from tisiago.seq2func.targets import STRAND_FILES
    from tisiago.seq2func.tracks import track_prefix
    from tisiago.seq2func.txcache import unit_of

    tracks_dir = Path(tracks_dir or cache.config.get("tracks_dir", "data/p6/tracks"))
    rows = cache.rows(split)
    missing = sorted(set(rows.transcript_id) - set(preds))
    if missing:
        raise ValueError(f"no predictions for {len(missing)} {split} transcripts")
    coords: dict[str, list] = {"+": [], "-": []}
    for i, row in enumerate(rows.itertuples()):
        unit = unit_of(row)
        coords[unit.strand].append((row.chrom, np.asarray(unit.positions, np.int64), i))
    manifests = {}
    for condition in conditions:
        k = cache.condition_index(condition)
        sizes = chrom_sizes_of(f"{track_prefix(tracks_dir, condition, 'tis', 'pooled')}.plus.bw")
        size_of = dict(sizes)
        files, totals = {}, {}
        for c, library in enumerate(LIBRARIES):
            prefix = pooled_prefix(out_dir, condition, library)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            totals[library] = 0.0
            for strand, suffix in STRAND_FILES.items():
                per_chrom: dict[str, tuple[list, list, list]] = {}
                for chrom, positions, order in coords[strand]:
                    values = np.asarray(preds[rows.transcript_id.iloc[order]][k, :, c], np.float64)
                    entry = per_chrom.setdefault(chrom, ([], [], []))
                    entry[0].append(positions + strand_shift(strand))
                    entry[1].append(values)
                    entry[2].append(np.full(len(positions), order))
                path = prefix.with_name(f"{prefix.name}.{suffix}.bw")
                files.setdefault(library, {})[strand] = path.name
                bw = pyBigWig.open(str(path), "w")
                bw.addHeader(sizes)
                try:
                    for chrom, _ in sizes:
                        if chrom not in per_chrom:
                            continue
                        pos = np.concatenate(per_chrom[chrom][0])
                        val = np.concatenate(per_chrom[chrom][1])
                        rank = np.concatenate(per_chrom[chrom][2])
                        sort = np.lexsort((rank, pos))
                        pos, val = pos[sort], val[sort]
                        first = np.ones(len(pos), bool)
                        first[1:] = pos[1:] != pos[:-1]
                        keep = first & (pos >= 0) & (pos < size_of[chrom])
                        if keep.any():
                            bw.addEntries(
                                chrom,
                                pos[keep].tolist(),
                                values=val[keep].tolist(),
                                span=1,
                            )
                            totals[library] += float(val[keep].sum())
                finally:
                    bw.close()
        manifest = {
            **(meta or {}),
            "condition": condition,
            "source": "fastloop",
            "mode": "float",
            "layout": "pooled",
            "split": split,
            "files": files,
            "cache": str(cache.path),
            "windows_dir": cache.config.get("windows_dir"),
            "tracks_dir": str(tracks_dir),
            "transcripts": int(len(rows)),
            "region": "MANE exons of the cached split (0 elsewhere)",
            "written_totals": totals,
        }
        _write_json(Path(out_dir) / condition / EXPORT_MANIFEST, manifest)
        manifests[condition] = manifest
    return manifests


def save_params(params, path: str | Path) -> None:
    """Haiku ``params`` as ``<module>/<name>`` arrays in one ``.npz``."""
    flat = {f"{m}/{n}": np.asarray(v) for m, sub in params.items() for n, v in sub.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)


def load_params(path: str | Path) -> dict:
    """Inverse of :func:`save_params`."""
    import jax.numpy as jnp

    out: dict[str, dict] = {}
    with np.load(path) as data:
        for key in data.files:
            module, name = key.rsplit("/", 1)
            out.setdefault(module, {})[name] = jnp.asarray(data[key])
    return out


def run_config(
    config: Mapping,
    cache: TxCache,
    results: str | Path,
    runs_dir: str | Path,
    log: Callable[[str], None] = print,
    export_dir: str | Path | None = None,
    export_conditions: Sequence[str] | None = None,
) -> dict:
    """Train, predict the val transcripts, score, append the results row, write the JSON.

    With ``export_dir``, also write ``<export_dir>/<name>/`` val bigWigs of
    ``export_conditions`` (default the held-out ones; :func:`export_bigwigs`) and
    ``params.npz`` + ``config.json``.
    """
    import jax

    t0 = time.time()
    digest = config_hash(config)
    model, params, history, train_seconds = train(config, cache, log)
    t1 = time.time()
    conditions = cache.conditions

    @jax.jit
    def apply(batch):
        return model.apply(params, batch)[0]

    ecfg = config["eval"]
    preds = predict_rows(
        apply, cache, cache.rows("val"), conditions, ecfg["min_bucket"], ecfg["max_length"]
    )
    sources = {"model": preds}
    ckpt = ckpt_predictions(cache)
    if ckpt is not None:
        sources["ckpt"] = ckpt
    table, peaks = score_predictions(cache, sources, conditions, config["heldout"])
    eval_seconds = time.time() - t1
    meta = {
        "config_hash": digest,
        "name": config["name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "cache": str(cache.path),
        "heldout": ",".join(config["heldout"]),
        "heldout_list": list(config["heldout"]),
        "steps": int(config["train"]["steps"]),
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "runtime_seconds": time.time() - t0,
    }
    columns = result_columns(conditions)
    row = results_row(meta, table, peaks, conditions)
    append_result(results, row, columns)
    record = {
        "meta": meta,
        "config": config,
        "history": history,
        "row": row,
        "reference_row": results_row(meta, table, peaks, conditions, "ckpt")
        if ckpt is not None
        else None,
        "scorecard": table.to_dict(orient="records"),
        "peaks": peaks.to_dict(orient="records"),
    }
    _write_json(Path(runs_dir) / f"{config['name']}-{digest}.json", record)
    if export_dir is not None:
        dest = Path(export_dir) / config["name"]
        save_params(params, dest / "params.npz")
        _write_json(dest / "config.json", {"config": config, "config_hash": digest})
        export_bigwigs(
            cache,
            preds,
            list(export_conditions or config["heldout"]),
            dest,
            meta={"name": config["name"], "config_hash": digest, "primary3": row["primary3"]},
        )
        log(f"exported {config['name']} val tracks to {dest}")
    log(
        f"{config['name']} ({digest}): primary {row['primary']:.1f}, "
        f"primary3 {row['primary3']:.1f}, "
        f"train {train_seconds:.0f}s, eval {eval_seconds:.0f}s"
    )
    return record


def reference_record(cache: TxCache, heldout: Sequence[str]) -> dict:
    """Fast-loop scores of the checkpoint's own predictions (``ckpt.npy``)."""
    ckpt = ckpt_predictions(cache)
    if ckpt is None:
        raise ValueError(f"{cache.path} has no ckpt.npy for its val shards")
    t0 = time.time()
    table, peaks = score_predictions(cache, {"ckpt": ckpt}, cache.conditions, heldout)
    meta = {
        "config_hash": "checkpoint",
        "name": f"ckpt:{cache.config.get('checkpoint')}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "cache": str(cache.path),
        "heldout": ",".join(heldout),
        "heldout_list": list(heldout),
        "eval_seconds": time.time() - t0,
    }
    row = results_row(meta, table, peaks, cache.conditions, "ckpt")
    return {
        "meta": meta,
        "row": row,
        "scorecard": table.to_dict(orient="records"),
        "peaks": peaks.to_dict(orient="records"),
    }


MAIN_RESULTS = Path("data/p6/fastloop/results.tsv")
MAIN_RUNS = Path("data/p6/fastloop/runs")
TEST_RESULTS = Path("data/p6/fastloop/results_test.tsv")
TEST_RUNS = Path("data/p6/fastloop/runs_test")


class SplitView:
    """A :class:`TxCache` whose ``rows("val")`` are ``split``'s rows (val scorer on ``split``)."""

    def __init__(self, cache: TxCache, split: str):
        """Wrap ``cache``; every other attribute is the cache's own."""
        self._cache, self.split = cache, split

    def rows(self, split: str) -> pd.DataFrame:
        """Index rows of ``split`` (``val`` -> the viewed split)."""
        return self._cache.rows(self.split if split == "val" else split)

    def __getattr__(self, name):
        """Every other attribute is the wrapped cache's."""
        return getattr(self._cache, name)


def load_export(export: str | Path) -> tuple[dict, dict]:
    """``(config, params)`` saved by :func:`run_config` under ``<export_dir>/<name>/``.

    The config is the exact training config (not re-merged into :data:`DEFAULTS`); its hash
    must equal the saved one.
    """
    export = Path(export)
    payload = json.loads((export / "config.json").read_text())
    config = payload["config"]
    resolve_config(config)  # registry check only
    if config_hash(config) != payload["config_hash"]:
        raise ValueError(f"{export}: config hash {config_hash(config)} != saved")
    return config, load_params(export / "params.npz")


def param_shapes(params) -> dict[str, tuple[int, ...]]:
    """``{"<module>/<name>": shape}`` of a Haiku parameter tree."""
    return {f"{m}/{n}": tuple(np.shape(v)) for m, sub in params.items() for n, v in sub.items()}


def rebuild_model(config: Mapping, cache: TxCache, params, split: str):
    """:func:`make_model` of ``config``; raises unless ``params`` has exactly its tree."""
    import jax

    model = make_model(config, context_of(cache))
    rows = cache.rows(split)
    if rows.empty:
        raise ValueError(f"{cache.path} has no {split} transcripts")
    row = next(rows.itertuples())
    ecfg = config["eval"]
    size = bucket_length(
        min(int(row.length), int(ecfg["max_length"])),
        int(ecfg["min_bucket"]),
        int(ecfg["max_length"]),
    )
    batch = stack_examples([crop_example(cache, row, cache.conditions[0], 0, size)], [row])
    expected = param_shapes(model.init(jax.random.PRNGKey(0), _device_batch(batch)))
    got = param_shapes(params)
    if expected != got:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        shapes = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
        raise ValueError(
            f"params do not match the config's model: missing {missing}, extra {extra}, "
            f"shape mismatch {shapes}"
        )
    return model


def check_pca(cache: TxCache, train_cache: str | Path | None) -> None:
    """Raise if ``cache``'s ``pca.npz`` differs from the training cache's (when that exists)."""
    if train_cache is None:
        return
    train_pca, own = Path(train_cache) / "pca.npz", cache.path / "pca.npz"
    if not train_pca.exists():
        return
    if not own.exists() or train_pca.read_bytes() != own.read_bytes():
        raise ValueError(f"{own} is not the training cache's {train_pca} (embedding inputs differ)")


def predict_export(
    export: str | Path,
    cache: TxCache,
    split: str = "test",
    results: str | Path = TEST_RESULTS,
    runs_dir: str | Path = TEST_RUNS,
    export_dir: str | Path | None = None,
    export_conditions: Sequence[str] | None = None,
    i_am_final: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Score a saved head on ``split`` of ``cache`` without training (module docstring)."""
    import jax

    if split not in ("val", "test"):
        raise ValueError(f"split must be val or test, not {split!r}")
    if split == "test":
        if not i_am_final:
            raise PermissionError("split test reads the held-out chromosomes; pass i_am_final")
        if cache.config.get("split") != "test":
            raise ValueError(f"{cache.path} is not a test cache (txcache --test)")
    results = Path(results)
    if results.name == MAIN_RESULTS.name or results.resolve() == MAIN_RESULTS.resolve():
        raise ValueError(f"{results}: predictions never go to the main results.tsv")
    t0 = time.time()
    config, params = load_export(export)
    digest = config_hash(config)
    check_pca(cache, config.get("cache"))
    model = rebuild_model(config, cache, params, split)

    @jax.jit
    def apply(batch):
        return model.apply(params, batch)[0]

    ecfg = config["eval"]
    conditions = cache.conditions
    preds = predict_rows(
        apply, cache, cache.rows(split), conditions, ecfg["min_bucket"], ecfg["max_length"]
    )
    view = SplitView(cache, split)
    sources = {"model": preds}
    ckpt = ckpt_predictions(view)
    if ckpt is not None:
        sources["ckpt"] = ckpt
    table, peaks = score_predictions(view, sources, conditions, config["heldout"])
    meta = {
        "config_hash": digest,
        "name": config["name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "cache": str(cache.path),
        "heldout": ",".join(config["heldout"]),
        "heldout_list": list(config["heldout"]),
        "steps": int(config["train"]["steps"]),
        "eval_seconds": time.time() - t0,
        "runtime_seconds": time.time() - t0,
        "split": split,
        "export": str(export),
        "train_cache": config.get("cache"),
    }
    columns = result_columns(conditions)
    row = results_row(meta, table, peaks, conditions)
    append_result(results, row, columns)
    record = {
        "meta": meta,
        "config": config,
        "row": row,
        "reference_row": results_row(meta, table, peaks, conditions, "ckpt")
        if ckpt is not None
        else None,
        "scorecard": table.to_dict(orient="records"),
        "peaks": peaks.to_dict(orient="records"),
    }
    _write_json(Path(runs_dir) / f"{config['name']}-{digest}-{split}.json", record)
    if export_dir is not None:
        dest = Path(export_dir) / config["name"]
        export_bigwigs(
            cache,
            preds,
            list(export_conditions or config["heldout"]),
            dest,
            split=split,
            meta={"name": config["name"], "config_hash": digest, "primary3": row["primary3"]},
        )
        log(f"exported {config['name']} {split} tracks to {dest}")
    log(f"{config['name']} ({digest}) on {split}: primary3 {row['primary3']:.1f}")
    return record


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path, help="YAML / JSON config files")
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help=f"default {MAIN_RESULTS} ({TEST_RESULTS} with --predict)",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=f"default {MAIN_RUNS} ({TEST_RUNS} with --predict)",
    )
    ap.add_argument("--cache", type=Path, default=None, help="override every config's cache")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (smoke)")
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="write <dir>/<name>/ val genomic float bigWigs + params (export_bigwigs)",
    )
    ap.add_argument(
        "--export-conditions",
        nargs="+",
        default=None,
        help="conditions to export (default: the config's held-out ones)",
    )
    ap.add_argument(
        "--reference",
        action="store_true",
        help="score the cache's checkpoint predictions (ckpt.npy) and write <runs-dir>/ckpt.json",
    )
    ap.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="<export_dir>/<name> (params.npz + config.json): score it on --split of --cache",
    )
    ap.add_argument("--split", choices=("val", "test"), default="test", help="with --predict")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    return ap


def main() -> None:
    """CLI: run each config in turn (one results row each)."""
    ap = build_parser()
    args = ap.parse_args()
    if args.predict is not None:
        if args.configs or args.reference:
            ap.error("--predict takes no config files and no --reference")
        if args.cache is None:
            ap.error("--predict needs --cache")
        if args.split == "test" and not args.i_am_final:
            ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
        predict_export(
            args.predict,
            TxCache(args.cache),
            args.split,
            args.results or TEST_RESULTS,
            args.runs_dir or TEST_RUNS,
            args.export_dir,
            args.export_conditions,
            i_am_final=args.i_am_final,
            log=lambda s: print(s, flush=True),
        )
        return
    args.results = args.results or MAIN_RESULTS
    args.runs_dir = args.runs_dir or MAIN_RUNS
    if not args.configs and not args.reference:
        ap.error("give config files or --reference")
    caches: dict[str, TxCache] = {}

    def cache_of(path) -> TxCache:
        key = str(path)
        if key not in caches:
            caches[key] = TxCache(path)
        return caches[key]

    if args.reference:
        cache = cache_of(args.cache or DEFAULTS["cache"])
        record = reference_record(cache, DEFAULTS["heldout"])
        _write_json(args.runs_dir / "ckpt_reference.json", record)
        print(json.dumps(record["row"], indent=2, default=str), flush=True)
    configs = [load_config(p) for p in args.configs]
    for config in configs:
        if args.cache:
            config["cache"] = str(args.cache)
        if args.steps is not None:
            config["train"]["steps"] = args.steps
        print(f"== {config['name']} ({config_hash(config)})", flush=True)
        run_config(
            config,
            cache_of(config["cache"]),
            args.results,
            args.runs_dir,
            log=lambda s: print(s, flush=True),
            export_dir=args.export_dir,
            export_conditions=args.export_conditions,
        )


if __name__ == "__main__":
    main()
