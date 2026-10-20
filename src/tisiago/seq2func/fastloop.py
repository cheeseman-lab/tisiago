"""Fast-loop trainer and scorer over a transcript cache (P6 fast loss/structure search).

Usage::

    python -m tisiago.seq2func.fastloop configs/p6fast/ref_rnacond.yaml [more.yaml ...]
    python -m tisiago.seq2func.fastloop --reference      # score the checkpoint's own predictions

A config (YAML or JSON) is merged into :data:`DEFAULTS` and composes registered components:

- ``inputs``: names from :data:`INPUTS` (``embedding``, ``embedding_white``, ``rna``,
  ``rna_sense``, ``rna_onesided``, ``position``), concatenated per position.
- ``model.trunk``: :data:`TRUNKS` (``mlp``, ``conv``, ``dilated``, ``attention``).
- ``model.tis`` / ``model.chx``: output parameterisations from :data:`OUTPUTS`: ``softplus``
  rate; ``factorised`` amount x softmax within blocks (``block`` int, two grids averaged, or
  ``"crop"``; ``temperature``); ``hazard`` scanning: ``P(start at i) = h_i prod_{j<i}(1 - h_j)``
  times the crop's summed amount. ``model.strand_heads`` gives each genomic strand its own
  output layer (the genomic head predicts both strands from one embedding); ``model.
  split_trunk`` gives CHX its own trunk (TIS keeps the first); ``model.chx_contrast`` feeds log ratios of the gradient-stopped CHX prediction to the TIS output.
- ``losses``: list of ``{kind, weight, ...}`` from :data:`LOSSES`: ``poisson`` (deviance of
  totals; ``scope`` sample / global), ``multinomial`` (``scope`` global = the batch, one
  condition, like a genomic window; sample = the transcript crop; local = blocks of ``block``),
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
real pooled signal within +-30 nt). The checkpoint's own predictions (``ckpt.npy``) are scored
alongside as ``ckpt``. One row per config is appended to ``--results`` (:data:`RESULT_COLUMNS`)
and the full record written to ``--runs-dir/<name>-<hash>.json``.
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
    if config["model"]["trunk"]["kind"] not in TRUNKS:
        raise ValueError(f"unknown trunk {config['model']['trunk']['kind']!r}")
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


def crop_example(
    cache: TxCache, row, condition: str, start: int, crop: int
) -> dict[str, np.ndarray]:
    """One transcript ``[start, start + crop)`` of ``condition``, zero-padded to ``crop``."""
    k = cache.condition_index(condition)
    n = max(0, min(crop, int(row.length) - start))
    lo = int(row.offset) + start
    out = {
        "emb": np.zeros((crop, cache.array(row.shard, "emb").shape[1]), np.float16),
        "rna": np.zeros((crop, 2), np.float32),
        "target": np.zeros((crop, 2), np.float32),
        "valid": np.zeros(crop, bool),
        "pos": np.zeros(crop, np.int32),
    }
    out["emb"][:n] = cache.array(row.shard, "emb")[lo : lo + n]
    out["rna"][:n] = cache.array(row.shard, "rna")[k, lo : lo + n]
    out["target"][:n, 0] = cache.array(row.shard, "tis")[k, lo : lo + n, 2]
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
            crop_example(self.cache, r, c, s, self.crop)
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
    for i, rate in enumerate(cfg.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))):
        y = hk.Conv1D(
            int(cfg.get("channels", 128)), 3, rate=int(rate), padding="SAME", name=f"dil{i}"
        )(h)
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


def make_model(config: Mapping, ctx: Mapping):
    """Haiku ``forward(batch) -> (rate (B, L, 2) [TIS, CHX], aux)`` of ``config``."""
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
        x = jnp.concatenate([INPUTS[n](batch, ctx) for n in config["inputs"]], axis=-1)
        x = jnp.where(valid[..., None], x.astype(jnp.float32), 0.0)
        h = trunk(x, valid, model["trunk"])
        h_chx = h
        if model.get("split_trunk"):
            with hk.name_scope("chx_trunk"):
                h_chx = trunk(x, valid, model["trunk"])
        n_chx, chx_fn = OUTPUTS[model["chx"]["kind"]]
        chx = chx_fn(head(h_chx, batch["strand"], n_chx, "chx_out"), valid, model["chx"])
        tis_in = h
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

        return local_multinomial(
            pred, target, _full_mask(pred, valid), channels, int(loss.get("block", 1024))
        )
    pred, target, mask = _masked(pred, target, valid, channels)
    pred, target, _ = _grouped(pred, target, mask, scope)
    tp, tt = pred.sum(axis=1), target.sum(axis=1)
    p_t = target / _floor(tt)[:, None, :]
    p_p = pred / _floor(tp)[:, None, :]
    return (_xlogy(p_t, p_t) - _xlogy(p_t, p_p)).sum(axis=1).mean()


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
        cache.preload("train", ("emb", "rna", "tis", "chx"))
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
        real, preds = [], {name: [] for name in sources}
        for tid, group in sites.groupby("transcript_id", sort=False):
            pos = group.tx_pos.to_numpy(int)
            real.append(site_profiles(reps["tis"]["pooled"][index[tid]], pos))
            for name, src in sources.items():
                preds[name].append(site_profiles(src[tid][k, :, 0], pos))
        if not real:
            continue
        real_p = np.concatenate(real)
        keep = real_p.sum(1) > 0
        peaks.append({"condition": condition, "source": "real", **peak_summary(real_p[keep])})
        for name, parts in preds.items():
            prof = np.concatenate(parts)[keep]
            peaks.append({"condition": condition, "source": name, **peak_summary(prof)})
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
    heldout = list(meta.get("heldout_list") or [])
    if heldout:
        values = [row.get(f"{heldout[0]}_tis_{s}", float("nan")) for s in PRIMARY_SCALES]
        row["primary"] = float(np.mean(values))
    return row


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


def run_config(
    config: Mapping,
    cache: TxCache,
    results: str | Path,
    runs_dir: str | Path,
    log: Callable[[str], None] = print,
) -> dict:
    """Train, predict the val transcripts, score, append the results row, write the JSON."""
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
    log(
        f"{config['name']} ({digest}): primary {row['primary']:.1f}, "
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


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path, help="YAML / JSON config files")
    ap.add_argument("--results", type=Path, default=Path("data/p6/fastloop/results.tsv"))
    ap.add_argument("--runs-dir", type=Path, default=Path("data/p6/fastloop/runs"))
    ap.add_argument("--cache", type=Path, default=None, help="override every config's cache")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (smoke)")
    ap.add_argument(
        "--reference",
        action="store_true",
        help="score the cache's checkpoint predictions (ckpt.npy) and write <runs-dir>/ckpt.json",
    )
    return ap


def main() -> None:
    """CLI: run each config in turn (one results row each)."""
    ap = build_parser()
    args = ap.parse_args()
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
        )


if __name__ == "__main__":
    main()
