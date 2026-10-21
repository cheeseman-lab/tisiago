"""Custom 1-bp track heads on AlphaGenome and a JAX train step (alphagenome-ft)."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from alphagenome.models import dna_output
from alphagenome.models.dna_model import Organism
from alphagenome_ft import (
    CustomHead,
    CustomHeadConfig,
    CustomHeadType,
    create_model_with_heads,
    create_optimizer,
    register_custom_head,
)
from alphagenome_ft.custom_heads import create_registered_head, get_custom_head_config
from alphagenome_ft.optimizer_utils import is_trainable_head_path, parameter_path_to_str
from alphagenome_research.model import dna_model

from tisiago.seq2func.loss import poisson_multinomial
from tisiago.seq2func.orf_features import orf_features
from tisiago.seq2func.rna_features import onesided_features, rna_features, strand_log_ratios
from tisiago.seq2func.targets import modulated_channels
from tisiago.seq2func.tx_features import NUM_TX_CHANNELS

ORGANISM = Organism.HOMO_SAPIENS
CONV_WIDTH = 31
MOD_HIDDEN = 64
MOD_HIDDEN_CHX = 256
MOD_SCALE = 1.5
INIT_SEQ_LEN = 2**14
RNACOND_WIDTH = 4 + 2
LOCAL_BLOCK = 1024
CHX_CONTRAST_WIDTHS = (15, 45, 150)
CHX_CONTRAST_EPS = 0.01
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TIS_CONTEXT_DILATIONS = DILATIONS[:-1]
ATTN_BLOCK = 1024
CONV9_KERNEL = 9
ATTN_HEADS = 4
ATTN_DIM = 64
ATTN_LAYERS = 2
_SEQUENCE_FREE_INIT = False


@contextmanager
def sequence_free_init(enabled: bool = True):
    """Let sequence heads stand in a zero sequence while agft initialises parameters."""
    global _SEQUENCE_FREE_INIT
    previous, _SEQUENCE_FREE_INIT = _SEQUENCE_FREE_INIT, enabled
    try:
        yield
    finally:
        _SEQUENCE_FREE_INIT = previous


class TrackHead(CustomHead):
    """1-bp embeddings -> Linear(256) -> relu -> Linear(num_tracks) -> softplus (float32)."""

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        x = embeddings.get_sequence_embeddings(resolution=1)
        x = hk.Linear(256, name="hidden")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)
        return jax.nn.softplus(x.astype(jnp.float32))

    def loss(self, predictions, batch):
        """Masked Poisson-multinomial on ``batch["targets"]`` / ``batch["mask"]``.

        Optional static ``frame_weight`` / ``frame_channels`` / ``frame_block`` batch entries
        add the codon-phase term (absent: weight 0, no channels); ``local_weight`` /
        ``local_channels`` / ``local_block`` the per-block multinomial term (absent: off).
        """
        return poisson_multinomial(
            predictions,
            batch["targets"],
            batch["mask"],
            frame_weight=batch.get("frame_weight", 0.0),
            frame_channels=batch.get("frame_channels", ()),
            frame_block=batch.get("frame_block", 10),
            local_weight=batch.get("local_weight", 0.0),
            local_channels=batch.get("local_channels", ()),
            local_block=batch.get("local_block", LOCAL_BLOCK),
            local2_weight=batch.get("local2_weight", 0.0),
            local2_channels=batch.get("local2_channels", ()),
            local2_block=batch.get("local2_block", 32),
        )


class ConvTrackHead(TrackHead):
    """concat(1-bp embeddings, optional extra channels) -> Conv1D(256, 31) -> relu -> Conv1D(C, 1).

    Loaded as ``conv``, it runs under agft's forward, which calls ``head(embeddings,
    organism_index)`` without the sequence, so it sees embeddings only. ``kwargs["sequence"]`` is
    concatenated when a caller provides it (:class:`SeqConvTrackHead` passes one-hot + features).
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        x = embeddings.get_sequence_embeddings(resolution=1).astype(jnp.float32)
        sequence = kwargs.get("sequence")
        if sequence is not None:
            x = jnp.concatenate([x, sequence.astype(jnp.float32)], axis=-1)
        x = hk.Conv1D(256, kernel_shape=CONV_WIDTH, padding="SAME", name="hidden")(x)
        x = jax.nn.relu(x)
        x = hk.Conv1D(self._num_tracks, kernel_shape=1, name="output")(x)
        return jax.nn.softplus(x.astype(jnp.float32))


class SeqConvTrackHead(ConvTrackHead):
    """concat(1-bp embeddings, one-hot, open-frame features) -> ConvTrackHead's conv stack.

    Needs ``kwargs["sequence"]`` ``(B, S, 4)``: :func:`load_model` installs a forward that passes
    the input one-hot (see :func:`install_sequence_forward`). Only inside
    :func:`sequence_free_init` (agft's parameter init) does a zero sequence stand in.
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        sequence = kwargs.get("sequence")
        if sequence is None:
            if not _SEQUENCE_FREE_INIT:
                raise ValueError("seqconv head needs the input sequence (sequence=...)")
            x = embeddings.get_sequence_embeddings(resolution=1)
            sequence = jnp.zeros((*x.shape[:2], 4), jnp.float32)
        sequence = sequence.astype(jnp.float32)
        extra = jnp.concatenate([sequence, orf_features(sequence)], axis=-1)
        return ConvTrackHead.predict(self, embeddings, organism_index, sequence=extra)


def _head_sequence(embeddings, kwargs, kind: str, width: int, extra: str):
    """``kwargs["sequence"]`` ``(B, S, width)`` for a sequence head (zeros inside init)."""
    sequence = kwargs.get("sequence")
    if sequence is None:
        if not _SEQUENCE_FREE_INIT:
            raise ValueError(f"{kind} head needs the input sequence (sequence=...)")
        x = embeddings.get_sequence_embeddings(resolution=1)
        sequence = jnp.zeros((*x.shape[:2], width), jnp.float32)
    if sequence.shape[-1] != width:
        raise ValueError(
            f"{kind} head needs sequences with {width} channels ({extra}), got {sequence.shape[-1]}"
        )
    return sequence


def _tx_sequence(embeddings, kwargs, kind: str):
    """``kwargs["sequence"]`` ``(B, S, 4 + 20)`` for a transcript head (zeros inside init)."""
    extra = "one-hot + transcript features: iterate(..., transcripts=...)"
    return _head_sequence(embeddings, kwargs, kind, 4 + NUM_TX_CHANNELS, extra)


class SeqConvTxTrackHead(ConvTrackHead):
    """concat(1-bp embeddings, one-hot, spliced open-frame features) -> ConvTrackHead's stack.

    Needs ``kwargs["sequence"]`` ``(B, S, 4 + 20)``: one-hot plus the transcript-coordinate
    features of :mod:`tisiago.seq2func.tx_features`, built by the data loader (``iterate(...,
    transcripts=...)``); the backbone sees only the one-hot (:func:`make_sequence_forward`).
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        sequence = _tx_sequence(embeddings, kwargs, "seqconv_tx")
        return ConvTrackHead.predict(self, embeddings, organism_index, sequence=sequence)


def _modulation(embeddings, sequence, hidden: int, num_out: int, scale=MOD_SCALE):
    """``exp(scale * tanh(delta))`` from the ``mod_hidden`` / zero-init ``mod_output`` convs.

    ``scale``: a scalar or a static per-output array ``(num_out,)``.
    """
    x = embeddings.get_sequence_embeddings(resolution=1).astype(jnp.float32)
    x = jnp.concatenate([x, sequence.astype(jnp.float32)], axis=-1)
    x = hk.Conv1D(hidden, kernel_shape=CONV_WIDTH, padding="SAME", name="mod_hidden")(x)
    x = jax.nn.relu(x)
    zeros = hk.initializers.Constant(0.0)
    delta = hk.Conv1D(num_out, kernel_shape=1, w_init=zeros, b_init=zeros, name="mod_output")(x)
    return jnp.exp(scale * jnp.tanh(delta.astype(jnp.float32)))


class ResidTxTrackHead(TrackHead):
    """:class:`TrackHead`'s output times a bounded, zero-initialised sequence modulation.

    The factor is ``exp(MOD_SCALE * tanh(delta))``, bounded to ``[exp(-MOD_SCALE),
    exp(MOD_SCALE)]`` = [0.22, 4.5] so the modulation cannot blow up the loss.

    Base branch: ``TrackHead``'s ``hidden`` / ``output`` (same leaf paths, so an ``mlp`` run
    initialises it). Modulation: concat(1-bp embeddings, one-hot + spliced transcript features,
    as :class:`SeqConvTxTrackHead`) -> Conv1D(64, 31) -> relu -> Conv1D(num_tracks, 1), the
    last layer zero-initialised (``mod_hidden`` / ``mod_output``), so predictions equal the
    base head's at step 0.
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        sequence = _tx_sequence(embeddings, kwargs, "resid_tx")
        base = TrackHead.predict(self, embeddings, organism_index)
        return base * _modulation(embeddings, sequence, MOD_HIDDEN, self._num_tracks)


class ResidChxTrackHead(TrackHead):
    """:class:`ResidTxTrackHead` with the modulation on selected channels only (CHX by default).

    The channels come from ``metadata["mod_channels"]`` (static indices, set by
    :func:`register_track_head`); every other channel returns the base output exactly. The
    modulation is Conv1D(256, 31) -> relu -> Conv1D(len(mod_channels), 1), zero-initialised,
    with the same leaf names as :class:`ResidTxTrackHead`. Channel ``k``'s factor is bounded by
    ``metadata["mod_scales"][k]`` (default :data:`MOD_SCALE` for every channel).
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        channels = tuple(self._metadata.get("mod_channels") or ())
        if not channels or len(set(channels)) != len(channels):
            raise ValueError(
                "resid_chx head needs unique metadata['mod_channels'] (CHX channel indices)"
            )
        scales = self._metadata.get("mod_scales")
        scale = MOD_SCALE if scales is None else jnp.asarray(scales, jnp.float32)
        sequence = _tx_sequence(embeddings, kwargs, "resid_chx")
        base = TrackHead.predict(self, embeddings, organism_index)
        factor = _modulation(embeddings, sequence, MOD_HIDDEN_CHX, len(channels), scale)
        index = jnp.asarray(channels)
        return base.at[..., index].multiply(factor, unique_indices=True)


class RnaCondTrackHead(TrackHead):
    """concat(1-bp embeddings, RNA features) -> Linear(256) -> relu -> Linear(C) -> softplus.

    Needs ``kwargs["sequence"]`` ``(B, S, 6)``: the one-hot plus the condition's pooled RNA counts
    (plus, minus) scaled to a 5e7-read library, built by ``iterate(..., rna_conditioned=True)``;
    the backbone sees only the one-hot (:func:`make_sequence_forward`). The RNA enters through
    :func:`tisiago.seq2func.rna_features.rna_features` (10 channels, computed here so they follow
    augmentation). Outputs TIS+, TIS-, CHX+, CHX- (``targets.rnacond_layout``).
    """

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, num_tracks)``."""
        extra = "one-hot + RNA plus/minus: iterate(..., rna_conditioned=True)"
        sequence = _head_sequence(embeddings, kwargs, "rnacond", RNACOND_WIDTH, extra)
        x = embeddings.get_sequence_embeddings(resolution=1).astype(jnp.float32)
        x = jnp.concatenate([x, rna_features(sequence[..., 4:])], axis=-1)
        x = hk.Linear(256, name="hidden")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)
        return jax.nn.softplus(x.astype(jnp.float32))


def _block_softmax(rate, logits, block: int, offset: int):
    """``rate`` summed per block (grid starting at ``-offset``), spread by softmax(``logits``)."""
    batch, length, channels = rate.shape
    padded = -(-(length + offset) // block) * block
    pad = [(0, 0), (offset, padded - length - offset), (0, 0)]
    valid = jnp.pad(jnp.ones((1, length, 1), bool), pad)
    shape = (batch, padded // block, block, channels)
    rate = jnp.pad(rate, pad).reshape(shape)
    logits = jnp.where(valid, jnp.pad(logits, pad), -1e30).reshape(shape)
    out = rate.sum(axis=2, keepdims=True) * jax.nn.softmax(logits, axis=2)
    return out.reshape(batch, padded, channels)[:, offset : offset + length]


def local_softmax(rate, logits, block: int = LOCAL_BLOCK):
    """Factorised track: per-block amount ``sum(rate)`` times a within-block softmax of ``logits``.

    ``(B, S, C)`` each. Two grids (offsets 0 and ``block // 2``) are averaged to soften block
    edges; the output sums to ``rate``'s sum over the window, per channel.
    """
    rate, logits = rate.astype(jnp.float32), logits.astype(jnp.float32)
    first = _block_softmax(rate, logits, block, 0)
    return 0.5 * (first + _block_softmax(rate, logits, block, block // 2))


def _dilated_trunk(h, dilations=DILATIONS, channels: int = 128):
    """Residual dilated-conv stack (kernel 3, ``dilations``), zero-initialised branches."""
    zeros = hk.initializers.Constant(0.0)
    for i, rate in enumerate(dilations):
        y = hk.Conv1D(channels, kernel_shape=3, rate=rate, padding="SAME", name=f"dil{i}")(h)
        y = jax.nn.relu(y)
        h = h + hk.Conv1D(h.shape[-1], 1, w_init=zeros, b_init=zeros, name=f"dil{i}_out")(y)
    return h


def _block_attention(h, offset: int, name: str):
    """Multi-head self-attention within blocks of :data:`ATTN_BLOCK` (grid at ``-offset``).

    Learned per-head relative-position bias; padding keys are masked; zero-initialised output
    projection.
    """
    batch, length, width = h.shape
    block, heads = ATTN_BLOCK, ATTN_HEADS
    dim = ATTN_DIM // heads
    padded = -(-(length + offset) // block) * block
    pad = [(0, 0), (offset, padded - length - offset), (0, 0)]
    x = jnp.pad(h, pad)
    valid = jnp.pad(jnp.ones((length,), bool), [(offset, padded - length - offset)])
    valid = valid.reshape(padded // block, 1, 1, block)

    def project(tag):
        y = hk.Linear(ATTN_DIM, name=f"{name}_{tag}")(x)
        return y.reshape(batch, padded // block, block, heads, dim)

    q, k, v = project("q"), project("k"), project("v")
    scores = jnp.einsum("bnqhd,bnkhd->bnhqk", q, k) / math.sqrt(dim)
    table = hk.get_parameter(
        f"{name}_rel_bias", (heads, 2 * block - 1), init=hk.initializers.Constant(0.0)
    )
    rel = jnp.arange(block)[:, None] - jnp.arange(block)[None, :] + block - 1
    scores = scores + table[:, rel][None, None]
    scores = jnp.where(valid[None], scores, -1e30)
    att = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bnhqk,bnkhd->bnqhd", att, v).reshape(batch, padded, ATTN_DIM)
    zeros = hk.initializers.Constant(0.0)
    out = hk.Linear(width, w_init=zeros, b_init=zeros, name=f"{name}_out")(out)
    return out[:, offset : offset + length]


def _attention_trunk(h):
    """:data:`ATTN_LAYERS` pre-norm residual block-attention layers, alternate layers shifted."""
    for layer in range(ATTN_LAYERS):
        offset = (ATTN_BLOCK // 2) * (layer % 2)
        name = f"attn{layer}"

        def sublayer(x, offset=offset, name=name):
            y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"{name}_norm")(x)
            return _block_attention(y, offset, name)

        h = h + hk.remat(sublayer)(h)
    return h


class RnaCondFactTrackHead(TrackHead):
    """``rnacond`` with a factorised TIS output (and optional extras, set by subclasses).

    Trunk: concat(1-bp embeddings, RNA features[, one-sided RNA features]) -> Linear(256) ->
    relu [-> dilated-conv or block-attention stack]. CHX+/- = softplus(Linear(2) ``output``).
    TIS+/- = :func:`local_softmax` of a softplus rate and logits from ``tis_output`` (Linear(4));
    with ``CHX_CONTRAST`` its input is concat(trunk, downstream/upstream log ratios of the
    head's own, gradient-stopped CHX prediction) through ``tis_hidden`` (Linear(128), relu).
    The ``conv9`` trunk swaps ``hidden`` for two stacked Conv1D(256, 9) layers, and ``FACT_CHX``
    makes CHX factorised too (``output`` Linear(4), same blocks and grids as TIS).
    Outputs TIS+, TIS-, CHX+, CHX- (``targets.rnacond_layout``).
    """

    ONESIDED = False
    CHX_CONTRAST = False
    FACT_CHX = False
    TIS_CONTEXT = False
    TRUNK = "mlp"

    def predict(self, embeddings, organism_index, **kwargs):
        """Non-negative per-base track predictions ``(B, S, 4)``."""
        if self._num_tracks != 4:
            raise ValueError(f"factorised rnacond heads have 4 tracks, not {self._num_tracks}")
        extra = "one-hot + RNA plus/minus: iterate(..., rna_conditioned=True)"
        sequence = _head_sequence(embeddings, kwargs, "rnacond", RNACOND_WIDTH, extra)
        rna = sequence[..., 4:]
        feats = [rna_features(rna)]
        if self.ONESIDED:
            feats.append(onesided_features(rna))
        x = embeddings.get_sequence_embeddings(resolution=1).astype(jnp.float32)
        x = jnp.concatenate([x, *feats], axis=-1)
        if self.TRUNK == "conv9":
            h = jax.nn.relu(hk.Conv1D(256, CONV9_KERNEL, padding="SAME", name="hidden")(x))
            h = jax.nn.relu(hk.Conv1D(256, CONV9_KERNEL, padding="SAME", name="hidden2")(h))
        else:
            h = jax.nn.relu(hk.Linear(256, name="hidden")(x))
        if self.TRUNK == "dilated":
            h = _dilated_trunk(h)
        elif self.TRUNK == "attention":
            h = _attention_trunk(h)
        if self.FACT_CHX:
            c = hk.Linear(4, name="output")(h).astype(jnp.float32)
            chx = local_softmax(jax.nn.softplus(c[..., :2]), c[..., 2:])
        else:
            chx = jax.nn.softplus(hk.Linear(2, name="output")(h).astype(jnp.float32))
        tis_in = h
        if self.TIS_CONTEXT:
            ctx = jax.nn.relu(hk.Linear(256, name="tis_ctx")(h))
            tis_in = _dilated_trunk(ctx, TIS_CONTEXT_DILATIONS)
        if self.CHX_CONTRAST:
            ratios = strand_log_ratios(
                jax.lax.stop_gradient(chx), CHX_CONTRAST_WIDTHS, CHX_CONTRAST_EPS
            )
            tis_in = jnp.concatenate([h, ratios], axis=-1)
            tis_in = jax.nn.relu(hk.Linear(128, name="tis_hidden")(tis_in))
        t = hk.Linear(4, name="tis_output")(tis_in).astype(jnp.float32)
        tis = local_softmax(jax.nn.softplus(t[..., :2]), t[..., 2:])
        return jnp.concatenate([tis, chx], axis=-1)


class RnaCondFactOneSidedTrackHead(RnaCondFactTrackHead):
    """:class:`RnaCondFactTrackHead` + one-sided RNA features."""

    ONESIDED = True


class RnaCondFactChxTrackHead(RnaCondFactOneSidedTrackHead):
    """+ one-sided RNA features + the CHX-contrast input to the TIS output."""

    CHX_CONTRAST = True


class RnaCondFactDilatedTrackHead(RnaCondFactOneSidedTrackHead):
    """+ one-sided RNA features + a dilated-conv trunk (receptive field ~2 kb)."""

    TRUNK = "dilated"


class RnaCondFactAttentionTrackHead(RnaCondFactOneSidedTrackHead):
    """+ one-sided RNA features + a block-attention trunk (1 kb blocks, shifted second layer)."""

    TRUNK = "attention"


class RnaCondFactChxDilatedTrackHead(RnaCondFactChxTrackHead):
    """:class:`RnaCondFactChxTrackHead` with the dilated-conv trunk."""

    TRUNK = "dilated"


class RnaCondFactChxAttentionTrackHead(RnaCondFactChxTrackHead):
    """:class:`RnaCondFactChxTrackHead` with the block-attention trunk."""

    TRUNK = "attention"


class RnaCondFactConvFactChxTrackHead(RnaCondFactOneSidedTrackHead):
    """Fast-loop recipe: two stacked width-9 convs (256), factorised TIS and CHX, no contrast."""

    TRUNK = "conv9"
    FACT_CHX = True


class RnaCondFactConvFactChxDilTisTrackHead(RnaCondFactConvFactChxTrackHead):
    """``rnacond_fc2`` + a TIS-only residual dilated context (1..256, 128 channels) on h."""

    TIS_CONTEXT = True


HEAD_KINDS = {
    "mlp": TrackHead,
    "conv": ConvTrackHead,
    "seqconv": SeqConvTrackHead,
    "seqconv_tx": SeqConvTxTrackHead,
    "resid_tx": ResidTxTrackHead,
    "resid_chx": ResidChxTrackHead,
    "rnacond": RnaCondTrackHead,
    "rnacond_f": RnaCondFactTrackHead,
    "rnacond_fo": RnaCondFactOneSidedTrackHead,
    "rnacond_foc": RnaCondFactChxTrackHead,
    "rnacond_fod": RnaCondFactDilatedTrackHead,
    "rnacond_foa": RnaCondFactAttentionTrackHead,
    "rnacond_focd": RnaCondFactChxDilatedTrackHead,
    "rnacond_foca": RnaCondFactChxAttentionTrackHead,
    "rnacond_fc2": RnaCondFactConvFactChxTrackHead,
    "rnacond_fc2d": RnaCondFactConvFactChxDilTisTrackHead,
}
RNACOND_KINDS = frozenset(k for k in HEAD_KINDS if k == "rnacond" or k.startswith("rnacond_"))
SEQUENCE_KINDS = frozenset({"seqconv", "seqconv_tx", "resid_tx", "resid_chx"} | RNACOND_KINDS)
SEQUENCE_WIDTHS = {
    "seqconv_tx": 4 + NUM_TX_CHANNELS,
    "resid_tx": 4 + NUM_TX_CHANNELS,
    "resid_chx": 4 + NUM_TX_CHANNELS,
    **{kind: RNACOND_WIDTH for kind in RNACOND_KINDS},
}
_REGISTERED_KINDS: dict[str, str] = {}


def _check_mod_channels(kind: str, num_tracks: int, mod_channels) -> tuple[int, ...] | None:
    """``mod_channels`` as a tuple: required (unique, in range) for ``resid_chx``, else None."""
    if not issubclass(HEAD_KINDS[kind], ResidChxTrackHead):
        if mod_channels is not None:
            raise ValueError(f"mod_channels is only for resid_chx heads, not {kind!r}")
        return None
    channels = tuple(int(c) for c in mod_channels or ())
    if not channels or len(set(channels)) != len(channels):
        raise ValueError(f"{kind} needs unique, non-empty mod_channels, got {mod_channels!r}")
    if not all(0 <= c < num_tracks for c in channels):
        raise ValueError(f"mod_channels {channels} out of range for num_tracks={num_tracks}")
    return channels


def _check_mod_scales(channels, mod_scales) -> tuple[float, ...] | None:
    """Per-channel modulation bounds: positive and finite, default :data:`MOD_SCALE`."""
    if channels is None:
        if mod_scales is not None:
            raise ValueError("mod_scales is only for resid_chx heads")
        return None
    if mod_scales is None:
        return (MOD_SCALE,) * len(channels)
    scales = tuple(float(s) for s in mod_scales)
    if len(scales) != len(channels) or not all(math.isfinite(s) and s > 0 for s in scales):
        raise ValueError(
            f"mod_scales must be {len(channels)} positive finite values, got {mod_scales!r}"
        )
    return scales


def resid_chx_spec(
    layout, libraries=("chx",), scale_tis: float = MOD_SCALE
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """``(mod_channels, mod_scales)`` of a ``resid_chx`` head modulating ``libraries``.

    Every channel of ``libraries`` in ``layout`` (both strands); TIS channels are bounded by
    ``scale_tis``, the others by :data:`MOD_SCALE`.
    """
    channels = modulated_channels(layout, libraries)
    scales = tuple(float(scale_tis) if layout[c][1] == "tis" else MOD_SCALE for c in channels)
    return channels, scales


def register_track_head(
    name: str, num_tracks: int, kind: str = "mlp", mod_channels=None, mod_scales=None
) -> None:
    """Register a track head of ``kind``; idempotent for the same arguments.

    ``mod_channels`` (``resid_chx`` only, required there) are the modulated channel indices and
    ``mod_scales`` their factor bounds (default :data:`MOD_SCALE` each), both stored in the head
    config's metadata.
    """
    if kind not in HEAD_KINDS:
        raise ValueError(f"unknown head kind {kind!r}; expected one of {sorted(HEAD_KINDS)}")
    channels = _check_mod_channels(kind, num_tracks, mod_channels)
    scales = _check_mod_scales(channels, mod_scales)
    existing = get_custom_head_config(name)
    if existing is not None:
        if existing.num_tracks != num_tracks:
            raise ValueError(
                f"head {name!r} already registered with num_tracks={existing.num_tracks}, "
                f"not {num_tracks}"
            )
        previous = _REGISTERED_KINDS.get(name, kind)
        if previous != kind:
            raise ValueError(
                f"head {name!r} already registered with kind={previous!r}, not {kind!r}"
            )
        previous_channels = (existing.metadata or {}).get("mod_channels")
        if previous_channels != channels:
            raise ValueError(
                f"head {name!r} already registered with mod_channels={previous_channels}, "
                f"not {channels}"
            )
        previous_scales = (existing.metadata or {}).get("mod_scales")
        if (
            channels is not None
            and tuple(previous_scales or (MOD_SCALE,) * len(channels)) != scales
        ):
            raise ValueError(
                f"head {name!r} already registered with mod_scales={previous_scales}, not {scales}"
            )
        return
    register_custom_head(
        name,
        HEAD_KINDS[kind],
        CustomHeadConfig(
            type=CustomHeadType.GENOME_TRACKS,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=num_tracks,
            metadata=None if channels is None else {"mod_channels": channels, "mod_scales": scales},
        ),
    )
    _REGISTERED_KINDS[name] = kind


def load_model(checkpoint: str, head: str, *, detach_backbone: bool = True, device=None):
    """Load the local all-folds checkpoint with one registered custom head (no downloads).

    Heads of a :data:`SEQUENCE_KINDS` kind get :func:`install_sequence_forward`.
    """
    sequence_head = _REGISTERED_KINDS.get(head) in SEQUENCE_KINDS
    with sequence_free_init(sequence_head):
        model = create_model_with_heads(
            "all_folds",
            heads=[head],
            checkpoint_path=checkpoint,
            organism_settings={ORGANISM: dna_model.OrganismSettings()},
            detach_backbone=detach_backbone,
            device=device,
        )
    if sequence_head:
        model = install_sequence_forward(model, [head], detach_backbone=detach_backbone)
    return model


def _alphagenome_backbone():
    """``backbone(dna_sequence, organism_index) -> Embeddings`` as agft's standard forward."""
    from alphagenome_research.model import model as model_lib
    from alphagenome_research.model.metadata import metadata as metadata_lib

    metadata = {organism: metadata_lib.load(organism) for organism in dna_model.Organism}

    def backbone(dna_sequence, organism_index):
        return model_lib.AlphaGenome(metadata)(dna_sequence, organism_index)[1]

    return backbone


def _stop_gradient_embeddings(embeddings):
    values = {
        f.name: jax.lax.stop_gradient(getattr(embeddings, f.name))
        for f in fields(embeddings)
        if getattr(embeddings, f.name) is not None
    }
    return replace(embeddings, **values)


def make_sequence_forward(heads, *, detach_backbone: bool, backbone=None):
    """The standard agft 0.1.12 head forward, with the input one-hot passed as ``sequence=``.

    Same Haiku transform and module names as ``create_model_with_heads`` (``alphagenome/...``
    backbone, ``head/<name>/...`` heads), so it applies to that model's ``(params, state)``.
    ``backbone`` (tests) replaces AlphaGenome inside the transform. The backbone gets
    ``dna_sequence[..., :4]``; heads get every channel (``seqconv_tx`` adds 20, ``rnacond`` 2).
    """
    backbone = backbone or _alphagenome_backbone()
    num_organisms = len(dna_model.Organism)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        embeddings = backbone(dna_sequence[..., :4], organism_index)
        if detach_backbone:
            embeddings = _stop_gradient_embeddings(embeddings)
        predictions = {}
        with hk.name_scope("head"):
            for name in heads:
                head = create_registered_head(name, metadata=None, num_organisms=num_organisms)
                predictions[name] = head(embeddings, organism_index, sequence=dna_sequence)
        return predictions, embeddings

    return forward


def install_sequence_forward(model, heads, *, detach_backbone: bool, backbone=None):
    """Replace ``model._predict`` with :func:`make_sequence_forward` (agft is not modified).

    The forward's abstract ``(params, state)`` must match the model's leaf for leaf; the new
    ``_predict`` keeps agft's signature and returns ``{head: predictions}``.
    """
    from tisiago.seq2func.checkpoint import check_restorable, tree_leaf_shapes

    forward = make_sequence_forward(heads, detach_backbone=detach_backbone, backbone=backbone)
    width = max(SEQUENCE_WIDTHS.get(_REGISTERED_KINDS.get(h), 4) for h in heads)
    abstract = jax.eval_shape(
        forward.init,
        jax.random.PRNGKey(0),
        jnp.zeros((1, INIT_SEQ_LEN, width), jnp.bfloat16),
        jnp.zeros((1,), jnp.int32),
    )
    check_restorable(tree_leaf_shapes(abstract), tree_leaf_shapes((model._params, model._state)))

    @jax.jit
    def apply(params, state, rng, sequences, organism_indices):
        (predictions, _), _ = forward.apply(params, state, rng, sequences, organism_indices)
        return predictions

    def predict(
        params,
        state,
        sequences,
        organism_indices,
        *,
        requested_outputs=(),
        negative_strand_mask=None,
        strand_reindexing=None,
        rng=None,
    ):
        return apply(params, state, rng, sequences, organism_indices)

    model._predict = predict
    return model


def load_finetuned(
    checkpoint_dir: str,
    base_checkpoint: str,
    head: str,
    num_tracks: int,
    kind: str = "mlp",
    *,
    detach_backbone: bool = True,
    device=None,
    mod_channels=None,
    mod_scales=None,
):
    """Rebuild the training model and restore a ``save_checkpoint(save_full_model=True)`` dir.

    agft 0.1.12 ``load_checkpoint`` wraps full-model checkpoints with the base model's forward,
    which has no custom heads, so the model is rebuilt with :func:`load_model` instead and every
    saved ``(params, state)`` leaf is checked against it before the restore. ``mod_channels`` /
    ``mod_scales``: see :func:`register_track_head` (``resid_chx``: the run's channels/bounds).
    """
    import orbax.checkpoint as ocp

    from tisiago.seq2func.checkpoint import check_restorable, saved_leaf_shapes, tree_leaf_shapes

    path = Path(checkpoint_dir).resolve() / "checkpoint"
    register_track_head(
        head, num_tracks, kind=kind, mod_channels=mod_channels, mod_scales=mod_scales
    )
    model = load_model(base_checkpoint, head, detach_backbone=detach_backbone, device=device)
    template = (model._params, model._state)
    check_restorable(tree_leaf_shapes(template), saved_leaf_shapes(path))
    target = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), template
    )
    params, state = ocp.StandardCheckpointer().restore(str(path), target=target)
    model._params, model._state = params, state
    return model


def restore_init(model, checkpoint_dir: str, head: str, *, reuse_head: bool | str) -> int:
    """Overwrite ``model``'s parameters with a finished run's ``best/`` (two-stage init).

    Backbone parameters and state always come from the checkpoint; ``head``'s leaves per
    ``reuse_head`` (``"all"``, ``"matching"``, ``"none"``; see ``checkpoint.init_leaves``), the
    rest keeping their fresh initialisation. Only the copied leaves are read (Orbax partial
    restore), onto the template leaves' devices and dtypes.

    Returns:
        The number of leaves copied.
    """
    import orbax.checkpoint as ocp

    from tisiago.seq2func.checkpoint import init_leaves, saved_leaf_shapes, tree_leaf_shapes

    path = Path(checkpoint_dir).resolve() / "checkpoint"
    template = (model._params, model._state)
    load = init_leaves(
        tree_leaf_shapes(template), saved_leaf_shapes(path), head, reuse_head=reuse_head
    )
    target: tuple[dict, dict] = ({}, {})
    for part, module, name in load:
        x = template[int(part)][module][name]
        sds = jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None))
        target[int(part)].setdefault(module, {})[name] = sds
    restored = ocp.PyTreeCheckpointer().restore(
        str(path),
        args=ocp.args.PyTreeRestore(
            item=target,
            restore_args=ocp.checkpoint_utils.construct_restore_args(target),
            partial_restore=True,
        ),
    )
    trees = tuple({m: dict(leaves) for m, leaves in tree.items()} for tree in template)
    for part, module, name in load:
        trees[int(part)][module][name] = restored[int(part)][module][name]
    model._params, model._state = trees
    return len(load)


def make_predict(model, head: str):
    """Jitted ``predict(params, state, sequences (B, S, 4)) -> (B, S, C)`` for ``head``."""
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

    return predict


def label_backbone_params(params, head: str, *, mod: bool = False):
    """Label leaves under ``head/<head>/`` as ``"head"`` and every other leaf ``"backbone"``.

    ``mod=True`` labels the head's leaves in a module whose name starts with ``mod_`` ``"mod"``.
    """

    def label(path, _value):
        parts = parameter_path_to_str(path).split("/")
        if not is_trainable_head_path("/".join(parts), {head}):
            return "backbone"
        return "mod" if mod and len(parts) > 1 and parts[-2].startswith("mod_") else "head"

    return jax.tree_util.tree_map_with_path(label, params)


def make_schedule(
    learning_rate: float, total_steps: int | None, warmup_steps: int = 0, freeze_steps: int = 0
):
    """Constant ``learning_rate``, or warmup-cosine decaying to ``learning_rate / 100``.

    ``freeze_steps`` > 0 makes the rate 0 for the first ``freeze_steps`` updates and the
    unchanged schedule after them.
    """
    if total_steps is None:
        schedule = learning_rate
    else:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=learning_rate * 0.01,
        )
    if freeze_steps <= 0:
        return schedule
    base = schedule if callable(schedule) else optax.constant_schedule(schedule)
    return lambda count: jnp.where(count < freeze_steps, 0.0, base(count))


def make_optimizer(
    params,
    head: str,
    *,
    learning_rate: float,
    weight_decay: float,
    heads_only: bool = True,
    backbone_learning_rate: float | None = None,
    total_steps: int | None = None,
    warmup_steps: int = 0,
    clip_norm: float | None = None,
    freeze_backbone_steps: int = 0,
    mod_learning_rate: float | None = None,
) -> optax.GradientTransformation:
    """AdamW on the head only, or on head + backbone with a separate backbone learning rate.

    ``total_steps`` switches every learning rate to a warmup-cosine schedule; ``clip_norm``
    clips the global gradient norm before the optimizer; ``freeze_backbone_steps`` holds the
    backbone learning rate at 0 for that many updates (head warm-up). ``mod_learning_rate``
    gives the head's ``mod_*`` modules their own rate (same schedule shape and weight decay;
    ``None``: the head's).
    """
    head_lr = make_schedule(learning_rate, total_steps, warmup_steps)
    if mod_learning_rate is not None:
        if heads_only:
            backbone = optax.set_to_zero()
        else:
            if backbone_learning_rate is None:
                backbone_learning_rate = learning_rate / 100
            backbone = optax.adamw(
                make_schedule(
                    backbone_learning_rate, total_steps, warmup_steps, freeze_backbone_steps
                ),
                weight_decay=weight_decay,
            )
        inner = optax.multi_transform(
            {
                "head": optax.adamw(head_lr, weight_decay=weight_decay),
                "mod": optax.adamw(
                    make_schedule(mod_learning_rate, total_steps, warmup_steps),
                    weight_decay=weight_decay,
                ),
                "backbone": backbone,
            },
            label_backbone_params(params, head, mod=True),
        )
        if clip_norm is None:
            return inner
        return optax.chain(optax.clip_by_global_norm(clip_norm), inner)
    if heads_only:
        return create_optimizer(
            params,
            trainable_head_names=(head,),
            learning_rate=head_lr,
            weight_decay=weight_decay,
            heads_only=True,
            gradient_clip_global_norm=clip_norm,
        )
    if backbone_learning_rate is None:
        backbone_learning_rate = learning_rate / 100
    inner = optax.multi_transform(
        {
            "head": optax.adamw(head_lr, weight_decay=weight_decay),
            "backbone": optax.adamw(
                make_schedule(
                    backbone_learning_rate, total_steps, warmup_steps, freeze_backbone_steps
                ),
                weight_decay=weight_decay,
            ),
        },
        label_backbone_params(params, head),
    )
    if clip_norm is None:
        return inner
    return optax.chain(optax.clip_by_global_norm(clip_norm), inner)


def make_train_step(
    model,
    head: str,
    *,
    learning_rate: float,
    weight_decay: float,
    heads_only: bool = True,
    backbone_learning_rate: float | None = None,
    total_steps: int | None = None,
    warmup_steps: int = 0,
    clip_norm: float | None = None,
    frame_weight: float = 0.0,
    frame_channels: tuple[int, ...] = (),
    frame_block: int = 10,
    freeze_backbone_steps: int = 0,
    mod_learning_rate: float | None = None,
    local_weight: float = 0.0,
    local_channels: tuple[int, ...] = (),
    local_block: int = LOCAL_BLOCK,
    local2_weight: float = 0.0,
    local2_channels: tuple[int, ...] = (),
    local2_block: int = 32,
):
    """Jitted AdamW step ``step(params, state, opt_state, batch) -> (params, opt_state, aux)``.

    ``heads_only=False`` also trains the backbone; load the model with ``detach_backbone=False``.
    ``frame_weight`` adds the codon-phase term on ``frame_channels``, ``local_weight`` the
    per-block multinomial on ``local_channels``, ``local2_weight`` a second one
    (see ``loss.py``).
    ``freeze_backbone_steps``, ``mod_learning_rate``: see :func:`make_optimizer`.
    """
    opt = make_optimizer(
        model._params,
        head,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        heads_only=heads_only,
        backbone_learning_rate=backbone_learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        clip_norm=clip_norm,
        freeze_backbone_steps=freeze_backbone_steps,
        mod_learning_rate=mod_learning_rate,
    )
    opt_state = opt.init(model._params)
    loss_fn = model.create_loss_fn_for_head(head)
    frame = {
        "frame_weight": float(frame_weight),
        "frame_channels": tuple(frame_channels),
        "frame_block": int(frame_block),
    }
    if local_weight:
        frame |= {
            "local_weight": float(local_weight),
            "local_channels": tuple(local_channels),
            "local_block": int(local_block),
        }
    if local2_weight:
        frame |= {
            "local2_weight": float(local2_weight),
            "local2_channels": tuple(local2_channels),
            "local2_block": int(local2_block),
        }
    strand_reindexing = model._metadata[ORGANISM].strand_reindexing
    org_index = dna_model.convert_to_organism_index(ORGANISM)

    def total_loss(params, state, batch):
        batch_size = batch["sequences"].shape[0]
        org = jnp.full((batch_size,), org_index, jnp.int32)
        preds = model._predict(
            params,
            state,
            batch["sequences"],
            org,
            requested_outputs=(),
            negative_strand_mask=jnp.zeros((batch_size,), bool),
            strand_reindexing=strand_reindexing,
        )[head]
        out = loss_fn(
            preds,
            {
                "targets": batch["targets"],
                "mask": batch["mask"],
                "organism_index": org,
                **frame,
            },
        )
        return out["loss"], out

    @jax.jit
    def step(params, state, opt_state, batch):
        (_, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(params, state, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    return step, opt_state
