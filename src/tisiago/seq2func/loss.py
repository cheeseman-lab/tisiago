"""Masked Poisson-multinomial loss in deviance form (0 at a perfect fit)."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

EPS = 1e-7


def _xlogy(x, y):
    return jnp.where(x > 0, x * jnp.log(jnp.maximum(y, EPS)), 0.0)


def codon_phase_multinomial(pred, target, mask, channels: Sequence[int], block: int = 10):
    """Counts-weighted KL of local 3-nt phase composition on ``channels``; annotation-free.

    Positions are trimmed to a multiple of ``3 * block`` and phase is position mod 3 on the
    window's own grid. Per block of ``block`` codons with observed counts ``n > 0`` and no masked
    position, the term is ``n * KL(observed phase distribution || predicted)``; the sum is divided
    by the counts of those blocks. ``block=1`` compares single triplets. 0 when nothing counts.
    """
    channels = tuple(int(c) for c in channels)
    length = (pred.shape[1] // (3 * block)) * 3 * block
    if not channels or length == 0:
        return jnp.zeros((), jnp.float32)
    idx = jnp.asarray(channels)
    shape = (pred.shape[0], length // (3 * block), block, 3, len(channels))
    pred = pred[:, :length][..., idx].astype(jnp.float32).reshape(shape).sum(axis=2)
    target = jnp.maximum(target[:, :length][..., idx].astype(jnp.float32), 0.0)
    target = target.reshape(shape).sum(axis=2)
    keep = mask[:, :length][..., idx].reshape(shape).all(axis=(2, 3))
    n_target = target.sum(axis=2)
    keep = keep & (n_target > 0)
    p_target = target / jnp.maximum(n_target[:, :, None, :], EPS)
    p_pred = pred / jnp.maximum(pred.sum(axis=2)[:, :, None, :], EPS)
    kl = (_xlogy(p_target, p_target) - _xlogy(p_target, p_pred)).sum(axis=2)
    total = jnp.where(keep, n_target, 0.0).sum()
    weighted = jnp.where(keep, n_target * kl, 0.0).sum()
    return jnp.where(total > 0, weighted / jnp.maximum(total, EPS), 0.0)


def local_multinomial(pred, target, mask, channels: Sequence[int], block: int = 1024):
    """Counts-weighted multinomial KL of the profile within blocks of ``block`` positions.

    On ``channels`` only, positions trimmed to a multiple of ``block`` on the window's own grid;
    masked positions are zeroed in both (as :func:`poisson_multinomial`). Per block and channel
    with observed counts ``n > 0`` the term is ``n * KL(observed || predicted)``; the sum is
    divided by the counts of those blocks. 0 when nothing counts.
    """
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
    p_target = target / jnp.maximum(n_target[:, :, None, :], EPS)
    p_pred = pred / jnp.maximum(pred.sum(axis=2)[:, :, None, :], EPS)
    kl = (_xlogy(p_target, p_target) - _xlogy(p_target, p_pred)).sum(axis=2)
    total = n_target.sum()
    weighted = jnp.where(n_target > 0, n_target * kl, 0.0).sum()
    return jnp.where(total > 0, weighted / jnp.maximum(total, EPS), 0.0)


def poisson_multinomial(
    pred,
    target,
    mask,
    positional_weight: float = 5.0,
    frame_weight: float = 0.0,
    frame_channels: Sequence[int] = (),
    frame_block: int = 10,
    local_weight: float = 0.0,
    local_channels: Sequence[int] = (),
    local_block: int = 1024,
) -> dict:
    """Poisson deviance on window totals plus weighted multinomial KL on the profile.

    ``"frame"`` is :func:`codon_phase_multinomial` on ``frame_channels``; it enters ``"loss"`` as
    ``frame_weight * frame`` only when ``frame_weight`` is non-zero. ``"local"`` is
    :func:`local_multinomial` on ``local_channels`` (0 unless ``local_weight``), entering as
    ``local_weight * local``.
    """
    frame = codon_phase_multinomial(pred, target, mask, frame_channels, frame_block)
    local = jnp.zeros((), jnp.float32)
    if local_weight:
        local = local_multinomial(pred, target, mask, local_channels, local_block)
    pred = jnp.where(mask, pred.astype(jnp.float32), 0.0)
    target = jnp.where(mask, jnp.maximum(target.astype(jnp.float32), 0.0), 0.0)
    total_pred = pred.sum(axis=1)
    total_target = target.sum(axis=1)
    poisson = 2.0 * (
        total_pred
        - total_target
        + _xlogy(total_target, total_target)
        - _xlogy(total_target, total_pred)
    )
    p_target = target / jnp.maximum(total_target[:, None, :], EPS)
    p_pred = pred / jnp.maximum(total_pred[:, None, :], EPS)
    kl = (_xlogy(p_target, p_target) - _xlogy(p_target, p_pred)).sum(axis=1)
    poisson, kl = poisson.mean(), kl.mean()
    loss = poisson + positional_weight * kl
    if frame_weight:
        loss = loss + frame_weight * frame
    if local_weight:
        loss = loss + local_weight * local
    return {"loss": loss, "poisson": poisson, "multinomial": kl, "frame": frame, "local": local}
