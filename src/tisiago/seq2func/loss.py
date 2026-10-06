"""Masked Poisson-multinomial loss in deviance form (0 at a perfect fit)."""

from __future__ import annotations

import jax.numpy as jnp

EPS = 1e-7


def _xlogy(x, y):
    return jnp.where(x > 0, x * jnp.log(jnp.maximum(y, EPS)), 0.0)


def poisson_multinomial(pred, target, mask, positional_weight: float = 5.0) -> dict:
    """Poisson deviance on window totals plus weighted multinomial KL on the profile."""
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
    return {"loss": poisson + positional_weight * kl, "poisson": poisson, "multinomial": kl}
