"""RNA-input features of the ``rnacond`` head, in JAX (jit-safe, static shapes).

The input is the condition's pooled RNA 5'-end counts ``(..., S, 2)`` (plus, minus), scaled to a
5e7-read library (``data.rna_depth_scale``; read units, so ``log1p`` compresses). Per
strand: ``log1p(counts)`` and ``log1p(box_mean_w(counts))`` for ``w`` in :data:`RNA_WIDTHS`, a
centred, same-length box mean with zero padding past the window ends (always divided by ``w``).
Output ``(..., S, 10)``: the five plus-strand features, then the five minus-strand features, so a
reverse complement of the input reverses the features and swaps the two strand blocks.
"""

from __future__ import annotations

RNA_WIDTHS = (9, 33, 129, 513)
RNA_FEATURE_NAMES = tuple(
    f"{strand}_{name}"
    for strand in ("plus", "minus")
    for name in ("log1p", *(f"box{w}" for w in RNA_WIDTHS))
)
NUM_RNA_FEATURES = len(RNA_FEATURE_NAMES)


def box_mean(x, width: int):
    """Centred box mean of odd ``width`` along axis -2 of ``x`` (``(..., S, C)``), via cumsum."""
    import jax.numpy as jnp

    if width < 1 or width % 2 == 0:
        raise ValueError(f"box width must be odd and positive, got {width}")
    half = width // 2
    pad = [(0, 0)] * x.ndim
    pad[-2] = (half + 1, half)
    total = jnp.cumsum(jnp.pad(x.astype(jnp.float32), pad), axis=-2)
    window = total[..., width:, :] - total[..., :-width, :]
    return jnp.maximum(window, 0.0) / width


def rna_features(rna):
    """``(..., S, 2)`` depth-scaled RNA counts (plus, minus) -> ``(..., S, 10)`` features."""
    import jax.numpy as jnp

    rna = jnp.maximum(rna.astype(jnp.float32), 0.0)
    blocks = []
    for strand in range(2):
        x = rna[..., strand : strand + 1]
        blocks.append(jnp.log1p(x))
        blocks.extend(jnp.log1p(box_mean(x, w)) for w in RNA_WIDTHS)
    return jnp.concatenate(blocks, axis=-1)


ONESIDED_WIDTHS = (33, 129, 513)
ONESIDED_RISE_WIDTH = 129
ONESIDED_RISE_THRESHOLD = 1.0
ONESIDED_CAP = 10_000
ONESIDED_FEATURE_NAMES = tuple(
    f"{strand}_{name}"
    for strand in ("plus", "minus")
    for name in (*(f"contrast{w}" for w in ONESIDED_WIDTHS), "rise_distance")
)
NUM_ONESIDED_FEATURES = len(ONESIDED_FEATURE_NAMES)


def _onesided_plus(x):
    """Plus-strand one-sided features of ``(B, S)`` counts -> ``(B, S, len(widths) + 1)``.

    ``contrast_w = log1p(mean of x[i:i+w]) - log1p(mean of x[i-w:i])`` (zero past the ends);
    ``rise_distance = log1p(min(i - j, cap))`` with ``j`` the last position ``<= i`` whose
    ``contrast`` at :data:`ONESIDED_RISE_WIDTH` exceeds :data:`ONESIDED_RISE_THRESHOLD` (``cap``
    when there is none).
    """
    import jax
    import jax.numpy as jnp

    widths = (*ONESIDED_WIDTHS, ONESIDED_RISE_WIDTH)
    pad = max(widths)
    n = x.shape[-1]
    total = jnp.cumsum(jnp.pad(x, [(0, 0), (pad + 1, pad)]), axis=-1)  # total[pad + k] = sum x[:k]

    def window(lo):  # sum of x[i + lo : i + lo + w] for every i, as a function of w
        return (
            lambda w: total[..., pad + lo + w : pad + lo + w + n]
            - total[..., pad + lo : pad + lo + n]
        )

    def contrast(w):
        down = jnp.maximum(window(0)(w), 0.0) / w
        up = jnp.maximum(window(-w)(w), 0.0) / w
        return jnp.log1p(down) - jnp.log1p(up)

    feats = [contrast(w) for w in ONESIDED_WIDTHS]
    rise = jax.lax.stop_gradient(contrast(ONESIDED_RISE_WIDTH))
    index = jnp.arange(n, dtype=jnp.float32)
    marked = jnp.where(rise > ONESIDED_RISE_THRESHOLD, index, -jnp.inf)
    last = jax.lax.cummax(marked, axis=marked.ndim - 1)
    distance = jnp.minimum(index - last, float(ONESIDED_CAP))
    feats.append(jnp.log1p(distance))
    return jnp.stack(feats, axis=-1)


def onesided_features(rna):
    """``(B, S, 2)`` depth-scaled RNA counts -> ``(B, S, 8)`` strand-aware one-sided features.

    Per strand (plus block, then minus block): upstream/downstream contrasts at
    :data:`ONESIDED_WIDTHS` and the distance from the last 5' coverage rise, with "upstream"
    meaning lower coordinates on the plus strand and higher on the minus strand (computed on the
    reversed track), so a reverse complement reverses the features and swaps the strand blocks.
    """
    import jax.numpy as jnp

    rna = jnp.maximum(rna.astype(jnp.float32), 0.0)
    plus = _onesided_plus(rna[..., 0])
    minus = _onesided_plus(rna[..., ::-1, 1])[..., ::-1, :]
    return jnp.concatenate([plus, minus], axis=-1)


def strand_log_ratios(x, widths, eps: float):
    """``(B, S, 2)`` non-negative tracks -> ``(B, S, 2 * len(widths))`` downstream/upstream ratios.

    Per strand (plus block, then minus block) and width ``w``: ``log((mean of the w positions
    from i downstream + eps) / (mean of the w positions before i upstream + eps))``, upstream
    being lower coordinates on the plus strand and higher on the minus strand; a reverse
    complement reverses the output and swaps the strand blocks (as :func:`onesided_features`).
    """
    import jax.numpy as jnp

    x = jnp.maximum(x.astype(jnp.float32), 0.0)
    pad = max(widths)

    def plus(v):
        n = v.shape[-1]
        total = jnp.cumsum(jnp.pad(v, [(0, 0), (pad + 1, pad)]), axis=-1)
        out = []
        for w in widths:
            down = total[..., pad + w : pad + w + n] - total[..., pad : pad + n]
            up = total[..., pad : pad + n] - total[..., pad - w : pad - w + n]
            down = jnp.maximum(down, 0.0) / w + eps
            out.append(jnp.log(down / (jnp.maximum(up, 0.0) / w + eps)))
        return jnp.stack(out, axis=-1)

    return jnp.concatenate([plus(x[..., 0]), plus(x[..., ::-1, 1])[..., ::-1, :]], axis=-1)
