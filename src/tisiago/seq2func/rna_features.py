"""RNA-input features of the ``rnacond`` head, in JAX (jit-safe, static shapes).

The input is the condition's raw pooled RNA 5'-end counts ``(..., S, 2)`` (plus, minus). Per
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
    """``(..., S, 2)`` raw RNA counts (plus, minus) -> ``(..., S, 10)`` features (see module)."""
    import jax.numpy as jnp

    rna = jnp.maximum(rna.astype(jnp.float32), 0.0)
    blocks = []
    for strand in range(2):
        x = rna[..., strand : strand + 1]
        blocks.append(jnp.log1p(x))
        blocks.extend(jnp.log1p(box_mean(x, w)) for w in RNA_WIDTHS)
    return jnp.concatenate(blocks, axis=-1)
