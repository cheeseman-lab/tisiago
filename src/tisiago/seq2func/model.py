"""Custom 1-bp track heads on AlphaGenome and a JAX train step (alphagenome-ft)."""

from __future__ import annotations

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
from alphagenome_ft.custom_heads import get_custom_head_config
from alphagenome_ft.optimizer_utils import is_trainable_head_path, parameter_path_to_str
from alphagenome_research.model import dna_model

from tisiago.seq2func.loss import poisson_multinomial

ORGANISM = Organism.HOMO_SAPIENS
CONV_WIDTH = 31


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
        add the codon-phase term (absent: weight 0, no channels).
        """
        return poisson_multinomial(
            predictions,
            batch["targets"],
            batch["mask"],
            frame_weight=batch.get("frame_weight", 0.0),
            frame_channels=batch.get("frame_channels", ()),
            frame_block=batch.get("frame_block", 10),
        )


class ConvTrackHead(TrackHead):
    """concat(1-bp embeddings, optional one-hot) -> Conv1D(256, 31) -> relu -> Conv1D(C, 1).

    agft 0.1.12 calls custom heads as ``head(embeddings, organism_index)`` and does not forward
    the input sequence, so in practice the head sees embeddings only; ``kwargs["sequence"]`` is
    concatenated when a caller provides it.
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


HEAD_KINDS = {"mlp": TrackHead, "conv": ConvTrackHead}
_REGISTERED_KINDS: dict[str, str] = {}


def register_track_head(name: str, num_tracks: int, kind: str = "mlp") -> None:
    """Register a track head of ``kind``; idempotent for the same ``(name, num_tracks, kind)``."""
    if kind not in HEAD_KINDS:
        raise ValueError(f"unknown head kind {kind!r}; expected one of {sorted(HEAD_KINDS)}")
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
        return
    register_custom_head(
        name,
        HEAD_KINDS[kind],
        CustomHeadConfig(
            type=CustomHeadType.GENOME_TRACKS,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=num_tracks,
        ),
    )
    _REGISTERED_KINDS[name] = kind


def load_model(checkpoint: str, head: str, *, detach_backbone: bool = True, device=None):
    """Load the local all-folds checkpoint with one registered custom head (no downloads)."""
    return create_model_with_heads(
        "all_folds",
        heads=[head],
        checkpoint_path=checkpoint,
        organism_settings={ORGANISM: dna_model.OrganismSettings()},
        detach_backbone=detach_backbone,
        device=device,
    )


def load_finetuned(
    checkpoint_dir: str,
    base_checkpoint: str,
    head: str,
    num_tracks: int,
    kind: str = "mlp",
    *,
    detach_backbone: bool = True,
    device=None,
):
    """Rebuild the training model and restore a ``save_checkpoint(save_full_model=True)`` dir.

    agft 0.1.12 ``load_checkpoint`` wraps full-model checkpoints with the base model's forward,
    which has no custom heads, so the model is rebuilt with :func:`load_model` instead and every
    saved ``(params, state)`` leaf is checked against it before the restore.
    """
    import orbax.checkpoint as ocp

    from tisiago.seq2func.checkpoint import check_restorable, saved_leaf_shapes, tree_leaf_shapes

    path = Path(checkpoint_dir).resolve() / "checkpoint"
    register_track_head(head, num_tracks, kind=kind)
    model = load_model(base_checkpoint, head, detach_backbone=detach_backbone, device=device)
    template = (model._params, model._state)
    check_restorable(tree_leaf_shapes(template), saved_leaf_shapes(path))
    target = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), template
    )
    params, state = ocp.StandardCheckpointer().restore(str(path), target=target)
    model._params, model._state = params, state
    return model


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


def label_backbone_params(params, head: str):
    """Label leaves under ``head/<head>/`` as ``"head"`` and every other leaf ``"backbone"``."""

    def label(path, _value):
        return "head" if is_trainable_head_path(parameter_path_to_str(path), {head}) else "backbone"

    return jax.tree_util.tree_map_with_path(label, params)


def make_schedule(learning_rate: float, total_steps: int | None, warmup_steps: int = 0):
    """Constant ``learning_rate``, or warmup-cosine decaying to ``learning_rate / 100``."""
    if total_steps is None:
        return learning_rate
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )


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
) -> optax.GradientTransformation:
    """AdamW on the head only, or on head + backbone with a separate backbone learning rate.

    ``total_steps`` switches every learning rate to a warmup-cosine schedule; ``clip_norm``
    clips the global gradient norm before the optimizer.
    """
    head_lr = make_schedule(learning_rate, total_steps, warmup_steps)
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
                make_schedule(backbone_learning_rate, total_steps, warmup_steps),
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
):
    """Jitted AdamW step ``step(params, state, opt_state, batch) -> (params, opt_state, aux)``.

    ``heads_only=False`` also trains the backbone; load the model with ``detach_backbone=False``.
    ``frame_weight`` adds the codon-phase term on ``frame_channels`` (see ``loss.py``).
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
    )
    opt_state = opt.init(model._params)
    loss_fn = model.create_loss_fn_for_head(head)
    frame = {
        "frame_weight": float(frame_weight),
        "frame_channels": tuple(frame_channels),
        "frame_block": int(frame_block),
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
