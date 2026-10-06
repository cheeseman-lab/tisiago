"""Custom 1-bp track head on AlphaGenome and a heads-only JAX train step (alphagenome-ft)."""

from __future__ import annotations

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
from alphagenome_research.model import dna_model

from tisiago.seq2func.loss import poisson_multinomial

ORGANISM = Organism.HOMO_SAPIENS


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
        """Masked Poisson-multinomial on ``batch["targets"]`` / ``batch["mask"]``."""
        return poisson_multinomial(predictions, batch["targets"], batch["mask"])


def register_track_head(name: str, num_tracks: int) -> None:
    """Register a :class:`TrackHead`; idempotent for the same ``(name, num_tracks)``."""
    existing = get_custom_head_config(name)
    if existing is not None:
        if existing.num_tracks != num_tracks:
            raise ValueError(
                f"head {name!r} already registered with num_tracks={existing.num_tracks}, "
                f"not {num_tracks}"
            )
        return
    register_custom_head(
        name,
        TrackHead,
        CustomHeadConfig(
            type=CustomHeadType.GENOME_TRACKS,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=num_tracks,
        ),
    )


def load_model(checkpoint: str, head: str, *, detach_backbone: bool = True):
    """Load the local all-folds checkpoint with one registered custom head (no downloads)."""
    return create_model_with_heads(
        "all_folds",
        heads=[head],
        checkpoint_path=checkpoint,
        organism_settings={ORGANISM: dna_model.OrganismSettings()},
        detach_backbone=detach_backbone,
    )


def make_train_step(
    model,
    head: str,
    *,
    learning_rate: float,
    weight_decay: float,
    heads_only: bool = True,
):
    """Jitted AdamW step ``step(params, state, opt_state, batch) -> (params, opt_state, aux)``."""
    opt = create_optimizer(
        model._params,
        trainable_head_names=(head,),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        heads_only=heads_only,
    )
    opt_state = opt.init(model._params)
    loss_fn = model.create_loss_fn_for_head(head)
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
            preds, {"targets": batch["targets"], "mask": batch["mask"], "organism_index": org}
        )
        return out["loss"], out

    @jax.jit
    def step(params, state, opt_state, batch):
        (_, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(params, state, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    return step, opt_state
