"""Custom 1-bp track heads on AlphaGenome and a JAX train step (alphagenome-ft)."""

from __future__ import annotations

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

ORGANISM = Organism.HOMO_SAPIENS
CONV_WIDTH = 31
INIT_SEQ_LEN = 2**14
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


HEAD_KINDS = {"mlp": TrackHead, "conv": ConvTrackHead, "seqconv": SeqConvTrackHead}
SEQUENCE_KINDS = frozenset({"seqconv"})
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
    ``backbone`` (tests) replaces AlphaGenome inside the transform.
    """
    backbone = backbone or _alphagenome_backbone()
    num_organisms = len(dna_model.Organism)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        embeddings = backbone(dna_sequence, organism_index)
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
    abstract = jax.eval_shape(
        forward.init,
        jax.random.PRNGKey(0),
        jnp.zeros((1, INIT_SEQ_LEN, 4), jnp.bfloat16),
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


def restore_init(model, checkpoint_dir: str, head: str, *, reuse_head: bool) -> int:
    """Overwrite ``model``'s parameters with a finished run's ``best/`` (two-stage init).

    Backbone parameters and state always come from the checkpoint; ``head``'s leaves only when
    ``reuse_head`` (otherwise they keep their fresh initialisation). Only the copied leaves are
    read (Orbax partial restore), onto the template leaves' devices and dtypes.

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


def label_backbone_params(params, head: str):
    """Label leaves under ``head/<head>/`` as ``"head"`` and every other leaf ``"backbone"``."""

    def label(path, _value):
        return "head" if is_trainable_head_path(parameter_path_to_str(path), {head}) else "backbone"

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
) -> optax.GradientTransformation:
    """AdamW on the head only, or on head + backbone with a separate backbone learning rate.

    ``total_steps`` switches every learning rate to a warmup-cosine schedule; ``clip_norm``
    clips the global gradient norm before the optimizer; ``freeze_backbone_steps`` holds the
    backbone learning rate at 0 for that many updates (head warm-up).
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
):
    """Jitted AdamW step ``step(params, state, opt_state, batch) -> (params, opt_state, aux)``.

    ``heads_only=False`` also trains the backbone; load the model with ``detach_backbone=False``.
    ``frame_weight`` adds the codon-phase term on ``frame_channels`` (see ``loss.py``).
    ``freeze_backbone_steps``: see :func:`make_optimizer`.
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
