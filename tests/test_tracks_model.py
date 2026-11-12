"""CPU-safe tests for the track head wiring (no checkpoint load)."""

import numpy as np
import pytest

pytest.importorskip("alphagenome_ft")
import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tisiago.tracks import model as model_mod  # noqa: E402
from tisiago.tracks.model import (  # noqa: E402
    RnaCondFactConvFactChxTrackHead,
    RnaCondFactOneSidedTrackHead,
    RnaCondFactTrackHead,
    TrackHead,
    label_backbone_params,
    make_optimizer,
    register_track_head,
)

# Only rnacond_fc2d is registered for training. Its base classes stay registered here as
# test-only kinds: "mlp" (TrackHead) is a cheap head for the shared training machinery, and
# the factorised bases pin the shapes and behaviour fc2d inherits.
TEST_KINDS = {
    "mlp": TrackHead,
    "rnacond_f": RnaCondFactTrackHead,
    "rnacond_fo": RnaCondFactOneSidedTrackHead,
    "rnacond_fc2": RnaCondFactConvFactChxTrackHead,
}


@pytest.fixture(autouse=True)
def test_kinds(monkeypatch):
    for kind, cls in TEST_KINDS.items():
        monkeypatch.setitem(model_mod.HEAD_KINDS, kind, cls)
        if kind != "mlp":
            monkeypatch.setitem(model_mod.SEQUENCE_WIDTHS, kind, model_mod.RNACOND_WIDTH)
    sequence = model_mod.SEQUENCE_KINDS | (set(TEST_KINDS) - {"mlp"})
    monkeypatch.setattr(model_mod, "SEQUENCE_KINDS", frozenset(sequence))


def test_head_loss_is_zero_at_target():
    def forward(pred, target):
        head = TrackHead(name="t", output_type=None, num_tracks=2)
        return head.loss(pred, {"targets": target, "mask": jnp.ones(target.shape, bool)})

    fn = hk.without_apply_rng(hk.transform(forward))
    y = jnp.asarray(np.random.default_rng(0).poisson(3, (1, 16, 2)).astype(np.float32))
    out = fn.apply(fn.init(None, y, y), y, y)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_head_loss_reads_frame_settings_from_batch():
    from tisiago.tracks.loss import poisson_multinomial

    def forward(pred, target, **frame):
        head = TrackHead(name="t", output_type=None, num_tracks=2)
        return head.loss(pred, {"targets": target, "mask": jnp.ones(target.shape, bool), **frame})

    fn = hk.without_apply_rng(hk.transform(forward))
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.poisson(3, (1, 30, 2)).astype(np.float32))
    pred = jnp.asarray(rng.uniform(1, 4, (1, 30, 2)).astype(np.float32))
    params = fn.init(None, pred, y)
    plain = fn.apply(params, pred, y)
    assert float(plain["frame"]) == 0.0
    frame = {"frame_weight": 2.0, "frame_channels": (1,), "frame_block": 5}
    out = fn.apply(params, pred, y, **frame)
    mask = jnp.ones(y.shape, bool)
    expected = poisson_multinomial(pred, y, mask, **frame)
    assert float(out["frame"]) == pytest.approx(float(expected["frame"]))
    assert float(out["frame"]) > 0
    assert float(out["loss"]) == pytest.approx(float(plain["loss"]) + 2.0 * float(out["frame"]))


def test_register_is_idempotent():
    register_track_head("p6_test_head", 6)
    register_track_head("p6_test_head", 6)


def test_register_rejects_different_num_tracks():
    register_track_head("p6_test_head_mismatch", 6)
    with pytest.raises(ValueError, match="num_tracks"):
        register_track_head("p6_test_head_mismatch", 2)


class _FakeEmbeddings:
    def __init__(self, x):
        self._x = x

    def get_sequence_embeddings(self, resolution):
        assert resolution == 1
        return self._x


def test_register_is_idempotent_and_rejects_kind_change():
    register_track_head("p6_test_fc2d_reg_kind", 4, kind="rnacond_fc2d")
    register_track_head("p6_test_fc2d_reg_kind", 4, kind="rnacond_fc2d")
    with pytest.raises(ValueError, match="kind"):
        register_track_head("p6_test_fc2d_reg_kind", 4, kind="mlp")
    with pytest.raises(ValueError, match="num_tracks"):
        register_track_head("p6_test_fc2d_reg_kind", 2, kind="rnacond_fc2d")


def test_register_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        register_track_head("p6_test_bad_kind", 2, kind="lstm")


def _fake_params():
    return {
        "head/h/~predict/hidden": {"w": jnp.ones((3, 2)), "b": jnp.ones((2,))},
        "head/other/~predict/output": {"w": jnp.ones((2, 2))},
        "alphagenome/sequence_encoder/conv": {"w": jnp.ones((2, 2))},
        "alphagenome/head/rna_seq/output": {"w": jnp.ones((2, 2))},
    }


def test_label_backbone_params():
    labels = label_backbone_params(_fake_params(), "h")
    assert labels["head/h/~predict/hidden"] == {"w": "head", "b": "head"}
    assert labels["head/other/~predict/output"] == {"w": "backbone"}
    assert labels["alphagenome/sequence_encoder/conv"] == {"w": "backbone"}
    assert labels["alphagenome/head/rna_seq/output"] == {"w": "backbone"}


def test_make_optimizer_full_uses_separate_learning_rates():
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(
        params,
        "h",
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        backbone_learning_rate=1e-4,
    )
    updates, _ = opt.update(grads, opt.init(params), params)
    # First Adam step moves each leaf by ~lr regardless of gradient scale.
    head_step = float(jnp.abs(updates["head/h/~predict/hidden"]["w"]).mean())
    bb_step = float(jnp.abs(updates["alphagenome/sequence_encoder/conv"]["w"]).mean())
    assert head_step == pytest.approx(1e-2, rel=1e-3)
    assert bb_step == pytest.approx(1e-4, rel=1e-3)


def test_make_optimizer_full_default_backbone_lr_is_lr_over_100():
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(params, "h", learning_rate=1e-2, weight_decay=0.0, heads_only=False)
    updates, _ = opt.update(grads, opt.init(params), params)
    bb_step = float(jnp.abs(updates["alphagenome/head/rna_seq/output"]["w"]).mean())
    assert bb_step == pytest.approx(1e-4, rel=1e-3)


def test_make_optimizer_heads_only_freezes_backbone():
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(params, "h", learning_rate=1e-2, weight_decay=0.0, heads_only=True)
    updates, _ = opt.update(grads, opt.init(params), params)
    assert float(jnp.abs(updates["alphagenome/sequence_encoder/conv"]["w"]).max()) == 0.0
    assert float(jnp.abs(updates["head/h/~predict/hidden"]["w"]).mean()) > 0


def test_write_json_atomic(tmp_path):
    from tisiago.tracks.checkpoint import write_json_atomic

    out = tmp_path / "sub" / "x.json"
    write_json_atomic({"reduction": 0.95}, out)
    assert out.read_text().count("0.95") == 1
    assert [p.name for p in out.parent.iterdir()] == ["x.json"]


def test_targets_and_head_kinds_import_without_pysam():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pysam'] = None; "
        "import tisiago.tracks.targets, tisiago.tracks.head_kinds"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_write_json_atomic_rejects_nan(tmp_path):
    from tisiago.tracks.checkpoint import finite_or_none, write_json_atomic

    out = tmp_path / "x.json"
    write_json_atomic({"losses": [finite_or_none(float("nan")), finite_or_none(1.0)]}, out)
    assert '"losses": [\n    null,\n    1.0\n  ]' in out.read_text()
    with pytest.raises(ValueError):
        write_json_atomic({"reduction": float("nan")}, tmp_path / "y.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json"]


def test_make_schedule_warmup_peak_and_floor():
    from tisiago.tracks.model import make_schedule

    sched = make_schedule(1e-2, total_steps=100, warmup_steps=10)
    assert float(sched(0)) == 0.0
    assert float(sched(10)) == pytest.approx(1e-2, rel=1e-5)
    assert float(sched(100)) == pytest.approx(1e-4, rel=1e-4)
    assert make_schedule(1e-2, total_steps=None, warmup_steps=0) == 1e-2


def test_make_optimizer_schedule_scales_full_step_updates():
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(
        params,
        "h",
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        backbone_learning_rate=1e-4,
        total_steps=100,
        warmup_steps=10,
    )
    state = opt.init(params)
    updates, state = opt.update(grads, state, params)
    assert float(jnp.abs(updates["head/h/~predict/hidden"]["w"]).max()) == 0.0
    for _ in range(9):
        updates, state = opt.update(grads, state, params)
    updates, state = opt.update(grads, state, params)  # count 10 = peak
    assert float(jnp.abs(updates["head/h/~predict/hidden"]["w"]).mean()) == pytest.approx(
        1e-2, rel=1e-2
    )
    assert float(jnp.abs(updates["alphagenome/sequence_encoder/conv"]["w"]).mean()) == (
        pytest.approx(1e-4, rel=1e-2)
    )


def test_make_optimizer_heads_only_with_schedule_keeps_backbone_zero():
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(
        params,
        "h",
        learning_rate=1e-2,
        weight_decay=0.1,
        heads_only=True,
        total_steps=50,
        warmup_steps=5,
        clip_norm=1.0,
    )
    state = opt.init(params)
    for _ in range(8):
        updates, state = opt.update(grads, state, params)
        assert float(jnp.abs(updates["alphagenome/sequence_encoder/conv"]["w"]).max()) == 0.0
        assert float(jnp.abs(updates["alphagenome/head/rna_seq/output"]["w"]).max()) == 0.0
    assert float(jnp.abs(updates["head/h/~predict/hidden"]["w"]).mean()) > 0


@pytest.mark.parametrize("heads_only", [True, False])
def test_clip_norm_bounds_huge_gradient_updates(heads_only):
    params = _fake_params()
    huge = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 1e6), params)
    opt = make_optimizer(
        params, "h", learning_rate=1e-2, weight_decay=0.0, heads_only=heads_only, clip_norm=1.0
    )
    updates, _ = opt.update(huge, opt.init(params), params)
    leaves = jax.tree_util.tree_leaves(updates)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    assert max(float(jnp.abs(x).max()) for x in leaves) <= 1e-2 * 1.01


def test_clip_norm_limits_spike_damage_and_leaves_small_gradients_alone():
    key = "head/h/~predict/hidden"
    params = {key: {"w": jnp.ones((4,))}}
    steady = {key: {"w": jnp.full((4,), 0.01)}}
    spike = {key: {"w": jnp.full((4,), 1e6)}}

    def run(clip_norm, grads_seq):
        opt = make_optimizer(params, "h", learning_rate=1e-2, weight_decay=0.0, clip_norm=clip_norm)
        state = opt.init(params)
        for g in grads_seq:
            updates, state = opt.update(g, state, params)
        return float(jnp.abs(updates[key]["w"]).max())

    # Below the threshold clipping is the identity.
    assert run(1.0, [steady] * 5) == pytest.approx(run(None, [steady] * 5), rel=1e-6)
    # A spike inflates Adam's second moment; clipping keeps later steady steps from stalling.
    after = [steady] * 20 + [spike] + [steady] * 20
    assert run(1.0, after) > 2 * run(None, after)


# --- the sequence-passing forward ---


def _fake_backbone(dim=8):
    from alphagenome_research.model.embeddings import Embeddings

    def backbone(dna_sequence, organism_index):
        x = hk.Linear(dim, name="fake_backbone")(dna_sequence.astype(jnp.float32))
        return Embeddings(embeddings_1bp=x)

    return backbone


class _FakeModel:
    """Just enough of CustomAlphaGenomeModel for make_predict / make_train_step."""

    def __init__(self, heads, num_tracks, seq_len=48, kind="mlp", width=4):
        import types

        from tisiago.tracks.model import ORGANISM, make_sequence_forward

        forward = make_sequence_forward(heads, detach_backbone=False, backbone=_fake_backbone())
        dummy = jnp.zeros((1, seq_len, width), jnp.float32)
        self._params, self._state = forward.init(
            jax.random.PRNGKey(0), dummy, jnp.zeros((1,), jnp.int32)
        )
        self._metadata = {ORGANISM: types.SimpleNamespace(strand_reindexing=None)}
        self._predict = None
        self._num_tracks = num_tracks
        self._kind = kind

    def create_loss_fn_for_head(self, head):
        from tisiago.tracks.model import HEAD_KINDS

        cls = HEAD_KINDS[self._kind]
        fn = hk.transform(
            lambda p, b: cls(name=head, output_type=None, num_tracks=self._num_tracks).loss(p, b)
        )
        return lambda p, b: fn.apply({}, None, p, b)


def _random_onehot(rng, batch, length):
    return jnp.asarray(np.eye(4, dtype=np.float32)[rng.integers(0, 4, (batch, length))])


def test_only_the_reference_head_is_registered_and_choices_match():
    from tisiago.tracks import evaluate, head_kinds
    from tisiago.tracks import train_genomic as train
    from tisiago.tracks.model import HEAD_KINDS, RnaCondFactConvFactChxDilTisTrackHead

    production = {k: v for k, v in HEAD_KINDS.items() if k not in TEST_KINDS}
    assert production == {"rnacond_fc2d": RnaCondFactConvFactChxDilTisTrackHead}
    assert model_mod.SEQUENCE_WIDTHS["rnacond_fc2d"] == 6
    for parser in (train.build_parser(),):
        action = next(a for a in parser._actions if a.dest == "head")
        assert set(action.choices) == set(production)
    for module in (evaluate, head_kinds):
        assert set(module.HEAD_CHOICES) == set(production)


def test_sequence_forward_passes_augmented_input_to_head():
    from alphagenome.models import dna_output
    from alphagenome_ft import CustomHeadConfig, CustomHeadType, register_custom_head

    from tisiago.tracks.augment import reverse_complement, shift
    from tisiago.tracks.model import make_predict

    class SpyHead(TrackHead):
        def predict(self, embeddings, organism_index, **kwargs):
            x = embeddings.get_sequence_embeddings(resolution=1)
            return kwargs["sequence"].astype(jnp.float32) + 0.0 * x[..., :4]

    name = "p6_test_spy_head"
    from alphagenome_ft.custom_heads import get_custom_head_config

    if get_custom_head_config(name) is None:
        register_custom_head(
            name,
            SpyHead,
            CustomHeadConfig(
                type=CustomHeadType.GENOME_TRACKS,
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=4,
            ),
        )
    from tisiago.tracks.model import install_sequence_forward

    model = _FakeModel([name], 4)
    install_sequence_forward(model, [name], detach_backbone=True, backbone=_fake_backbone())
    predict = make_predict(model, name)
    rng = np.random.default_rng(0)
    onehot = np.asarray(_random_onehot(rng, 1, 48))[0]
    targets = rng.poisson(1, (48, 2)).astype(np.float32)
    seq_rc, _, _ = reverse_complement(onehot, targets, targets > 0, [(0, 1)])
    seq_aug, _, _ = shift(seq_rc, targets, targets > 0, 5)
    out = np.asarray(predict(model._params, model._state, jnp.asarray(seq_aug[None])))
    np.testing.assert_array_equal(out[0], seq_aug)


def test_install_sequence_forward_rejects_mismatched_params():
    from tisiago.tracks.model import install_sequence_forward

    register_track_head("p6_test_fc2d_guard", 4, kind="rnacond_fc2d")
    model = _FakeModel(["p6_test_fc2d_guard"], 4, kind="rnacond_fc2d", width=6)
    with pytest.raises(ValueError, match="does not match"):
        install_sequence_forward(
            model, ["p6_test_fc2d_guard"], detach_backbone=True, backbone=_fake_backbone(16)
        )


@pytest.mark.parametrize("kind", ["mlp", "rnacond_fc2d"])
def test_load_model_installs_sequence_forward_only_for_sequence_kinds(monkeypatch, kind):
    import types

    head = f"p6_test_load_{kind}"
    register_track_head(head, 4, kind=kind)
    sentinel = object()
    seen = {}

    def fake_create(version, *, heads, detach_backbone, device, **kwargs):
        seen["free_init"] = model_mod._SEQUENCE_FREE_INIT
        return types.SimpleNamespace(_predict=sentinel)

    def fake_install(model, heads, *, detach_backbone):
        seen["installed"] = (tuple(heads), detach_backbone)
        model._predict = "sequence-forward"
        return model

    monkeypatch.setattr(model_mod, "create_model_with_heads", fake_create)
    monkeypatch.setattr(model_mod, "install_sequence_forward", fake_install)
    model = model_mod.load_model("ckpt", head, detach_backbone=False)
    assert model_mod._SEQUENCE_FREE_INIT is False
    if kind == "rnacond_fc2d":
        assert seen == {"free_init": True, "installed": ((head,), False)}
        assert model._predict == "sequence-forward"
    else:
        assert seen == {"free_init": False}
        assert model._predict is sentinel


def test_sequence_forward_matches_agft_forward_numerically(monkeypatch):
    """The forward from agft's create_model_with_heads vs the replicated one (stand-in backbone)."""
    import types

    from alphagenome_research.model import dna_model
    from alphagenome_research.model import model as model_lib
    from alphagenome_research.model.embeddings import Embeddings
    from alphagenome_research.model.metadata import metadata as metadata_lib

    class StandInAlphaGenome(hk.Module):
        def __init__(self, metadata, name="alphagenome"):
            super().__init__(name=name)

        def __call__(self, dna_sequence, organism_index):
            x = hk.Linear(8, name="stand_in")(dna_sequence.astype(jnp.float32))
            return {}, Embeddings(embeddings_1bp=x)

    cpu = jax.devices("cpu")[0]
    base = types.SimpleNamespace(
        _device_context=types.SimpleNamespace(_device=cpu),
        _metadata={},
        _one_hot_encoder=None,
        _fasta_extractors=None,
        _output_metadata_by_organism=None,
        _variant_scorers=None,
        _params={},
        _state={},
        _predict=None,
        _predict_variant=None,
    )
    monkeypatch.setattr(model_lib, "AlphaGenome", StandInAlphaGenome)
    monkeypatch.setattr(metadata_lib, "load", lambda organism: None)
    monkeypatch.setattr(dna_model, "create", lambda *a, **k: base)

    head = "p6_test_agft_equiv"
    register_track_head(head, 3, kind="mlp")
    model = model_mod.load_model("ckpt", head, detach_backbone=True)
    rng = np.random.default_rng(0)
    seq = _random_onehot(rng, 2, 64)
    org = jnp.zeros((2,), jnp.int32)
    kwargs = dict(
        requested_outputs=(),
        negative_strand_mask=jnp.zeros((2,), bool),
        strand_reindexing=None,
    )
    agft_out = np.asarray(model._predict(model._params, model._state, seq, org, **kwargs)[head])
    model_mod.install_sequence_forward(model, [head], detach_backbone=True)
    ours = np.asarray(model._predict(model._params, model._state, seq, org, **kwargs)[head])
    assert agft_out.shape == (2, 64, 3) and agft_out.std() > 0
    np.testing.assert_array_equal(ours, agft_out)


# --- two-stage init: backbone freeze and restore from a finished run ---


def _run_updates(n, **kwargs):
    params = _fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(
        params,
        "h",
        learning_rate=1e-2,
        weight_decay=1e-4,
        heads_only=False,
        backbone_learning_rate=1e-4,
        total_steps=40,
        warmup_steps=5,
        clip_norm=1.0,
        **kwargs,
    )
    state = opt.init(params)
    out = []
    for _ in range(n):
        updates, state = opt.update(grads, state, params)
        out.append(updates)
    return out


def test_freeze_backbone_zero_is_identical_to_default():
    default, frozen0 = _run_updates(8), _run_updates(8, freeze_backbone_steps=0)
    for a, b in zip(default, frozen0, strict=True):
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_freeze_backbone_steps_holds_backbone_then_releases():
    default, frozen = _run_updates(12), _run_updates(12, freeze_backbone_steps=8)
    bb = "alphagenome/sequence_encoder/conv"
    for i in range(12):
        np.testing.assert_array_equal(
            np.asarray(frozen[i]["head/h/~predict/hidden"]["w"]),
            np.asarray(default[i]["head/h/~predict/hidden"]["w"]),
        )
        moved = float(jnp.abs(frozen[i][bb]["w"]).max())
        assert (moved == 0.0) if i < 8 else (moved > 0.0)


def test_make_schedule_freeze_masks_then_follows_schedule():
    from tisiago.tracks.model import make_schedule

    base = make_schedule(1e-2, total_steps=100, warmup_steps=10)
    frozen = make_schedule(1e-2, total_steps=100, warmup_steps=10, freeze_steps=20)
    assert float(frozen(19)) == 0.0
    assert float(frozen(20)) == pytest.approx(float(base(20)))
    assert float(make_schedule(1e-3, None, freeze_steps=5)(5)) == pytest.approx(1e-3)


def _save_model(model, directory):
    import orbax.checkpoint as ocp

    ck = ocp.StandardCheckpointer()
    ck.save(str(directory.resolve() / "checkpoint"), (model._params, model._state))
    ck.wait_until_finished()


def _perturbed(tree, delta):
    return jax.tree_util.tree_map(lambda x: x + delta, tree)


def test_restore_init_same_head_loads_everything(tmp_path):
    from tisiago.tracks.model import restore_init

    head = "p6_test_init_same"
    register_track_head(head, 2, kind="mlp")
    source = _FakeModel([head], 2, kind="mlp")
    source._params = _perturbed(source._params, 0.5)
    _save_model(source, tmp_path / "best")
    target = _FakeModel([head], 2, kind="mlp")
    n = restore_init(target, str(tmp_path / "best"), head, reuse_head=True)
    assert n == len(jax.tree_util.tree_leaves((source._params, source._state)))
    for a, b in zip(
        jax.tree_util.tree_leaves(target._params),
        jax.tree_util.tree_leaves(source._params),
        strict=True,
    ):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_restore_init_new_head_kind_starts_fresh(tmp_path):
    from tisiago.tracks.model import restore_init

    old, new = "p6_test_init_mlp", "p6_test_init_fc2d"
    register_track_head(old, 2, kind="mlp")
    register_track_head(new, 4, kind="rnacond_fc2d")
    source = _FakeModel([old], 2, kind="mlp")
    source._params = _perturbed(source._params, 0.5)
    _save_model(source, tmp_path / "best")
    target = _FakeModel([new], 4, kind="rnacond_fc2d", width=6)
    fresh = jax.tree_util.tree_map(np.asarray, target._params)
    restore_init(target, str(tmp_path / "best"), new, reuse_head=False)
    for module, leaves in target._params.items():
        for name, value in leaves.items():
            want = (
                fresh[module][name]
                if module.startswith(f"head/{new}/")
                else (source._params[module][name])
            )
            np.testing.assert_array_equal(np.asarray(value), np.asarray(want))
    assert any(m.startswith(f"head/{new}/") for m in target._params)
    with pytest.raises(ValueError, match="missing"):
        restore_init(target, str(tmp_path / "best"), new, reuse_head=True)


# --- RNA-conditioned heads: embeddings + RNA features of the condition's pooled RNA input ---


def _rna_sequences(rng, batch, length):
    onehot = np.asarray(_random_onehot(rng, batch, length))
    rna = rng.poisson(1.5, (batch, length, 2)).astype(np.float32)
    return jnp.asarray(np.concatenate([onehot, rna], axis=-1))


def test_rnacond_checkpoint_round_trip(tmp_path):
    from tisiago.tracks.model import install_sequence_forward, make_predict, restore_init

    head = "p6_test_fc2d_ck"
    register_track_head(head, 4, kind="rnacond_fc2d")

    def build():
        model = _FakeModel([head], 4, kind="rnacond_fc2d", width=6)
        return install_sequence_forward(
            model, [head], detach_backbone=False, backbone=_fake_backbone()
        )

    source = build()
    source._params = _perturbed(source._params, 0.25)
    _save_model(source, tmp_path / "best")
    same = build()
    n = restore_init(same, str(tmp_path / "best"), head, reuse_head="all")
    assert n == len(jax.tree_util.tree_leaves((source._params, source._state)))
    seq = _rna_sequences(np.random.default_rng(1), 1, 48)
    np.testing.assert_array_equal(
        np.asarray(make_predict(same, head)(same._params, same._state, seq)),
        np.asarray(make_predict(source, head)(source._params, source._state, seq)),
    )


# --- factorised rnacond heads (TIS = per-block amount x within-block softmax) ---

FACT_KINDS = ("rnacond_f", "rnacond_fo")


def test_local_softmax_preserves_totals_and_concentrates():
    from tisiago.tracks.model import local_softmax

    rng = np.random.default_rng(0)
    rate = jnp.asarray(rng.gamma(1.0, 1.0, (2, 50, 2)), jnp.float32)
    logits = jnp.asarray(rng.normal(size=(2, 50, 2)), jnp.float32)
    out = np.asarray(local_softmax(rate, logits, 16))
    assert out.shape == (2, 50, 2) and bool((out >= 0).all())
    np.testing.assert_allclose(out.sum(axis=1), np.asarray(rate).sum(axis=1), rtol=1e-5)
    # constant logits and a constant rate: the rate itself (partial edge blocks included)
    flat = np.asarray(local_softmax(jnp.full((1, 50, 1), 2.0), jnp.zeros((1, 50, 1)), 16))
    np.testing.assert_allclose(flat, 2.0, rtol=1e-6)
    # a dominant logit takes its blocks' mass
    peak = jnp.zeros((1, 64, 1)).at[0, 20, 0].set(50.0)
    out = np.asarray(local_softmax(jnp.ones((1, 64, 1)), peak, 16))
    assert out[0, 20, 0] > 0.9 * 16


def _fact_forward(kind):
    from tisiago.tracks.model import HEAD_KINDS

    def forward(emb, seq):
        head = HEAD_KINDS[kind](name="r", output_type=None, num_tracks=4)
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), sequence=seq)

    return hk.without_apply_rng(hk.transform(forward))


@pytest.mark.parametrize("kind", FACT_KINDS)
def test_fact_heads_registered_shapes_and_rna_use(kind):
    from tisiago.tracks.model import SEQUENCE_WIDTHS
    from tisiago.tracks.rna_features import NUM_ONESIDED_FEATURES, NUM_RNA_FEATURES

    assert SEQUENCE_WIDTHS[kind] == 6
    register_track_head(f"p6_test_{kind}_reg", 4, kind=kind)
    fn = _fact_forward(kind)
    rng = np.random.default_rng(1)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)), jnp.float32)
    seq = _rna_sequences(rng, 1, 40)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    n_feat = NUM_RNA_FEATURES + (0 if kind == "rnacond_f" else NUM_ONESIDED_FEATURES)
    assert params["r/~predict/hidden"]["w"].shape == (8 + n_feat, 256)
    assert params["r/~predict/output"]["w"].shape == (256, 2)
    assert "r/~predict/tis_hidden" not in params
    pred = np.asarray(fn.apply(params, emb, seq))
    assert pred.shape == (1, 40, 4) and np.isfinite(pred).all() and (pred >= 0).all()
    other = np.asarray(fn.apply(params, emb, seq.at[..., 4:].set(0.0)))
    assert not np.allclose(pred, other)
    # CHX is softplus(output) of the trunk; TIS sums to its softplus rate over the window
    hidden = params["r/~predict/hidden"]
    assert hidden["w"].shape[0] == 8 + n_feat


def _fc2_setup():
    fn = _fact_forward("rnacond_fc2")
    rng = np.random.default_rng(5)
    emb = jnp.asarray(rng.normal(size=(1, 64, 8)), jnp.float32)
    seq = _rna_sequences(rng, 1, 64)
    return fn, emb, seq, fn.init(jax.random.PRNGKey(0), emb, seq)


def test_fc2_registered_and_conv9_trunk_shapes():
    from tisiago.tracks.model import SEQUENCE_WIDTHS
    from tisiago.tracks.rna_features import NUM_ONESIDED_FEATURES, NUM_RNA_FEATURES

    assert SEQUENCE_WIDTHS["rnacond_fc2"] == 6
    register_track_head("p6_test_fc2_reg", 4, kind="rnacond_fc2")
    fn, emb, seq, params = _fc2_setup()
    n_in = 8 + NUM_RNA_FEATURES + NUM_ONESIDED_FEATURES
    assert params["r/~predict/hidden"]["w"].shape == (9, n_in, 256)
    assert params["r/~predict/hidden2"]["w"].shape == (9, 256, 256)
    assert params["r/~predict/output"]["w"].shape == (256, 4)
    assert not any("tis_hidden" in k for k in params)
    pred = np.asarray(fn.apply(params, emb, seq))
    assert pred.shape == (1, 64, 4) and np.isfinite(pred).all() and (pred >= 0).all()
    other = np.asarray(fn.apply(params, emb, seq.at[..., 4:].set(0.0)))
    assert not np.allclose(pred, other)


def test_fc2_chx_is_factorised_per_block_and_grads_finite():
    fn, emb, seq, params = _fc2_setup()
    pred = np.asarray(fn.apply(params, emb, seq))
    grads = jax.grad(lambda p: fn.apply(p, emb, seq).sum())(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()
    assert float(jnp.abs(grads["r/~predict/hidden2"]["w"]).sum()) > 0
    assert (pred[..., 2:].sum(axis=1) > 0).all()


def test_fc2_chx_block_sums_equal_amount():
    """Re-derive the CHX amount from the head's own trunk and compare with block sums."""
    from tisiago.tracks.model import CONV9_KERNEL, LOCAL_BLOCK

    fn, emb, seq, params = _fc2_setup()
    pred = np.asarray(fn.apply(params, emb, seq))
    from tisiago.tracks.rna_features import onesided_features, rna_features

    rna = seq[..., 4:]
    x = jnp.concatenate([emb, rna_features(rna), onesided_features(rna)], axis=-1)

    def conv(h, name):
        w, b = params[f"r/~predict/{name}"]["w"], params[f"r/~predict/{name}"]["b"]
        out = jax.lax.conv_general_dilated(
            h, w, (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC")
        )
        return jax.nn.relu(out + b)

    assert CONV9_KERNEL == 9
    h = conv(conv(x, "hidden"), "hidden2")
    out = h @ params["r/~predict/output"]["w"] + params["r/~predict/output"]["b"]
    amount = np.asarray(jax.nn.softplus(out[..., :2]).sum(axis=1))
    # 64-nt window < one block: both grids put the whole window in one block
    assert 64 <= LOCAL_BLOCK
    np.testing.assert_allclose(pred[..., 2:].sum(axis=1), amount, rtol=1e-4)


@pytest.mark.parametrize("kind", (*FACT_KINDS, "rnacond_fc2", "rnacond_fc2d"))
def test_fact_jitted_train_step_with_local_loss(kind):
    from tisiago.tracks.model import install_sequence_forward, make_predict, make_train_step

    head = f"p6_test_{kind}_train"
    register_track_head(head, 4, kind=kind)
    model = _FakeModel([head], 4, kind=kind, width=6)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _rna_sequences(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 4)).astype(np.float32)),
        "mask": jnp.ones((2, 48, 4), bool),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        local_weight=5.0,
        local_channels=(0, 1),
        local_block=16,
    )
    params, state = model._params, model._state
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"])) and float(aux["local"]) > 0
    leaves = jax.tree_util.tree_leaves(new_params)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    before = np.asarray(params[f"head/{head}/~predict/tis_output"]["w"])
    assert not np.allclose(before, np.asarray(new_params[f"head/{head}/~predict/tis_output"]["w"]))
    pred = make_predict(model, head)(new_params, state, batch["sequences"])
    assert pred.shape == (2, 48, 4) and bool(jnp.isfinite(pred).all())


def test_train_step_without_local_weight_matches_previous_loss():
    """local_weight 0 leaves the loss exactly as before (no local term)."""
    from tisiago.tracks.model import install_sequence_forward, make_train_step

    head = "p6_test_fc2d_nolocal"
    register_track_head(head, 4, kind="rnacond_fc2d")
    model = _FakeModel([head], 4, kind="rnacond_fc2d", width=6)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _rna_sequences(rng, 1, 48),
        "targets": jnp.asarray(rng.poisson(2, (1, 48, 4)).astype(np.float32)),
        "mask": jnp.ones((1, 48, 4), bool),
    }
    kw = dict(learning_rate=1e-2, weight_decay=0.0, heads_only=False)
    step_a, opt_a = make_train_step(model, head, **kw)
    step_b, opt_b = make_train_step(model, head, local_weight=0.0, local_channels=(0, 1), **kw)
    _, _, aux_a = step_a(model._params, model._state, opt_a, batch)
    _, _, aux_b = step_b(model._params, model._state, opt_b, batch)
    assert float(aux_a["loss"]) == float(aux_b["loss"])
    assert float(aux_a["loss"]) == pytest.approx(
        float(aux_a["poisson"]) + 5.0 * float(aux_a["multinomial"]), rel=1e-6
    )


def test_fc2d_context_branch_only_feeds_tis():
    from tisiago.tracks import head_kinds

    assert head_kinds.HEAD_CHOICES == ("rnacond_fc2d",)
    fn, emb, seq, params = _fc2_setup()
    fn_d = _fact_forward("rnacond_fc2d")
    pd = fn_d.init(jax.random.PRNGKey(0), emb, seq)
    assert set(pd) - set(params) >= {"r/~predict/tis_ctx", "r/~predict/dil8", "r/~predict/dil8_out"}
    assert "r/~predict/dil9" not in pd and pd["r/~predict/tis_ctx"]["w"].shape == (256, 256)
    assert pd["r/~predict/dil0"]["w"].shape == (3, 256, 128)
    assert pd["r/~predict/output"]["w"].shape == (256, 4)
    assert pd["r/~predict/tis_output"]["w"].shape == (256, 4)
    pred = np.asarray(fn_d.apply(pd, emb, seq))
    assert pred.shape == (1, 64, 4) and np.isfinite(pred).all() and (pred >= 0).all()
    # zero-initialised branches: CHX equals fc2's with identical shared weights
    shared = {k: v for k, v in pd.items() if k in params}
    ref = np.asarray(fn.apply({**params, **{k: shared[k] for k in params}}, emb, seq))
    np.testing.assert_allclose(pred[..., 2:], ref[..., 2:], rtol=1e-5, atol=1e-6)

    def loss(p):
        return fn_d.apply(p, emb, seq).sum()

    grads = jax.grad(loss)(pd)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    assert float(jnp.abs(grads["r/~predict/dil8_out"]["w"]).sum()) > 0
    # the CHX output ignores the context branch: perturbing it leaves CHX unchanged
    bumped = {
        **pd,
        "r/~predict/dil8_out": {
            **pd["r/~predict/dil8_out"],
            "w": pd["r/~predict/dil8_out"]["w"] + 0.5,
        },
    }
    out = np.asarray(fn_d.apply(bumped, emb, seq))
    np.testing.assert_allclose(out[..., 2:], pred[..., 2:], rtol=1e-6)
    assert not np.allclose(out[..., :2], pred[..., :2])


def test_train_step_local2_off_matches_and_on_adds():
    from tisiago.tracks.model import install_sequence_forward, make_train_step

    head = "p6_test_fc2d_local2"
    register_track_head(head, 4, kind="rnacond_fc2d")
    model = _FakeModel([head], 4, kind="rnacond_fc2d", width=6)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _rna_sequences(rng, 1, 64),
        "targets": jnp.asarray(rng.poisson(2, (1, 64, 4)).astype(np.float32)),
        "mask": jnp.ones((1, 64, 4), bool),
    }
    kw = dict(
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        local_weight=3.0,
        local_channels=(0, 1),
        local_block=32,
    )
    aux = {}
    for name, extra in (
        ("base", {}),
        ("off", dict(local2_weight=0.0, local2_channels=(0,), local2_block=8)),
        ("on", dict(local2_weight=4.0, local2_channels=(0, 1), local2_block=16)),
    ):
        step, opt = make_train_step(model, head, **kw, **extra)
        aux[name] = step(model._params, model._state, opt, batch)[2]
    assert float(aux["base"]["loss"]) == float(aux["off"]["loss"])
    assert float(aux["base"]["local2"]) == 0.0 and float(aux["on"]["local2"]) > 0
    expected = float(aux["base"]["loss"]) + 4.0 * float(aux["on"]["local2"])
    assert float(aux["on"]["loss"]) == pytest.approx(expected, rel=1e-5)
