"""CPU-safe tests for the track head wiring (no checkpoint load)."""

import numpy as np
import pytest

pytest.importorskip("alphagenome_ft")
import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tisiago.seq2func.model import (  # noqa: E402
    ConvTrackHead,
    TrackHead,
    label_backbone_params,
    make_optimizer,
    register_track_head,
)


def test_head_loss_is_zero_at_target():
    def forward(pred, target):
        head = TrackHead(name="t", output_type=None, num_tracks=2)
        return head.loss(pred, {"targets": target, "mask": jnp.ones(target.shape, bool)})

    fn = hk.without_apply_rng(hk.transform(forward))
    y = jnp.asarray(np.random.default_rng(0).poisson(3, (1, 16, 2)).astype(np.float32))
    out = fn.apply(fn.init(None, y, y), y, y)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_head_loss_reads_frame_settings_from_batch():
    from tisiago.seq2func.loss import poisson_multinomial

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


def test_conv_head_shape_and_zero_loss_at_target():
    def forward(emb, target):
        head = ConvTrackHead(name="c", output_type=None, num_tracks=2)
        pred = head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32))
        loss = head.loss(target, {"targets": target, "mask": jnp.ones(target.shape, bool)})
        return pred, loss

    fn = hk.without_apply_rng(hk.transform(forward))
    emb = jnp.asarray(np.random.default_rng(0).normal(size=(1, 64, 8)), jnp.bfloat16)
    y = jnp.asarray(np.random.default_rng(1).poisson(3, (1, 64, 2)).astype(np.float32))
    params = fn.init(jax.random.PRNGKey(0), emb, y)
    pred, loss = fn.apply(params, emb, y)
    assert pred.shape == (1, 64, 2) and pred.dtype == jnp.float32
    assert bool((pred >= 0).all())
    assert params["c/~predict/hidden"]["w"].shape == (31, 8, 256)
    assert float(loss["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_conv_head_concats_sequence_kwarg():
    def forward(emb, seq):
        head = ConvTrackHead(name="c", output_type=None, num_tracks=2)
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), sequence=seq)

    fn = hk.without_apply_rng(hk.transform(forward))
    emb = jnp.zeros((1, 32, 8), jnp.float32)
    seq = jnp.zeros((1, 32, 4), jnp.float32)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    assert params["c/~predict/hidden"]["w"].shape == (31, 12, 256)
    assert fn.apply(params, emb, seq).shape == (1, 32, 2)


def test_register_conv_is_idempotent_and_rejects_kind_change():
    register_track_head("p6_test_conv", 2, kind="conv")
    register_track_head("p6_test_conv", 2, kind="conv")
    with pytest.raises(ValueError, match="kind"):
        register_track_head("p6_test_conv", 2, kind="mlp")
    with pytest.raises(ValueError, match="num_tracks"):
        register_track_head("p6_test_conv", 4, kind="conv")


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


@pytest.mark.parametrize(
    ("final", "expected"),
    [(0.9, True), (1.0, True), (1.2, False), (float("nan"), False)],
)
def test_passes_profile(final, expected):
    from tisiago.seq2func.overfit import passes_profile

    assert passes_profile(final) is expected


def test_fetch_onehot_pads_past_chromosome_end(tmp_path):
    from tisiago.seq2func.overfit import fetch_onehot

    fasta = tmp_path / "g.fa"
    fasta.write_text(">c\nACGTacgt\n")
    onehot = fetch_onehot(str(fasta), "c", -2, 10)
    assert onehot.shape == (12, 4)
    assert onehot[:2].sum() == 0 and onehot[-2:].sum() == 0
    assert onehot[2:10].argmax(1).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_write_json_atomic(tmp_path):
    from tisiago.seq2func.overfit import write_json_atomic

    out = tmp_path / "sub" / "x.json"
    write_json_atomic({"reduction": 0.95}, out)
    assert out.read_text().count("0.95") == 1
    assert [p.name for p in out.parent.iterdir()] == ["x.json"]


def test_targets_and_overfit_import_without_pysam():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pysam'] = None; "
        "import tisiago.seq2func.targets, tisiago.seq2func.overfit"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fetch_onehot_window_entirely_past_chromosome_end(tmp_path):
    from tisiago.seq2func.overfit import fetch_onehot

    fasta = tmp_path / "g.fa"
    fasta.write_text(">c\nACGTacgt\n")
    onehot = fetch_onehot(str(fasta), "c", 20, 26)
    assert onehot.shape == (6, 4)
    assert onehot.sum() == 0


@pytest.mark.parametrize(
    ("initial", "final", "expected"),
    [
        (float("nan"), 1.0, False),
        (10.0, float("nan"), False),
        (0.0, 0.0, False),
        (10.0, 0.5, True),
        (10.0, 5.0, False),
    ],
)
def test_passes_gate(initial, final, expected):
    from tisiago.seq2func.overfit import passes_gate

    assert passes_gate(initial, final) is expected


def test_write_json_atomic_rejects_nan(tmp_path):
    from tisiago.seq2func.overfit import finite_or_none, write_json_atomic

    out = tmp_path / "x.json"
    write_json_atomic({"losses": [finite_or_none(float("nan")), finite_or_none(1.0)]}, out)
    assert '"losses": [\n    null,\n    1.0\n  ]' in out.read_text()
    with pytest.raises(ValueError):
        write_json_atomic({"reduction": float("nan")}, tmp_path / "y.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json"]


def test_make_schedule_warmup_peak_and_floor():
    from tisiago.seq2func.model import make_schedule

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


# --- seqconv: sequence-aware head and the sequence-passing forward ---


def _fake_backbone(dim=8):
    from alphagenome_research.model.embeddings import Embeddings

    def backbone(dna_sequence, organism_index):
        x = hk.Linear(dim, name="fake_backbone")(dna_sequence.astype(jnp.float32))
        return Embeddings(embeddings_1bp=x)

    return backbone


class _FakeModel:
    """Just enough of CustomAlphaGenomeModel for make_predict / make_train_step."""

    def __init__(self, heads, num_tracks, seq_len=48, kind="seqconv", width=4):
        import types

        from tisiago.seq2func.model import ORGANISM, make_sequence_forward

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
        from tisiago.seq2func.model import HEAD_KINDS

        cls = HEAD_KINDS[self._kind]
        fn = hk.transform(
            lambda p, b: cls(name=head, output_type=None, num_tracks=self._num_tracks).loss(p, b)
        )
        return lambda p, b: fn.apply({}, None, p, b)


def _random_onehot(rng, batch, length):
    return jnp.asarray(np.eye(4, dtype=np.float32)[rng.integers(0, 4, (batch, length))])


def test_seqconv_registered_and_choices_match():
    from tisiago.seq2func import evaluate, overfit, periodicity, train
    from tisiago.seq2func.model import HEAD_KINDS, SEQUENCE_KINDS, SeqConvTrackHead

    assert HEAD_KINDS["seqconv"] is SeqConvTrackHead
    assert SEQUENCE_KINDS == {"seqconv", "seqconv_tx", "resid_tx"}
    for parser in (train.build_parser(),):
        action = next(a for a in parser._actions if a.dest == "head")
        assert set(action.choices) == set(HEAD_KINDS)
    for module in (evaluate, overfit, periodicity):
        assert set(module.HEAD_CHOICES) == set(HEAD_KINDS)


def test_seqconv_head_input_is_embeddings_onehot_and_orf_features():
    from tisiago.seq2func.model import SeqConvTrackHead
    from tisiago.seq2func.orf_features import NUM_FEATURES

    def forward(emb, seq):
        head = SeqConvTrackHead(name="s", output_type=None, num_tracks=2)
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), sequence=seq)

    fn = hk.without_apply_rng(hk.transform(forward))
    emb = jnp.zeros((1, 40, 8), jnp.bfloat16)
    seq = _random_onehot(np.random.default_rng(0), 1, 40)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    assert params["s/~predict/hidden"]["w"].shape == (31, 8 + 4 + NUM_FEATURES, 256)
    pred = fn.apply(params, emb, seq)
    assert pred.shape == (1, 40, 2) and pred.dtype == jnp.float32
    # Same embeddings, different sequence -> different predictions: the sequence is used.
    other = fn.apply(params, emb, _random_onehot(np.random.default_rng(1), 1, 40))
    assert not np.allclose(np.asarray(pred), np.asarray(other))


def test_seqconv_head_requires_sequence_outside_sequence_free_init():
    from tisiago.seq2func.model import SeqConvTrackHead, sequence_free_init

    def forward(emb):
        head = SeqConvTrackHead(name="s", output_type=None, num_tracks=2)
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32))

    fn = hk.without_apply_rng(hk.transform(forward))
    emb = jnp.zeros((1, 16, 8), jnp.float32)
    with pytest.raises(ValueError, match="sequence"):
        fn.init(jax.random.PRNGKey(0), emb)
    with sequence_free_init(True):
        params = fn.init(jax.random.PRNGKey(0), emb)
    assert params["s/~predict/hidden"]["w"].shape == (31, 8 + 22, 256)
    with pytest.raises(ValueError, match="sequence"):
        fn.apply(params, emb)


def test_sequence_forward_passes_augmented_input_to_head():
    from alphagenome.models import dna_output
    from alphagenome_ft import CustomHeadConfig, CustomHeadType, register_custom_head

    from tisiago.seq2func.augment import reverse_complement, shift
    from tisiago.seq2func.model import make_predict

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
    from tisiago.seq2func.model import install_sequence_forward

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
    from tisiago.seq2func.model import install_sequence_forward

    register_track_head("p6_test_seqconv_guard", 2, kind="seqconv")
    model = _FakeModel(["p6_test_seqconv_guard"], 2)
    with pytest.raises(ValueError, match="does not match"):
        install_sequence_forward(
            model, ["p6_test_seqconv_guard"], detach_backbone=True, backbone=_fake_backbone(16)
        )


def test_seqconv_jitted_train_step_has_finite_grads():
    from tisiago.seq2func.model import install_sequence_forward, make_predict, make_train_step

    head = "p6_test_seqconv_train"
    register_track_head(head, 2, kind="seqconv")
    model = _FakeModel([head], 2)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _random_onehot(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 2)).astype(np.float32)),
        "mask": jnp.ones((2, 48, 2), bool),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        frame_weight=5.0,
        frame_channels=(0,),
        frame_block=2,
    )
    params, state = model._params, model._state

    def loss(p):
        preds = model._predict(
            p,
            state,
            batch["sequences"],
            jnp.zeros((2,), jnp.int32),
            requested_outputs=(),
            negative_strand_mask=jnp.zeros((2,), bool),
            strand_reindexing=None,
        )[head]
        return model.create_loss_fn_for_head(head)(
            preds, {"targets": batch["targets"], "mask": batch["mask"]}
        )["loss"]

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert float(jnp.abs(grads[f"head/{head}/~predict/hidden"]["w"]).sum()) > 0
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"])) and float(aux["frame"]) >= 0
    before = np.asarray(params[f"head/{head}/~predict/hidden"]["w"])
    after = np.asarray(new_params[f"head/{head}/~predict/hidden"]["w"])
    assert not np.allclose(before, after)
    pred = make_predict(model, head)(new_params, state, batch["sequences"])
    assert pred.shape == (2, 48, 2) and bool(jnp.isfinite(pred).all())


@pytest.mark.parametrize("kind", ["mlp", "conv", "seqconv", "seqconv_tx", "resid_tx"])
def test_load_model_installs_sequence_forward_only_for_sequence_kinds(monkeypatch, kind):
    import types

    from tisiago.seq2func import model as model_mod

    head = f"p6_test_load_{kind}"
    register_track_head(head, 2, kind=kind)
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
    if kind in ("seqconv", "seqconv_tx", "resid_tx"):
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

    from tisiago.seq2func import model as model_mod

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


# --- seqconv_tx: spliced transcript features as extra input channels ---


def _tx_sequences(rng, batch, length):
    onehot = np.asarray(_random_onehot(rng, batch, length))
    extra = rng.uniform(size=(batch, length, 20)).astype(np.float32)
    return jnp.asarray(np.concatenate([onehot, extra], axis=-1))


def test_seqconv_tx_head_input_is_embeddings_and_all_sequence_channels():
    from tisiago.seq2func.model import HEAD_KINDS, SeqConvTxTrackHead, sequence_free_init

    assert HEAD_KINDS["seqconv_tx"] is SeqConvTxTrackHead

    def forward(emb, seq=None):
        head = SeqConvTxTrackHead(name="s", output_type=None, num_tracks=2)
        kwargs = {} if seq is None else {"sequence": seq}
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), **kwargs)

    fn = hk.without_apply_rng(hk.transform(forward))
    emb = jnp.zeros((1, 40, 8), jnp.bfloat16)
    rng = np.random.default_rng(0)
    seq = _tx_sequences(rng, 1, 40)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    assert params["s/~predict/hidden"]["w"].shape == (31, 8 + 24, 256)
    pred = fn.apply(params, emb, seq)
    assert pred.shape == (1, 40, 2) and pred.dtype == jnp.float32
    # The transcript channels are used: same one-hot, other features -> other predictions.
    other = seq.at[..., 4:].set(0.0)
    assert not np.allclose(np.asarray(pred), np.asarray(fn.apply(params, emb, other)))
    with pytest.raises(ValueError, match="24 channels"):
        fn.apply(params, emb, seq[..., :4])
    with pytest.raises(ValueError, match="sequence"):
        fn.init(jax.random.PRNGKey(0), emb)
    with sequence_free_init(True):
        free = fn.init(jax.random.PRNGKey(0), emb)
    assert free["s/~predict/hidden"]["w"].shape == (31, 8 + 24, 256)


def test_sequence_forward_gives_backbone_onehot_only():
    from tisiago.seq2func.model import make_sequence_forward

    seen = {}

    def backbone(dna_sequence, organism_index):
        from alphagenome_research.model.embeddings import Embeddings

        seen["shape"] = dna_sequence.shape
        return Embeddings(embeddings_1bp=hk.Linear(8, name="fb")(dna_sequence))

    head = "p6_test_tx_backbone"
    register_track_head(head, 2, kind="seqconv_tx")
    forward = make_sequence_forward([head], detach_backbone=True, backbone=backbone)
    seq = _tx_sequences(np.random.default_rng(0), 1, 32)
    params, state = forward.init(jax.random.PRNGKey(0), seq, jnp.zeros((1,), jnp.int32))
    assert seen["shape"] == (1, 32, 4)
    assert params["fb"]["w"].shape == (4, 8)
    assert params[f"head/{head}/~predict/hidden"]["w"].shape == (31, 8 + 24, 256)


def test_seqconv_tx_jitted_train_step_has_finite_grads():
    from tisiago.seq2func.model import install_sequence_forward, make_predict, make_train_step

    head = "p6_test_seqconv_tx_train"
    register_track_head(head, 2, kind="seqconv_tx")
    model = _FakeModel([head], 2, kind="seqconv_tx", width=24)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _tx_sequences(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 2)).astype(np.float32)),
        "mask": jnp.ones((2, 48, 2), bool),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        frame_weight=5.0,
        frame_channels=(0,),
        frame_block=2,
    )
    params, state = model._params, model._state

    def loss(p):
        preds = model._predict(
            p,
            state,
            batch["sequences"],
            jnp.zeros((2,), jnp.int32),
            requested_outputs=(),
            negative_strand_mask=jnp.zeros((2,), bool),
            strand_reindexing=None,
        )[head]
        return model.create_loss_fn_for_head(head)(
            preds, {"targets": batch["targets"], "mask": batch["mask"]}
        )["loss"]

    grads = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    w = grads[f"head/{head}/~predict/hidden"]["w"]
    assert float(jnp.abs(w[:, -20:]).sum()) > 0  # gradient reaches the transcript channels
    assert float(jnp.abs(grads["fake_backbone"]["w"]).sum()) > 0
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"])) and float(aux["frame"]) >= 0
    pred = make_predict(model, head)(new_params, state, batch["sequences"])
    assert pred.shape == (2, 48, 2) and bool(jnp.isfinite(pred).all())


# --- resid_tx: mlp base branch times exp(zero-initialised sequence modulation) ---


def _resid_model(head, seq_len=48):
    register_track_head(head, 2, kind="resid_tx")
    model = _FakeModel([head], 2, kind="resid_tx", seq_len=seq_len, width=24)
    from tisiago.seq2func.model import install_sequence_forward

    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    return model


def test_resid_tx_equals_mlp_at_init_and_modulates_after():
    from tisiago.seq2func.model import HEAD_KINDS, ResidTxTrackHead

    assert HEAD_KINDS["resid_tx"] is ResidTxTrackHead

    def forward(kind, emb, seq=None):
        head = HEAD_KINDS[kind](name="h", output_type=None, num_tracks=2)
        kwargs = {} if seq is None else {"sequence": seq}
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), **kwargs)

    mlp = hk.without_apply_rng(hk.transform(lambda e: forward("mlp", e)))
    resid = hk.without_apply_rng(hk.transform(lambda e, q: forward("resid_tx", e, q)))
    rng = np.random.default_rng(0)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)).astype(np.float32))
    seq = _tx_sequences(rng, 1, 40)
    mlp_params = mlp.init(jax.random.PRNGKey(0), emb)
    params = resid.init(jax.random.PRNGKey(1), emb, seq)
    for key in mlp_params:
        assert params[key]["w"].shape == mlp_params[key]["w"].shape
    assert params["h/~predict/mod_hidden"]["w"].shape == (31, 8 + 24, 64)
    for leaf in params["h/~predict/mod_output"].values():
        assert float(jnp.abs(leaf).max()) == 0.0
    base_params = {**params, **mlp_params}
    out = resid.apply(base_params, emb, seq)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(mlp.apply(mlp_params, emb)))
    assert out.dtype == jnp.float32
    bumped = {**base_params, "h/~predict/mod_output": {"w": jnp.ones((1, 64, 2)), "b": jnp.ones(2)}}
    assert not np.allclose(np.asarray(resid.apply(bumped, emb, seq)), np.asarray(out))
    with pytest.raises(ValueError, match="24 channels"):
        resid.apply(params, emb, seq[..., :4])


def test_resid_tx_modulation_is_bounded_for_large_mod_output_weights():
    from tisiago.seq2func.model import HEAD_KINDS, MOD_SCALE

    def forward(kind, emb, seq=None):
        head = HEAD_KINDS[kind](name="h", output_type=None, num_tracks=2)
        kwargs = {} if seq is None else {"sequence": seq}
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), **kwargs)

    mlp = hk.without_apply_rng(hk.transform(lambda e: forward("mlp", e)))
    resid = hk.without_apply_rng(hk.transform(lambda e, q: forward("resid_tx", e, q)))
    rng = np.random.default_rng(0)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)).astype(np.float32))
    seq = _tx_sequences(rng, 1, 40)
    mlp_params = mlp.init(jax.random.PRNGKey(0), emb)
    params = {**resid.init(jax.random.PRNGKey(1), emb, seq), **mlp_params}
    base = np.asarray(mlp.apply(mlp_params, emb))
    for sign in (1.0, -1.0):
        big = {"w": sign * 1e4 * jnp.ones((1, 64, 2)), "b": sign * 1e4 * jnp.ones(2)}
        out = np.asarray(resid.apply({**params, "h/~predict/mod_output": big}, emb, seq))
        assert np.isfinite(out).all()
        np.testing.assert_array_less(out, base * np.exp(MOD_SCALE) * (1 + 1e-5) + 1e-30)
        np.testing.assert_array_less(base * np.exp(-MOD_SCALE) * (1 - 1e-5) - 1e-30, out)
    assert MOD_SCALE == 1.5


def test_resid_tx_jitted_train_step_has_nonzero_mod_grads_with_frame_loss():
    from tisiago.seq2func.model import make_predict, make_train_step

    head = "p6_test_resid_tx_train"
    model = _resid_model(head)
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _tx_sequences(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 2)).astype(np.float32)),
        "mask": jnp.ones((2, 48, 2), bool),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        frame_weight=5.0,
        frame_channels=(0,),
        frame_block=2,
    )
    params, state = model._params, model._state
    loss_fn = model.create_loss_fn_for_head(head)

    def loss(p):
        preds = model._predict(
            p,
            state,
            batch["sequences"],
            jnp.zeros((2,), jnp.int32),
            requested_outputs=(),
            negative_strand_mask=jnp.zeros((2,), bool),
            strand_reindexing=None,
        )[head]
        return loss_fn(
            preds,
            {
                "targets": batch["targets"],
                "mask": batch["mask"],
                "frame_weight": 5.0,
                "frame_channels": (0,),
                "frame_block": 2,
            },
        )["loss"]

    grads = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    scope = f"head/{head}/~predict/"
    assert float(jnp.abs(grads[scope + "mod_output"]["w"]).sum()) > 0
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"]))
    # The zero-initialised last layer moved, so the next forward differs from the base branch.
    assert float(jnp.abs(new_params[scope + "mod_output"]["w"]).sum()) > 0
    pred = make_predict(model, head)(new_params, state, batch["sequences"])
    assert pred.shape == (2, 48, 2) and bool(jnp.isfinite(pred).all())


def test_resid_tx_checkpoint_round_trip_and_matching_init(tmp_path):
    from tisiago.seq2func.model import make_predict, restore_init

    head = "p6_test_resid_tx_ck"
    source = _resid_model(head)
    source._params = _perturbed(source._params, 0.25)
    _save_model(source, tmp_path / "best")
    same = _resid_model(head)
    n = restore_init(same, str(tmp_path / "best"), head, reuse_head="all")
    assert n == len(jax.tree_util.tree_leaves((source._params, source._state)))
    seq = _tx_sequences(np.random.default_rng(1), 1, 48)
    np.testing.assert_array_equal(
        np.asarray(make_predict(same, head)(same._params, same._state, seq)),
        np.asarray(make_predict(source, head)(source._params, source._state, seq)),
    )

    mlp_head = "p6_test_resid_tx_mlp"
    register_track_head(mlp_head, 2, kind="mlp")
    base = _FakeModel([mlp_head], 2, kind="mlp")
    base._params = _perturbed(base._params, 0.5)
    _save_model(base, tmp_path / "mlp_best")
    name = mlp_head + "_r"
    target = _resid_model(name)
    # the mlp source's head scope renamed to the resid arm (the same arm in a real run)
    base_scope = f"head/{mlp_head}/~predict/"
    saved = {k.replace(base_scope, f"head/{name}/~predict/"): v for k, v in base._params.items()}
    renamed = _FakeModel([mlp_head], 2, kind="mlp")
    renamed._params, renamed._state = saved, base._state
    _save_model(renamed, tmp_path / "mlp_as_resid")
    fresh = jax.tree_util.tree_map(np.asarray, target._params)
    restore_init(target, str(tmp_path / "mlp_as_resid"), name, reuse_head="matching")
    scope = f"head/{name}/~predict/"
    for module, leaves in target._params.items():
        for leaf, value in leaves.items():
            if module in (scope + "hidden", scope + "output"):
                want = saved[module][leaf]
            elif module.startswith(scope):
                want = fresh[module][leaf]  # mod_* stay fresh
            else:
                want = base._params[module][leaf]
            np.testing.assert_array_equal(np.asarray(value), np.asarray(want))


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
    from tisiago.seq2func.model import make_schedule

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
    from tisiago.seq2func.model import restore_init

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
    from tisiago.seq2func.model import restore_init

    old, new = "p6_test_init_mlp", "p6_test_init_seqconv"
    register_track_head(old, 2, kind="mlp")
    register_track_head(new, 2, kind="seqconv")
    source = _FakeModel([old], 2, kind="mlp")
    source._params = _perturbed(source._params, 0.5)
    _save_model(source, tmp_path / "best")
    target = _FakeModel([new], 2, kind="seqconv")
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
