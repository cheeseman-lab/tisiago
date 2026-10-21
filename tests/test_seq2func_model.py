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
    assert SEQUENCE_KINDS == {
        "seqconv",
        "seqconv_tx",
        "resid_tx",
        "resid_chx",
        "rnacond",
        "rnacond_f",
        "rnacond_fo",
        "rnacond_foc",
        "rnacond_fod",
        "rnacond_foa",
        "rnacond_focd",
        "rnacond_foca",
        "rnacond_fc2",
        "rnacond_fc2d",
    }
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


@pytest.mark.parametrize(
    "kind", ["mlp", "conv", "seqconv", "seqconv_tx", "resid_tx", "resid_chx", "rnacond"]
)
def test_load_model_installs_sequence_forward_only_for_sequence_kinds(monkeypatch, kind):
    import types

    from tisiago.seq2func import model as model_mod

    head = f"p6_test_load_{kind}"
    register_track_head(head, 2, kind=kind, mod_channels=(1,) if kind == "resid_chx" else None)
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
    if kind in ("seqconv", "seqconv_tx", "resid_tx", "resid_chx", "rnacond"):
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


# --- resid_chx: resid_tx's modulation on the CHX channels only, own learning rate ---


def _chx_forward(kind, emb, seq=None, channels=(1, 3), scales=None):
    from tisiago.seq2func.model import HEAD_KINDS

    metadata = {"mod_channels": channels} if kind == "resid_chx" else None
    if scales is not None:
        metadata["mod_scales"] = scales
    head = HEAD_KINDS[kind](name="h", output_type=None, num_tracks=4, metadata=metadata)
    kwargs = {} if seq is None else {"sequence": seq}
    return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), **kwargs)


def test_resid_chx_registered_like_resid_tx():
    from tisiago.seq2func import overfit
    from tisiago.seq2func.model import (
        HEAD_KINDS,
        SEQUENCE_KINDS,
        SEQUENCE_WIDTHS,
        ResidChxTrackHead,
    )

    assert HEAD_KINDS["resid_chx"] is ResidChxTrackHead
    assert "resid_chx" in SEQUENCE_KINDS
    assert SEQUENCE_WIDTHS["resid_chx"] == SEQUENCE_WIDTHS["resid_tx"]
    assert "resid_chx" in overfit.HEAD_CHOICES and "resid_chx" in overfit.TRANSCRIPT_KINDS
    assert overfit.MOD_KINDS == {"resid_tx", "resid_chx"}
    assert overfit.CHX_KINDS == {"resid_chx"}


def test_resid_chx_equals_mlp_at_init_and_modulates_only_chx_channels():
    from tisiago.seq2func.model import MOD_HIDDEN_CHX, MOD_SCALE

    mlp = hk.without_apply_rng(hk.transform(lambda e: _chx_forward("mlp", e)))
    chx = hk.without_apply_rng(hk.transform(lambda e, q: _chx_forward("resid_chx", e, q)))
    rng = np.random.default_rng(0)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)).astype(np.float32))
    seq = _tx_sequences(rng, 1, 40)
    mlp_params = mlp.init(jax.random.PRNGKey(0), emb)
    params = chx.init(jax.random.PRNGKey(1), emb, seq)
    assert MOD_HIDDEN_CHX == 256
    assert params["h/~predict/mod_hidden"]["w"].shape == (31, 8 + 24, MOD_HIDDEN_CHX)
    assert params["h/~predict/mod_output"]["w"].shape == (1, MOD_HIDDEN_CHX, 2)
    for leaf in params["h/~predict/mod_output"].values():
        assert float(jnp.abs(leaf).max()) == 0.0
    params = {**params, **mlp_params}
    base = np.asarray(mlp.apply(mlp_params, emb))
    out = np.asarray(chx.apply(params, emb, seq))
    np.testing.assert_array_equal(out, base)
    for sign in (1.0, -1.0):
        big = {"w": sign * 1e4 * jnp.ones((1, MOD_HIDDEN_CHX, 2)), "b": sign * 1e4 * jnp.ones(2)}
        out = np.asarray(chx.apply({**params, "h/~predict/mod_output": big}, emb, seq))
        np.testing.assert_array_equal(out[..., [0, 2]], base[..., [0, 2]])
        assert not np.allclose(out[..., [1, 3]], base[..., [1, 3]])
        assert np.isfinite(out).all()
        np.testing.assert_array_less(out, base * np.exp(MOD_SCALE) * (1 + 1e-5) + 1e-30)
        np.testing.assert_array_less(base * np.exp(-MOD_SCALE) * (1 - 1e-5) - 1e-30, out)


def test_resid_chx_needs_mod_channels():
    from tisiago.seq2func.model import sequence_free_init

    chx = hk.without_apply_rng(
        hk.transform(lambda e, q: _chx_forward("resid_chx", e, q, channels=()))
    )
    emb = jnp.zeros((1, 8, 8), jnp.float32)
    with pytest.raises(ValueError, match="mod_channels"), sequence_free_init():
        chx.init(jax.random.PRNGKey(0), emb, None)
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_chx_none", 4, kind="resid_chx")
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_chx_range", 4, kind="resid_chx", mod_channels=(1, 4))
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_chx_dup", 4, kind="resid_chx", mod_channels=(1, 1))
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_chx_mlp", 4, kind="mlp", mod_channels=(1,))
    register_track_head("p6_test_chx_ok", 4, kind="resid_chx", mod_channels=[1, 3])
    register_track_head("p6_test_chx_ok", 4, kind="resid_chx", mod_channels=(1, 3))
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_chx_ok", 4, kind="resid_chx", mod_channels=(0, 2))


def _mod_fake_params():
    return {
        **_fake_params(),
        "head/h/~predict/mod_hidden": {"w": jnp.ones((3, 2)), "b": jnp.ones((2,))},
        "head/h/~predict/mod_output": {"w": jnp.ones((2, 2))},
        "head/other/~predict/mod_output": {"w": jnp.ones((2, 2))},
    }


def test_label_backbone_params_mod_label():
    params = _mod_fake_params()
    plain = label_backbone_params(params, "h")
    assert plain["head/h/~predict/mod_output"] == {"w": "head"}
    labels = label_backbone_params(params, "h", mod=True)
    assert labels["head/h/~predict/mod_hidden"] == {"w": "mod", "b": "mod"}
    assert labels["head/h/~predict/mod_output"] == {"w": "mod"}
    assert labels["head/h/~predict/hidden"] == {"w": "head", "b": "head"}
    assert labels["head/other/~predict/mod_output"] == {"w": "backbone"}
    assert labels["alphagenome/sequence_encoder/conv"] == {"w": "backbone"}


@pytest.mark.parametrize("heads_only", [True, False])
def test_make_optimizer_mod_learning_rate(heads_only):
    params = _mod_fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = make_optimizer(
        params,
        "h",
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=heads_only,
        backbone_learning_rate=1e-4,
        mod_learning_rate=1e-3,
        total_steps=40,
        warmup_steps=0,
        clip_norm=1e6,
    )
    updates, _ = opt.update(grads, opt.init(params), params)

    def step(module):
        return float(jnp.abs(updates[module]["w"]).mean())

    assert step("head/h/~predict/hidden") == pytest.approx(1e-2, rel=1e-3)
    assert step("head/h/~predict/mod_hidden") == pytest.approx(1e-3, rel=1e-3)
    assert step("head/h/~predict/mod_output") == pytest.approx(1e-3, rel=1e-3)
    bb = step("alphagenome/sequence_encoder/conv")
    assert bb == 0.0 if heads_only else bb == pytest.approx(1e-4, rel=1e-3)


def test_make_optimizer_mod_learning_rate_none_matches_default():
    params = _mod_fake_params()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for heads_only in (True, False):
        kw = dict(learning_rate=1e-2, weight_decay=1e-4, heads_only=heads_only, total_steps=10)
        a, b = (
            make_optimizer(params, "h", **kw),
            make_optimizer(params, "h", mod_learning_rate=None, **kw),
        )
        ua, _ = a.update(grads, a.init(params), params)
        ub, _ = b.update(grads, b.init(params), params)
        for x, y in zip(jax.tree_util.tree_leaves(ua), jax.tree_util.tree_leaves(ub), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _chx_model(head, seq_len=48, channels=(1,)):
    register_track_head(head, 2, kind="resid_chx", mod_channels=channels)
    model = _FakeModel([head], 2, kind="resid_chx", seq_len=seq_len, width=24)
    from tisiago.seq2func.model import install_sequence_forward

    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    return model


@pytest.mark.parametrize("backbone_lr", [0.0, 1e-3])
def test_resid_chx_step_with_zero_head_lr_moves_only_mod_leaves(backbone_lr):
    from tisiago.seq2func.model import make_train_step

    head = "p6_test_chx_step"
    model = _chx_model(head)
    rng = np.random.default_rng(0)
    batch = {
        "sequences": _tx_sequences(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 2)).astype(np.float32)),
        "mask": jnp.ones((2, 48, 2), bool),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=0.0,
        weight_decay=1e-4,
        heads_only=False,
        backbone_learning_rate=backbone_lr,
        mod_learning_rate=1e-2,
        frame_weight=5.0,
        frame_channels=(1,),
        frame_block=2,
    )
    params, state = model._params, model._state
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"]))
    scope = f"head/{head}/~predict/"
    for module, leaves in params.items():
        for name, value in leaves.items():
            moved = not np.array_equal(np.asarray(value), np.asarray(new_params[module][name]))
            if module == scope + "mod_output":
                assert moved
            elif module.startswith(scope + "mod_"):
                pass  # mod_hidden: zero gradient behind the zero mod_output, weight decay only
            elif module.startswith(scope):
                assert not moved, module  # the base branch at head lr 0
            elif backbone_lr == 0.0:
                assert not moved, module
    if backbone_lr:
        assert not np.array_equal(
            np.asarray(params["fake_backbone"]["w"]), np.asarray(new_params["fake_backbone"]["w"])
        )


def test_resid_chx_checkpoint_round_trip_and_matching_init(tmp_path, monkeypatch):
    from alphagenome_ft.custom_heads import get_custom_head_config

    from tisiago.seq2func import model as model_mod
    from tisiago.seq2func.model import make_predict, restore_init

    head = "p6_test_chx_ck"
    source = _chx_model(head)
    source._params = _perturbed(source._params, 0.25)
    _save_model(source, tmp_path / "best")
    seq = _tx_sequences(np.random.default_rng(1), 1, 48)
    want = np.asarray(make_predict(source, head)(source._params, source._state, seq))

    def fake_load_model(checkpoint, name, *, detach_backbone=True, device=None):
        model = _FakeModel([name], 2, kind="resid_chx", width=24)
        return model_mod.install_sequence_forward(
            model, [name], detach_backbone=detach_backbone, backbone=_fake_backbone()
        )

    monkeypatch.setattr(model_mod, "load_model", fake_load_model)
    loaded = model_mod.load_finetuned(
        str(tmp_path / "best"), "weights", head, 2, "resid_chx", mod_channels=(1,)
    )
    assert tuple(get_custom_head_config(head).metadata["mod_channels"]) == (1,)
    np.testing.assert_array_equal(
        np.asarray(make_predict(loaded, head)(loaded._params, loaded._state, seq)), want
    )
    with pytest.raises(ValueError, match="mod_channels"):
        model_mod.load_finetuned(
            str(tmp_path / "best"), "weights", head, 2, "resid_chx", mod_channels=(0,)
        )

    mlp_head = "p6_test_chx_mlp"
    register_track_head(mlp_head, 2, kind="mlp")
    base = _FakeModel([mlp_head], 2, kind="mlp")
    base._params = _perturbed(base._params, 0.5)
    name = mlp_head + "_r"
    base_scope = f"head/{mlp_head}/~predict/"
    saved = {k.replace(base_scope, f"head/{name}/~predict/"): v for k, v in base._params.items()}
    renamed = _FakeModel([mlp_head], 2, kind="mlp")
    renamed._params, renamed._state = saved, base._state
    _save_model(renamed, tmp_path / "mlp_as_chx")
    target = _chx_model(name)
    fresh = jax.tree_util.tree_map(np.asarray, target._params)
    restore_init(target, str(tmp_path / "mlp_as_chx"), name, reuse_head="matching")
    scope = f"head/{name}/~predict/"
    for module, leaves in target._params.items():
        for leaf, value in leaves.items():
            if module in (scope + "hidden", scope + "output"):
                want = saved[module][leaf]
            elif module.startswith(scope):
                want = fresh[module][leaf]
            else:
                want = base._params[module][leaf]
            np.testing.assert_array_equal(np.asarray(value), np.asarray(want))
    # matching init leaves predictions equal to the mlp source's (mod_output zero)
    mlp_pred = np.asarray(_mlp_predict(base, mlp_head, seq))
    chx_pred = np.asarray(make_predict(target, name)(target._params, target._state, seq))
    np.testing.assert_array_equal(chx_pred, mlp_pred)


# --- resid_chx on more libraries: per-channel modulation bound (--mod-scale-tis) ---


def test_resid_chx_per_channel_scales_bound_each_channel():
    from tisiago.seq2func.model import MOD_HIDDEN_CHX, MOD_SCALE

    mlp = hk.without_apply_rng(hk.transform(lambda e: _chx_forward("mlp", e)))
    chx = hk.without_apply_rng(
        hk.transform(lambda e, q: _chx_forward("resid_chx", e, q, scales=(MOD_SCALE, 3.0)))
    )
    rng = np.random.default_rng(0)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)).astype(np.float32))
    seq = _tx_sequences(rng, 1, 40)
    mlp_params = mlp.init(jax.random.PRNGKey(0), emb)
    params = {**chx.init(jax.random.PRNGKey(1), emb, seq), **mlp_params}
    base = np.asarray(mlp.apply(mlp_params, emb))
    np.testing.assert_array_equal(np.asarray(chx.apply(params, emb, seq)), base)
    for sign in (1.0, -1.0):
        big = {"w": sign * 1e4 * jnp.ones((1, MOD_HIDDEN_CHX, 2)), "b": sign * 1e4 * jnp.ones(2)}
        out = np.asarray(chx.apply({**params, "h/~predict/mod_output": big}, emb, seq))
        np.testing.assert_array_equal(out[..., [0, 2]], base[..., [0, 2]])
        for channel, scale in ((1, MOD_SCALE), (3, 3.0)):
            ratio = out[..., channel] / base[..., channel]
            np.testing.assert_allclose(ratio, np.exp(sign * scale), rtol=1e-5)


def test_resid_chx_default_scales_match_old_metadata():
    from tisiago.seq2func.model import MOD_HIDDEN_CHX, MOD_SCALE

    old = hk.without_apply_rng(hk.transform(lambda e, q: _chx_forward("resid_chx", e, q)))
    new = hk.without_apply_rng(
        hk.transform(lambda e, q: _chx_forward("resid_chx", e, q, scales=(MOD_SCALE,) * 2))
    )
    rng = np.random.default_rng(3)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)).astype(np.float32))
    seq = _tx_sequences(rng, 1, 40)
    params = old.init(jax.random.PRNGKey(1), emb, seq)
    params["h/~predict/mod_output"] = {
        "w": jnp.asarray(rng.normal(size=(1, MOD_HIDDEN_CHX, 2)).astype(np.float32)),
        "b": jnp.asarray([0.3, -0.7], jnp.float32),
    }
    np.testing.assert_array_equal(
        np.asarray(old.apply(params, emb, seq)), np.asarray(new.apply(params, emb, seq))
    )
    register_track_head("p6_test_chx_old", 4, kind="resid_chx", mod_channels=(1, 3))
    register_track_head(
        "p6_test_chx_old", 4, kind="resid_chx", mod_channels=(1, 3), mod_scales=(MOD_SCALE,) * 2
    )


def test_resid_chx_mod_scales_validation():
    with pytest.raises(ValueError, match="mod_scales"):
        register_track_head(
            "p6_test_sc_len", 4, kind="resid_chx", mod_channels=(1, 3), mod_scales=(1.5,)
        )
    for bad in ((1.5, 0.0), (1.5, -2.0), (1.5, float("nan")), (1.5, float("inf"))):
        with pytest.raises(ValueError, match="mod_scales"):
            register_track_head(
                "p6_test_sc_bad", 4, kind="resid_chx", mod_channels=(1, 3), mod_scales=bad
            )
    with pytest.raises(ValueError, match="mod_scales"):
        register_track_head("p6_test_sc_mlp", 4, kind="mlp", mod_scales=(1.5,))
    register_track_head(
        "p6_test_sc_ok", 4, kind="resid_chx", mod_channels=(1, 3), mod_scales=(1.5, 3)
    )
    with pytest.raises(ValueError, match="mod_scales"):
        register_track_head(
            "p6_test_sc_ok", 4, kind="resid_chx", mod_channels=(1, 3), mod_scales=(1.5, 2.0)
        )


def test_resid_chx_spec_from_libraries_and_tis_scale():
    from tisiago.seq2func.model import MOD_SCALE, resid_chx_spec
    from tisiago.seq2func.targets import channel_layout, library_channels

    layout = channel_layout(["HeLa", "K562"])
    channels, scales = resid_chx_spec(layout)
    assert channels == library_channels(layout, "chx") and scales == (MOD_SCALE,) * 4
    channels, scales = resid_chx_spec(layout, ("chx", "tis"), scale_tis=3.0)
    assert channels == (0, 1, 2, 3, 6, 7, 8, 9)
    assert scales == tuple(3.0 if layout[c][1] == "tis" else MOD_SCALE for c in channels)
    assert scales.count(3.0) == 4


def test_resid_chx_round_trip_with_scales(tmp_path, monkeypatch):
    from alphagenome_ft.custom_heads import get_custom_head_config

    from tisiago.seq2func import model as model_mod
    from tisiago.seq2func.model import MOD_SCALE, make_predict

    head = "p6_test_chx_ck_sc"
    register_track_head(head, 2, kind="resid_chx", mod_channels=(0, 1), mod_scales=(3.0, MOD_SCALE))
    source = _FakeModel([head], 2, kind="resid_chx", seq_len=48, width=24)
    model_mod.install_sequence_forward(
        source, [head], detach_backbone=False, backbone=_fake_backbone()
    )
    source._params = _perturbed(source._params, 0.25)
    _save_model(source, tmp_path / "best")
    seq = _tx_sequences(np.random.default_rng(1), 1, 48)
    want = np.asarray(make_predict(source, head)(source._params, source._state, seq))

    def fake_load_model(checkpoint, name, *, detach_backbone=True, device=None):
        model = _FakeModel([name], 2, kind="resid_chx", width=24)
        return model_mod.install_sequence_forward(
            model, [name], detach_backbone=detach_backbone, backbone=_fake_backbone()
        )

    monkeypatch.setattr(model_mod, "load_model", fake_load_model)
    loaded = model_mod.load_finetuned(
        str(tmp_path / "best"),
        "weights",
        head,
        2,
        "resid_chx",
        mod_channels=(0, 1),
        mod_scales=(3.0, MOD_SCALE),
    )
    metadata = get_custom_head_config(head).metadata
    assert tuple(metadata["mod_scales"]) == (3.0, MOD_SCALE)
    np.testing.assert_array_equal(
        np.asarray(make_predict(loaded, head)(loaded._params, loaded._state, seq)), want
    )
    with pytest.raises(ValueError, match="mod_scales"):
        model_mod.load_finetuned(
            str(tmp_path / "best"), "weights", head, 2, "resid_chx", mod_channels=(0, 1)
        )


def _mlp_predict(model, head, seq):
    from tisiago.seq2func.model import install_sequence_forward, make_predict

    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    return make_predict(model, head)(model._params, model._state, seq[..., :4])


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


# --- rnacond: embeddings + RNA features of the condition's raw pooled RNA input ---


def _rna_sequences(rng, batch, length):
    onehot = np.asarray(_random_onehot(rng, batch, length))
    rna = rng.poisson(1.5, (batch, length, 2)).astype(np.float32)
    return jnp.asarray(np.concatenate([onehot, rna], axis=-1))


def _rnacond_forward(emb, seq=None, num_tracks=4):
    from tisiago.seq2func.model import RnaCondTrackHead

    head = RnaCondTrackHead(name="r", output_type=None, num_tracks=num_tracks)
    kwargs = {} if seq is None else {"sequence": seq}
    return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), **kwargs)


def test_rnacond_registered_as_sequence_kind():
    from tisiago.seq2func.model import (
        HEAD_KINDS,
        SEQUENCE_KINDS,
        SEQUENCE_WIDTHS,
        RnaCondTrackHead,
    )

    assert HEAD_KINDS["rnacond"] is RnaCondTrackHead
    assert "rnacond" in SEQUENCE_KINDS and SEQUENCE_WIDTHS["rnacond"] == 6
    register_track_head("p6_test_rnacond_reg", 4, kind="rnacond")
    register_track_head("p6_test_rnacond_reg", 4, kind="rnacond")
    with pytest.raises(ValueError, match="kind"):
        register_track_head("p6_test_rnacond_reg", 4, kind="mlp")
    with pytest.raises(ValueError, match="mod_channels"):
        register_track_head("p6_test_rnacond_mod", 4, kind="rnacond", mod_channels=(2,))


def test_rnacond_head_input_is_embeddings_and_rna_features():
    from tisiago.seq2func.rna_features import NUM_RNA_FEATURES, rna_features

    fn = hk.without_apply_rng(hk.transform(_rnacond_forward))
    emb = jnp.asarray(np.random.default_rng(2).normal(size=(1, 40, 8)), jnp.bfloat16)
    rng = np.random.default_rng(0)
    seq = _rna_sequences(rng, 1, 40)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    assert NUM_RNA_FEATURES == 10
    assert params["r/~predict/hidden"]["w"].shape == (8 + 10, 256)
    assert params["r/~predict/output"]["w"].shape == (256, 4)
    pred = fn.apply(params, emb, seq)
    assert pred.shape == (1, 40, 4) and pred.dtype == jnp.float32 and bool((pred >= 0).all())
    # the RNA channels are used; the one-hot reaches the head only through the backbone
    other_rna = seq.at[..., 4:].set(0.0)
    assert not np.allclose(np.asarray(pred), np.asarray(fn.apply(params, emb, other_rna)))
    other_dna = seq.at[..., :4].set(_random_onehot(np.random.default_rng(9), 1, 40))
    np.testing.assert_array_equal(np.asarray(pred), np.asarray(fn.apply(params, emb, other_dna)))
    # the documented definition, by hand
    x = jnp.concatenate([emb.astype(jnp.float32), rna_features(seq[..., 4:])], axis=-1)
    hidden, output = params["r/~predict/hidden"], params["r/~predict/output"]
    h = jax.nn.relu(x @ hidden["w"] + hidden["b"])
    expected = jax.nn.softplus(h @ output["w"] + output["b"])
    np.testing.assert_allclose(np.asarray(pred), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_rnacond_head_requires_six_channel_sequence():
    from tisiago.seq2func.model import sequence_free_init

    fn = hk.without_apply_rng(hk.transform(_rnacond_forward))
    emb = jnp.zeros((1, 16, 8), jnp.float32)
    seq = _rna_sequences(np.random.default_rng(0), 1, 16)
    with pytest.raises(ValueError, match="sequence"):
        fn.init(jax.random.PRNGKey(0), emb)
    with sequence_free_init(True):
        free = fn.init(jax.random.PRNGKey(0), emb)
    assert free["r/~predict/hidden"]["w"].shape == (8 + 10, 256)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    with pytest.raises(ValueError, match="6 channels"):
        fn.apply(params, emb, seq[..., :4])


def test_rnacond_jitted_train_step_has_finite_grads_and_frame_loss():
    from tisiago.seq2func.model import install_sequence_forward, make_predict, make_train_step

    head = "p6_test_rnacond_train"
    register_track_head(head, 4, kind="rnacond")
    model = _FakeModel([head], 4, kind="rnacond", width=6)
    install_sequence_forward(model, [head], detach_backbone=False, backbone=_fake_backbone())
    rng = np.random.default_rng(0)
    mask = np.ones((2, 48, 4), bool)
    mask[0, :10, :2] = False
    batch = {
        "sequences": _rna_sequences(rng, 2, 48),
        "targets": jnp.asarray(rng.poisson(2, (2, 48, 4)).astype(np.float32)),
        "mask": jnp.asarray(mask),
    }
    step, opt_state = make_train_step(
        model,
        head,
        learning_rate=1e-2,
        weight_decay=0.0,
        heads_only=False,
        frame_weight=5.0,
        frame_channels=(2, 3),
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
    assert w.shape == (8 + 10, 256)
    assert float(jnp.abs(w[-10:]).sum()) > 0  # gradient reaches the RNA-feature rows
    assert float(jnp.abs(grads["fake_backbone"]["w"]).sum()) > 0
    assert grads["fake_backbone"]["w"].shape == (4, 8)  # the backbone sees the one-hot only
    new_params, _, aux = step(params, state, opt_state, batch)
    assert np.isfinite(float(aux["loss"])) and float(aux["frame"]) > 0
    before = np.asarray(params[f"head/{head}/~predict/hidden"]["w"])
    assert not np.allclose(before, np.asarray(new_params[f"head/{head}/~predict/hidden"]["w"]))
    pred = make_predict(model, head)(new_params, state, batch["sequences"])
    assert pred.shape == (2, 48, 4) and bool(jnp.isfinite(pred).all())


def test_rnacond_checkpoint_round_trip(tmp_path):
    from tisiago.seq2func.model import install_sequence_forward, make_predict, restore_init

    head = "p6_test_rnacond_ck"
    register_track_head(head, 4, kind="rnacond")

    def build():
        model = _FakeModel([head], 4, kind="rnacond", width=6)
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

FACT_KINDS = (
    "rnacond_f",
    "rnacond_fo",
    "rnacond_foc",
    "rnacond_fod",
    "rnacond_foa",
    "rnacond_focd",
    "rnacond_foca",
)


def test_local_softmax_preserves_totals_and_concentrates():
    from tisiago.seq2func.model import local_softmax

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
    from tisiago.seq2func.model import HEAD_KINDS

    def forward(emb, seq):
        head = HEAD_KINDS[kind](name="r", output_type=None, num_tracks=4)
        return head(_FakeEmbeddings(emb), jnp.zeros((1,), jnp.int32), sequence=seq)

    return hk.without_apply_rng(hk.transform(forward))


@pytest.mark.parametrize("kind", FACT_KINDS)
def test_fact_heads_registered_shapes_and_rna_use(kind):
    from tisiago.seq2func import overfit
    from tisiago.seq2func.model import SEQUENCE_WIDTHS
    from tisiago.seq2func.rna_features import NUM_ONESIDED_FEATURES, NUM_RNA_FEATURES

    assert SEQUENCE_WIDTHS[kind] == 6 and kind in overfit.RNA_KINDS
    register_track_head(f"p6_test_{kind}_reg", 4, kind=kind)
    fn = _fact_forward(kind)
    rng = np.random.default_rng(1)
    emb = jnp.asarray(rng.normal(size=(1, 40, 8)), jnp.float32)
    seq = _rna_sequences(rng, 1, 40)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    n_feat = NUM_RNA_FEATURES + (0 if kind == "rnacond_f" else NUM_ONESIDED_FEATURES)
    assert params["r/~predict/hidden"]["w"].shape == (8 + n_feat, 256)
    assert params["r/~predict/output"]["w"].shape == (256, 2)
    assert ("r/~predict/tis_hidden" in params) == kind.startswith("rnacond_foc")
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
    from tisiago.seq2func import evaluate, overfit, periodicity
    from tisiago.seq2func.model import SEQUENCE_WIDTHS
    from tisiago.seq2func.rna_features import NUM_ONESIDED_FEATURES, NUM_RNA_FEATURES

    assert SEQUENCE_WIDTHS["rnacond_fc2"] == 6 and "rnacond_fc2" in overfit.RNA_KINDS
    assert "rnacond_fc2" in evaluate.HEAD_CHOICES and "rnacond_fc2" in periodicity.HEAD_CHOICES
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
    from tisiago.seq2func.model import CONV9_KERNEL, LOCAL_BLOCK

    fn, emb, seq, params = _fc2_setup()
    pred = np.asarray(fn.apply(params, emb, seq))
    from tisiago.seq2func.rna_features import onesided_features, rna_features

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


def test_fact_chx_contrast_stops_tis_gradient_into_chx_output():
    fn = _fact_forward("rnacond_foc")
    rng = np.random.default_rng(2)
    emb = jnp.asarray(rng.normal(size=(1, 64, 8)), jnp.float32)
    seq = _rna_sequences(rng, 1, 64)
    params = fn.init(jax.random.PRNGKey(0), emb, seq)
    grads = jax.grad(lambda p: fn.apply(p, emb, seq)[..., :2].sum())(params)
    np.testing.assert_array_equal(np.asarray(grads["r/~predict/output"]["w"]), 0.0)
    assert float(jnp.abs(grads["r/~predict/tis_hidden"]["w"]).sum()) > 0
    # but TIS does read the CHX prediction: changing the CHX output weights changes TIS
    moved = jax.tree_util.tree_map(lambda x: x, params)
    moved["r/~predict/output"] = {
        "w": params["r/~predict/output"]["w"] * 3.0,
        "b": params["r/~predict/output"]["b"] + 1.0,
    }
    a, b = fn.apply(params, emb, seq), fn.apply(moved, emb, seq)
    assert not np.allclose(np.asarray(a[..., :2]), np.asarray(b[..., :2]))


@pytest.mark.parametrize(
    ("kind", "shallow_kind"),
    [
        ("rnacond_fod", "rnacond_fo"),
        ("rnacond_foa", "rnacond_fo"),
        ("rnacond_focd", "rnacond_foc"),
        ("rnacond_foca", "rnacond_foc"),
    ],
)
def test_fact_deep_trunks_start_as_the_mlp_trunk(kind, shallow_kind):
    """Zero-initialised residual branches: at init the trunk equals the shallow head's."""
    rng = np.random.default_rng(3)
    emb = jnp.asarray(rng.normal(size=(1, 48, 8)), jnp.float32)
    seq = _rna_sequences(rng, 1, 48)
    deep = _fact_forward(kind)
    params = deep.init(jax.random.PRNGKey(0), emb, seq)
    plain = _fact_forward(shallow_kind)
    shallow = {k: v for k, v in params.items() if k in plain.init(jax.random.PRNGKey(0), emb, seq)}
    np.testing.assert_allclose(
        np.asarray(deep.apply(params, emb, seq)),
        np.asarray(plain.apply(shallow, emb, seq)),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("kind", (*FACT_KINDS, "rnacond_fc2"))
def test_fact_jitted_train_step_with_local_loss(kind):
    from tisiago.seq2func.model import install_sequence_forward, make_predict, make_train_step

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
    """local_weight 0 leaves the rnacond loss exactly as before (no local term)."""
    from tisiago.seq2func.model import install_sequence_forward, make_train_step

    head = "p6_test_rnacond_nolocal"
    register_track_head(head, 4, kind="rnacond")
    model = _FakeModel([head], 4, kind="rnacond", width=6)
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
    from tisiago.seq2func import overfit

    assert "rnacond_fc2d" in overfit.RNACOND_FACT_KINDS
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
    from tisiago.seq2func.model import install_sequence_forward, make_train_step

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
