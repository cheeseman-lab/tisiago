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
