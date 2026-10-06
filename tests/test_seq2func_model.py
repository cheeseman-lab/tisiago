"""CPU-safe tests for the track head wiring (no checkpoint load)."""

import numpy as np
import pytest

pytest.importorskip("alphagenome_ft")
import haiku as hk  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tisiago.seq2func.model import TrackHead, register_track_head  # noqa: E402


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
