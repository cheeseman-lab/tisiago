"""Tests for the spliced-input backbone trainer (CPU, synthetic cache, stub backbone)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("haiku")
pytest.importorskip("optax")
jnp = jax.numpy

from tests.test_tracks_fastloop import (  # noqa: E402, F401
    CONDITIONS,
    _bw_values,
    _real_tracks,
    _test_cache,
    cache,
)
from tisiago.tracks import fastloop  # noqa: E402
from tisiago.tracks import train_spliced as txtrain  # noqa: E402
from tisiago.tracks.txcache import TxCache  # noqa: E402

MIN, MAX = 64, 128


@pytest.fixture
def txc(cache):  # noqa: F811
    """The fast-loop cache plus ``seq.npy`` (some N) in every shard."""
    rng = np.random.default_rng(3)
    for shard in cache.index.shard.unique():
        size = int(cache.index[cache.index.shard == shard].length.sum())
        codes = rng.integers(0, 5, size=size).astype(np.int8)
        np.save(cache.path / "shards" / shard / "seq.npy", codes)
    return TxCache(cache.path)


def _row(c, tid):
    return next(r for r in c.index.itertuples() if r.transcript_id == tid)


def test_buckets_and_tiles_partition_the_transcript():
    assert txtrain.bucket_of(50, MIN, MAX) == 64 and txtrain.bucket_of(65, MIN, MAX) == 128
    assert txtrain.bucket_of(400, MIN, MAX) == 128
    assert txtrain.tiles(100, 128) == [(0, 0, 100)]
    for length in (129, 200, 400, 1000):
        tiles = txtrain.tiles(length, 128)
        owner = np.full(length, -1)
        for k, (start, lo, hi) in enumerate(tiles):
            assert start <= lo < hi <= start + 128 and np.all(owner[lo:hi] == -1)
            owner[lo:hi] = k
        assert (owner >= 0).all()
        centres = np.array([s + 63.5 for s, _, _ in tiles])
        dist = np.abs(np.arange(length)[:, None] - centres[None])
        nearest = dist.argmin(1)  # ties to the earlier window
        np.testing.assert_array_equal(owner, nearest)


def test_transcript_example_crops_pads_and_selects_conditions(txc):
    row = _row(txc, "A")  # length 300, train shard offset 0
    ex = txtrain.transcript_example(txc, row, ("HeLa", "U2OS"), 250, 128)
    assert ex["valid"].sum() == 50 and not ex["valid"][50:].any()
    seq = np.load(txc.path / "shards" / "train_chr1" / "seq.npy")
    np.testing.assert_array_equal(ex["seq"][:50], seq[250:300])
    assert (ex["seq"][50:] == fastloop.SEQ_N).all()
    # rna[:, :, 0] encodes shard position + 1000 x cache condition index
    np.testing.assert_array_equal(ex["rna"][0, :50, 0], np.arange(250, 300))
    np.testing.assert_array_equal(ex["rna"][1, :50, 0], np.arange(250, 300) + 2000)
    assert (ex["rna"][:, 50:] == 0).all() and (ex["target"][:, 50:] == 0).all()
    tis = np.load(txc.path / "shards" / "train_chr1" / "tis.npy")
    np.testing.assert_array_equal(ex["target"][1, :50, 0], tis[2, 250:300, 2])
    np.testing.assert_array_equal(ex["pos"][:50], np.arange(250, 300))
    assert ex["expressed"].tolist() == [True, True] and ex["rna_tx"].shape == (2, 2)
    old = fastloop.crop_example(txc, row, "U2OS", 250, 128)
    np.testing.assert_allclose(ex["rna_tx"][1], old["rna_tx"])
    np.testing.assert_array_equal(ex["exon"], old["exon"])
    np.testing.assert_array_equal(ex["target"][1], old["target"])


def test_predict_transcripts_stitches_tiles_exactly(txc):
    rows = txc.rows("val")

    def predict(ex):  # per tile: (cache RNA code, absolute transcript position)
        pos = np.broadcast_to(ex["pos"], ex["rna"].shape[:2])
        return np.stack([ex["rna"][..., 0], pos], axis=-1)

    preds = txtrain.predict_transcripts(predict, txc, rows, txc.conditions, MIN, MAX)
    for r in rows.itertuples():
        pred = preds[r.transcript_id]
        assert pred.shape == (3, r.length, 2)
        np.testing.assert_array_equal(pred[0, :, 1], np.arange(r.length))
        rna = np.load(txc.path / "shards" / r.shard / "rna.npy")
        np.testing.assert_array_equal(pred[..., 0], rna[:, r.offset : r.offset + r.length, 0])


def test_sampler_is_seeded_and_never_reads_heldout_conditions(txc):
    config = fastloop.resolve_config({"heldout": ["K562"]})
    train = fastloop.train_conditions_of(config, txc.conditions)
    assert train == ("HeLa", "U2OS")
    sampler = txtrain.TrainSampler(txc, train, seed=1, min_bucket=MIN, max_bucket=MAX)
    assert "C" not in set(sampler.rows.transcript_id)  # expressed in K562 only
    seen = set()
    for i in range(40):
        ex = sampler.example(i)
        row, start, size = sampler.draw(i)
        assert ex["rna"].shape[0] == 2 and ex["seq"].shape == (size,)
        n = int(ex["valid"].sum())
        shard_pos = row.offset + start + np.arange(n)
        k = np.rint((ex["rna"][:, :n, 0] - shard_pos[None]) / 1000).astype(int)
        seen |= set(np.unique(k).tolist())
        assert row.split == "train"
        np.testing.assert_array_equal(sampler.example(i)["seq"], ex["seq"])
    assert seen == {0, 2}  # HeLa, U2OS; never K562 (1)


def test_example_loss_ignores_unexpressed_conditions():
    rng = np.random.default_rng(0)
    pred = jnp.asarray(rng.gamma(1.0, size=(2, 32, 2)), jnp.float32)
    batch = {
        "target": jnp.asarray(rng.poisson(1.0, size=(2, 32, 2)), jnp.float32),
        "valid": jnp.ones((2, 32), bool),
    }
    losses = fastloop.DEFAULTS["losses"]
    both, _ = txtrain.example_loss(pred, batch, {}, losses, jnp.array([True, False]))
    one, _ = fastloop.total_loss(
        pred[:1], {"target": batch["target"][:1], "valid": batch["valid"][:1]}, {}, losses
    )
    assert float(both) == pytest.approx(float(one), rel=1e-6)


def test_pca_projection_equals_the_cache_pca(tmp_path, txc):
    rng = np.random.default_rng(1)
    pca = {"mean": rng.normal(size=16).astype("f4"), "components": rng.normal(size=(8, 16))}
    np.savez(txc.path / "pca.npz", **pca, explained_variance=np.ones(8))
    proj = txtrain.pca_projection(txc, 16, 8)
    x = rng.normal(size=(5, 16)).astype("f4")
    np.testing.assert_allclose(
        x @ proj["w"] + proj["b"], (x - pca["mean"]) @ pca["components"].T, rtol=1e-4, atol=1e-4
    )
    assert txtrain.pca_projection(txc, 32, 8) is None


def test_export_is_fastloop_export_of_the_same_predictions(tmp_path, txc):
    import pyBigWig

    from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
    from tisiago.tracks.txcache import unit_of

    tracks = _real_tracks(tmp_path / "tracks")
    txc.config["tracks_dir"] = str(tracks)
    rows = txc.rows("val")

    def predict(ex):
        return np.stack([ex["rna"][..., 1], ex["target"][..., 1] + ex["pos"]], axis=-1)

    preds = txtrain.predict_transcripts(predict, txc, rows, txc.conditions, MIN, MAX)
    txtrain.export_val(txc, preds, ["K562"], tmp_path / "a")
    fastloop.export_bigwigs(txc, preds, ["K562"], tmp_path / "b", tracks_dir=tracks)
    k = txc.condition_index("K562")
    for c, lib in enumerate(fastloop.LIBRARIES):
        for suffix in ("plus", "minus"):
            name = f"K562/{lib}_pooled.{suffix}.bw"
            np.testing.assert_array_equal(
                _bw_values(tmp_path / "a" / name), _bw_values(tmp_path / "b" / name)
            )
        for r in rows.itertuples():
            unit = unit_of(r)
            suffix, shift = ("plus", 0) if r.strand == "+" else ("minus", RIBOTISH_MINUS_SHIFT)
            bw = pyBigWig.open(str(tmp_path / "a" / "K562" / f"{lib}_pooled.{suffix}.bw"))
            got = [bw.values("chr1", int(p) + shift, int(p) + shift + 1)[0] for p in unit.positions]
            bw.close()
            np.testing.assert_allclose(got, preds[r.transcript_id][k, :, c], rtol=1e-6)


def _argv(txc, out, results, *extra):
    return [
        "--cache", str(txc.path), "--heldout", "K562", "--out", str(out),
        "--results", str(results), "--min-bucket", str(MIN), "--max-bucket", str(MAX),
        "--steps", "2", "--eval-every", "1", "--log-every", "1", "--warmup", "1",
        *extra,
    ]  # fmt: skip


def _stub(args):
    return txtrain.stub_backbone(dim=16)


def _tree(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_two_step_smoke_with_stub_backbone_checkpoints_and_resumes(tmp_path, txc):
    tracks = _real_tracks(tmp_path / "tracks")
    txc.config["tracks_dir"] = str(tracks)
    (txc.path / "cache.json").write_text(
        json.dumps({"conditions": list(CONDITIONS), "tracks_dir": str(tracks)})
    )
    full, part = tmp_path / "full" / "run", tmp_path / "part" / "run"
    export = tmp_path / "export"
    txtrain.main(_argv(txc, full, tmp_path / "r1.tsv", "--export-dir", str(export)), _stub)
    record = json.loads((full / "record.json").read_text())
    assert record["meta"]["steps"] == 2 and len(record["evals"]) == 2
    assert record["meta"]["heldout"] == "K562" and np.isfinite(record["meta"]["final_loss"])
    assert record["reference_row"] is not None  # the cache's ckpt.npy scored alongside
    head = open(tmp_path / "r1.tsv").readline().rstrip("\n").split("\t")
    assert head == fastloop.result_columns(CONDITIONS)
    assert (export / "run" / "K562" / "tis_pooled.plus.bw").exists()
    assert not (export / "run" / "HeLa").exists()
    for d in ("best", "last"):
        assert (full / d / "params.npz").exists()
    assert (full / "last" / "opt_state.npz").exists()
    run = json.loads((full / "run.json").read_text())
    assert run["train_conditions"] == ["HeLa", "U2OS"]

    txtrain.main(_argv(txc, part, tmp_path / "r2.tsv", "--stop-after", "1"), _stub)
    assert not (part / "record.json").exists()
    assert json.loads((part / "last" / "state.json").read_text())["step"] == 1
    txtrain.main(_argv(txc, part, tmp_path / "r2.tsv"), _stub)
    resumed = json.loads((part / "record.json").read_text())
    assert [e["step"] for e in resumed["evals"]] == [1, 2]
    a, b = _tree(full / "last" / "params.npz"), _tree(part / "last" / "params.npz")
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6)
    assert resumed["row"]["primary3"] == pytest.approx(record["row"]["primary3"], nan_ok=True)
    with pytest.raises(FileExistsError, match="finished"):
        txtrain.main(_argv(txc, part, tmp_path / "r2.tsv"), _stub)
    other = tmp_path / "other"
    txtrain.main(_argv(txc, other, tmp_path / "r3.tsv", "--stop-after", "1"), _stub)
    with pytest.raises(ValueError, match="is run"):
        txtrain.main(_argv(txc, other, tmp_path / "r3.tsv", "--seed", "5"), _stub)


def test_refuses_a_test_cache(tmp_path, txc):
    test = _test_cache(tmp_path / "test_cache")
    with pytest.raises(ValueError, match="test cache"):
        txtrain.check_cache(test)
    with pytest.raises(SystemExit):
        txtrain.main(_argv(test, tmp_path / "out", tmp_path / "r.tsv"), _stub)
    assert not (tmp_path / "out").exists()
    config = fastloop.resolve_config({"name": "t"})
    with pytest.raises(ValueError, match="test cache"):
        txtrain.train_run(test, config, _stub(None), tmp_path / "out", txtrain.Settings())


def test_init_must_come_from_the_same_fold(tmp_path):
    run = tmp_path / "run"
    (run / "best").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"heldout_conditions": ["HeLa"]}))
    with pytest.raises(ValueError, match="held out"):
        txtrain.check_init_heldout(run / "best", ["K562"])
    txtrain.check_init_heldout(run / "best", ["HeLa"])


def _trained_run(tmp_path, txc, pca=False):
    """A finished 2-step stub run (``<tmp>/runs/run``) with its val export; the tracks dir."""
    tracks = _real_tracks(tmp_path / "tracks")
    txc.config["tracks_dir"] = str(tracks)
    cfg = {"conditions": list(CONDITIONS), "tracks_dir": str(tracks)}
    (txc.path / "cache.json").write_text(json.dumps(cfg))
    if pca:
        rng = np.random.default_rng(2)
        np.savez(
            txc.path / "pca.npz", mean=rng.normal(size=16), components=rng.normal(size=(8, 16))
        )
    run = tmp_path / "runs" / "run"
    txtrain.main(
        _argv(txc, run, tmp_path / "results.tsv", "--export-dir", str(tmp_path / "export")), _stub
    )
    return run, tracks


def _txc_test(root, tracks, train=None):
    """The fast-loop synthetic test cache plus seq.npy (and the training cache's pca.npz)."""
    test = _test_cache(root)
    size = int(test.index.length.sum())
    codes = np.random.default_rng(4).integers(0, 5, size=size).astype(np.int8)
    np.save(test.path / "shards" / "test_chr2" / "seq.npy", codes)
    config = {"conditions": list(CONDITIONS), "split": "test", "tracks_dir": str(tracks)}
    (test.path / "cache.json").write_text(json.dumps(config))
    if train is not None and (train / "pca.npz").exists():
        (test.path / "pca.npz").write_bytes((train / "pca.npz").read_bytes())
    return TxCache(test.path)


def test_predict_val_reproduces_the_runs_own_val_record(tmp_path, txc):
    run, _ = _trained_run(tmp_path, txc)
    out = tmp_path / "pred_val"
    txtrain.main(
        ["--predict", str(run), "--cache", str(txc.path), "--split", "val",
         "--results", str(tmp_path / "results_val.tsv"), "--export-dir", str(out)],
        _stub,
    )  # fmt: skip
    record = json.loads((run / "record.json").read_text())
    again = json.loads((run / "record_val.json").read_text())
    assert again["meta"]["config_hash"] == record["meta"]["config_hash"]
    assert again["meta"]["best_step"] == record["meta"]["best_step"]
    for key in ("scorecard", "peaks"):
        pd.testing.assert_frame_equal(
            pd.DataFrame(again[key]), pd.DataFrame(record[key]), check_exact=True
        )
    assert again["reference_row"] is not None
    for key in fastloop.result_columns(txc.conditions)[11:]:
        a, b = record["row"][key], again["row"][key]
        assert (a == b) or (a is None and b is None) or (math.isnan(a) and math.isnan(b)), key
    for lib in fastloop.LIBRARIES:
        for strand in ("plus", "minus"):
            name = f"{lib}_pooled.{strand}.bw"
            np.testing.assert_array_equal(
                _bw_values(out / "run" / "K562" / name),
                _bw_values(tmp_path / "export" / "run" / "K562" / name),
            )
    frame = pd.read_csv(tmp_path / "results_val.tsv", sep="\t")
    assert len(frame) == 1 and frame.name[0] == "run"
    assert len(pd.read_csv(tmp_path / "results.tsv", sep="\t")) == 1  # training row untouched


def test_predict_scores_a_test_cache_end_to_end(tmp_path, txc):
    import pyBigWig

    from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
    from tisiago.tracks.txcache import unit_of

    run, tracks = _trained_run(tmp_path, txc, pca=True)
    test = _txc_test(tmp_path / "x_test", tracks, train=txc.path)
    out = tmp_path / "pred_test"
    results = tmp_path / "results_test.tsv"
    txtrain.main(
        ["--predict", str(run), "--cache", str(test.path), "--split", "test", "--i-am-final",
         "--results", str(results), "--export-dir", str(out)],
        _stub,
    )  # fmt: skip
    record = json.loads((run / "record_test.json").read_text())
    table = pd.DataFrame(record["scorecard"])
    assert {"model", "ckpt"} <= set(table.source) and set(table.condition) == set(CONDITIONS)
    model_rows = table[(table.source == "model") & (table.scale == "nt")]
    assert len(model_rows) == 2 * len(CONDITIONS) and np.isfinite(model_rows.value).all()
    assert record["meta"]["split"] == "test" and record["meta"]["cache"] == str(test.path)
    assert set(pd.DataFrame(record["peaks"]).source) == {"real", "model", "ckpt"}
    frame = pd.read_csv(results, sep="\t")
    assert len(frame) == 1 and frame.cache[0] == str(test.path)
    assert list(frame.columns) == fastloop.result_columns(CONDITIONS)
    # exported test tracks are the best/ parameters' own tiled predictions
    payload, settings = txtrain.load_run(run)
    backbone = _stub(None)
    net = txtrain.Net(payload["config"], backbone, settings.proj_dim)
    row0 = next(test.rows("test").itertuples())
    ex = txtrain.transcript_example(test, row0, CONDITIONS, 0, MIN)
    template = {"backbone": backbone.params, "head": net.init_head(jax.random.PRNGKey(0), ex)}
    params = txtrain.load_tree(template, run / "best" / "params.npz")
    preds = txtrain.predict_transcripts(
        txtrain.predictor(net.forward(), params, backbone.state),
        test, test.rows("test"), CONDITIONS, MIN, MAX,
    )  # fmt: skip
    k = test.condition_index("K562")
    for r in test.rows("test").itertuples():
        unit = unit_of(r)
        suffix, shift = ("plus", 0) if r.strand == "+" else ("minus", RIBOTISH_MINUS_SHIFT)
        bw = pyBigWig.open(str(out / "run" / "K562" / f"tis_pooled.{suffix}.bw"))
        got = [bw.values("chr2", int(p) + shift, int(p) + shift + 1)[0] for p in unit.positions]
        bw.close()
        np.testing.assert_allclose(got, preds[r.transcript_id][k, :, 0], rtol=1e-6)
    manifest = json.loads((out / "run" / "K562" / "predict_tracks.json").read_text())
    assert manifest["split"] == "test" and manifest["transcripts"] == 2


def test_predict_refusals(tmp_path, txc):
    run, tracks = _trained_run(tmp_path, txc, pca=True)
    test = _txc_test(tmp_path / "x_test", tracks, train=txc.path)
    base = ["--predict", str(run), "--results", str(tmp_path / "r_test.tsv")]

    def refused(*extra):
        with pytest.raises(SystemExit):
            txtrain.main([*base, *extra], _stub)

    refused("--cache", str(test.path), "--split", "test")  # no --i-am-final
    refused("--cache", str(txc.path), "--split", "test", "--i-am-final")  # not a test cache
    refused("--cache", str(test.path), "--split", "val")  # val on a test cache
    for flag in (["--steps", "5"], ["--out", str(tmp_path / "o")], ["--seed=1"], ["--preload"]):
        refused("--cache", str(test.path), "--split", "test", "--i-am-final", *flag)
    with pytest.raises(SystemExit):  # predict flags without --predict
        txtrain.main(_argv(txc, tmp_path / "o2", tmp_path / "r.tsv", "--i-am-final"), _stub)
    assert not (tmp_path / "o2").exists() and not (tmp_path / "r_test.tsv").exists()
    kw = {"log": lambda s: None}
    with pytest.raises(PermissionError, match="i_am_final"):
        txtrain.predict_run(run, test, _stub(None), "test", tmp_path / "r_test.tsv", **kw)
    with pytest.raises(ValueError, match="not a test cache"):
        txtrain.predict_run(
            run, txc, _stub(None), "test", tmp_path / "r_test.tsv", i_am_final=True, **kw
        )
    with pytest.raises(ValueError, match="main results"):
        txtrain.predict_run(run, txc, _stub(None), "val", tmp_path / "results.tsv", **kw)
    # the test cache's PCA must be the training cache's, byte for byte
    np.savez(test.path / "pca.npz", mean=np.zeros(16), components=np.ones((8, 16)))
    with pytest.raises(ValueError, match="not the training cache"):
        txtrain.predict_run(
            run, test, _stub(None), "test", tmp_path / "r_test.tsv", i_am_final=True, **kw
        )
    # a backbone of another shape does not load into best/
    with pytest.raises(ValueError, match="shape|leaves"):
        txtrain.predict_run(run, txc, txtrain.stub_backbone(dim=8), "val", tmp_path / "r.tsv", **kw)
    # an edited run.json fails its hash
    payload = json.loads((run / "run.json").read_text())
    payload["settings"]["seed"] = 7
    (run / "run.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="run hash"):
        txtrain.load_run(run)
    assert not (tmp_path / "r_test.tsv").exists()
