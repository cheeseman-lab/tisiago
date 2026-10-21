"""Tests for the P6 fast-loop trainer/scorer over a transcript cache (CPU, synthetic cache)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("haiku")
pytest.importorskip("optax")
jnp = jax.numpy

from tisiago.seq2func import fastloop  # noqa: E402
from tisiago.seq2func.loss import local_multinomial, poisson_multinomial  # noqa: E402
from tisiago.seq2func.scorecard import Unit, library_rows  # noqa: E402
from tisiago.seq2func.txcache import TxCache  # noqa: E402

CONDITIONS = ("HeLa", "K562", "U2OS")
DIM = 8


def _transcripts():
    # (id, strand, split, length, cds_start, cds_end, expressed per condition)
    return [
        ("A", "+", "train", 300, 30, 270, (True, True, True)),
        ("B", "-", "train", 120, 12, 105, (True, False, True)),
        ("C", "+", "train", 50, 5, 44, (False, True, False)),
        ("V1", "+", "val", 400, 40, 340, (True, True, True)),
        ("V2", "-", "val", 260, 20, 230, (True, True, False)),
    ]


@pytest.fixture
def cache(tmp_path):
    """A tiny cache: values encode (condition, position) so crops can be checked exactly."""
    rng = np.random.default_rng(0)
    root = tmp_path / "cache"
    rows, offsets = [], {}
    for tid, strand, split, length, cs, ce, expressed in _transcripts():
        shard = f"{split}_chr1"
        offset = offsets.get(shard, 0)
        offsets[shard] = offset + length
        rows.append(
            {
                "transcript_id": tid,
                "gene_id": f"G{tid}",
                "gene_name": tid,
                "chrom": "chr1",
                "strand": strand,
                "split": split,
                "start": 0,
                "end": length,
                "length": length,
                "cds_start": cs,
                "cds_end": ce,
                "exons": [[0, length]],
                "cds": [[cs, ce]],
                "junctions": [],
                **{f"expressed_{c}": e for c, e in zip(CONDITIONS, expressed, strict=True)},
                "shard": shard,
                "offset": offset,
            }
        )
    index = pd.DataFrame(rows)
    for shard, size in offsets.items():
        d = root / "shards" / shard
        d.mkdir(parents=True)
        emb = rng.normal(size=(size, DIM)).astype(np.float16)
        emb[:, 0] = np.arange(size) % 1000
        np.save(d / "emb.npy", emb)
        rna = rng.poisson(3.0, size=(len(CONDITIONS), size, 2)).astype(np.float32)
        rna[:, :, 0] = np.arange(size)[None] + 1000 * np.arange(len(CONDITIONS))[:, None]
        np.save(d / "rna.npy", rna)
        for lib, lam in (("tis", 0.3), ("chx", 2.0)):
            reps = rng.poisson(lam, size=(len(CONDITIONS), size, 2)).astype(np.float32)
            np.save(d / f"{lib}.npy", np.concatenate([reps, reps.sum(-1, keepdims=True)], -1))
        if shard.startswith("val"):
            np.save(d / "ckpt.npy", rng.gamma(1.0, size=(len(CONDITIONS), size, 2)).astype("f4"))
    index.to_parquet(root / "index.parquet", index=False)
    (root / "cache.json").write_text(json.dumps({"conditions": list(CONDITIONS)}))
    sites = pd.DataFrame(
        {
            "condition": ["K562", "K562", "HeLa"],
            "transcript_id": ["V1", "V2", "V1"],
            "tx_pos": [40, 25, 100],
            "codon": ["ATG", "CTG", "ATG"],
            "tistype": ["Annotated", "Novel", "Novel"],
        }
    )
    sites.to_parquet(root / "sites.parquet", index=False)
    return TxCache(root)


def _config(**override):
    base = {
        "name": "t",
        "data": {"crop": 128, "batch": 4},
        "train": {"steps": 3, "warmup": 1, "log_every": 1},
        "eval": {"min_bucket": 64, "max_length": 256},
    }
    return fastloop.resolve_config(fastloop.deep_merge(base, override))


def test_resolve_config_validates_names_and_hash_ignores_name():
    config = _config()
    assert config["inputs"] == ["embedding", "rna"] and config["data"]["crop"] == 128
    renamed = {**config, "name": "other"}
    assert fastloop.config_hash(config) == fastloop.config_hash(renamed)
    assert fastloop.config_hash(config) != fastloop.config_hash(_config(seed=1))
    with pytest.raises(ValueError, match="unknown input"):
        _config(inputs=["nope"])
    with pytest.raises(ValueError, match="unknown loss"):
        _config(losses=[{"kind": "nope"}])
    with pytest.raises(ValueError, match="unknown tis output"):
        _config(model={"tis": {"kind": "nope"}})
    with pytest.raises(ValueError, match="unknown config keys"):
        _config(bogus=1)


def test_load_config_from_yaml_names_by_stem(tmp_path):
    path = tmp_path / "ref.yaml"
    path.write_text("model:\n  tis: {kind: factorised, block: 64}\n")
    config = fastloop.load_config(path)
    assert config["name"] == "ref" and config["model"]["tis"]["block"] == 64
    assert config["model"]["chx"]["kind"] == "softplus"


def test_train_conditions_exclude_heldout(cache):
    assert fastloop.train_conditions_of(_config(), CONDITIONS) == ("HeLa", "U2OS")
    with pytest.raises(ValueError, match="held out"):
        fastloop.train_conditions_of(_config(train_conditions=["K562"]), CONDITIONS)


def test_crop_example_slices_pads_and_masks(cache):
    rows = cache.index.set_index("transcript_id", drop=False)
    a = rows.loc["A"]
    ex = fastloop.crop_example(cache, a, "U2OS", 250, 128)
    assert ex["valid"].sum() == 50 and not ex["valid"][50:].any()
    np.testing.assert_array_equal(ex["pos"][:50], np.arange(250, 300))
    np.testing.assert_array_equal(ex["rna"][:50, 0], 2000 + np.arange(250, 300))
    np.testing.assert_array_equal(ex["emb"][:50, 0], np.arange(250, 300))
    assert (ex["rna"][50:] == 0).all() and (ex["target"][50:] == 0).all()
    tis = cache.array("train_chr1", "tis")[2, 250:300, 2]
    np.testing.assert_array_equal(ex["target"][:50, 0], tis)
    b = rows.loc["B"]  # offset 300 in the train shard
    ex = fastloop.crop_example(cache, b, "HeLa", 0, 128)
    np.testing.assert_array_equal(ex["rna"][:120, 0], 300 + np.arange(120))


def test_sampler_is_seeded_single_condition_and_expressed_only(cache):
    config = _config()
    one, two = fastloop.Sampler(cache, config), fastloop.Sampler(cache, config)
    for _ in range(5):
        rows, conditions, starts = one.draw()
        assert len(set(conditions)) == 1 and conditions[0] in ("HeLa", "U2OS")
        for r, c, s in zip(rows, conditions, starts, strict=True):
            assert r.split == "train" and bool(getattr(r, f"expressed_{c}"))
            assert 0 <= s <= max(0, r.length - 128)
    b1, b2 = one.batch(), two.batch()
    two.draw()  # different stream position from here on
    for _ in range(5):
        one.draw()
    b1, b2 = fastloop.Sampler(cache, config).batch(), fastloop.Sampler(cache, config).batch()
    for key in b1:
        np.testing.assert_array_equal(b1[key], b2[key])
    assert b1["emb"].shape == (4, 128, DIM) and b1["strand"].shape == (4,)
    start = fastloop.Sampler(cache, _config(data={"crop_mode": "start"}))
    assert all(s == 0 for s in start.draw()[2])


def test_prefetched_yields_in_order_and_reraises():
    counter = iter(range(10))
    assert list(fastloop.prefetched(lambda: next(counter), 4)) == [0, 1, 2, 3]

    def boom():
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        list(fastloop.prefetched(boom, 2))


def test_softplus_output_known_values_and_masked():
    p = jnp.array([[[0.0], [1.0], [5.0]]])
    valid = jnp.array([[True, True, False]])
    out = fastloop.OUTPUTS["softplus"][1](p, valid, {})
    np.testing.assert_allclose(out, [[math.log(2.0), math.log1p(math.e), 0.0]], rtol=1e-6)


def test_factorised_output_preserves_block_amounts_and_masks_padding():
    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.normal(size=(2, 10, 2)).astype(np.float32))
    valid = jnp.asarray(np.arange(10)[None] < np.array([[10], [7]]))
    amount = np.log1p(np.exp(np.asarray(p[..., 0]))) * np.asarray(valid)
    cfg = {"block": 4, "two_grids": False}
    out = np.asarray(fastloop.OUTPUTS["factorised"][1](p, valid, cfg))
    assert (out[1, 7:] == 0).all()
    for b in range(2):
        for lo in range(0, 10, 4):
            assert out[b, lo : lo + 4].sum() == pytest.approx(amount[b, lo : lo + 4].sum(), 1e-5)
    both = np.asarray(fastloop.OUTPUTS["factorised"][1](p, valid, {"block": 4}))
    np.testing.assert_allclose(both.sum(1), amount.sum(1), rtol=1e-5)
    crop = np.asarray(fastloop.OUTPUTS["factorised"][1](p, valid, {"block": "crop"}))
    np.testing.assert_allclose(crop.sum(1), amount.sum(1), rtol=1e-5)
    logits = np.where(np.asarray(valid[1]), np.asarray(p[1, :, 1]), -np.inf)
    soft = np.exp(logits - logits.max())
    np.testing.assert_allclose(crop[1], amount[1].sum() * soft / soft.sum(), rtol=1e-5)
    hot = np.asarray(
        fastloop.OUTPUTS["factorised"][1](p, valid, {"block": "crop", "temperature": 0.1})
    )
    assert hot.max() > crop.max()


def test_hazard_scanning_on_hand_inputs():
    valid = jnp.array([[True, True, True, False]])
    probs = fastloop.scan_probabilities(jnp.zeros((1, 4)), valid)
    np.testing.assert_allclose(probs, [[0.5, 0.25, 0.125, 0.0]], rtol=1e-6)
    p = jnp.stack([jnp.full((1, 4), 2.0), jnp.zeros((1, 4))], axis=-1)
    out = fastloop.OUTPUTS["hazard"][1](p, valid, {})
    total = 3 * math.log1p(math.exp(2.0))
    np.testing.assert_allclose(out, [[0.5 * total, 0.25 * total, 0.125 * total, 0.0]], rtol=1e-5)
    sure = fastloop.scan_probabilities(jnp.array([[-30.0, 30.0, 0.0]]), jnp.ones((1, 3), bool))
    np.testing.assert_allclose(sure, [[0.0, 1.0, 0.0]], atol=1e-6)


def _loss_inputs():
    rng = np.random.default_rng(2)
    target = jnp.asarray(rng.poisson(1.5, size=(3, 64, 2)).astype(np.float32))
    pred = jnp.asarray(rng.gamma(2.0, size=(3, 64, 2)).astype(np.float32))
    valid = jnp.asarray(np.arange(64)[None] < np.array([[64], [60], [40]]))
    return pred, target, valid


AUX = {"nb_log_r": jnp.log(jnp.array([5.0, 5.0]))}


def test_poisson_and_multinomial_match_the_genomic_loss_on_one_group():
    pred, target, valid = _loss_inputs()
    mask = jnp.broadcast_to(valid[..., None], pred.shape)
    ref = poisson_multinomial(pred, target, mask, positional_weight=1.0)
    per_sample = {"scope": "sample"}
    assert float(fastloop.loss_poisson(pred, target, valid, AUX, per_sample)) == pytest.approx(
        float(ref["poisson"]), rel=1e-5
    )
    assert float(fastloop.loss_multinomial(pred, target, valid, AUX, per_sample)) == pytest.approx(
        float(ref["multinomial"]), rel=1e-5
    )
    flat = [x.reshape(1, -1, 2) for x in (pred, target, mask)]
    ref = poisson_multinomial(*flat, positional_weight=1.0)
    glob = {"scope": "global"}
    assert float(fastloop.loss_poisson(pred, target, valid, AUX, glob)) == pytest.approx(
        float(ref["poisson"]), rel=1e-5
    )
    assert float(fastloop.loss_multinomial(pred, target, valid, AUX, glob)) == pytest.approx(
        float(ref["multinomial"]), rel=1e-5
    )
    local = {"scope": "local", "block": 16, "libraries": ["tis"]}
    expected = local_multinomial(pred, target, mask, (0,), 16)
    assert float(fastloop.loss_multinomial(pred, target, valid, AUX, local)) == pytest.approx(
        float(expected), rel=1e-6
    )


def test_losses_known_values_on_hand_inputs():
    target = jnp.array([[[2.0, 0.0], [0.0, 0.0]]])
    valid = jnp.ones((1, 2), bool)
    pred = jnp.array([[[1.0, 1.0], [1.0, 1.0]]])
    # totals: TIS pred 2 vs 2 -> 0; CHX pred 2 vs 0 -> 2 * 2 = 4; mean over channels = 2
    assert float(fastloop.loss_poisson(pred, target, valid, AUX, {})) == pytest.approx(2.0)
    # TIS profile [1, 0] vs [0.5, 0.5]: KL = log 2; CHX empty: 0; mean = log(2) / 2
    assert float(fastloop.loss_multinomial(pred, target, valid, AUX, {})) == pytest.approx(
        math.log(2.0) / 2, rel=1e-6
    )
    # per-nt Poisson deviance: TIS 2*(1-2+2log2) + 2*1 ; CHX 2*2 ; / 2 counts
    expected = (2 * (1 - 2 + 2 * math.log(2)) + 2 + 4) / 2
    assert float(fastloop.loss_poisson_nt(pred, target, valid, AUX, {})) == pytest.approx(
        expected, rel=1e-5
    )


def test_nb_and_poisson_nt_are_lowest_near_the_truth():
    pred, target, valid = _loss_inputs()
    good = jnp.maximum(target, 0.05)
    for name in ("nb", "poisson_nt"):
        fn = fastloop.LOSSES[name]
        assert float(fn(good, target, valid, AUX, {})) < float(fn(pred, target, valid, AUX, {}))


def test_coupling_is_zero_when_tis_matches_the_chx_rise():
    valid = jnp.ones((1, 300), bool)
    chx = jnp.where(jnp.arange(300) >= 150, 5.0, 0.0)[None]
    rise = jax.nn.relu(fastloop.chx_log_ratios(chx, (45,))[..., 0])
    pred = jnp.stack([rise, chx], -1)
    loss = {"width": 45}
    assert float(fastloop.loss_coupling(pred, pred, valid, AUX, loss)) == pytest.approx(
        0.0, abs=1e-5
    )
    flat = pred.at[..., 0].set(1.0)
    assert float(fastloop.loss_coupling(flat, flat, valid, AUX, loss)) > 0.1


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"model": {"tis": {"kind": "factorised", "block": 32}, "chx_contrast": True}},
        {"model": {"tis": {"kind": "hazard"}, "trunk": {"kind": "conv", "width": 5}}},
        {"model": {"trunk": {"kind": "dilated", "dilations": [1, 2], "channels": 8}}},
        {"model": {"trunk": {"kind": "attention", "block": 32, "layers": 2, "dim": 8}}},
        {"inputs": ["embedding_white", "rna_sense", "rna_onesided", "position"]},
        {"model": {"strand_heads": False, "trunk": {"layers": 2, "hidden": 16}}},
        {"model": {"split_trunk": True, "chx_contrast": True}},
        {"model": {"chx_trunk": {"kind": "conv", "width": 7, "hidden": 8}}},
    ],
)
def test_every_component_gives_finite_gradients_for_every_loss(cache, override):
    config = _config(**override)
    losses = [
        {"kind": "poisson", "scope": "sample"},
        {"kind": "poisson", "scope": "global"},
        {"kind": "multinomial", "scope": "global"},
        {"kind": "multinomial", "scope": "sample"},
        {"kind": "multinomial", "scope": "local", "block": 32},
        {"kind": "frame", "block": 2},
        {"kind": "poisson_nt"},
        {"kind": "nb"},
        {"kind": "coupling", "width": 15},
    ]
    model = fastloop.make_model(config, fastloop.context_of(cache))
    batch = fastloop._device_batch(fastloop.Sampler(cache, config).batch())
    params = model.init(jax.random.PRNGKey(0), batch)

    def objective(params):
        pred, aux = model.apply(params, batch)
        return fastloop.total_loss(pred, batch, aux, losses)

    (value, parts), grads = jax.value_and_grad(objective, has_aux=True)(params)
    assert np.isfinite(float(value)) and len(parts) == len(losses)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves)
    pred, _ = model.apply(params, batch)
    assert pred.shape == (4, 128, 2) and (np.asarray(pred) >= 0).all()
    assert (np.asarray(pred)[~np.asarray(batch["valid"])] == 0).all()


def test_strand_heads_select_the_output_layer_by_strand(cache):
    config = _config(inputs=["embedding"])
    model = fastloop.make_model(config, fastloop.context_of(cache))
    batch = fastloop._device_batch(fastloop.Sampler(cache, config).batch())
    params = model.init(jax.random.PRNGKey(0), batch)
    flipped = {**batch, "strand": 1 - batch["strand"]}
    a, b = model.apply(params, batch)[0], model.apply(params, flipped)[0]
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_split_trunk_separates_tis_and_chx_parameters(cache):
    config = _config(model={"split_trunk": True})
    model = fastloop.make_model(config, fastloop.context_of(cache))
    batch = fastloop._device_batch(fastloop.Sampler(cache, config).batch())
    params = model.init(jax.random.PRNGKey(0), batch)
    assert any(k.startswith("chx_trunk/") for k in params)

    def tis_only(params):
        pred, _ = model.apply(params, batch)
        return pred[..., 0].sum()

    grads = jax.grad(tis_only)(params)
    chx_trunk = [v for k, v in grads.items() if k.startswith("chx_trunk/")]
    assert all(np.allclose(np.asarray(g), 0) for m in chx_trunk for g in m.values())


def test_chx_trunk_config_and_validation(cache):
    config = _config(model={"chx_trunk": {"kind": "conv", "width": 7, "hidden": 8, "layers": 2}})
    model = fastloop.make_model(config, fastloop.context_of(cache))
    batch = fastloop._device_batch(fastloop.Sampler(cache, config).batch())
    params = model.init(jax.random.PRNGKey(0), batch)
    assert params["chx_trunk/conv1"]["w"].shape == (7, 8, 8)
    with pytest.raises(ValueError, match="chx_trunk"):
        fastloop.resolve_config({"model": {"chx_trunk": {"kind": "nope"}}})


def test_bucket_length():
    assert fastloop.bucket_length(10, 64, 256) == 64
    assert fastloop.bucket_length(65, 64, 256) == 128
    assert fastloop.bucket_length(256, 64, 256) == 256


def test_predict_rows_full_length_in_chunks(cache):
    rows = cache.rows("val")

    def apply(batch):  # prediction = position, condition-specific RNA sense
        return jnp.stack([batch["pos"].astype(jnp.float32), batch["rna"][..., 0]], -1)

    out = fastloop.predict_rows(apply, cache, rows, CONDITIONS, min_bucket=64, max_length=128)
    v1 = out["V1"]
    assert v1.shape == (3, 400, 2)
    np.testing.assert_array_equal(v1[0, :, 0], np.arange(400))
    np.testing.assert_array_equal(v1[2, :, 1], 2000 + np.arange(400))


def _vectors(cache, condition, library, preds):
    val = cache.rows("val")
    units = val[val[f"expressed_{condition}"]]
    k = CONDITIONS.index(condition)
    c = fastloop.CHANNEL[library]
    lib = cache.array("val_chr1", library)
    out = {"rep1": [], "rep2": [], "model": []}
    real_units = []
    for r in units.itertuples():
        lo, hi = r.offset, r.offset + r.length
        out["rep1"].append(np.asarray(lib[k, lo:hi, 0], float))
        out["rep2"].append(np.asarray(lib[k, lo:hi, 1], float))
        out["model"].append(np.asarray(preds[r.transcript_id][k, :, c], float))
        # an equivalent single-exon scorecard Unit with the same CDS range
        exons = ((0, r.length),)
        cds = (
            ((r.cds_start, r.cds_end),)
            if r.strand == "+"
            else ((r.length - r.cds_end, r.length - r.cds_start),)
        )
        real_units.append(
            Unit(r.transcript_id, r.gene_id, r.gene_name, "chr1", r.strand, exons, cds)
        )
    return real_units, out


def test_scoring_equals_scorecard_library_rows(cache):
    rng = np.random.default_rng(3)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in cache.rows("val").itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    for condition in CONDITIONS:
        for library in ("tis", "chx"):
            units, vectors = _vectors(cache, condition, library, preds)
            assert [u.cds_range for u in units] == [
                (r.cds_start, r.cds_end)
                for r in cache.rows("val").itertuples()
                if getattr(r, f"expressed_{condition}")
            ]
            expected = pd.DataFrame(library_rows(units, vectors, library, min_reads=5))
            got = table[(table.condition == condition) & (table.library == library)]
            got = got.drop(columns=["condition", "heldout"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(got, expected)
    assert table[table.condition == "K562"].heldout.all()
    assert not table[table.condition == "HeLa"].heldout.any()
    assert set(peaks.source) == {"real", "model"}


def test_peak_summary_and_site_profiles():
    vec = np.zeros(100)
    vec[[40, 41, 90]] = [3.0, 1.0, 2.0]
    prof = fastloop.site_profiles(vec, [40, 90, 5])
    assert prof.shape == (3, 61) and prof[0, 30] == 3.0 and prof[2, 25] == 0.0
    summary = fastloop.peak_summary(prof)
    assert summary["sites"] == 2  # the empty profile around 5 is dropped
    assert summary["argmax0"] == 1.0
    assert summary["meta_share"] == pytest.approx((0.75 + 1.0) / 2)


def test_results_row_schema_and_append(tmp_path, cache):
    columns = fastloop.result_columns(CONDITIONS)
    assert columns[:3] == ["config_hash", "name", "timestamp"]
    assert "K562_tis_codon" in columns and "K562_chx_frame" in columns
    assert "K562_tis_frame" not in columns and "HeLa_tis_peak_meta" in columns
    assert len(columns) == len(set(columns))
    table = pd.DataFrame(
        [
            {
                "condition": "K562",
                "heldout": True,
                "library": "tis",
                "source": "model",
                "scale": s,
                "value": 0.5,
                "yardstick": 0.8,
                "percent": p,
                "n": 3,
            }
            for s, p in (("codon", 60.0), ("nt", 40.0))
        ]
    )
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, pd.DataFrame(), CONDITIONS)
    assert row["primary"] == pytest.approx(50.0) and row["K562_tis_codon"] == 60.0
    assert set(row) == set(columns)
    path = tmp_path / "results.tsv"
    fastloop.append_result(path, row, columns)
    fastloop.append_result(path, row, columns)
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == columns and len(frame) == 2
    with pytest.raises(ValueError, match="different header"):
        fastloop.append_result(path, row, columns[:-1])


def test_run_config_end_to_end_writes_row_and_json(tmp_path, cache):
    config = _config(name="smoke")
    results, runs = tmp_path / "results.tsv", tmp_path / "runs"
    record = fastloop.run_config(config, cache, results, runs, log=lambda s: None)
    frame = pd.read_csv(results, sep="\t")
    assert len(frame) == 1 and frame.name[0] == "smoke"
    assert frame.config_hash[0] == fastloop.config_hash(config)
    saved = json.loads((runs / f"smoke-{fastloop.config_hash(config)}.json").read_text())
    assert saved["history"][-1]["step"] == 3
    assert {"model", "ckpt"} <= {r["source"] for r in saved["scorecard"]}
    assert record["reference_row"]["name"] == "smoke"
    preload = _config(name="smoke", data={"preload": False})
    again = fastloop.run_config(preload, cache, tmp_path / "r2.tsv", runs, log=lambda s: None)
    assert again["history"][-1]["loss"] == pytest.approx(record["history"][-1]["loss"], rel=1e-5)


def test_reference_record_scores_ckpt(cache):
    record = fastloop.reference_record(cache, ["K562"])
    assert record["meta"]["config_hash"] == "checkpoint"
    rows = pd.DataFrame(record["scorecard"])
    assert set(rows.source) == {"ckpt"} and set(rows.condition) == set(CONDITIONS)
    two_units = rows.condition.isin(["HeLa", "K562"]) & (rows.scale == "gene")
    assert np.isfinite(rows.value[two_units]).all()


def test_class_peak_summary_splits_atg_and_near_cognate():
    vec = np.zeros(200)
    vec[[40, 100, 131]] = [2.0, 1.0, 3.0]
    prof = fastloop.site_profiles(vec, [40, 100, 130])
    classes = fastloop.start_classes(["ATG", "ctg", "GTG"])
    assert list(classes) == ["atg", "nc", "nc"]
    summary = fastloop.class_peak_summary(prof, classes)
    assert summary["argmax0_atg"] == 1.0 and summary["sites_atg"] == 1
    assert summary["argmax0_nc"] == 0.5 and summary["sites_nc"] == 2
    assert summary["meta_share_nc"] == pytest.approx((1.0 + 0.0) / 2)
    empty = fastloop.class_peak_summary(prof[:1], classes[:1])
    assert empty["sites_nc"] == 0 and math.isnan(empty["argmax0_nc"])


def test_class_peaks_rows_keep_phase1_columns_and_primary2(cache):
    rng = np.random.default_rng(4)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in cache.rows("val").itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    columns = fastloop.result_columns(CONDITIONS)
    old = [c for c in columns if c == "primary2" or "_atg" in c or "_nc" in c]
    assert columns[-len(old) :] == old  # phase-2 columns are appended after phase 1
    assert "K562_tis_peak_argmax0_nc_real" in columns and "HeLa_tis_peak_meta_atg" in columns
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, peaks, CONDITIONS)
    assert set(row) == set(columns)
    model = peaks[(peaks.source == "model") & (peaks.condition == "K562")].iloc[0]
    real = peaks[(peaks.source == "real") & (peaks.condition == "K562")].iloc[0]
    assert row["K562_tis_peak_argmax0"] == model.argmax0  # phase-1 column unchanged
    for cls in ("atg", "nc"):
        if model[f"sites_{cls}"]:
            assert row[f"K562_tis_peak_argmax0_{cls}"] == model[f"argmax0_{cls}"]
            assert row[f"K562_tis_peak_meta_{cls}_real"] == real[f"meta_share_{cls}"]
    assert model.sites == model.sites_atg + model.sites_nc
    fake = dict(row, K562_tis_nt=60.0, K562_tis_peak_argmax0_nc=0.1)
    fake["K562_tis_peak_argmax0_nc_real"] = 0.4
    assert fastloop.primary2(fake, "K562") == pytest.approx((60.0 + 25.0) / 2)
    fake["K562_tis_peak_argmax0_nc"] = 0.8
    assert fastloop.primary2(fake, "K562") == pytest.approx((60.0 + 100.0) / 2)
