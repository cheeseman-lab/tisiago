"""Tests for the fast-loop trainer/scorer over a transcript cache (CPU, synthetic cache)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("haiku")
pytest.importorskip("optax")
jnp = jax.numpy

from tisiago.tracks import fastloop  # noqa: E402
from tisiago.tracks.loss import local_multinomial, poisson_multinomial  # noqa: E402
from tisiago.tracks.scorecard import Unit, library_rows  # noqa: E402
from tisiago.tracks.txcache import TxCache  # noqa: E402

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
        {
            "model": {
                "chx_trunk": {"kind": "conv", "width": 7, "hidden": 8},
                "chx_onehot": {"width": 5, "channels": 4},
                "chx_context": {"kind": "dilated", "dilations": [1, 4], "channels": 8},
            }
        },
        {"model": {"tis_context": {"kind": "dilated", "dilations": [1, 4], "channels": 8}}},
        {
            "model": {
                "tis": {"kind": "factorised", "block": 32},
                "chx": {"kind": "factorised", "block": 32},
                "chx_flux": {"mode": "logit", "init_beta": 0.5, "stop_gradient_init": False},
            }
        },
        {
            "model": {
                "chx": {"kind": "factorised", "block": 32},
                "chx_flux": {"mode": "replace", "frame_kernel": 5, "stop_onehot": {"channels": 4}},
            }
        },
        {
            "inputs": ["embedding", "rna", "onehot"],
            "model": {
                "chx": {"kind": "factorised", "block": 32},
                "chx_flux": {"stop_codons": "fixed", "learned_hazard": False, "frame_kernel": 1},
            },
        },
        {"inputs": ["embedding", "rna", "rna_onset", "rna_relative"]},
        {
            "inputs": ["embedding", "rna", "exon", "onehot"],
            "model": {
                "tis_context": {"kind": "dilated", "dilations": [1, 4], "causal": "up"},
                "tis_onehot": {"width": 5, "channels": 4, "layers": 2},
            },
        },
        {
            "model": {
                "tis_context": {"kind": "dilated", "dilations": [1, 4], "channels": 8},
                "tis_onehot": {"width": 5, "channels": 4, "into": "context"},
                "tis_exon": True,
            },
        },
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
        {"kind": "multinomial", "scope": "local", "block": 32, "count_power": 0.0},
        {"kind": "multinomial", "scope": "local", "block": 32, "target_power": 0.5},
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
    p3 = columns.index("primary3")
    old = [c for c in columns[:p3] if c == "primary2" or "_atg" in c or "_nc" in c]
    assert columns[p3 - len(old) : p3] == old  # phase-2 columns are appended after phase 1
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


def test_weighted_local_multinomial_reduces_to_counts_weighted_and_weights_blocks():
    rng = np.random.default_rng(5)
    pred = jnp.asarray(rng.gamma(1.0, size=(2, 64, 2)), jnp.float32)
    target = jnp.asarray(rng.poisson(0.5, size=(2, 64, 2)), jnp.float32)
    mask = jnp.ones((2, 64, 2), bool)
    same = fastloop.weighted_local_multinomial(pred, target, mask, (0, 1), 16)
    assert float(same) == pytest.approx(float(local_multinomial(pred, target, mask, (0, 1), 16)))
    # blocks with counts 1 (point mass) and 9 (over 3 of 4), predicted uniform
    t = np.zeros((1, 8, 1), np.float32)
    t[0, 0, 0], t[0, 4:7, 0] = 1.0, 3.0
    pr = jnp.ones((1, 8, 1), jnp.float32)
    m = jnp.ones((1, 8, 1), bool)
    kl = np.array([math.log(4.0), math.log(4.0 / 3.0)])
    uni = fastloop.weighted_local_multinomial(pr, jnp.asarray(t), m, (0,), 4, count_power=0.0)
    assert float(uni) == pytest.approx(kl.mean())
    half = fastloop.weighted_local_multinomial(pr, jnp.asarray(t), m, (0,), 4, count_power=0.5)
    assert float(half) == pytest.approx((kl * [1.0, 3.0]).sum() / 4.0)
    # tempering a point mass changes nothing; tempering 1:3 to sqrt flattens it
    t2 = np.zeros((1, 4, 1), np.float32)
    t2[0, :2, 0] = [1.0, 3.0]
    flat = fastloop.weighted_local_multinomial(
        jnp.asarray(np.sqrt(t2)), jnp.asarray(t2), m[:, :4], (0,), 4, target_power=0.5
    )
    assert float(flat) == pytest.approx(0.0, abs=1e-6)


def test_rna_relative_input_is_relative_to_the_transcript_head(cache):
    rows = cache.index.set_index("transcript_id", drop=False)
    ex = fastloop.crop_example(cache, rows.loc["A"], "U2OS", 250, 128)
    tx = cache.array("train_chr1", "rna")[2, 0:300, 0]
    np.testing.assert_allclose(ex["rna_tx"], [tx[:129].mean(), tx.mean()], rtol=1e-6)
    batch = fastloop._device_batch(fastloop.stack_examples([ex], [rows.loc["A"]]))
    rel = np.asarray(fastloop.INPUTS["rna_relative"](batch, {}))
    assert rel.shape == (1, 128, 4) and np.isfinite(rel).all()
    onset = np.asarray(fastloop.INPUTS["rna_onset"](batch, {}))
    assert onset.shape == (1, 128, 6)


def test_local_multinomial_offset_shifts_the_block_grid():
    rng = np.random.default_rng(6)
    pred = jnp.asarray(rng.gamma(1.0, size=(2, 70, 2)), jnp.float32)
    target = jnp.asarray(rng.poisson(0.5, size=(2, 70, 2)), jnp.float32)
    valid = jnp.ones((2, 70), bool)
    loss = {"kind": "multinomial", "scope": "local", "block": 16, "libraries": ["tis"]}
    got = fastloop.loss_multinomial(pred, target, valid, AUX, {**loss, "offset": 8})
    want = local_multinomial(pred[:, 8:], target[:, 8:], jnp.ones((2, 62, 2), bool), (0,), 16)
    assert float(got) == pytest.approx(float(want))
    assert float(fastloop.loss_multinomial(pred, target, valid, AUX, loss)) != pytest.approx(
        float(got)
    )


def test_rep_summary_conditions_on_replicate_2_and_splits_exons():
    vec = np.zeros(300)

    def prof(peaks):
        v = vec.copy()
        v[peaks] = 1.0
        return fastloop.site_profiles(v, [50, 150, 250])

    # site 50: rep2 hits, rep1 hits, pred hits (+1); 150: rep2 hits, rep1 misses, pred misses;
    # 250: rep2 misses (excluded)
    rep2 = prof([50, 150, 270])
    rep1 = prof([50, 140, 250])
    pred = prof([51, 160, 250])
    classes = fastloop.start_classes(["CTG", "GTG", "CTG"])
    regions = fastloop.exon_regions([50, 150, 250], 100)
    assert list(regions) == ["first", "later", "later"]
    assert list(fastloop.exon_regions([5], None)) == ["first"]
    out = fastloop.rep_summary(pred, rep1, rep2, classes, regions)
    assert out["rep_n_nc"] == 2 and out["rep_hit_nc"] == 0.5 and out["rep_ref_nc"] == 0.5
    assert out["rep_ratio_nc"] == 1.0
    assert out["rep_hit_nc_first"] == 1.0 and out["rep_n_nc_first"] == 1
    assert out["rep_hit_nc_later"] == 0.0 and out["rep_ref_nc_later"] == 0.0
    assert math.isnan(out["rep_ratio_nc_later"]) and out["rep_both_nc"] == 1.0
    assert out["rep_n_atg"] == 0 and math.isnan(out["rep_hit_atg"])


def test_phase3_columns_appended_and_primary3(cache):
    rng = np.random.default_rng(6)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in cache.rows("val").itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    columns = fastloop.result_columns(CONDITIONS)
    start = columns.index("primary3")
    first_utr5 = next(i for i, c in enumerate(columns) if "_utr5_" in c)
    assert all("_rep_" in c for c in columns[start + 1 : first_utr5])
    first_sharp = next(i for i, c in enumerate(columns) if "_tis_sharp_" in c)
    assert all("_utr5_" in c for c in columns[first_utr5:first_sharp])
    assert columns.index("primary2") < start and "K562_tis_rep_hit_nc_later" in columns
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, peaks, CONDITIONS)
    assert set(row) == set(columns)
    model = peaks[(peaks.source == "model") & (peaks.condition == "K562")].iloc[0]
    for key in ("rep_hit_nc", "rep_ref_atg", "rep_n_nc_first"):
        assert row[f"K562_tis_{key}"] == pytest.approx(model[key], nan_ok=True)
    real = peaks[(peaks.source == "real") & (peaks.condition == "K562")].iloc[0]
    assert real.rep_hit_nc == pytest.approx(real.rep_ref_nc, nan_ok=True)
    fake = dict(row, K562_tis_nt=60.0, K562_tis_rep_hit_nc=0.3, K562_tis_rep_ref_nc=0.6)
    assert fastloop.primary3(fake, "K562") == pytest.approx((60.0 + 50.0) / 2)
    fake["K562_tis_rep_hit_nc"] = 0.9
    assert fastloop.primary3(fake, "K562") == pytest.approx((60.0 + 100.0) / 2)


def _real_tracks(root, conditions=CONDITIONS, size=5000):
    import pyBigWig

    for c in conditions:
        (root / c).mkdir(parents=True)
        for strand in ("plus", "minus"):
            bw = pyBigWig.open(str(root / c / f"tis_reppooled.{strand}.bw"), "w")
            bw.addHeader([("chr1", size), ("chr2", size)])
            bw.close()
    return root


def test_export_bigwigs_writes_sense_predictions_at_genomic_positions(tmp_path, cache):
    import pyBigWig

    from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
    from tisiago.tracks.txcache import unit_of

    tracks = _real_tracks(tmp_path / "tracks")
    rng = np.random.default_rng(7)
    rows = cache.rows("val")
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in rows.itertuples()
    }
    out = tmp_path / "export"
    manifests = fastloop.export_bigwigs(cache, preds, ["K562"], out, tracks_dir=tracks)
    assert manifests["K562"]["layout"] == "pooled"
    k = cache.condition_index("K562")
    for c, lib in enumerate(fastloop.LIBRARIES):
        for r in rows.itertuples():
            unit = unit_of(r)
            suffix, shift = ("plus", 0) if r.strand == "+" else ("minus", RIBOTISH_MINUS_SHIFT)
            bw = pyBigWig.open(str(out / "K562" / f"{lib}_pooled.{suffix}.bw"))
            got = np.array(
                [bw.values("chr1", int(p) + shift, int(p) + shift + 1)[0] for p in unit.positions]
            )
            bw.close()
            np.testing.assert_allclose(got, preds[r.transcript_id][k, :, c], rtol=1e-6)
    saved = json.loads((out / "K562" / "predict_tracks.json").read_text())
    assert saved["split"] == "val" and saved["files"]["tis"]["-"] == "tis_pooled.minus.bw"
    with pytest.raises(ValueError, match="no predictions"):
        fastloop.export_bigwigs(cache, {}, ["K562"], out, tracks_dir=tracks)


def test_run_config_export_and_params_roundtrip(tmp_path, cache, monkeypatch):
    tracks = _real_tracks(tmp_path / "tracks")
    cache.config["tracks_dir"] = str(tracks)
    config = _config(name="exp")
    out = tmp_path / "export"
    fastloop.run_config(
        config, cache, tmp_path / "r.tsv", tmp_path / "runs", log=lambda s: None, export_dir=out
    )
    assert (out / "exp" / "K562" / "tis_pooled.plus.bw").exists()
    assert not (out / "exp" / "HeLa").exists()
    params = fastloop.load_params(out / "exp" / "params.npz")
    model = fastloop.make_model(config, fastloop.context_of(cache))
    rows = cache.rows("val").iloc[:1]
    pred = fastloop.predict_rows(lambda b: model.apply(params, b)[0], cache, rows, ["K562"], 64)
    assert np.isfinite(next(iter(pred.values()))).all()


def test_exon_features_on_hand_junctions():
    feats = fastloop.exon_features([10, 30], np.array([0, 9, 10, 29, 35]), 40)
    np.testing.assert_allclose(feats[:, 0] * 3, np.log1p([0, 0, 1, 1, 2]), rtol=1e-6)
    np.testing.assert_array_equal(feats[:, 1], [1, 1, 0, 0, 0])
    np.testing.assert_allclose(np.expm1(feats[:, 2] * 10), [0, 9, 0, 19, 5], atol=1e-4)
    np.testing.assert_allclose(np.expm1(feats[:, 3] * 10), [10, 1, 20, 1, 5], atol=1e-4)
    np.testing.assert_allclose(np.expm1(feats[:, 4] * 10), [10, 1, 0, 1, 5], atol=1e-4)
    none = fastloop.exon_features([], np.array([3]), 8)
    np.testing.assert_allclose(np.expm1(none[0, [2, 3]] * 10), [3, 5], atol=1e-4)
    assert np.expm1(none[0, 4] * 10) == pytest.approx(fastloop.JUNCTION_CAP, rel=1e-4)


def test_seq_codes_are_cropped_when_cached_and_n_otherwise(cache):
    rows = cache.index.set_index("transcript_id", drop=False)
    ex = fastloop.crop_example(cache, rows.loc["A"], "HeLa", 0, 64)
    assert (ex["seq"] == fastloop.SEQ_N).all()
    codes = (np.arange(300 + 120 + 50) % 4).astype(np.int8)
    np.save(cache.path / "shards" / "train_chr1" / "seq.npy", codes)
    fresh = TxCache(cache.path)
    ex = fastloop.crop_example(fresh, rows.loc["B"], "HeLa", 0, 128)
    np.testing.assert_array_equal(ex["seq"][:120], codes[300:420])
    assert (ex["seq"][120:] == fastloop.SEQ_N).all()
    onehot = np.asarray(fastloop._input_onehot({"seq": jnp.asarray(ex["seq"])}, {}))
    assert onehot[:120].sum(1).min() == 1 and onehot[120:].sum() == 0


@pytest.mark.parametrize("causal,keep", [("up", slice(0, 60)), ("down", slice(70, 128))])
def test_causal_tis_context_ignores_the_other_side(cache, causal, keep):
    config = _config(
        inputs=["embedding"],
        model={
            "tis_context": {"kind": "dilated", "dilations": [1, 2, 4], "causal": causal},
            "tis": {"kind": "softplus"},
        },
    )
    model = fastloop.make_model(config, fastloop.context_of(cache))
    batch = fastloop._device_batch(fastloop.Sampler(cache, config).batch())
    batch["valid"] = jnp.ones_like(batch["valid"])
    params = model.init(jax.random.PRNGKey(0), batch)
    changed = dict(batch)
    if causal == "up":
        changed["emb"] = batch["emb"].at[:, 60:].add(5.0)
    else:
        changed["emb"] = batch["emb"].at[:, :70].add(5.0)
    a = np.asarray(model.apply(params, batch)[0])[..., 0]
    b = np.asarray(model.apply(params, changed)[0])[..., 0]
    np.testing.assert_allclose(a[:, keep], b[:, keep], rtol=1e-5, atol=1e-6)
    assert not np.allclose(a, b)


def test_tis_target_min_and_geo_use_both_replicates(cache):
    rows = cache.index.set_index("transcript_id", drop=False)
    tis = cache.array("train_chr1", "tis")[0, :300]
    ex = fastloop.crop_example(cache, rows.loc["A"], "HeLa", 0, 300, "min")
    np.testing.assert_allclose(ex["target"][:, 0], 2 * np.minimum(tis[:, 0], tis[:, 1]))
    ex = fastloop.crop_example(cache, rows.loc["A"], "HeLa", 0, 300, "geo")
    np.testing.assert_allclose(ex["target"][:, 0], 2 * np.sqrt(tis[:, 0] * tis[:, 1]), rtol=1e-6)
    pooled = fastloop.crop_example(cache, rows.loc["A"], "HeLa", 0, 300)
    np.testing.assert_allclose(pooled["target"][:, 0], tis[:, 2])
    np.testing.assert_allclose(ex["target"][:, 1], pooled["target"][:, 1])
    sampler = fastloop.Sampler(cache, _config(data={"tis_target": "min"}))
    assert sampler.tis_target == "min"
    with pytest.raises(ValueError, match="tis_target"):
        fastloop.crop_example(cache, rows.loc["A"], "HeLa", 0, 8, "max")


def _test_cache(root, seed=1):
    """A test cache (txcache --test layout): two test_chr8 transcripts, ckpt, sites."""
    rng = np.random.default_rng(seed)
    rows, offset = [], 0
    for tid, strand, length, cs, ce in (("T1", "+", 220, 20, 200), ("T2", "-", 180, 15, 160)):
        rows.append(
            {
                "transcript_id": tid,
                "gene_id": f"G{tid}",
                "gene_name": tid,
                "chrom": "chr2",
                "strand": strand,
                "split": "test",
                "start": 1000 + offset,
                "end": 1000 + offset + length,
                "length": length,
                "cds_start": cs,
                "cds_end": ce,
                "exons": [[1000 + offset, 1000 + offset + length]],
                "cds": [[1000 + offset + cs, 1000 + offset + ce]],
                "junctions": [],
                **{f"expressed_{c}": True for c in CONDITIONS},
                "shard": "test_chr2",
                "offset": offset,
            }
        )
        offset += length
    d = root / "shards" / "test_chr2"
    d.mkdir(parents=True)
    np.save(d / "emb.npy", rng.normal(size=(offset, DIM)).astype(np.float16))
    np.save(d / "rna.npy", rng.poisson(3.0, size=(len(CONDITIONS), offset, 2)).astype("f4"))
    for lib, lam in (("tis", 0.3), ("chx", 2.0)):
        reps = rng.poisson(lam, size=(len(CONDITIONS), offset, 2)).astype(np.float32)
        np.save(d / f"{lib}.npy", np.concatenate([reps, reps.sum(-1, keepdims=True)], -1))
    np.save(d / "ckpt.npy", rng.gamma(1.0, size=(len(CONDITIONS), offset, 2)).astype("f4"))
    pd.DataFrame(rows).to_parquet(root / "index.parquet", index=False)
    (root / "cache.json").write_text(json.dumps({"conditions": list(CONDITIONS), "split": "test"}))
    pd.DataFrame(
        {
            "condition": ["K562", "K562"],
            "transcript_id": ["T1", "T2"],
            "tx_pos": [20, 30],
            "codon": ["ATG", "CTG"],
            "tistype": ["Annotated", "Novel"],
        }
    ).to_parquet(root / "sites.parquet", index=False)
    return TxCache(root)


def _trained_export(tmp_path, cache, **override):
    tracks = _real_tracks(tmp_path / "tracks")
    cache.config["tracks_dir"] = str(tracks)
    config = _config(name="exp", cache=str(cache.path), **override)
    record = fastloop.run_config(
        config,
        cache,
        tmp_path / "results.tsv",
        tmp_path / "runs",
        log=lambda s: None,
        export_dir=tmp_path / "export",
    )
    return config, record, tracks


def _bw_values(path, chrom="chr1", size=5000):
    import pyBigWig

    bw = pyBigWig.open(str(path))
    try:
        return np.nan_to_num(np.array(bw.values(chrom, 0, size)))
    finally:
        bw.close()


def test_predict_export_on_val_reproduces_run_config_exactly(tmp_path, cache):
    config, record, _ = _trained_export(tmp_path, cache, inputs=["embedding_white", "rna"])
    out = tmp_path / "pred_val"
    again = fastloop.predict_export(
        tmp_path / "export" / "exp",
        cache,
        split="val",
        results=tmp_path / "results_test.tsv",
        runs_dir=tmp_path / "runs_test",
        export_dir=out,
        log=lambda s: None,
    )
    assert again["meta"]["config_hash"] == fastloop.config_hash(config)
    for key in ("scorecard", "peaks"):
        pd.testing.assert_frame_equal(
            pd.DataFrame(again[key]), pd.DataFrame(record[key]), check_exact=True
        )
    for key in fastloop.result_columns(cache.conditions)[11:]:
        a, b = record["row"][key], again["row"][key]
        assert (a == b) or (math.isnan(a) and math.isnan(b)), key
    for lib in fastloop.LIBRARIES:
        for strand in ("plus", "minus"):
            name = f"{lib}_pooled.{strand}.bw"
            np.testing.assert_array_equal(
                _bw_values(out / "exp" / "K562" / name),
                _bw_values(tmp_path / "export" / "exp" / "K562" / name),
            )
    frame = pd.read_csv(tmp_path / "results_test.tsv", sep="\t")
    assert len(frame) == 1 and frame.name[0] == "exp"
    assert (tmp_path / "runs_test" / f"exp-{fastloop.config_hash(config)}-val.json").exists()


def test_predict_export_guards(tmp_path, cache):
    _trained_export(tmp_path, cache)
    export = tmp_path / "export" / "exp"
    kw = {"runs_dir": tmp_path / "rt", "log": lambda s: None}
    with pytest.raises(PermissionError, match="i_am_final"):
        fastloop.predict_export(export, cache, "test", tmp_path / "r_test.tsv", **kw)
    with pytest.raises(ValueError, match="not a test cache"):
        fastloop.predict_export(
            export, cache, "test", tmp_path / "r_test.tsv", i_am_final=True, **kw
        )
    with pytest.raises(ValueError, match="main results"):
        fastloop.predict_export(export, cache, "val", tmp_path / "results.tsv", **kw)
    # params of another architecture do not load into this config's model
    params = fastloop.load_params(export / "params.npz")
    other = _config(name="exp", model={"trunk": {"kind": "mlp", "hidden": 32, "layers": 1}})
    with pytest.raises(ValueError, match="do not match"):
        fastloop.rebuild_model(other, cache, params, "val")
    fastloop.rebuild_model(fastloop.load_export(export)[0], cache, params, "val")
    # a changed saved config fails its hash
    payload = json.loads((export / "config.json").read_text())
    payload["config"]["seed"] = 99
    (export / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="config hash"):
        fastloop.load_export(export)


def test_check_pca_requires_the_training_cache_pca(tmp_path, cache):
    train = tmp_path / "train"
    train.mkdir()
    fastloop.check_pca(cache, train)  # no training PCA: nothing to compare
    np.savez(train / "pca.npz", explained_variance=np.ones(DIM))
    with pytest.raises(ValueError, match="not the training cache"):
        fastloop.check_pca(cache, train)
    (cache.path / "pca.npz").write_bytes((train / "pca.npz").read_bytes())
    fastloop.check_pca(cache, train)


def test_predict_export_scores_test_rows_end_to_end(tmp_path, cache):
    import pyBigWig

    from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
    from tisiago.tracks.txcache import unit_of

    config, _, tracks = _trained_export(tmp_path, cache)
    test = _test_cache(tmp_path / "x_test")
    test.config["tracks_dir"] = str(tracks)
    out = tmp_path / "pred_test"
    record = fastloop.predict_export(
        tmp_path / "export" / "exp",
        test,
        split="test",
        results=tmp_path / "results_test.tsv",
        runs_dir=tmp_path / "runs_test",
        export_dir=out,
        i_am_final=True,
        log=lambda s: None,
    )
    table = pd.DataFrame(record["scorecard"])
    assert {"model", "ckpt", "rep1", "rep2"} >= set(table.source) >= {"model", "ckpt"}
    assert set(table.condition) == set(CONDITIONS)
    model_rows = table[(table.source == "model") & (table.scale == "nt")]
    assert len(model_rows) == 2 * len(CONDITIONS) and np.isfinite(model_rows.value).all()
    assert record["meta"]["split"] == "test" and record["meta"]["cache"] == str(test.path)
    peaks = pd.DataFrame(record["peaks"])
    assert set(peaks.source) == {"real", "model", "ckpt"}
    # exported test tracks hold the head's own predictions at the test transcripts' positions
    model = fastloop.make_model(config, fastloop.context_of(test))
    params = fastloop.load_params(tmp_path / "export" / "exp" / "params.npz")
    preds = fastloop.predict_rows(
        jax.jit(lambda b: model.apply(params, b)[0]), test, test.rows("test"), CONDITIONS, 64, 256
    )
    k = test.condition_index("K562")
    for r in test.rows("test").itertuples():
        unit = unit_of(r)
        suffix, shift = ("plus", 0) if r.strand == "+" else ("minus", RIBOTISH_MINUS_SHIFT)
        bw = pyBigWig.open(str(out / "exp" / "K562" / f"tis_pooled.{suffix}.bw"))
        got = np.array(
            [bw.values("chr2", int(p) + shift, int(p) + shift + 1)[0] for p in unit.positions]
        )
        bw.close()
        np.testing.assert_allclose(got, preds[r.transcript_id][k, :, 0], rtol=1e-6)
    manifest = json.loads((out / "exp" / "K562" / "predict_tracks.json").read_text())
    assert manifest["split"] == "test" and manifest["transcripts"] == 2
    frame = pd.read_csv(tmp_path / "results_test.tsv", sep="\t")
    assert len(frame) == 1 and frame.cache[0] == str(test.path)


def test_predict_cli_requires_the_final_flag(tmp_path, monkeypatch):
    import sys

    argv = ["fastloop", "--predict", str(tmp_path / "e"), "--cache", str(tmp_path / "c")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        fastloop.main()
    monkeypatch.setattr(sys, "argv", [*argv[:3]])
    with pytest.raises(SystemExit):
        fastloop.main()


def test_orf_stop_and_frame():
    # ATG CCC TAA GG: stop codon ends at 9
    seq = np.array([0, 3, 2, 1, 1, 1, 3, 0, 0, 2, 2], np.int8)
    assert fastloop.orf_stop(seq, 0) == 9
    assert fastloop.orf_stop(seq, 1) == 10  # no in-frame stop: last full codon
    chx = np.array([4.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 9.0, 9.0])
    assert fastloop.orf_frame(chx, 0, 9) == pytest.approx(7.0 / 10.0)
    assert math.isnan(fastloop.orf_frame(np.zeros(11), 0, 9))


def test_utr5_summary_conditions_on_replicate_2():
    peak = np.zeros((4, 61))
    peak[:, 30] = 1.0
    off = np.zeros((4, 61))
    off[:, 10] = 1.0
    rep2 = np.stack([peak[0], peak[1], peak[2], off[3]])
    rep1 = np.stack([peak[0], off[1], peak[2], peak[3]])
    pred = np.stack([peak[0], peak[1], off[2], peak[3]])
    utr5 = np.array([True, True, False, True])
    frames = np.array([0.6, 0.4, 0.9, np.nan])
    out = fastloop.utr5_summary(pred, rep1, rep2, utr5, frames)
    assert out["utr5_tis_rep_n"] == 2  # sites 0, 1 (site 3 misses in rep2; site 2 not 5'UTR)
    assert out["utr5_tis_rep_hit"] == 1.0 and out["utr5_tis_rep_ref"] == 0.5
    assert out["utr5_tis_rep_ratio"] == 2.0
    assert out["utr5_orf_n"] == 2 and out["utr5_orf_frame"] == pytest.approx(0.5)
    assert out["utr5_orf_framed"] == 0.5


def test_utr5_columns_filled_from_cache_with_seq(cache):
    rows = cache.rows("val")
    size = int(rows.offset.max() + rows.length.max())
    seq = np.zeros(size, np.int8)  # all A: no stop codon, the ORF runs to the last codon
    np.save(cache.path / "shards" / "val_chr1" / "seq.npy", seq)
    sites = cache.sites.copy()
    sites.loc[sites.transcript_id == "V2", "tistype"] = "5'UTR"
    cache.sites = sites
    rng = np.random.default_rng(7)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in rows.itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, peaks, CONDITIONS)
    k562 = peaks[(peaks.condition == "K562")].set_index("source")
    assert k562.loc["model", "utr5_orf_n"] == 1
    v2 = rows[rows.transcript_id == "V2"].iloc[0]
    k = cache.condition_index("K562")
    lo = int(v2.offset)
    chx = np.asarray(cache.array("val_chr1", "chx")[k, lo : lo + int(v2.length), 2], np.float64)
    stop = 25 + (int(v2.length) - 25) // 3 * 3
    assert row["K562_utr5_orf_frame_real"] == pytest.approx(fastloop.orf_frame(chx, 25, stop))
    pred = preds["V2"][k, :, 1]
    assert row["K562_utr5_orf_frame"] == pytest.approx(fastloop.orf_frame(pred, 25, stop))
    # existing columns are unchanged by the new ones
    base = fastloop.results_row(
        meta, table, peaks.drop(columns=[c for c in peaks if c.startswith("utr5_")]), CONDITIONS
    )
    for col in fastloop.result_columns(CONDITIONS):
        if "_utr5_" not in col:
            assert row[col] == pytest.approx(base[col], nan_ok=True) or row[col] == base[col]


def test_append_result_extends_an_older_header(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text("a\tb\n1\t2\n")
    fastloop.append_result(path, {"a": 3, "b": 4, "c": 5.0}, ["a", "b", "c"])
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == ["a", "b", "c"]
    assert frame.a.tolist() == [1, 3] and math.isnan(frame.c[0]) and frame.c[1] == 5.0
    with pytest.raises(ValueError, match="different header"):
        fastloop.append_result(path, {"a": 1}, ["a", "x", "c", "d"])


def test_chx_onehot_needs_chx_trunk():
    with pytest.raises(ValueError, match="chx_onehot needs"):
        _config(model={"chx_onehot": {"width": 5}})


def _flux_naive(init, stop, keep, bg, valid):
    out = np.zeros_like(init)
    for b in range(init.shape[0]):
        for i in range(init.shape[1]):
            if not valid[b, i]:
                continue
            prev = out[b, i - 3] if i >= 3 and valid[b, i - 3] else 0.0
            out[b, i] = prev * (1 - stop[b, i]) * keep + init[b, i] + bg
    return out


def test_flux_scan_equals_a_naive_loop_and_masks_padding():
    rng = np.random.default_rng(0)
    init = rng.gamma(0.5, size=(3, 47)).astype(np.float32)
    stop = rng.uniform(0, 0.5, size=(3, 47)).astype(np.float32)
    valid = np.ones((3, 47), bool)
    valid[1, 40:] = False
    valid[2, 20:] = False
    got = np.asarray(fastloop.flux_scan(init, stop, 0.9, 0.01, valid))
    np.testing.assert_allclose(got, _flux_naive(init, stop, 0.9, 0.01, valid), rtol=1e-5)
    assert (got[~valid] == 0).all()
    other = init.copy()
    other[~valid] = 100.0
    again = np.asarray(fastloop.flux_scan(other, stop, 0.9, 0.01, valid))
    np.testing.assert_array_equal(again, got)


def test_flux_without_stops_is_the_in_frame_cumsum_and_keeps_the_init_frame():
    rng = np.random.default_rng(1)
    init = rng.gamma(0.5, size=(1, 30)).astype(np.float32)
    valid = np.ones((1, 30), bool)
    got = np.asarray(fastloop.flux_scan(init, np.zeros_like(init), 1.0, 0.0, valid))
    for frame in range(3):
        np.testing.assert_allclose(got[0, frame::3], np.cumsum(init[0, frame::3]), rtol=1e-5)
    spike = np.zeros((1, 30), np.float32)
    spike[0, 7] = 1.0
    flux = np.asarray(fastloop.flux_scan(spike, np.full_like(spike, 0.1), 0.95, 0.0, valid))[0]
    on = np.flatnonzero(flux > 0)
    assert on[0] == 7 and (on % 3 == 1).all() and len(on) == len(range(7, 30, 3))


def test_frame_blur_identity_and_shift():
    x = jnp.arange(10.0)[None]
    np.testing.assert_allclose(np.asarray(fastloop.frame_blur(x, jnp.array([0.0, 1.0, 0.0]))), x)
    shifted = np.asarray(fastloop.frame_blur(x, jnp.array([0.0, 0.0, 1.0])))[0]
    np.testing.assert_allclose(shifted, [*range(1, 10), 0])


def _flux_model_pair(cache, flux):
    model_cfg = {
        "tis": {"kind": "factorised", "block": 32},
        "chx": {"kind": "factorised", "block": 32},
        "chx_trunk": {"kind": "conv", "width": 7, "hidden": 8},
    }
    base = _config(model=model_cfg)
    with_flux = _config(model={**model_cfg, "chx_flux": flux})
    ctx = fastloop.context_of(cache)
    batch = fastloop._device_batch(fastloop.Sampler(cache, base).batch())
    return [fastloop.make_model(c, ctx) for c in (base, with_flux)], batch


def test_chx_flux_logit_at_beta_zero_equals_the_base_model_at_init(cache):
    (base, flux), batch = _flux_model_pair(cache, {"mode": "logit"})
    key = jax.random.PRNGKey(3)
    p_base, p_flux = base.init(key, batch), flux.init(key, batch)
    assert any(k.startswith("chx_flux") or "chx_flux" in k for k in p_flux)
    for k, v in p_base.items():
        for name, leaf in v.items():
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(p_flux[k][name]))
    a, b = base.apply(p_base, batch)[0], flux.apply(p_flux, batch)[0]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("stop_gradient", [True, False])
def test_chx_flux_couples_tis_only_without_stop_gradient(cache, stop_gradient):
    flux = {"mode": "replace", "stop_gradient_init": stop_gradient}
    (_, model), batch = _flux_model_pair(cache, flux)
    params = model.init(jax.random.PRNGKey(0), batch)

    def chx_only(params):
        pred, _ = model.apply(params, batch)
        return (pred[..., 1] * jnp.arange(pred.shape[1])).sum()

    grads = jax.grad(chx_only)(params)
    tis_out = np.abs(np.asarray(grads["tis_out"]["w"])).sum()
    assert (tis_out == 0) == stop_gradient


def test_chx_flux_validation():
    with pytest.raises(ValueError, match="factorised CHX"):
        _config(model={"chx_flux": {"mode": "logit"}})
    chx = {"kind": "factorised", "block": 32}
    with pytest.raises(ValueError, match="mode"):
        _config(model={"chx": chx, "chx_flux": {"mode": "nope"}})
    with pytest.raises(ValueError, match="frame_kernel"):
        _config(model={"chx": chx, "chx_flux": {"frame_kernel": 4}})
    assert _config(model={"chx": chx, "chx_flux": {"frame_kernel": 1}})
    with pytest.raises(ValueError, match="chx_contrast"):
        _config(model={"chx": chx, "chx_contrast": True, "chx_flux": {"mode": "logit"}})


def test_codon_hazard_sees_the_codon_starting_at_each_nt():
    import haiku as hk

    fn = hk.without_apply_rng(hk.transform(lambda x, v: fastloop.codon_hazard(x, v, 4)))
    onehot = jax.nn.one_hot(jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]]), 4)
    valid = jnp.ones((1, 8), bool)
    params = fn.init(jax.random.PRNGKey(0), onehot, valid)
    params["chx_flux_codon_out"]["w"] = jnp.ones_like(params["chx_flux_codon_out"]["w"])
    base = np.asarray(fn.apply(params, onehot, valid))[0]
    changed = onehot.at[0, 5].set(jnp.array([1.0, 0, 0, 0]))
    after = np.asarray(fn.apply(params, changed, valid))[0]
    differs = np.flatnonzero(~np.isclose(base, after))
    assert set(differs) <= {3, 4, 5} and len(differs) > 0


def test_fixed_stop_codons_end_in_frame_flux_and_pass_other_frames():
    # A C G T = 0 1 2 3, N = 4. TAA at 6 (frame 0), TGA at 13 (frame 1), TAN at 20 (not a stop),
    # TAG cut by padding at the end.
    seq = np.full((1, 30), 0, np.int8)
    seq[0, 6:9] = (3, 0, 0)
    seq[0, 13:16] = (3, 2, 0)
    seq[0, 20:23] = (3, 0, 4)
    seq[0, 26:28] = (3, 0)
    seq[0, 28] = 2
    valid = np.ones((1, 30), bool)
    valid[0, 28:] = False
    starts = np.asarray(fastloop.stop_codon_starts(jnp.asarray(seq), jnp.asarray(valid)))[0]
    assert np.flatnonzero(starts).tolist() == [6, 13]
    ended = fastloop.stop_ended(jnp.asarray(seq), jnp.asarray(valid))
    assert np.flatnonzero(np.asarray(ended)[0]).tolist() == [9, 16]
    init = np.zeros((1, 30), np.float32)
    init[0, [0, 1, 2]] = 1.0  # one ribosome load per frame upstream
    stop = jnp.where(ended, 1.0, 0.0)
    flux = np.asarray(fastloop.flux_scan(init, stop, 1.0, 0.0, valid))[0]
    np.testing.assert_allclose(flux[0:9:3], 1.0)  # frame 0 up to and including the TAA
    assert (flux[9:28:3] == 0).all()  # frame 0 ends after its stop
    np.testing.assert_allclose(flux[1:14:3], 1.0)  # frame 1 passes TAA, keeps flux on TGA
    assert (flux[16:28:3] == 0).all()
    np.testing.assert_allclose(flux[2:28:3], 1.0)  # frame 2 passes every stop and the TAN
    assert (flux[28:] == 0).all()


def test_fixed_stop_flux_model_without_learned_hazard(cache, tmp_path):
    flux = {"mode": "replace", "stop_codons": "fixed", "learned_hazard": False}
    (_, model), batch = _flux_model_pair(cache, flux)
    params = model.init(jax.random.PRNGKey(0), batch)
    assert not any("chx_flux_stop" in k for k in params)
    pred, _ = model.apply(params, batch)
    assert np.isfinite(np.asarray(pred)).all()
    with pytest.raises(ValueError, match="stop_codons"):
        _config(model={"chx": {"kind": "factorised"}, "chx_flux": {"stop_codons": "x"}})
    with pytest.raises(ValueError, match="needs learned_hazard"):
        bad = {"learned_hazard": False, "stop_onehot": {"channels": 4}}
        _config(model={"chx": {"kind": "factorised"}, "chx_flux": bad})


def test_sharp_summary_per_start_class():
    sharp = np.zeros(61)
    sharp[30] = 3.0
    sharp[31] = 1.0
    broad = np.zeros(61)
    broad[28:33] = 1.0
    far = np.zeros(61)
    far[30] = 1.0
    far[5] = 9.0  # outside +-15: ignored
    prof = np.stack([sharp, broad, far, np.zeros(61)])
    classes = np.array(["atg", "nc", "nc", "atg"], dtype=object)
    utr5 = np.array([False, True, True, False])
    out = fastloop.sharp_summary(prof, classes, utr5)
    assert out["sharp_n_atg"] == 1  # the empty site is dropped
    assert out["sharp_share15_atg"] == pytest.approx(0.75)
    assert out["sharp_argmax15_atg"] == 1.0
    assert out["sharp_n_nc"] == 2 and out["sharp_share15_nc"] == pytest.approx((0.2 + 1.0) / 2)
    assert out["sharp_argmax15_nc"] == 0.5  # broad's argmax is its first max (28)
    assert out["sharp_n_utr5"] == 2 and out["sharp_share15_utr5"] == out["sharp_share15_nc"]
    empty = fastloop.sharp_summary(prof[:0], classes[:0], utr5[:0])
    assert empty["sharp_n_utr5"] == 0 and math.isnan(empty["sharp_share15_atg"])


def test_top_share_and_peakiness():
    vec = np.zeros(200)
    vec[[3, 50]] = [6.0, 2.0]
    vec[100:110] = 0.2
    assert fastloop.top_share(vec) == pytest.approx((6.0 + 2.0) / 10.0)  # top 2 of 200
    assert fastloop.top_share(np.ones(10)) == pytest.approx(0.1)  # at least one position
    assert math.isnan(fastloop.top_share(np.zeros(5)))
    real = [vec * 10, np.full(200, 0.1)]  # second has 20 reads < 50: skipped
    flat = [np.ones(200), np.ones(200)]
    out = fastloop.peakiness_summary(real)
    assert out["top1_n"] == 1 and out["top1"] == pytest.approx(0.8)
    assert math.isnan(out["top1_resampled"])
    pred = fastloop.peakiness_summary(real, flat)
    assert pred["top1"] == pytest.approx(0.01)
    assert 0.01 < pred["top1_resampled"] < 0.2  # counting noise at 100 reads
    again = fastloop.peakiness_summary(real, flat)
    assert again == pred  # fixed seed


def test_sharp_columns_filled_and_existing_unchanged(cache):
    rows = cache.rows("val")
    sites = cache.sites.copy()
    sites.loc[sites.transcript_id == "V2", "tistype"] = "5'UTR"
    cache.sites = sites
    rng = np.random.default_rng(3)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in rows.itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, peaks, CONDITIONS)
    columns = fastloop.result_columns(CONDITIONS)
    k562 = peaks[peaks.condition == "K562"].set_index("source")
    for cls in fastloop.SHARP_CLASSES:
        assert row[f"K562_tis_sharp_n_{cls}"] == k562.loc["model", f"sharp_n_{cls}"]
        for stat in fastloop.SHARP_STATS:
            got = row[f"K562_tis_sharp_{stat}_{cls}_real"]
            assert got == pytest.approx(k562.loc["real", f"sharp_{stat}_{cls}"], nan_ok=True)
    assert row["K562_tis_top1_real"] == pytest.approx(k562.loc["real", "top1"], nan_ok=True)
    assert row["K562_tis_top1"] == pytest.approx(k562.loc["model", "top1"], nan_ok=True)
    new = [c for c in columns if "_tis_sharp_" in c or "_tis_top1" in c]
    later = [c for c in columns if c.endswith(fastloop.CDS_PARTS) or "_chx_pause_" in c]
    assert columns[-len(new) - len(later) : -len(later)] == new  # appended after the earlier ones
    base = fastloop.results_row(
        meta,
        table,
        peaks.drop(columns=[c for c in peaks if c.startswith(("sharp_", "top1"))]),
        CONDITIONS,
    )
    for col in columns:
        if col not in new:
            assert row[col] == pytest.approx(base[col], nan_ok=True) or row[col] == base[col]


def test_rep_summary_splits_in_cds_without_changing_the_unsplit_keys():
    vec = np.zeros(300)

    def prof(peaks):
        v = vec.copy()
        v[peaks] = 1.0
        return fastloop.site_profiles(v, [50, 150, 250])

    rep2, rep1, pred = prof([50, 150, 250]), prof([50, 150, 250]), prof([51, 160, 250])
    classes = fastloop.start_classes(["CTG", "GTG", "CTG"])
    regions = fastloop.exon_regions([50, 150, 250], 100)
    in_cds = np.array([False, True, True])
    plain = fastloop.rep_summary(pred, rep1, rep2, classes, regions)
    split = fastloop.rep_summary(pred, rep1, rep2, classes, regions, in_cds)
    for key, value in plain.items():
        assert split[key] == pytest.approx(value, nan_ok=True)
    assert split["rep_n_nc_incds"] == 2 and split["rep_hit_nc_incds"] == 0.5
    assert split["rep_n_nc_later_incds"] == 2 and split["rep_ratio_nc_later_incds"] == 0.5
    assert split["rep_n_nc_outcds"] == 1 and split["rep_hit_nc_first_outcds"] == 1.0
    assert split["rep_n_nc_later_outcds"] == 0 and math.isnan(split["rep_hit_nc_later_outcds"])
    assert set(split) - set(plain) == {
        f"rep_{stat}_{cls}{region}{part}"
        for stat in fastloop.REP_STATS
        for cls in fastloop.START_CLASSES
        for region in fastloop.EXON_REGIONS
        for part in fastloop.CDS_PARTS
    }


def test_in_cds_frame_and_peak_heights():
    got = fastloop.in_cds_frame([10, 13, 14, 40, 43, 7], 10, 43)
    assert got.tolist() == [False, True, False, True, False, False]  # start codon itself excluded
    x = np.zeros(100)
    x[50] = 61.0
    h = fastloop.peak_heights(x)
    assert h[50] == pytest.approx(61.0) and h[0] == 0.0
    flat = fastloop.peak_heights(np.ones(100))
    assert flat[50] == pytest.approx(1.0) and flat[0] == pytest.approx(61 / 31)  # zero past ends


def test_chx_pause_summary_depth_matched_heights():
    length, cds = 300, (40, 260)
    rep2 = np.zeros(length)
    rep2[[100, 150]] = 10.0
    rep2[[20, 255]] = 10.0  # outside the CDS interior: not peaks
    rep1 = np.ones(length)
    rep1[100] = 31.0  # height at 100 = 31 / (91 / 61); at 150 = 1
    ref = (31 * 61 / 91 + 1.0) / 2
    real = fastloop.chx_pause_summary(None, [rep1], [rep2], [cds])
    assert real["chx_pause_n"] == 2 and real["chx_pause_ref"] == pytest.approx(ref)
    assert real["chx_pause_ratio"] == pytest.approx(1.0)
    spike = np.zeros(length)
    spike[100] = 1.0  # every drawn read on one nt: height 61 there, 0 at 150
    out = fastloop.chx_pause_summary([spike], [rep1], [rep2], [cds])
    assert out["chx_pause_pred"] == pytest.approx(61 / 2)
    assert out["chx_pause_ratio"] == pytest.approx(61 / 2 / ref)
    flat = fastloop.chx_pause_summary([np.ones(length)], [rep1], [rep2], [cds])
    assert flat == fastloop.chx_pause_summary([np.ones(length)], [rep1], [rep2], [cds])  # seeded
    assert flat["chx_pause_pred"] < 3.0
    none = fastloop.chx_pause_summary([spike], [rep1], [np.zeros(length)], [cds])
    assert none["chx_pause_n"] == 0 and math.isnan(none["chx_pause_ratio"])


def test_pause_columns_filled_and_existing_unchanged(cache, monkeypatch):
    monkeypatch.setattr(fastloop, "PAUSE_MIN_HEIGHT", 2.0)  # the synthetic CHX is Poisson(2)
    monkeypatch.setattr(fastloop, "PAUSE_MIN_READS", 4.0)
    rows = cache.rows("val")
    rng = np.random.default_rng(4)
    preds = {
        r.transcript_id: rng.gamma(1.0, size=(3, r.length, 2)).astype(np.float32)
        for r in rows.itertuples()
    }
    table, peaks = fastloop.score_predictions(
        cache, {"model": preds}, CONDITIONS, heldout=["K562"], min_reads=5
    )
    meta = {"config_hash": "abc", "name": "x", "heldout_list": ["K562"], "steps": 3}
    row = fastloop.results_row(meta, table, peaks, CONDITIONS)
    columns = fastloop.result_columns(CONDITIONS)
    new = [c for c in columns if c.endswith(fastloop.CDS_PARTS) or "_chx_pause_" in c]
    assert columns[-len(new) :] == new  # appended after every existing column
    k562 = peaks[peaks.condition == "K562"].set_index("source")
    for stat in fastloop.PAUSE_STATS:
        got, want = row[f"K562_chx_pause_{stat}"], k562.loc["model", f"chx_pause_{stat}"]
        assert got == pytest.approx(want, nan_ok=True)
    assert k562.loc["real", "chx_pause_n"] > 0
    assert k562.loc["real", "chx_pause_ratio"] == pytest.approx(1.0)
    for key in ("rep_n_nc_incds", "rep_n_atg_outcds", "rep_hit_atg_first_outcds"):
        assert row[f"K562_tis_{key}"] == pytest.approx(k562.loc["model", key], nan_ok=True)
    base = fastloop.results_row(
        meta,
        table,
        peaks.drop(
            columns=[
                c for c in peaks if c.startswith("chx_pause_") or c.endswith(fastloop.CDS_PARTS)
            ]
        ),
        CONDITIONS,
    )
    for col in columns:
        if col not in new:
            assert row[col] == pytest.approx(base[col], nan_ok=True) or row[col] == base[col]


def _pair(cache, a: dict, b: dict):
    model_cfg = {
        "tis": {"kind": "factorised", "block": 32},
        "chx": {"kind": "factorised", "block": 32},
        "chx_trunk": {"kind": "conv", "width": 7, "hidden": 8},
    }
    configs = [_config(model={**model_cfg, **extra}) for extra in (a, b)]
    ctx = fastloop.context_of(cache)
    batch = fastloop._device_batch(fastloop.Sampler(cache, configs[0]).batch())
    return [fastloop.make_model(c, ctx) for c in configs], batch


def _shared_params_equal(p_base, p_new):
    for k, v in p_base.items():
        for name, leaf in v.items():
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(p_new[k][name]))


# configs/tracks/spliced_flux_pos.yaml; the §16 config hashed ec13b3768a88 before its name and
# cache path moved (identical model, losses and training otherwise).
G7_HASH = "26ef36ef60c8"
FLUX = {"mode": "replace", "stop_codons": "fixed", "learned_hazard": False, "frame_kernel": 1}


def test_dwell_codon_starts_equal_to_the_flux_model_and_learns(cache):
    dwell = {**FLUX, "dwell_codon": {"channels": 4}}
    (base, model), batch = _pair(cache, {"chx_flux": FLUX}, {"chx_flux": dwell})
    key = jax.random.PRNGKey(1)
    p_base, p_new = base.init(key, batch), model.init(key, batch)
    assert {k for k in p_new if "dwell_codon" in k} == {
        "chx_flux_dwell_codon",
        "chx_flux_dwell_codon_out",
    }
    _shared_params_equal(p_base, p_new)
    np.testing.assert_array_equal(
        np.asarray(base.apply(p_base, batch)[0]), np.asarray(model.apply(p_new, batch)[0])
    )

    def chx_loss(params):
        pred, _ = model.apply(params, batch)
        return (pred[..., 1] * jnp.arange(pred.shape[1])).sum()

    grads = jax.grad(chx_loss)(p_new)
    assert np.abs(np.asarray(grads["chx_flux_dwell_codon_out"]["b"])).sum() > 0  # no seq: N
    with pytest.raises(ValueError, match="dwell_codon"):
        _config(model={"chx": {"kind": "factorised"}, "chx_flux": {**FLUX, "dwell_codon": 4}})


def test_codon_dwell_sees_the_p_and_a_site_codons():
    import haiku as hk

    fn = hk.without_apply_rng(hk.transform(lambda x, v: fastloop.codon_dwell(x, v, 4)))
    onehot = jax.nn.one_hot(jnp.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]), 4)
    valid = jnp.ones((1, 12), bool)
    params = fn.init(jax.random.PRNGKey(0), onehot, valid)
    assert np.allclose(np.asarray(fn.apply(params, onehot, valid)), 0.0)  # zero init
    params["chx_flux_dwell_codon_out"]["w"] = jnp.ones_like(params["chx_flux_dwell_codon_out"]["w"])
    params["chx_flux_dwell_codon"]["b"] = jnp.ones_like(params["chx_flux_dwell_codon"]["b"])
    base = np.asarray(fn.apply(params, onehot, valid))[0]
    changed = onehot.at[0, 8].set(jnp.array([0.0, 0, 0, 1]))
    after = np.asarray(fn.apply(params, changed, valid))[0]
    differs = np.flatnonzero(~np.isclose(base, after))
    assert set(differs) <= {3, 4, 5, 6, 7, 8} and len(differs) > 0


@pytest.mark.parametrize("flux", [None, FLUX])
def test_tis_leak_adds_scaled_model_chx_to_tis(cache, flux):
    extra = {"chx_flux": flux} if flux else {}
    (base, model), batch = _pair(cache, extra, {**extra, "tis_leak": {"init": 0.0}})
    key = jax.random.PRNGKey(2)
    p_base, p_new = base.init(key, batch), model.init(key, batch)
    assert set(p_new) - set(p_base) == {"~"} or "tis_leak" in p_new.get("~", {})
    _shared_params_equal({k: v for k, v in p_base.items() if k != "~"}, p_new)
    a, b = np.asarray(base.apply(p_base, batch)[0]), np.asarray(model.apply(p_new, batch)[0])
    np.testing.assert_array_equal(a[..., 1], b[..., 1])  # CHX unchanged
    valid = np.asarray(batch["valid"])
    lam = np.log1p(np.exp(fastloop.TIS_LEAK_BIAS))
    tis, chx = np.where(valid, a[..., 0], 0), np.where(valid, a[..., 1], 0)
    scale = tis.sum(-1, keepdims=True) / (chx.sum(-1, keepdims=True) + fastloop.EPS)
    np.testing.assert_allclose(b[..., 0], np.where(valid, tis + lam * scale * chx, 0), rtol=1e-5)

    def tis_loss(params):
        pred, _ = model.apply(params, batch)
        return (pred[..., 0] * jnp.arange(pred.shape[1])).sum()

    grads = jax.grad(tis_loss)(p_new)
    assert float(grads["~"]["tis_leak"]) != 0.0
    assert np.abs(np.asarray(grads["chx_out"]["w"])).sum() > 0  # TIS data trains the CHX path
    with pytest.raises(ValueError, match="tis_leak"):
        _config(model={"tis_leak": 0.1})


def test_new_options_leave_existing_config_hashes_unchanged():
    raw = fastloop.load_config("configs/tracks/spliced_flux_pos.yaml")
    assert fastloop.config_hash(raw) == G7_HASH
