"""Tests for writing predicted tracks as per-replicate Ribo-TISH-convention bigWigs."""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

from tisiago.seq2func import predict_tracks
from tisiago.seq2func.predict_tracks import (
    owned_spans,
    predict_condition,
    region_of,
    replicate_fractions,
)
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT, psite
from tisiago.seq2func.samples import parse_para
from tisiago.seq2func.targets import channel_layout
from tisiago.seq2func.tracks import read_bigwig_counts, track_prefix, write_bigwigs

SIZES = [("chr1", 300), ("chr7", 400)]
WINDOWS = pd.DataFrame({"chrom": ["chr7"] * 3, "start": [0, 50, 100], "end": [100, 150, 200]})


def _genes():
    return pd.DataFrame(
        {
            "condition": "K562",
            "gene_id": ["G1"],
            "chrom": "chr7",
            "start": [10],
            "end": [190],
            "strand": ["+"],
            "expressed": [True],
        }
    )


def _write_reps(tracks, plus, minus, reps=(1, 2), libraries=("tis", "chx", "rna")):
    """Real tracks; ``plus``/``minus`` map chrom -> {position: count} (minus in file coords)."""
    for library in libraries:
        for rep in reps:
            counts = {}
            for chrom, size in SIZES:
                # one read near each chromosome end (outside every window) keeps no strand empty
                counts[(chrom, "+")] = Counter({size - 5: 1, **plus.get(chrom, {})})
                counts[(chrom, "-")] = Counter({size - 5: 1, **minus.get(chrom, {})})
            write_bigwigs(counts, SIZES, track_prefix(tracks, "K562", library, rep))


@pytest.fixture
def world(tmp_path):
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "A" * 300 + "\n>chr7\n" + ("ATGAAACCC" * 50)[:400] + "\n")
    Fasta(str(fasta)).close()
    return dict(fasta=str(fasta), tracks=tmp_path / "tracks", out=tmp_path / "pred")


def test_owned_spans_nearest_centre():
    windows = pd.DataFrame(
        {
            "chrom": ["chr7", "chr7", "chr7", "chr7", "chr1"],
            "start": [0, 50, 100, 300, 0],
            "end": [100, 150, 200, 400, 100],
        }
    )
    spans = owned_spans(windows)
    # base 74 (centre 74.5) is nearer window 0's centre 50; base 75 nearer window 1's 100
    assert spans == [(0, 75), (75, 125), (125, 200), (300, 400), (0, 100)]


def test_region_of_merges_split_windows():
    assert region_of(WINDOWS) == {"chr7": [(0, 200)]}


def test_replicate_fractions_count_only_the_region(tmp_path):
    tracks = tmp_path / "tracks"
    # rep1: 3 in region (+), 1 at minus file pos 200 (canonical 199, in region), 50 on chr1
    # rep2: 4 in region, 7 at plus 250 (outside), minus file pos 0 is canonical -1 (outside)
    _write_reps(tracks, {"chr7": {10: 3}, "chr1": {5: 50}}, {"chr7": {200: 1}}, reps=(1,))
    _write_reps(tracks, {"chr7": {20: 4, 250: 7}}, {"chr7": {0: 9}}, reps=(2,))
    fractions = replicate_fractions(tracks, "K562", region_of(WINDOWS))
    assert fractions["tis"]["fractions"] == pytest.approx([0.5, 0.5])
    assert fractions["tis"]["totals"] == [4.0, 4.0]


def _oracle(batch):
    return batch["targets"]


def _psites(bam_path):
    pysam = pytest.importorskip("pysam")
    from tisiago.seq2func.synthetic_bam import write_para

    para = bam_path.with_name(bam_path.name + ".para.py")
    write_para(para)
    offsets, m0 = parse_para(para)
    found = Counter()
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            key = (read.reference_name, "-" if read.is_reverse else "+")
            found[(*key, psite(read, offsets, m0))] += 1
    return found


def test_oracle_expected_tracks_reproduce_real_replicates(world):
    plus = {"chr7": {10: 4, 77: 2, 199: 6, 250: 8}, "chr1": {3: 2}}
    minus = {"chr7": {31: 2, 76: 4, 200: 2, 201: 8}}
    _write_reps(world["tracks"], plus, minus)
    # pooled = rep1 + rep2 (identical replicates)
    _write_reps(
        world["tracks"],
        {k: {p: 2 * v for p, v in d.items()} for k, d in plus.items()},
        {k: {p: 2 * v for p, v in d.items()} for k, d in minus.items()},
        reps=("pooled",),
    )
    manifest = predict_condition(
        _oracle,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=world["out"],
        mode="expected",
    )
    assert manifest["fractions"]["chx"]["fractions"] == pytest.approx([0.5, 0.5])
    for library in ("tis", "chx"):
        for rep in (1, 2):
            pred = track_prefix(world["out"], "K562", library, rep)
            got_plus = read_bigwig_counts(f"{pred}.plus.bw")
            got_minus = read_bigwig_counts(f"{pred}.minus.bw")
            # region [0, 200): plus 250 and chr1 are outside; minus file 201 = canonical 200
            assert dict(got_plus["chr7"]) == {10: 4.0, 77: 2.0, 199: 6.0}
            assert dict(got_minus["chr7"]) == {31: 2.0, 76: 4.0, 200: 2.0}
            assert not got_plus["chr1"]
            pytest.importorskip("pysam")
            from tisiago.seq2func.synthetic_bam import tracks_to_bam

            bam = world["out"] / f"{library}_rep{rep}.bam"
            tracks_to_bam(f"{pred}.plus.bw", f"{pred}.minus.bw", bam)
            assert _psites(bam) == Counter(
                {
                    ("chr7", "+", 10): 4,
                    ("chr7", "+", 77): 2,
                    ("chr7", "+", 199): 6,
                    ("chr7", "-", 31): 2,
                    ("chr7", "-", 76): 4,
                    ("chr7", "-", 200): 2,
                }
            )
    assert not (world["out"] / "K562" / "rna_rep1.plus.bw").exists()
    assert manifest["stitching"] == "nearest_centre"


def test_minus_strand_prediction_written_at_ribotish_position(world):
    _write_reps(world["tracks"], {"chr7": {10: 1}}, {}, reps=(1, 2, "pooled"))
    layout = channel_layout(["K562"])
    minus_tis = layout.index(("K562", "tis", "-"))

    calls = []

    def one_minus_site(batch):
        pred = np.zeros_like(batch["targets"])
        start = int(WINDOWS.start[len(calls)])
        calls.append(start)
        if 0 <= 60 - start < pred.shape[1]:
            pred[0, 60 - start, minus_tis] = 8.0  # canonical P-site 60 (windows 0 and 1)
        return pred

    predict_condition(
        one_minus_site,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=world["out"],
        mode="expected",
    )
    got = read_bigwig_counts(f"{track_prefix(world['out'], 'K562', 'tis', 1)}.minus.bw")
    assert calls == [0, 50, 100]
    assert dict(got["chr7"]) == {60 + RIBOTISH_MINUS_SHIFT: 4.0}  # written once (window 0 owns)


def test_replicate_scaling_follows_real_depth(world):
    _write_reps(world["tracks"], {"chr7": {10: 1}}, {}, reps=(1,))
    _write_reps(world["tracks"], {"chr7": {10: 3}}, {}, reps=(2,))
    _write_reps(world["tracks"], {"chr7": {10: 4}}, {}, reps=("pooled",))

    def flat(batch):
        return np.full_like(batch["targets"], 8.0)

    manifest = predict_condition(
        flat,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=world["out"],
        mode="expected",
    )
    assert manifest["fractions"]["tis"]["fractions"] == pytest.approx([0.25, 0.75])
    rep1 = read_bigwig_counts(f"{track_prefix(world['out'], 'K562', 'tis', 1)}.plus.bw")
    rep2 = read_bigwig_counts(f"{track_prefix(world['out'], 'K562', 'tis', 2)}.plus.bw")
    assert set(rep1["chr7"].values()) == {2.0}
    assert set(rep2["chr7"].values()) == {6.0}
    assert sorted(rep1["chr7"]) == list(range(200))  # each position written once
    assert manifest["expected_totals"]["tis"] == pytest.approx(8.0 * 200 * 2)


def _sampled(world, out, seed, copies=1):
    def flat(batch):
        return np.full_like(batch["targets"], 6.0)

    return predict_condition(
        flat,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=out,
        mode="sample",
        seed=seed,
        copies=copies,
    )


def test_sampling_is_seeded_and_reproducible(world):
    _write_reps(world["tracks"], {"chr7": {10: 2}}, {}, reps=(1, 2))
    _write_reps(world["tracks"], {"chr7": {10: 4}}, {}, reps=("pooled",))
    runs = {
        name: _sampled(world, world["out"].with_name(name), seed)
        for name, seed in (("a", 7), ("b", 7), ("c", 8))
    }

    def read(name, rep):
        prefix = track_prefix(world["out"].with_name(name), "K562", "tis", rep)
        return read_bigwig_counts(f"{prefix}.plus.bw")["chr7"]

    assert read("a", 1) == read("b", 1)
    assert read("a", 1) != read("c", 1)
    assert read("a", 1) != read("a", 2)
    values = np.array(list(read("a", 1).values()))
    assert np.all(values == np.round(values)) and np.all(values > 0)
    assert runs["a"]["seed"] == 7 and runs["a"]["mode"] == "sample"
    # mean 3 per position over 200 positions: the sampled total is near 600
    assert 500 < sum(read("a", 1).values()) < 700


def test_copies_draw_even_counts(world):
    _write_reps(world["tracks"], {"chr7": {10: 2}}, {}, reps=(1, 2))
    _write_reps(world["tracks"], {"chr7": {10: 4}}, {}, reps=("pooled",))
    _sampled(world, world["out"], seed=1, copies=2)
    prefix = track_prefix(world["out"], "K562", "chx", 2)
    values = np.array(list(read_bigwig_counts(f"{prefix}.plus.bw")["chr7"].values()))
    assert len(values) and np.all(values % 2 == 0)


def test_manifest_written(world):
    _write_reps(world["tracks"], {"chr7": {10: 2}}, {}, reps=(1, 2, "pooled"))
    _sampled(world, world["out"], seed=3)
    manifest = json.loads((world["out"] / "K562" / "predict_tracks.json").read_text())
    assert manifest["windows"] == 3 and manifest["chroms"] == ["chr7"]


def test_test_split_refuses_without_final_flag(monkeypatch, capsys):
    argv = ["predict_tracks", "--checkpoint", "x/best", "--arm", "condition:K562"]
    monkeypatch.setattr(sys, "argv", [*argv, "--split", "test"])
    with pytest.raises(SystemExit):
        predict_tracks.main()
    assert "--i-am-final" in capsys.readouterr().err


def _flat_model(value):
    def predict(batch):
        return np.full_like(batch["targets"], value)

    return predict


def _run(world, out, predict_fn=None, **kwargs):
    return predict_condition(
        predict_fn,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=out,
        **kwargs,
    )


def test_match_depth_scales_to_real_pooled_total_and_records_factor(world):
    _write_reps(world["tracks"], {"chr7": {10: 800}}, {}, reps=(1, 2, "pooled"))
    # real pooled total in the region: 1600 per library; flat 8 over 200 bases x 2 strands = 3200
    off = _run(world, world["out"].with_name("off"), _flat_model(8.0), mode="expected")
    on = _run(world, world["out"], _flat_model(8.0), mode="expected", match_depth=True)
    assert off["match_depth"] is False and off["depth_factors"] is None
    assert on["match_depth"] is True and on["source"] == "model"
    assert on["depth_factors"] == pytest.approx({"tis": 0.5, "chx": 0.5})
    assert on["unscaled_expected_totals"]["tis"] == pytest.approx(3200.0)
    assert on["expected_totals"]["tis"] == pytest.approx(1600.0)
    for name, per_position in (("off", 4.0), ("out", 2.0)):
        out = world["out"] if name == "out" else world["out"].with_name(name)
        prefix = track_prefix(out, "K562", "tis", 1)
        values = read_bigwig_counts(f"{prefix}.plus.bw")["chr7"].values()
        assert set(values) == {per_position}
    saved = json.loads((world["out"] / "K562" / "predict_tracks.json").read_text())
    assert saved["depth_factors"]["chx"] == pytest.approx(0.5)


def test_match_depth_sampled_total_near_real_total(world):
    _write_reps(world["tracks"], {"chr7": {10: 500}}, {}, reps=(1, 2, "pooled"))
    manifest = _run(world, world["out"], _flat_model(6.0), mode="sample", seed=2, match_depth=True)
    for library in ("tis", "chx"):
        assert abs(sum(manifest["written_totals"][library]) - 1000) < 150


def test_real_resample_expectation_is_real_pooled_split_by_fraction(world):
    # equal replicate totals (8 each); minus file 201 is canonical 200, outside the region
    _write_reps(world["tracks"], {"chr7": {10: 6}}, {"chr7": {200: 2, 201: 5}}, reps=(1,))
    _write_reps(world["tracks"], {"chr7": {10: 2}}, {"chr7": {200: 6, 201: 5}}, reps=(2,))
    manifest = _run(world, world["out"], None, mode="expected", source="real-resample")
    assert manifest["source"] == "real-resample"
    assert manifest["expected_totals"]["tis"] == pytest.approx(16.0)
    for library in ("tis", "chx"):
        for rep in (1, 2):
            prefix = track_prefix(world["out"], "K562", library, rep)
            assert dict(read_bigwig_counts(f"{prefix}.plus.bw")["chr7"]) == {10: 4.0}
            # file coordinates preserved (not shifted twice); 201 stays out of the region
            assert dict(read_bigwig_counts(f"{prefix}.minus.bw")["chr7"]) == {200: 4.0}


def test_real_resample_minus_strand_roundtrips_through_psites(world):
    pytest.importorskip("pysam")
    from tisiago.seq2func.synthetic_bam import tracks_to_bam

    _write_reps(world["tracks"], {}, {"chr7": {100: 4}}, reps=(1, 2))
    _run(world, world["out"], None, mode="expected", source="real-resample")
    prefix = track_prefix(world["out"], "K562", "tis", 1)
    bam = world["out"] / "tis_rep1.bam"
    tracks_to_bam(f"{prefix}.plus.bw", f"{prefix}.minus.bw", bam)
    assert _psites(bam) == Counter({("chr7", "-", 100): 4})


def test_real_resample_sampling_is_seeded(world):
    _write_reps(world["tracks"], {"chr7": {10: 300, 50: 300}}, {}, reps=(1, 2))

    def read(name, seed, rep=1):
        out = world["out"].with_name(name)
        _run(world, out, None, mode="sample", seed=seed, source="real-resample")
        prefix = track_prefix(out, "K562", "tis", rep)
        return read_bigwig_counts(f"{prefix}.plus.bw")["chr7"]

    a, b, c = read("a", 5), read("b", 5), read("c", 6)
    assert a == b and a != c
    assert set(a) == {10, 50}
    assert 400 < sum(a.values()) < 800  # Poisson around 600 per replicate


def test_real_resample_needs_no_model_or_genome(world):
    _write_reps(world["tracks"], {"chr7": {10: 2}}, {}, reps=(1, 2))
    manifest = predict_condition(
        None,
        WINDOWS,
        tracks_dir=world["tracks"],
        condition="K562",
        out_dir=world["out"],
        mode="expected",
        source="real-resample",
    )
    assert manifest["windows"] == 3


def test_unknown_source_rejected(world):
    with pytest.raises(ValueError, match="source"):
        _run(world, world["out"], None, source="oracle")


def test_real_resample_cli_runs_without_checkpoint(world, monkeypatch, tmp_path):
    _write_reps(world["tracks"], {"chr7": {10: 4}}, {}, reps=(1, 2))
    windows_dir = tmp_path / "win"
    windows_dir.mkdir()
    WINDOWS.to_parquet(windows_dir / "val.parquet")
    argv = [
        "predict_tracks",
        "--source",
        "real-resample",
        "--condition",
        "K562",
        "--tracks-dir",
        str(world["tracks"]),
        "--windows-dir",
        str(windows_dir),
        "--out-dir",
        str(world["out"]),
        "--mode",
        "expected",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    predict_tracks.main()
    saved = json.loads((world["out"] / "K562" / "predict_tracks.json").read_text())
    assert saved["source"] == "real-resample" and saved["checkpoint"] is None
    assert saved["tracks_dir"] == str(world["tracks"])


def test_float_mode_writes_pooled_expectation_unrounded(world):
    _write_reps(world["tracks"], {"chr7": {10: 1}}, {}, reps=(1, 2, "pooled"))
    layout = channel_layout(["K562"])
    plus_tis, minus_chx = layout.index(("K562", "tis", "+")), layout.index(("K562", "chx", "-"))

    def ramp(batch):
        pred = np.zeros_like(batch["targets"], dtype=np.float64)
        pred[0, :, plus_tis] = 0.25 + np.arange(pred.shape[1]) / 1000.0  # never an integer
        pred[0, :, minus_chx] = 0.125
        return pred

    manifest = _run(world, world["out"], ramp, mode="float")
    assert manifest["mode"] == "float" and manifest["layout"] == "pooled"
    tis = predict_tracks.pooled_prefix(world["out"], "K562", "tis")
    chx = predict_tracks.pooled_prefix(world["out"], "K562", "chx")
    assert tis == world["out"] / "K562" / "tis_pooled"
    got_plus = read_bigwig_counts(f"{tis}.plus.bw")["chr7"]
    got_minus = read_bigwig_counts(f"{chx}.minus.bw")["chr7"]
    assert sorted(got_plus) == list(range(200))  # every region position written once
    assert not any(float(v).is_integer() for v in got_plus.values())  # no rounding, no sampling
    assert set(got_minus.values()) == {0.125}
    assert sorted(got_minus) == [p + RIBOTISH_MINUS_SHIFT for p in range(200)]
    assert manifest["written_totals"]["chx"] == [pytest.approx(0.125 * 200)]
    assert not (world["out"] / "K562" / "tis_rep1.plus.bw").exists()
    saved = json.loads((world["out"] / "K562" / "predict_tracks.json").read_text())
    assert saved["mode"] == "float" and saved["layout"] == "pooled"
    assert saved["files"]["tis"] == {"+": "tis_pooled.plus.bw", "-": "tis_pooled.minus.bw"}


def test_float_mode_matches_oracle_pooled_values(world):
    plus = {"chr7": {10: 3, 77: 1}}
    minus = {"chr7": {31: 2, 201: 5}}
    _write_reps(world["tracks"], plus, minus)
    _write_reps(
        world["tracks"],
        {k: {p: 2 * v for p, v in d.items()} for k, d in plus.items()},
        {k: {p: 2 * v for p, v in d.items()} for k, d in minus.items()},
        reps=("pooled",),
    )
    _run(world, world["out"], _oracle, mode="float")
    prefix = predict_tracks.pooled_prefix(world["out"], "K562", "tis")
    got_plus = {p: v for p, v in read_bigwig_counts(f"{prefix}.plus.bw")["chr7"].items() if v}
    got_minus = {p: v for p, v in read_bigwig_counts(f"{prefix}.minus.bw")["chr7"].items() if v}
    # oracle targets are the pooled real counts (rep1 + rep2) in canonical coordinates
    assert got_plus == {10: 6.0, 77: 2.0}
    assert got_minus == {31: 4.0}  # file 201 is canonical 200, outside the region


def test_float_mode_cli_default_out_dir(monkeypatch, tmp_path):
    seen = {}

    def fake_predict(predict_fn, windows, **kwargs):
        seen.update(kwargs)
        return {}

    windows_dir = tmp_path / "win"
    windows_dir.mkdir()
    WINDOWS.to_parquet(windows_dir / "val.parquet")
    monkeypatch.setattr(predict_tracks, "predict_condition", fake_predict)
    argv = [
        "predict_tracks",
        "--source",
        "real-resample",
        "--condition",
        "K562",
        "--tracks-dir",
        str(tmp_path / "tracks"),
        "--windows-dir",
        str(windows_dir),
        "--checkpoint",
        str(tmp_path / "run" / "best"),
        "--mode",
        "float",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    predict_tracks.main()
    assert seen["mode"] == "float"
    assert seen["out_dir"] == tmp_path / "run" / "predicted_val_float"


# --- rnacond: TIS/CHX tracks of any condition from the condition's RNA ---


def test_rnacond_float_mode_matches_oracle_and_feeds_rna(world):
    plus = {"chr7": {10: 3, 77: 1}}
    minus = {"chr7": {31: 2}}
    _write_reps(world["tracks"], plus, minus)
    _write_reps(world["tracks"], {"chr7": {10: 6, 77: 2}}, {"chr7": {31: 4}}, reps=("pooled",))
    seen = []

    def oracle(batch):
        seen.append(batch["sequences"][0])
        return batch["targets"]

    manifest = predict_condition(
        oracle,
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks"],
        genes=_genes(),
        condition="K562",
        conditions=("K562",),
        out_dir=world["out"],
        mode="float",
        rna_conditioned=True,
    )
    widths = [x.shape[-1] for x in seen]
    # the pooled RNA in CPM: 6 + 2 (+) and 4 (-) reads in the window, plus 2 x 2 end reads
    scale = 1e6 / 16
    assert seen[0][10, 4] == pytest.approx(6 * scale) and seen[0][31, 5] == pytest.approx(4 * scale)
    assert widths == [6, 6, 6]
    assert manifest["rna_conditioned"] is True
    assert manifest["rna_cpm_scale"] == pytest.approx(scale)
    for library in ("tis", "chx"):
        prefix = predict_tracks.pooled_prefix(world["out"], "K562", library)
        got_plus = {p: v for p, v in read_bigwig_counts(f"{prefix}.plus.bw")["chr7"].items() if v}
        got_minus = {p: v for p, v in read_bigwig_counts(f"{prefix}.minus.bw")["chr7"].items() if v}
        assert got_plus == {10: 6.0, 77: 2.0}
        assert got_minus == {31: 4.0}
    assert not (world["out"] / "K562" / "rna_pooled.plus.bw").exists()


def test_rnacond_channels_map_to_tis_and_chx(world):
    _write_reps(world["tracks"], {"chr7": {10: 1}}, {}, reps=(1, 2, "pooled"))

    def by_channel(batch):
        pred = np.zeros((1, batch["targets"].shape[1], 4))
        pred[0, :, 0], pred[0, :, 3] = 0.5, 0.25  # TIS+ and CHX-
        return pred

    _run(world, world["out"], by_channel, mode="float", rna_conditioned=True)
    tis = predict_tracks.pooled_prefix(world["out"], "K562", "tis")
    chx = predict_tracks.pooled_prefix(world["out"], "K562", "chx")
    assert set(read_bigwig_counts(f"{tis}.plus.bw")["chr7"].values()) == {0.5}
    assert set(read_bigwig_counts(f"{chx}.minus.bw")["chr7"].values()) == {0.25}
    assert not any(read_bigwig_counts(f"{tis}.minus.bw")["chr7"].values())


def test_rnacond_cli_any_condition(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_predict(predict_fn, windows, **kwargs):
        seen.update(kwargs)
        return {}

    def fake_load(checkpoint, arm, *, base_checkpoint, head_kind="mlp", device=None):
        seen["loaded"] = (arm, head_kind)
        return "predict"

    from tisiago.seq2func import evaluate

    windows_dir = tmp_path / "win"
    windows_dir.mkdir()
    WINDOWS.to_parquet(windows_dir / "val.parquet")
    expression = tmp_path / "expression.parquet"
    pd.concat([_genes(), _genes().assign(condition="HeLa")]).to_parquet(expression)
    monkeypatch.setattr(predict_tracks, "predict_condition", fake_predict)
    monkeypatch.setattr(evaluate, "load_predict_fn", fake_load)
    base = [
        "predict_tracks",
        "--arm",
        "rnacond:K562",
        "--checkpoint",
        str(tmp_path / "run" / "best"),
        "--tracks-dir",
        str(tmp_path / "tracks"),
        "--windows-dir",
        str(windows_dir),
        "--expression",
        str(expression),
        "--genome",
        "g.fa",
        "--base-checkpoint",
        "w",
        "--head",
        "rnacond",
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--condition", "K562"])
    predict_tracks.main()
    assert seen["loaded"] == ("rnacond:K562", "rnacond")
    assert seen["condition"] == "K562" and seen["conditions"] == ("K562",)
    assert seen["rna_conditioned"] is True and set(seen["genes"].condition) == {"K562"}
    assert seen["meta"]["heldout"] is True
    assert seen["out_dir"] == tmp_path / "run" / "predicted_val"
    monkeypatch.setattr(sys, "argv", [*base, "--condition", "HeLa"])
    predict_tracks.main()
    assert seen["condition"] == "HeLa" and seen["meta"]["heldout"] is False
    for bad in ([], ["--condition", "Nope"]):
        monkeypatch.setattr(sys, "argv", [*base, *bad])
        with pytest.raises(SystemExit):
            predict_tracks.main()
    assert "--condition" in capsys.readouterr().err
