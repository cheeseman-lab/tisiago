"""Tests for held-out evaluation and the replicate ceiling."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

from tisiago.seq2func import evaluate
from tisiago.seq2func.evaluate import (
    EXCLUDED,
    OUTSIDE,
    assign_spans,
    ceiling_metrics,
    evaluate_checkpoint,
    smooth_profile,
    uniform_metrics,
)
from tisiago.seq2func.masking import expression_mask
from tisiago.seq2func.metrics import profile_kl
from tisiago.seq2func.qc import _frame_counts
from tisiago.seq2func.targets import channel_layout
from tisiago.seq2func.tracks import track_prefix, write_bigwigs

SIZE = 200
SIZES = [("chr7", SIZE)]
WINDOWS = pd.DataFrame({"chrom": ["chr7"] * 3, "start": [0, 50, 100], "end": [100, 150, 200]})


def _genes(condition="HeLa"):
    # G1 inside window 0 only; G2 inside windows 0 and 1 (nearer window 1's centre 100);
    # G3 inside windows 1 and 2; G4 straddles every window boundary it touches -> excluded;
    # G5 is unexpressed; G6 on the minus strand inside window 2.
    return pd.DataFrame(
        {
            "condition": condition,
            "gene_id": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            "chrom": "chr7",
            "start": [5, 60, 110, 40, 10, 160, 120],
            "end": [30, 95, 140, 160, 20, 190, 145],
            "strand": ["+", "+", "+", "+", "+", "-", "+"],
            "expressed": [True, True, True, True, False, True, True],
        }
    )


def _counts(seed):
    rng = np.random.default_rng(seed)
    plus = Counter({int(p): int(rng.integers(1, 20)) for p in rng.choice(SIZE, 80, replace=False)})
    minus = Counter({int(p): int(rng.integers(1, 20)) for p in rng.choice(SIZE, 80, replace=False)})
    return {("chr7", "+"): plus, ("chr7", "-"): minus}


def _write(tracks, condition, rep, seed):
    for k, library in enumerate(("tis", "chx", "rna")):
        write_bigwigs(_counts(seed + k), SIZES, track_prefix(tracks, condition, library, rep))


def test_assign_spans_nearest_centre_and_excluded():
    where = assign_spans(_genes(), WINDOWS)
    assert where.tolist()[:4] == [0, 1, 1, EXCLUDED]
    # G3 [110, 140) centre 125 is inside windows 1 (centre 100) and 2 (centre 150): tie -> first
    assert where.tolist()[2] == 1
    assert where.tolist()[5] == 2
    far = pd.DataFrame({"chrom": ["chr7", "chr8"], "start": [0, 0], "end": [10, 10]})
    assert assign_spans(far, WINDOWS.iloc[2:]).tolist() == [OUTSIDE, OUTSIDE]


def test_ceiling_identical_replicates(tmp_path):
    tracks = tmp_path / "tracks"
    _write(tracks, "HeLa", 1, seed=0)
    _write(tracks, "HeLa", 2, seed=0)
    table = ceiling_metrics(tracks, ["HeLa"], _genes(), WINDOWS)
    assert len(table) == len(channel_layout(["HeLa"]))
    # profile prediction = smoothed rep2 (+ pseudocount): KL against rep1 is the per-window mean
    # of profile_kl(smooth(rep), rep, mask), finite and > 0 even for identical replicates
    layout = channel_layout(["HeLa"])
    reader = evaluate.TrackReader(tracks, ["HeLa"], rep=1)
    expected = np.mean(
        [
            profile_kl(
                smooth_profile(track := reader.read("chr7", w.start, w.end)),
                track,
                expression_mask(layout, _genes(), "chr7", w.start, w.end),
            )
            for w in WINDOWS.itertuples()
        ],
        axis=0,
    )
    np.testing.assert_allclose(table.profile_kl, expected, rtol=1e-12)
    assert (table.profile_kl > 0).all()
    assert (table.ceiling_smooth_bp == 5).all() and (table.ceiling_pseudocount == 1e-3).all()
    assert (table.ceiling_version == evaluate.REFERENCE_VERSION).all()
    plus = table[table.strand == "+"]
    assert np.allclose(plus.log_pearson, 1.0)
    # G1, G2, G3, G7 are + expressed genes fully inside a window; G4 is excluded and counted
    assert plus.n_genes.tolist() == [4, 4, 4]
    assert plus.genes_excluded.tolist() == [1, 1, 1]
    minus = table[table.strand == "-"]
    assert minus.n_genes.tolist() == [1, 1, 1]
    assert minus.log_pearson.isna().all()  # < 3 genes -> NaN, not a crash
    assert (table.kl_windows == 3).all()


def test_ceiling_differs_for_different_replicates(tmp_path):
    tracks = tmp_path / "tracks"
    _write(tracks, "HeLa", 1, seed=0)
    _write(tracks, "HeLa", 2, seed=10)
    table = ceiling_metrics(tracks, ["HeLa"], _genes(), WINDOWS)
    assert (table.profile_kl > 0).all()
    assert (table.log_pearson[table.strand == "+"] < 1.0).all()


def test_gene_totals_are_not_double_counted(tmp_path):
    # rep1: one count per base inside G2; rep2 doubles it. A gene counted in two windows would
    # double its totals in both replicates; check the totals directly.
    tracks = tmp_path / "tracks"
    flat = Counter({p: 1 for p in range(SIZE)})
    for rep in (1, 2):
        for library in ("tis", "chx", "rna"):
            write_bigwigs(
                {("chr7", "+"): flat, ("chr7", "-"): flat},
                SIZES,
                track_prefix(tracks, "HeLa", library, rep),
            )
    acc = evaluate.ChannelMetrics(channel_layout(["HeLa"]), _genes(), WINDOWS)
    reader = evaluate.TrackReader(tracks, ["HeLa"], rep=1)
    for i, row in enumerate(WINDOWS.itertuples()):
        track = reader.read(row.chrom, row.start, row.end)
        acc.add(i, track, track, np.ones_like(track, dtype=bool))
    lengths = (acc.genes.end - acc.genes.start).to_numpy()
    inside = acc.where >= 0
    assert np.allclose(acc.obs_totals[inside, 0], lengths[inside])


def test_frame_counts_shift_zero_for_canonical_psites():
    cds = [("chr1", "-", [(10, 16)])]
    canonical = {"chr1": Counter({15: 5, 12: 3, 14: 2})}
    assert _frame_counts(canonical, cds, shift=0) == (8, 10)
    shifted = {"chr1": Counter({16: 5, 13: 3, 15: 2})}
    assert _frame_counts(shifted, cds) == (8, 10)


@pytest.fixture
def world(tmp_path):
    seq = ("ATGAAACCC" * 30)[:SIZE]
    fasta = tmp_path / "g.fa"
    fasta.write_text(f">chr7\n{seq}\n")
    Fasta(str(fasta)).close()
    tracks = tmp_path / "tracks"
    _write(tracks, "HeLa", "pooled", seed=0)
    # CHX pooled: + CDS [60, 90) fully in frame; - CDS [160, 190) codon first bases 189, 186,
    # ... plus 4 off-frame at 170. Minus bigWigs hold Ribo-TISH positions (canonical + 1).
    minus = Counter({p + 1: 2 for p in range(189, 159, -3)})
    minus[171] = 4
    chx = {("chr7", "+"): Counter({p: 3 for p in range(60, 90, 3)}), ("chr7", "-"): minus}
    write_bigwigs(chx, SIZES, track_prefix(tracks, "HeLa", "chx", "pooled"))
    cds = [("chr7", "+", [(60, 90)]), ("chr7", "-", [(160, 190)]), ("chr7", "+", [(40, 160)])]
    cds_genes = ["G2", "G6", "G4"]
    return dict(fasta=str(fasta), tracks_dir=tracks, genes=_genes(), cds=cds, cds_genes=cds_genes)


def test_evaluate_checkpoint_perfect_fake_model(world):
    layout = channel_layout(["HeLa"])

    def oracle(batch):
        return batch["targets"]

    table = evaluate_checkpoint(
        None,
        "condition:HeLa",
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks_dir"],
        genes=world["genes"],
        cds=world["cds"],
        cds_genes=world["cds_genes"],
        predict_fn=oracle,
    )
    assert len(table) == len(layout)
    assert np.allclose(table.profile_kl, 0.0, atol=1e-9)
    assert np.allclose(table.log_pearson[table.strand == "+"], 1.0)
    chx = table[table.library == "chx"].set_index("strand")
    assert chx.loc["+", "frame0_obs"] == pytest.approx(1.0)
    assert chx.loc["+", "frame0_pred"] == pytest.approx(1.0)
    assert chx.loc["-", "frame0_obs"] == pytest.approx(20 / 24)
    # the + CDS spanning [40, 160) is never fully inside one window -> not counted
    assert chx.loc["+", "frame0_cds"] == 1
    assert chx.loc["-", "frame0_cds"] == 1
    assert table[table.library != "chx"].frame0_pred.isna().all()


def test_evaluate_checkpoint_flat_model_is_worse(world):
    def flat(batch):
        return np.ones_like(batch["targets"])

    table = evaluate_checkpoint(
        None,
        "condition:HeLa",
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks_dir"],
        genes=world["genes"],
        cds=world["cds"],
        cds_genes=world["cds_genes"],
        predict_fn=flat,
    )
    assert (table.profile_kl > 0.1).all()
    chx = table[table.library == "chx"].set_index("strand")
    assert chx.loc["+", "frame0_pred"] == pytest.approx(1 / 3)


def test_test_split_refuses_without_final_flag(tmp_path, monkeypatch, capsys):
    argv = ["evaluate", "--arm", "shared", "--checkpoint", str(tmp_path), "--split", "test"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-dir", str(tmp_path / "eval")])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code != 0
    assert "--i-am-final" in capsys.readouterr().err
    assert not (tmp_path / "eval").exists()


def test_cli_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    out.mkdir()
    (out / "condition_hela_val.tsv").write_text("x\n")
    argv = ["evaluate", "--arm", "condition:HeLa", "--checkpoint", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", [*argv, "--split", "val", "--out-dir", str(out)])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code != 0
    assert (out / "condition_hela_val.tsv").read_text() == "x\n"


def test_ceiling_only_refuses_existing_ceiling(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    out.mkdir()
    (out / "ceiling_val.tsv").write_text("x\n")
    (out / "uniform_val.tsv").write_text("x\n")
    argv = ["evaluate", "--ceiling-only", "--split", "val", "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code != 0
    assert (out / "ceiling_val.tsv").read_text() == "x\n"


def test_output_paths():
    out = evaluate.output_paths("data/p6/eval", "condition:RPE1_Que", "val")
    assert str(out["model"]) == "data/p6/eval/condition_rpe1_que_val.tsv"
    assert str(out["ceiling"]) == "data/p6/eval/ceiling_val.tsv"


def test_mean_kl_skips_nan_windows():
    assert math.isnan(evaluate.channel_kl(np.full((2, 1), np.nan))[0][0])


def test_load_predict_fn_rebuilds_model_and_restores_checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("alphagenome_ft")
    model_mod = pytest.importorskip("tisiago.seq2func.model")
    calls = {}

    class FakeModel:
        _params, _state = "params", "state"

    def fake_load_finetuned(
        checkpoint, base, head, num_tracks, kind, device=None, mod_channels=None
    ):
        calls["load"] = (checkpoint, base, head, num_tracks, kind)
        calls["device"] = device
        calls["mod_channels"] = mod_channels
        return FakeModel()

    def fake_make_predict(model, head):
        calls["head"] = head
        return lambda params, state, seq: np.zeros((1, seq.shape[1], 6)) + (params == "params")

    monkeypatch.setattr(model_mod, "load_finetuned", fake_load_finetuned)
    monkeypatch.setattr(model_mod, "make_predict", fake_make_predict)
    predict = evaluate.load_predict_fn(
        tmp_path, "condition:HeLa", base_checkpoint="weights", head_kind="conv", device="cpu0"
    )
    assert calls["load"] == (str(tmp_path), "weights", "p6_hela", 6, "conv")
    assert calls["mod_channels"] is None
    assert calls["device"] == "cpu0"
    assert calls["head"] == "p6_hela"
    out = predict({"sequences": np.zeros((1, 8, 4), np.float32)})
    assert out.shape == (1, 8, 6) and (out == 1).all()


def test_load_predict_fn_passes_chx_channels_for_resid_chx(monkeypatch, tmp_path):
    pytest.importorskip("alphagenome_ft")
    model_mod = pytest.importorskip("tisiago.seq2func.model")
    from tisiago.seq2func.targets import channel_layout, library_channels

    calls = {}

    class FakeModel:
        _params, _state = "params", "state"

    def fake_load_finetuned(checkpoint, base, head, num_tracks, kind, **kwargs):
        calls.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(model_mod, "load_finetuned", fake_load_finetuned)
    monkeypatch.setattr(model_mod, "make_predict", lambda model, head: None)
    evaluate.load_predict_fn(
        tmp_path / "best", "condition:K562", base_checkpoint="w", head_kind="resid_chx"
    )
    want = library_channels(channel_layout(["K562"]), "chx")
    assert calls["mod_channels"] == want and len(want) == 2
    (tmp_path / "run.json").write_text(json.dumps({"mod_channels": [0, 1]}))
    with pytest.raises(ValueError, match="mod_channels"):
        evaluate.load_predict_fn(
            tmp_path / "best", "condition:K562", base_checkpoint="w", head_kind="resid_chx"
        )
    (tmp_path / "run.json").write_text(json.dumps({"mod_channels": list(want)}))
    evaluate.load_predict_fn(
        tmp_path / "best", "condition:K562", base_checkpoint="w", head_kind="resid_chx"
    )


def test_head_kind_read_from_run_json(tmp_path):
    (tmp_path / "best").mkdir()
    assert evaluate.head_kind_of(tmp_path / "best") == "mlp"
    (tmp_path / "run.json").write_text('{"args": {"head": "conv"}}')
    assert evaluate.head_kind_of(tmp_path / "best") == "conv"


def test_frame0_only_counts_cds_of_expressed_genes(world):
    # Same CDS, but the + CDS now belongs to the unexpressed G5: it must drop out of both
    # predicted and observed frame-0 for the + channel.
    table = evaluate_checkpoint(
        None,
        "condition:HeLa",
        WINDOWS,
        fasta=world["fasta"],
        tracks_dir=world["tracks_dir"],
        genes=world["genes"],
        cds=world["cds"],
        cds_genes=["G5", "G6", "G4"],
        predict_fn=lambda batch: batch["targets"],
    )
    chx = table[table.library == "chx"].set_index("strand")
    assert chx.loc["+", "frame0_cds"] == 0
    assert np.isnan(chx.loc["+", "frame0_obs"]) and np.isnan(chx.loc["+", "frame0_pred"])
    assert chx.loc["-", "frame0_cds"] == 1


def test_channel_metrics_rejects_cds_without_genes():
    with pytest.raises(ValueError, match="cds_genes"):
        evaluate.ChannelMetrics(channel_layout(["HeLa"]), _genes(), WINDOWS, [("chr7", "+", [])])


def test_load_mane_cds_returns_gene_ids(tmp_path):
    gtf = tmp_path / "g.gtf"
    tag = 'tag "MANE_Select";'
    gtf.write_text(
        f'chr7\tH\tCDS\t11\t20\t.\t+\t0\tgene_id "G1.2"; transcript_id "T1.1"; {tag}\n'
        f'chr7\tH\tCDS\t31\t40\t.\t+\t0\tgene_id "G1.2"; transcript_id "T1.1"; {tag}\n'
        f'chr7\tH\texon\t1\t50\t.\t+\t.\tgene_id "G1.2"; transcript_id "T1.1"; {tag}\n'
        'chr7\tH\tCDS\t5\t9\t.\t-\t0\tgene_id "G2.1"; transcript_id "T2.1";\n'
    )
    cds, genes = evaluate.load_mane_cds(gtf)
    assert cds == [("chr7", "+", [(10, 20), (30, 40)])]
    assert genes == ["G1.2"]


@pytest.fixture
def cli_world(tmp_path, monkeypatch):
    from tisiago.seq2func.samples import CONDITIONS

    tracks = tmp_path / "tracks"
    for k, condition in enumerate(CONDITIONS):
        _write(tracks, condition, 1, seed=k)
        _write(tracks, condition, 2, seed=k + 50)
        _write(tracks, condition, "pooled", seed=k + 100)
    gtf = tmp_path / "g.gtf"
    gtf.write_text(
        'chr7\tH\tCDS\t61\t90\t.\t+\t0\tgene_id "G2"; transcript_id "T2"; tag "MANE_Select";\n'
    )
    (tmp_path / "windows").mkdir()
    WINDOWS.to_parquet(tmp_path / "windows" / "val.parquet")
    expression = tmp_path / "expression.parquet"
    pd.concat([_genes(c) for c in CONDITIONS], ignore_index=True).to_parquet(expression)
    base = [
        "evaluate",
        "--split",
        "val",
        "--windows-dir",
        str(tmp_path / "windows"),
        "--expression",
        str(expression),
        "--out-dir",
        str(tmp_path / "eval"),
        "--gtf",
        str(gtf),
    ]

    def run(*extra):
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        evaluate.main()

    return dict(run=run, out=tmp_path / "eval", tracks=tracks, root=tmp_path)


def test_max_windows_ceiling_never_takes_the_shared_name(cli_world):
    cli_world["run"](
        "--ceiling-only", "--max-windows", "2", "--tracks-dir", str(cli_world["tracks"])
    )
    out = cli_world["out"]
    assert sorted(p.name for p in out.iterdir()) == ["ceiling_val_max2.tsv", "uniform_val_max2.tsv"]
    for kind in ("ceiling", "uniform"):
        table = pd.read_csv(out / f"{kind}_val_max2.tsv", sep="\t")
        assert set(table.windows) == {2}
        assert set(table.tracks_dir) == {str(cli_world["tracks"])}
        assert set(table.source) == {kind}
        assert len(table) == 36
        chx = table[(table.library == "chx") & (table.strand == "+")]
        assert (chx.frame0_cds == 1).all() and chx.frame0_obs.notna().all()
        assert chx.frame0_pred.notna().all()
    assert pd.read_csv(out / "uniform_val_max2.tsv", sep="\t").log_pearson.isna().all()


def test_reused_ceiling_must_match_window_count(cli_world, capsys):
    out = cli_world["out"]
    cli_world["run"]("--ceiling-only", "--tracks-dir", str(cli_world["tracks"]))
    full = pd.read_csv(out / "ceiling_val.tsv", sep="\t")
    assert set(full.windows) == {3}
    full.assign(windows=2).to_csv(out / "ceiling_val.tsv", sep="\t", index=False)
    model = ["--arm", "condition:HeLa", "--checkpoint", str(cli_world["root"] / "best")]
    fakes = ["--genome", "g.fa", "--gtf", "g.gtf", "--base-checkpoint", "w"]
    with pytest.raises(SystemExit) as exc:
        cli_world["run"](*model, *fakes, "--tracks-dir", str(cli_world["tracks"]))
    assert exc.value.code != 0
    assert "computed on [2] windows, not 3" in capsys.readouterr().err
    assert not (out / "condition_hela_val.tsv").exists()
    # --overwrite recomputes it
    cli_world["run"]("--ceiling-only", "--overwrite", "--tracks-dir", str(cli_world["tracks"]))
    assert set(pd.read_csv(out / "ceiling_val.tsv", sep="\t").windows) == {3}


def test_reused_ceiling_must_match_tracks_dir(cli_world, capsys):
    cli_world["run"]("--ceiling-only", "--tracks-dir", str(cli_world["tracks"]))
    model = ["--arm", "condition:HeLa", "--checkpoint", str(cli_world["root"] / "best")]
    fakes = ["--genome", "g.fa", "--gtf", "g.gtf", "--base-checkpoint", "w"]
    with pytest.raises(SystemExit):
        cli_world["run"](*model, *fakes, "--tracks-dir", str(cli_world["root"] / "other"))
    assert "computed from tracks" in capsys.readouterr().err


def test_tracks_dir_defaults_to_run_json(tmp_path):
    (tmp_path / "best").mkdir()
    assert evaluate.tracks_dir_of(tmp_path / "best") == Path("data/p6/tracks")
    assert evaluate.tracks_dir_of(None) == Path("data/p6/tracks")
    (tmp_path / "run.json").write_text('{"args": {"tracks_dir": "/ssd/p6_tracks"}}')
    assert evaluate.tracks_dir_of(tmp_path / "best") == Path("/ssd/p6_tracks")


def test_write_tsv_temp_name_is_per_process(tmp_path, monkeypatch):
    seen = []
    real = evaluate.os.replace
    monkeypatch.setattr(evaluate.os, "replace", lambda a, b: (seen.append(str(a)), real(a, b)))
    evaluate.write_tsv(pd.DataFrame({"a": [1]}), tmp_path / "x.tsv")
    assert seen and f".{evaluate.os.getpid()}." in seen[0]
    assert (tmp_path / "x.tsv").read_text() == "a\n1\n"


def test_smooth_profile_centred_boxcar_plus_pseudocount():
    track = np.zeros((8, 2))
    track[3, 0] = 5.0
    track[0, 1] = 10.0
    out = smooth_profile(track)
    np.testing.assert_allclose(out[:, 0], [1e-3, 1.001, 1.001, 1.001, 1.001, 1.001, 1e-3, 1e-3])
    # edges: mode="same" zero-pads, so a spike at 0 spreads to positions 0..2 only
    np.testing.assert_allclose(out[:, 1], [2.001, 2.001, 2.001, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])


def test_uniform_kl_matches_closed_form(tmp_path):
    # KL(p || uniform over n unmasked bases) = sum p log p + log n
    tracks = tmp_path / "tracks"
    counts = {("chr7", "+"): Counter({10: 1, 20: 3}), ("chr7", "-"): Counter({31: 2, 41: 2})}
    for library in ("tis", "chx", "rna"):
        write_bigwigs(counts, SIZES, track_prefix(tracks, "HeLa", library, "pooled"))
    genes = _genes().iloc[[0]].assign(start=0, end=100)  # one expressed + gene: no masking
    window = WINDOWS.iloc[[0]]
    table = uniform_metrics(tracks, ["HeLa"], genes, window)
    p_plus, p_minus = np.array([0.25, 0.75]), np.array([0.5, 0.5])
    plus = (p_plus * np.log(p_plus)).sum() + np.log(100)
    minus = (p_minus * np.log(p_minus)).sum() + np.log(100)
    np.testing.assert_allclose(table.profile_kl, [plus, minus] * 3, rtol=1e-6)
    assert table.log_pearson.isna().all()
    assert (table.ceiling_version == evaluate.REFERENCE_VERSION).all()


@pytest.mark.parametrize("version", [None, 1])
def test_old_or_unversioned_ceiling_is_refused(cli_world, capsys, version):
    out = cli_world["out"]
    tracks = ["--tracks-dir", str(cli_world["tracks"])]
    cli_world["run"]("--ceiling-only", *tracks)
    table = pd.read_csv(out / "ceiling_val.tsv", sep="\t")
    if version is None:
        table = table.drop(columns="ceiling_version")
    else:
        table = table.assign(ceiling_version=version)
    table.to_csv(out / "ceiling_val.tsv", sep="\t", index=False)
    # a model run would reuse it
    model = ["--arm", "condition:HeLa", "--checkpoint", str(cli_world["root"] / "best")]
    fakes = ["--genome", "g.fa", "--base-checkpoint", "w"]
    with pytest.raises(SystemExit) as exc:
        cli_world["run"](*model, *fakes, *tracks)
    assert exc.value.code != 0
    assert "ceiling_version" in capsys.readouterr().err
    assert not (out / "condition_hela_val.tsv").exists()
    # a ceiling-only run that only needs to add the uniform reference also refuses the old one
    (out / "uniform_val.tsv").unlink()
    with pytest.raises(SystemExit):
        cli_world["run"]("--ceiling-only", *tracks)
    assert "ceiling_version" in capsys.readouterr().err
    assert not (out / "uniform_val.tsv").exists()
    cli_world["run"]("--ceiling-only", "--overwrite", *tracks)
    fresh = pd.read_csv(out / "ceiling_val.tsv", sep="\t")
    assert set(fresh.ceiling_version) == {evaluate.REFERENCE_VERSION}


def test_ceiling_frame0_pred_uses_raw_rep2_not_smoothed(tmp_path):
    # rep2 CHX + track: 3 reads on each codon first base of CDS [60, 90) plus 1 off-frame read;
    # raw frame-0 = 30 / 31. The 5-bp boxcar would blur it to ~1/3.
    tracks = tmp_path / "tracks"
    periodic = Counter({p: 3 for p in range(60, 90, 3)})
    periodic[61] = 1
    for rep in (1, 2):
        _write(tracks, "HeLa", rep, seed=0)
        write_bigwigs(
            {("chr7", "+"): periodic, ("chr7", "-"): Counter({5: 1})},
            SIZES,
            track_prefix(tracks, "HeLa", "chx", rep),
        )
    table = ceiling_metrics(
        tracks, ["HeLa"], _genes(), WINDOWS, cds=[("chr7", "+", [(60, 90)])], cds_genes=["G2"]
    )
    chx = table[(table.library == "chx") & (table.strand == "+")].iloc[0]
    assert chx.frame0_cds == 1
    assert chx.frame0_pred == pytest.approx(30 / 31)
    assert chx.frame0_obs == pytest.approx(30 / 31)


def test_evaluate_checkpoint_passes_transcript_features(world):
    from tisiago.seq2func.tx_features import Transcript, TranscriptFeatures

    transcripts = TranscriptFeatures([Transcript("T", "G", "chr7", "+", ((10, 60), (80, 120)))])
    widths = []

    def oracle(batch):
        widths.append(batch["sequences"].shape[-1])
        assert batch["sequences"][0, 10:60, 4 + 9].all()
        return batch["targets"]

    evaluate_checkpoint(
        None,
        "condition:HeLa",
        WINDOWS.iloc[:1],
        fasta=world["fasta"],
        tracks_dir=world["tracks_dir"],
        genes=world["genes"],
        predict_fn=oracle,
        transcripts=transcripts,
    )
    assert widths == [24]


def test_load_transcripts_only_for_transcript_heads(tmp_path):
    gtf = tmp_path / "g.gtf"
    gtf.write_text(
        'chr7\tH\texon\t11\t60\t.\t+\t.\tgene_id "G"; transcript_id "T"; tag "MANE_Select";\n'
    )
    assert evaluate.load_transcripts("mlp", None) is None
    assert evaluate.load_transcripts("seqconv", gtf) is None
    with pytest.raises(ValueError, match="gtf"):
        evaluate.load_transcripts("seqconv_tx", None)
    loaded = evaluate.load_transcripts("seqconv_tx", gtf)
    assert [t.transcript_id for t in loaded.overlapping("chr7", 0, 100)] == ["T"]
