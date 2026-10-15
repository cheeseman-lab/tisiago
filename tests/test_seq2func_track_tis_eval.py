"""Tests for scoring candidate start codons by TIS-track signal on the baseline protocol."""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")  # CPU stage: the conda tisiago env
from tisiago.seq2func import track_tis_eval  # noqa: E402
from tisiago.seq2func.predict_tracks import pooled_prefix, predict_condition  # noqa: E402
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT  # noqa: E402
from tisiago.seq2func.track_tis_eval import (  # noqa: E402
    aggregate,
    candidate_centre,
    condition_scores,
    coverage,
    estimate_offset,
    offset_profile,
    run_protocol,
    select_variant,
)
from tisiago.seq2func.tracks import track_prefix, write_bigwigs  # noqa: E402
from tisiago.tiling import a_plus_of  # noqa: E402

SIZES = [("chr7", 400), ("chr8", 400)]


def _write(prefix, plus=None, minus=None):
    """1-bp tracks; ``plus``/``minus`` map chrom -> {file position: value}."""
    counts = {}
    for chrom, positions in (plus or {}).items():
        counts[(chrom, "+")] = Counter(positions)
    for chrom, positions in (minus or {}).items():
        counts[(chrom, "-")] = Counter(positions)
    write_bigwigs(counts, SIZES, prefix)


def _real(tracks, condition, plus=None, minus=None):
    """Two replicates whose sum is ``plus``/``minus`` (rep1 one third, rep2 two thirds)."""
    for rep, share in ((1, 1 / 3), (2, 2 / 3)):
        _write(
            track_prefix(tracks, condition, "tis", rep),
            {c: {p: v * share for p, v in d.items()} for c, d in (plus or {}).items()},
            {c: {p: v * share for p, v in d.items()} for c, d in (minus or {}).items()},
        )


def _rows(chrom, gstart, strand):
    return pd.DataFrame({"chrom": chrom, "gstart": gstart, "strand": strand})


def test_candidate_centre_matches_a_plus_and_ribotish_minus_convention():
    gstart = np.array([100, 201, 57, 58])
    strand = np.array(["+", "-", "+", "-"])
    centre = candidate_centre(gstart, strand, 0)
    a_plus = [a_plus_of(int(g), str(s)) for g, s in zip(gstart, strand, strict=True)]
    assert centre.tolist() == [a_plus[0], a_plus[1] + RIBOTISH_MINUS_SHIFT, 57, 57 + 1]
    # offsets run 3' in transcript orientation: up the plus strand, down the minus strand
    assert candidate_centre(gstart, strand, 2).tolist() == [102, 199, 59, 56]
    assert candidate_centre(gstart, strand, -1).tolist() == [99, 202, 56, 59]


def test_condition_scores_read_each_strand_at_its_own_file_position(tmp_path):
    tracks = tmp_path / "tracks"
    # + codon at 100: peak 4 at the A, 1 at A+1 (in the peak), 3 at A+20 (context only)
    # - codon gstart 201 (A at canonical 200 = file 201): 6 there, 3 at file 202 (in the peak)
    _real(tracks, "K562", {"chr7": {100: 3, 101: 1, 120: 3}}, {"chr7": {201: 6, 202: 3}})
    rows = _rows("chr7", [100, 201, 300], ["+", "-", "+"])
    paths = track_tis_eval.real_paths(tracks)
    scores = condition_scores(rows, paths, "K562", offset=0, eps=1e-3)
    assert scores["peak"] == pytest.approx([4.0, 9.0, 0.0])
    context = np.array([7.0, 9.0, 0.0]) / 61
    assert scores["enrich"] == pytest.approx(scores["peak"] / (context + 1e-3))
    # offset 19 moves the + peak onto 120 (window 119..121) and the - peak away from 201..202
    shifted = condition_scores(rows, paths, "K562", offset=19, eps=1e-3)
    assert shifted["peak"] == pytest.approx([3.0, 0.0, 0.0])


def test_minus_offset_moves_toward_lower_genomic_positions(tmp_path):
    tracks = tmp_path / "tracks"
    # - codon A at canonical 200; 3 nt downstream (3') is canonical 197 = file 198
    _real(tracks, "K562", minus={"chr7": {198: 6}})
    rows = _rows("chr7", [201], ["-"])
    paths = track_tis_eval.real_paths(tracks)
    assert condition_scores(rows, paths, "K562", offset=3, eps=1e-3)["peak"] == pytest.approx([6])
    assert condition_scores(rows, paths, "K562", offset=-3, eps=1e-3)["peak"] == pytest.approx([0])


def test_scores_near_chromosome_edges_and_missing_chromosomes(tmp_path):
    tracks = tmp_path / "tracks"
    _real(tracks, "K562", {"chr7": {0: 3, 399: 6}})
    rows = _rows(["chr7", "chr7", "chrY"], [0, 399, 10], ["+", "+", "+"])
    scores = condition_scores(rows, track_tis_eval.real_paths(tracks), "K562", 0, eps=1e-3)
    assert scores["peak"] == pytest.approx([3.0, 6.0, 0.0])


def test_scores_do_not_depend_on_how_rows_are_grouped_into_reads(tmp_path, monkeypatch):
    tracks = tmp_path / "tracks"
    _real(tracks, "K562", {"chr7": {10: 3, 40: 6, 200: 3, 390: 6}}, {"chr7": {301: 3}})
    rows = _rows("chr7", [10, 200, 390, 41, 302], ["+", "+", "+", "+", "-"])
    paths = track_tis_eval.real_paths(tracks)
    together = condition_scores(rows, paths, "K562", 0, eps=1e-3)
    monkeypatch.setattr(track_tis_eval, "BLOCK_GAP", 0)  # every far-apart row its own read
    blocks = list(track_tis_eval._blocks(np.array([10, 200, 390, 41]), 30))
    assert [sorted(b.tolist()) for b, _, _ in blocks] == [[0, 3], [1], [2]]
    apart = condition_scores(rows, paths, "K562", 0, eps=1e-3)
    assert together["peak"] == pytest.approx([3.0, 3.0, 6.0, 6.0, 3.0])
    for kind in ("peak", "enrich"):
        assert apart[kind] == pytest.approx(together[kind])
    profile = offset_profile(rows, paths, ("K562",), half=2)
    monkeypatch.setattr(track_tis_eval, "BLOCK_GAP", 10**9)
    assert offset_profile(rows, paths, ("K562",), half=2) == pytest.approx(profile)


def test_offset_profile_recovers_planted_offset_on_both_strands(tmp_path):
    tracks = tmp_path / "tracks"
    # true TIS peak 1 nt 3' of the A on both strands: + at A+1, - at canonical A-1 = file A
    _real(tracks, "K562", {"chr7": {101: 6, 151: 3}}, {"chr7": {200: 6}})
    _real(tracks, "HeLa", {"chr7": {151: 3, 160: 1}}, {"chr8": {50: 3}})
    rows = _rows(["chr7", "chr7", "chr7", "chr8"], [100, 150, 201, 51], ["+", "+", "-", "-"])
    profile = offset_profile(rows, track_tis_eval.real_paths(tracks), ("K562", "HeLa"), half=5)
    assert profile.shape == (2, 2, 11)
    assert estimate_offset(profile) == 1
    assert int(np.argmax(profile[:, 1].sum(axis=0))) - 5 == 1  # minus strand alone
    assert profile[0, 0, 5 + 1] == pytest.approx(9.0)  # K562 plus at d=+1: 6 + 3


def test_aggregate_max_over_conditions_and_reference():
    per_condition = {
        "K562": {"peak": np.array([1.0, 5.0]), "enrich": np.array([2.0, 0.5])},
        "HeLa": {"peak": np.array([3.0, 0.0]), "enrich": np.array([1.0, 4.0])},
    }
    variants = aggregate(per_condition)
    assert variants["peak_max"].tolist() == [3.0, 5.0]
    assert variants["enrich_max"].tolist() == [2.0, 4.0]
    assert variants["peak_K562"].tolist() == [1.0, 5.0]
    assert variants["enrich_K562"].tolist() == [2.0, 0.5]


def test_coverage_uses_the_codon_a_in_the_window_union():
    region = {"chr7": [(0, 200)]}
    rows = _rows(["chr7", "chr7", "chr7", "chr8"], [199, 200, 200, 10], ["+", "+", "-", "+"])
    rows["label_tis"] = [1, 0, 1, 0]
    covered = coverage(rows, region)
    assert covered.tolist() == [True, False, True, False]  # minus gstart 200 is A at 199
    table = track_tis_eval.coverage_table(rows.label_tis.to_numpy(), covered)
    assert table["positives"] == {"n": 2, "covered": 2, "fraction": 1.0}
    assert table["negatives"] == {"n": 2, "covered": 0, "fraction": 0.0}


def _block(scores, labels, transcripts):
    return {
        "scores": {k: np.asarray(v, dtype=float) for k, v in scores.items()},
        "labels": np.asarray(labels),
        "transcripts": np.asarray(transcripts),
    }


def test_select_variant_by_recall_then_auprc_among_candidates():
    labels = [1, 0, 1, 0, 0, 0]
    tx = ["a", "a", "b", "b", "c", "c"]
    scores = {
        "peak_max": [0.9, 0.1, 0.2, 0.3, 0.0, 0.0],  # recall 1.0 at 1 FP / 3 tx
        "enrich_max": [0.9, 0.0, 0.8, 0.0, 0.0, 0.0],  # recall 1.0, perfect ranking
        "peak_K562": [0.9, 0.0, 0.9, 0.0, 0.0, 0.0],  # perfect too, but reference only
    }
    block = _block(scores, labels, tx)
    name, table = select_variant(block["scores"], block["labels"], block["transcripts"])
    assert name == "enrich_max"
    assert table["peak_max"]["recall"] == 1.0 and table["peak_max"]["auprc"] < 1.0
    assert set(table) == set(scores)


def test_protocol_selects_and_thresholds_on_operating_rows_only():
    tx = ["a", "a", "b", "b"]
    operating = _block(
        {"peak_max": [0.9, 0.1, 0.8, 0.2], "enrich_max": [0.1, 0.9, 0.2, 0.8]}, [1, 0, 1, 0], tx
    )
    # on test the ranking flips: enrich would win if selection peeked at test
    test = _block(
        {"peak_max": [0.1, 0.9, 0.2, 0.85], "enrich_max": [0.9, 0.1, 0.8, 0.2]}, [1, 0, 1, 0], tx
    )
    result = run_protocol(operating, test, budget=1.0)
    assert result["selected"] == "peak_max"
    assert result["threshold"] == pytest.approx(0.8)  # tightest threshold at full recall
    assert result["test"]["peak_max"]["all"]["recall"] == 0.0
    assert result["test"]["peak_max"]["all"]["threshold"] == pytest.approx(0.8)
    assert result["test"]["enrich_max"]["all"]["recall"] == 1.0  # reported, never selected
    alone = run_protocol(operating, None, budget=1.0)
    assert alone["selected"] == "peak_max" and "test" not in alone


def test_predicted_float_tracks_score_like_the_real_tracks_they_reproduce(tmp_path):
    """Oracle round trip: float pooled tracks of the real pooled counts give real-track scores."""
    from pyfaidx import Fasta

    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr7\n" + ("ATGAAACCC" * 50)[:400] + "\n>chr8\n" + "A" * 400 + "\n")
    Fasta(str(fasta)).close()
    tracks = tmp_path / "tracks"
    # a read at 395 on each strand keeps no track empty (the window reader needs data)
    plus = {"chr7": {100: 3, 101: 6, 150: 3, 395: 3}}
    minus = {"chr7": {201: 6, 202: 3, 120: 3, 395: 3}}
    _real(tracks, "K562", plus, minus)
    _write(track_prefix(tracks, "K562", "tis", "pooled"), plus, minus)
    for library in ("chx", "rna"):
        for rep in (1, 2, "pooled"):
            _write(
                track_prefix(tracks, "K562", library, rep), {"chr7": {395: 1}}, {"chr7": {395: 1}}
            )
    windows = pd.DataFrame({"chrom": ["chr7"] * 2, "start": [0, 150], "end": [250, 400]})
    genes = pd.DataFrame(
        {
            "condition": "K562",
            "gene_id": ["G1"],
            "chrom": "chr7",
            "start": [10],
            "end": [390],
            "strand": ["+"],
            "expressed": [True],
        }
    )
    out = tmp_path / "pred"
    predict_condition(
        lambda batch: batch["targets"],
        windows,
        fasta=str(fasta),
        tracks_dir=tracks,
        genes=genes,
        condition="K562",
        conditions=("K562",),
        out_dir=out,
        mode="float",
    )
    assert (pooled_prefix(out, "K562", "tis").with_name("tis_pooled.minus.bw")).exists()
    rows = _rows("chr7", [100, 150, 201, 121, 300], ["+", "+", "-", "-", "-"])
    real = condition_scores(rows, track_tis_eval.real_paths(tracks), "K562", 0, eps=1e-3)
    predicted = condition_scores(
        rows, track_tis_eval.predicted_paths({"K562": out}), "K562", 0, eps=1e-3
    )
    assert real["peak"] == pytest.approx([9.0, 3.0, 9.0, 3.0, 0.0])
    for kind in ("peak", "enrich"):
        assert predicted[kind] == pytest.approx(real[kind])


def _world(tmp_path):
    """Tiny store, row selections, windows, real tracks and baseline outputs."""
    rng = np.random.default_rng(0)
    records = []
    for split, chrom in (("val", "chr7"), ("test", "chr8")):
        for t in range(8):
            tx = f"{split}{t}"
            for k in range(10):
                gstart = 20 + 40 * t + 3 * k
                strand = "+" if t % 2 == 0 else "-"
                label = int(k == 2)
                cls = "AUG" if k in (2, 5) else "near_cognate"
                records.append((tx, chrom, strand, k * 3, gstart, cls, split, label))
    manifest = pd.DataFrame(
        records,
        columns=[
            "transcript_id",
            "chrom",
            "strand",
            "mrna_index",
            "gstart",
            "codon_class",
            "split",
            "label_tis",
        ],
    )
    manifest.insert(0, "row_idx", np.arange(len(manifest)))
    store = tmp_path / "store"
    store.mkdir()
    manifest.to_parquet(store / "manifest.parquet")
    val = np.flatnonzero(manifest.split == "val")
    is_cal = manifest.transcript_id.to_numpy()[val] < "val4"
    selections = tmp_path / "base" / "row_selections.npz"
    selections.parent.mkdir()
    test_rows = np.flatnonzero(manifest.split == "test")
    np.savez(
        selections,
        unique=np.arange(len(manifest)),
        calibration=val[is_cal],
        operating=val[~is_cal],
        test=test_rows,
    )
    # true starts carry a peak of 8 at A+1 (transcript orientation); other codons 0/1 noise
    tracks = tmp_path / "tracks"
    for condition in ("K562", "HeLa"):
        plus, minus = {}, {}
        for row in manifest.itertuples():
            centre = int(candidate_centre(np.array([row.gstart]), np.array([row.strand]), 1)[0])
            value = 8.0 if row.label_tis else float(rng.integers(0, 2))
            target = plus if row.strand == "+" else minus
            target.setdefault(row.chrom, {})[centre] = value + target.get(row.chrom, {}).get(
                centre, 0.0
            )
        _real(tracks, condition, plus, minus)
    windows = tmp_path / "windows"
    windows.mkdir()
    pd.DataFrame({"chrom": ["chr7"], "start": [0], "end": [400]}).to_parquet(
        windows / "val.parquet"
    )
    # the test window misses the last test transcript (rows 0 coverage)
    pd.DataFrame({"chrom": ["chr8"], "start": [0], "end": [300]}).to_parquet(
        windows / "test.parquet"
    )
    base = selections.parent
    pd.DataFrame(
        [
            {"arm": "ag_w8k", "seed": "ensemble", "threshold": 0.2, "auprc": 0.304},
            {"arm": "ag_w8k", "seed": "0", "threshold": 0.3, "auprc": 0.28},
        ]
    ).assign(recall=0.306, fp_per_transcript=0.99).to_csv(
        base / "metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {"arm": "ag_w8k", "stratum": "codon_class:AUG", "recall": 0.58, "auprc": 0.51},
            {"arm": "ag_txp", "stratum": "codon_class:AUG", "recall": 0.60, "auprc": 0.50},
        ]
    ).to_csv(base / "stratified_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([{"arm": "ag_w8k", "metric": "recall", "ci_low": 0.29, "ci_high": 0.32}]).to_csv(
        base / "bootstrap_ci.tsv", sep="\t", index=False
    )
    return dict(store=store, base=base, tracks=tracks, windows=windows, out=tmp_path / "out")


def _argv(world, *extra):
    return [
        "track_tis_eval",
        "--store",
        str(world["store"]),
        "--baseline-dir",
        str(world["base"]),
        "--tracks-dir",
        str(world["tracks"]),
        "--windows-dir",
        str(world["windows"]),
        "--out-dir",
        str(world["out"]),
        "--conditions",
        "K562",
        "HeLa",
        "--sources",
        "real",
        "--bootstrap",
        "20",
        "--profile-half",
        "5",
        *extra,
    ]


def test_cli_refuses_test_split_without_final_flag(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(world, "--split", "test"))
    with pytest.raises(SystemExit):
        track_tis_eval.main()
    assert "--i-am-final" in capsys.readouterr().err
    assert not world["out"].exists()


def test_cli_val_reports_offset_coverage_and_operating_selection(tmp_path, monkeypatch):
    world = _world(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(world, "--split", "val"))
    track_tis_eval.main()
    report = json.loads((world["out"] / "report.json").read_text())
    assert report["offset"]["offset"] == 1
    assert report["offset"]["source"] == "real" and report["offset"]["rows"] == "calibration"
    assert report["coverage"]["operating"]["positives"]["fraction"] == 1.0
    assert "test" not in report["coverage"]
    real = report["sources"]["real"]
    assert real["selected"] in ("peak_max", "enrich_max")
    assert real["operating"][real["selected"]]["recall"] == 1.0
    assert "test" not in real
    assert (world["out"] / "report.md").read_text().startswith("# P6 track TIS score")


def test_cli_test_reports_strata_cis_coverage_and_baseline(tmp_path, monkeypatch):
    world = _world(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(world, "--split", "test", "--i-am-final"))
    track_tis_eval.main()
    report = json.loads((world["out"] / "report.json").read_text())
    covered = report["coverage"]["test"]
    assert covered["positives"]["n"] == 8 and covered["positives"]["covered"] == 7
    real = report["sources"]["real"]
    selected = real["test"][real["selected"]]
    assert selected["all"]["recall"] == pytest.approx(7 / 8)  # uncovered positive scores 0
    assert set(selected) >= {"all", "codon_class:AUG", "codon_class:near_cognate"}
    ci = real["bootstrap_ci"]["all"]["recall"]
    assert ci["ci_low"] <= selected["all"]["recall"] <= ci["ci_high"]
    assert "codon_class:AUG" in real["bootstrap_ci"]
    baseline = report["baseline"]
    assert baseline["arm"] == "ag_w8k" and baseline["all"]["auprc"] == pytest.approx(0.304)
    assert baseline["strata"]["codon_class:AUG"]["recall"] == pytest.approx(0.58)
    assert baseline["bootstrap_ci"]["recall"] == {"ci_low": 0.29, "ci_high": 0.32}
    assert baseline["files"]["metrics"].endswith("metrics.tsv")
    md = (world["out"] / "report.md").read_text()
    assert "baseline ag_w8k" in md and "real" in md
