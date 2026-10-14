"""Tests for the same-pipeline comparison of Ribo-TISH calls on predicted vs real tracks."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")  # CPU stage: Ribo-TISH inputs need pysam
from tisiago.seq2func import predicted_calls  # noqa: E402
from tisiago.seq2func.predicted_calls import (  # noqa: E402
    call_report,
    restrict_gtf,
    restrict_tracks,
    run_predicted_calls,
)
from tisiago.seq2func.roundtrip import PASS_JACCARD  # noqa: E402
from tisiago.seq2func.tracks import read_bigwig_counts, track_prefix, write_bigwigs  # noqa: E402

SIZES = [("chr1", 300), ("chr7", 400)]
REGION = {"chr7": [(0, 200)]}
COLS = ["Tid", "Start", "StartCodon", "TisType", "TISPvalue", "RiboPvalue", "FisherQvalue"]


def _write(tracks, condition="K562", plus=None, minus=None):
    plus = plus or {("chr7", 10): 3, ("chr7", 250): 5, ("chr1", 20): 7}
    minus = minus or {("chr7", 200): 2, ("chr7", 201): 4}
    for library in ("tis", "chx"):
        for rep in (1, 2):
            counts = {(c, s): Counter() for c, _ in SIZES for s in "+-"}
            for (chrom, p), v in plus.items():
                counts[(chrom, "+")][p] = v
            for (chrom, p), v in minus.items():
                counts[(chrom, "-")][p] = v
            write_bigwigs(counts, SIZES, track_prefix(tracks, condition, library, rep))


def test_restrict_tracks_keeps_the_region_in_ribotish_coordinates(tmp_path):
    _write(tmp_path / "real")
    restrict_tracks(tmp_path / "real", tmp_path / "restricted", "K562", REGION)
    for library in ("tis", "chx"):
        for rep in (1, 2):
            prefix = track_prefix(tmp_path / "restricted", "K562", library, rep)
            plus = read_bigwig_counts(f"{prefix}.plus.bw")
            minus = read_bigwig_counts(f"{prefix}.minus.bw")
            assert dict(plus["chr7"]) == {10: 3.0}
            assert not plus["chr1"]
            # minus file 200 = canonical 199 (inside); 201 = canonical 200 (outside)
            assert dict(minus["chr7"]) == {200: 2.0}
    import pyBigWig

    bw = pyBigWig.open(f"{track_prefix(tmp_path / 'restricted', 'K562', 'tis', 1)}.plus.bw")
    assert list(bw.chroms().items()) == SIZES  # same header as the real tracks
    bw.close()


def test_restrict_gtf_keeps_split_chromosomes(tmp_path):
    gtf = tmp_path / "a.gtf"
    gtf.write_text(
        "##header\nchr1\tX\texon\t1\t10\t.\t+\t.\tx\nchr7\tX\texon\t1\t10\t.\t+\t.\ty\n"
        "chr17\tX\texon\t1\t10\t.\t+\t.\tz\n"
    )
    out = restrict_gtf(gtf, ["chr7"], tmp_path / "r.gtf")
    assert out.read_text() == "##header\nchr7\tX\texon\t1\t10\t.\t+\t.\ty\n"


def _table(rows):
    return pd.DataFrame(rows, columns=COLS)


SIG = (0.001, 0.001, 0.001)


def test_call_report_fields_overall_and_by_group():
    predicted = _table(
        [
            ("T1", 10, "ATG", "Annotated", *SIG),
            ("T1", 40, "CTG", "5'UTR", *SIG),
            ("T2", 5, "ATG", "Novel", *SIG),
            ("T3", 9, "ATG", "Novel", 0.5, 0.001, 0.001),  # filtered out
        ]
    )
    real = _table(
        [
            ("T1", 10, "ATG", "Annotated", *SIG),
            ("T1", 40, "CTG", "5'UTR", *SIG),
            ("T3", 9, "ATG", "Novel", *SIG),
            ("T4", 2, "GTG", "Novel", *SIG),
        ]
    )
    report = call_report(predicted, real)
    overall = report["overall"]
    assert overall["n_observed"] == 3 and overall["n_reference"] == 4
    assert overall["n_shared"] == 2
    assert overall["jaccard"] == pytest.approx(2 / 5)
    assert overall["recall"] == pytest.approx(2 / 4)
    assert overall["precision"] == pytest.approx(2 / 3)
    atg = report["by_start_codon"]["ATG"]
    assert (atg["n_observed"], atg["n_reference"], atg["n_shared"]) == (2, 2, 1)
    near = report["by_start_codon"]["near_cognate"]
    assert (near["n_observed"], near["n_reference"], near["n_shared"]) == (1, 2, 1)
    novel = report["by_tis_type"]["Novel"]
    assert (novel["n_observed"], novel["n_reference"], novel["n_shared"]) == (1, 2, 0)
    assert novel["precision"] == 0.0
    assert report["pass_jaccard"] == PASS_JACCARD
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def test_call_report_without_tis_type_and_empty_groups():
    cols = ["Tid", "Start", "StartCodon", "TISPvalue", "RiboPvalue", "FisherQvalue"]
    table = pd.DataFrame([("T1", 10, "ATG", *SIG)], columns=cols)
    report = call_report(table, table)
    assert report["passed"] is True and report["by_tis_type"] is None
    near = report["by_start_codon"]["near_cognate"]
    assert near["recall"] is None and near["precision"] is None
    json.dumps(report, allow_nan=False)


PATH_FLAGS = {"-t", "-b", "--tispara", "--ribopara", "-g", "-f", "-o", "-e"}


@pytest.fixture
def calls_world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(Path("real"))
    _write(Path("pred"), plus={("chr7", 10): 3}, minus={("chr7", 200): 2})
    gtf, fasta = Path("ref.gtf"), Path("ref.fa")
    gtf.write_text("chr1\tX\texon\t1\t10\t.\t+\t.\tx\nchr7\tX\texon\t1\t10\t.\t+\t.\ty\n")
    fasta.touch()
    windows = pd.DataFrame({"chrom": ["chr7", "chr7"], "start": [0, 100], "end": [100, 200]})
    argvs = []
    real_rows = [("T1", 10, "ATG", "Annotated", *SIG), ("T2", 4, "CTG", "Novel", *SIG)]
    pred_rows = [("T1", 10, "ATG", "Annotated", *SIG)]

    def fake_run(argv, check=False, cwd=None, **kwargs):
        if argv[1:] == ["--version"]:
            return SimpleNamespace(stdout="ribotish 0.2.8+tisiago.1\n", stderr="")
        assert Path(cwd).is_absolute()
        for flag, value in zip(argv, argv[1:]):
            if flag in PATH_FLAGS:
                assert all(Path(v).is_absolute() for v in value.split(",")), (flag, value)
        argvs.append((Path(cwd).name, [a.replace(str(cwd), "<work>") for a in argv]))
        rows = pred_rows if Path(cwd).name == "predicted" else real_rows
        _table(rows).to_csv(Path(cwd) / "K562_TIS_predict_all.txt", sep="\t", index=False)

    monkeypatch.setattr("tisiago.seq2func.roundtrip.subprocess.run", fake_run)
    return SimpleNamespace(gtf=gtf, fasta=fasta, windows=windows, argvs=argvs)


def test_run_predicted_calls_same_flags_and_region(calls_world):
    w = calls_world
    report = run_predicted_calls(
        "K562",
        Path("pred"),
        Path("real"),
        w.windows,
        Path("work"),
        gtf=w.gtf,
        fasta=w.fasta,
        ribotish="ribotish",
        nproc=4,
    )
    assert [name for name, _ in w.argvs] == ["predicted", "real"]
    assert w.argvs[0][1] == w.argvs[1][1]  # identical flags, inputs differ only by work dir
    argv = w.argvs[0][1]
    assert argv[argv.index("-p") + 1] == "4"
    gtf_used = Path("work") / "annotation.chr7.gtf"
    assert gtf_used.read_text() == "chr7\tX\texon\t1\t10\t.\t+\t.\ty\n"
    assert argv[argv.index("-g") + 1] == str(gtf_used.resolve())
    with pysam.AlignmentFile("work/real/tis_rep1.bam", "rb") as bam:
        reads = [(r.reference_name, r.is_reverse) for r in bam.fetch(until_eof=True)]
    assert Counter(reads) == Counter({("chr7", False): 3, ("chr7", True): 2})
    o = report["overall"]
    assert (o["n_observed"], o["n_reference"], o["n_shared"]) == (1, 2, 1)
    assert report["passed"] is False
    assert report["chroms"] == ["chr7"] and report["windows"] == 2
    assert report["ribotish_version"] == "ribotish 0.2.8+tisiago.1"


def test_cli_writes_report_and_refuses_test_without_final(calls_world, monkeypatch, capsys):
    w = calls_world
    Path("win").mkdir()
    w.windows.to_parquet("win/val.parquet")
    (Path("pred") / "K562" / "predict_tracks.json").write_text(
        json.dumps({"split": "val", "windows_dir": "win", "tracks_dir": "real"})
    )
    base = ["predicted_calls", "--condition", "K562", "--predicted-dir", "pred"]
    flags = ["--gtf", str(w.gtf), "--genome", str(w.fasta), "--ribotish", "ribotish"]
    monkeypatch.setattr(sys, "argv", [*base, *flags, "--split", "test"])
    with pytest.raises(SystemExit):
        predicted_calls.main()
    assert "--i-am-final" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*base, *flags])
    predicted_calls.main()
    report = json.loads(Path("pred/calls/K562/report.json").read_text())
    assert report["split"] == "val" and report["overall"]["n_shared"] == 1
    assert report["tracks_dir"] == "real"


def test_cli_rejects_split_that_differs_from_the_prediction(calls_world, monkeypatch, capsys):
    (Path("pred") / "K562" / "predict_tracks.json").write_text(
        json.dumps({"split": "val", "windows_dir": "win", "tracks_dir": "real"})
    )
    argv = ["predicted_calls", "--condition", "K562", "--predicted-dir", "pred"]
    monkeypatch.setattr(sys, "argv", [*argv, "--split", "test", "--i-am-final"])
    with pytest.raises(SystemExit):
        predicted_calls.main()
    assert "split" in capsys.readouterr().err
