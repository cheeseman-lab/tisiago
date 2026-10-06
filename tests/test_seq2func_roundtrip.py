"""Tests for round-trip input preparation."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tisiago.seq2func.roundtrip import PASS_JACCARD, prepare_inputs, run_condition
from tisiago.seq2func.tracks import track_prefix, write_bigwigs


def test_prepare_inputs_builds_rep_bams_and_paras(tmp_path):
    tracks = tmp_path / "tracks"
    for library in ("tis", "chx"):
        for rep in (1, 2):
            counts = {("chr1", "+"): Counter({10: rep}), ("chr1", "-"): Counter()}
            write_bigwigs(counts, [("chr1", 100)], track_prefix(tracks, "HeLa", library, rep))
    inputs = prepare_inputs("HeLa", tracks, tmp_path / "work")
    assert [p.name for p in inputs["tis_bams"]] == ["tis_rep1.bam", "tis_rep2.bam"]
    assert [p.name for p in inputs["chx_paras"]] == [
        "chx_rep1.bam.para.py",
        "chx_rep2.bam.para.py",
    ]
    assert all(p.exists() for paths in inputs.values() for p in paths)


PATH_FLAGS = {"-t", "-b", "--tispara", "--ribopara", "-g", "-f", "-o", "-e"}


def test_run_condition_uses_absolute_paths_with_relative_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for library in ("tis", "chx"):
        for rep in (1, 2):
            counts = {("chr1", "+"): Counter({10: rep}), ("chr1", "-"): Counter()}
            write_bigwigs(
                counts, [("chr1", 100)], track_prefix(Path("tracks"), "HeLa", library, rep)
            )
    cols = "Tid\tStart\tTISPvalue\tRiboPvalue\tFisherQvalue\n"
    rows = "T1\t5\t0.001\t0.001\t0.001\nT2\t7\t0.001\t0.001\t0.001\n"
    reference = Path("run") / "HeLa" / "pooled" / "HeLa_pooled.candidates.tsv"
    reference.parent.mkdir(parents=True)
    reference.write_text(cols + rows)
    gtf, fasta = Path("ref.gtf"), Path("ref.fa")
    gtf.touch()
    fasta.touch()
    seen = {}
    argvs = []

    def fake_run(argv, check=False, cwd=None, **kwargs):
        if argv[1:] == ["--version"]:
            return SimpleNamespace(stdout="ribotish 0.2.8+tisiago.1\n", stderr="")
        argvs.append(list(argv))
        for flag, value in zip(argv, argv[1:]):
            if flag in PATH_FLAGS:
                assert all(Path(v).is_absolute() for v in value.split(",")), (flag, value)
                seen[flag] = value
        assert Path(cwd).is_absolute()
        (Path(cwd) / "HeLa_TIS_predict_all.txt").write_text(cols + rows)

    monkeypatch.setattr("tisiago.seq2func.roundtrip.subprocess.run", fake_run)
    report = run_condition(
        "HeLa", Path("tracks"), Path("work"), reference, "ribotish", gtf=gtf, fasta=fasta
    )
    assert set(seen) == PATH_FLAGS
    assert "-p" not in argvs[0]
    assert report["jaccard"] == 1.0
    assert report["jaccard"] >= PASS_JACCARD
    assert report["passed"] is True
    assert report["ribotish_version"] == "ribotish 0.2.8+tisiago.1"
    assert report["reference"] == str(reference.resolve())
    run_condition(
        "HeLa", Path("tracks"), Path("work"), reference, "ribotish", nproc=8, gtf=gtf, fasta=fasta
    )
    assert argvs[1][argvs[1].index("-p") + 1] == "8"


def test_run_condition_missing_reference_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="reference"):
        run_condition("HeLa", tmp_path, tmp_path / "w", tmp_path / "none.tsv", "ribotish")
