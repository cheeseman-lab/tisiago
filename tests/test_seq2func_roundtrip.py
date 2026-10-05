"""Tests for round-trip input preparation."""

from collections import Counter
from pathlib import Path

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
    atis = Path("atis")
    (atis / "ribotish" / "combined").mkdir(parents=True)
    (atis / "reference").mkdir()
    (atis / "reference" / "gencode.v49.primary_assembly.annotation.gtf").touch()
    (atis / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa").touch()
    (atis / "ribotish" / "combined" / "HeLa_TIS_predict_all.txt").write_text(
        cols + "T1\t5\t0.001\t0.001\t0.001\nT2\t7\t0.001\t0.001\t0.001\n"
    )
    seen = {}

    def fake_run(argv, check, cwd):
        for flag, value in zip(argv, argv[1:]):
            if flag in PATH_FLAGS:
                assert all(Path(v).is_absolute() for v in value.split(",")), (flag, value)
                seen[flag] = value
        assert Path(cwd).is_absolute()
        (Path(cwd) / "HeLa_TIS_predict_all.txt").write_text(
            cols + "T1\t5\t0.001\t0.001\t0.001\nT2\t7\t0.001\t0.001\t0.001\n"
        )

    monkeypatch.setattr("tisiago.seq2func.roundtrip.subprocess.run", fake_run)
    report = run_condition("HeLa", Path("tracks"), Path("work"), atis, "ribotish")
    assert set(seen) == PATH_FLAGS
    assert report["jaccard"] == 1.0
    assert report["jaccard"] >= PASS_JACCARD
    assert report["passed"] is True
