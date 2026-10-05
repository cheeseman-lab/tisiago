"""Tests for round-trip input preparation."""

from collections import Counter

from tisiago.seq2func.roundtrip import prepare_inputs
from tisiago.seq2func.tracks import track_prefix, write_bigwigs


def test_prepare_inputs_builds_rep_bams_and_paras(tmp_path):
    tracks = tmp_path / "tracks"
    for library in ("tis", "chx"):
        for rep in (1, 2):
            counts = {("chr1", "+"): Counter({10: rep}), ("chr1", "-"): Counter()}
            write_bigwigs(counts, [("chr1", 100)], track_prefix(tracks, "HeLa", library, rep))
    inputs = prepare_inputs("HeLa", tracks, tmp_path / "work")
    assert [p.name for p in inputs["tis_bams"]] == ["tis_rep1.bam", "tis_rep2.bam"]
    assert [p.name for p in inputs["chx_paras"]] == ["chx_rep1.bam.para.py", "chx_rep2.bam.para.py"]
    assert all(p.exists() for paths in inputs.values() for p in paths)
