"""Tests for stranded 1-bp bigWig track building."""

import json
from collections import Counter
from types import SimpleNamespace

import pysam

from tisiago.seq2func.psite import rna_end5
from tisiago.seq2func.tracks import (
    build_sample,
    count_positions,
    pool,
    read_bigwig_counts,
    track_prefix,
    write_bigwigs,
)

SIZES = [("chr1", 1_000), ("chr2", 500)]


def _bam(tmp_path, reads):
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in SIZES]}
    path = tmp_path / "x.bam"
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for i, (ref, start, reverse) in enumerate(reads):
            read = pysam.AlignedSegment(bam.header)
            read.query_name = f"r{i}"
            read.query_sequence = "A" * 10
            read.reference_id = ref
            read.reference_start = start
            read.cigartuples = [(0, 10)]
            read.mapping_quality = 255
            read.flag = 16 if reverse else 0
            read.set_tag("NH", 1)
            read.set_tag("MD", "10")
            bam.write(read)
    pysam.index(str(path))
    return path


def test_counts_roundtrip_through_bigwig(tmp_path):
    bam = _bam(tmp_path, [(0, 5, False), (0, 5, False), (0, 20, True), (1, 7, False)])
    counts = count_positions(bam, rna_end5)
    assert counts[("chr1", "+")][5] == 2
    assert counts[("chr1", "-")][29] == 1
    total = write_bigwigs(counts, SIZES, tmp_path / "t")
    assert total == 4
    plus = read_bigwig_counts(tmp_path / "t.plus.bw")
    minus = read_bigwig_counts(tmp_path / "t.minus.bw")
    assert plus["chr1"][5] == 2 and plus["chr2"][7] == 1
    assert minus["chr1"][29] == 1
    assert sum(minus["chr2"].values()) == 0


def test_build_sample_rpf_places_psites_and_writes_record(tmp_path):
    bam = _bam(tmp_path, [(0, 5, False), (0, 5, False), (0, 20, True), (1, 7, False)])
    para = tmp_path / "x.bam.para.py"
    para.write_text("offdict = {10: 3}\n")
    sample = SimpleNamespace(
        condition="HeLa_ctrl", library="rpf", rep=1, sample_id="S1", bam=bam, para=para
    )
    record = build_sample(sample, tmp_path / "out")
    assert record == {
        "condition": "HeLa_ctrl",
        "library": "rpf",
        "rep": 1,
        "sample_id": "S1",
        "total": 4,
    }
    prefix = track_prefix(tmp_path / "out", "HeLa_ctrl", "rpf", 1)
    assert json.loads(prefix.with_suffix(".json").read_text()) == record
    plus = read_bigwig_counts(f"{prefix}.plus.bw")
    minus = read_bigwig_counts(f"{prefix}.minus.bw")
    assert plus["chr1"] == Counter({8: 2}) and plus["chr2"] == Counter({10: 1})
    assert minus["chr1"] == Counter({27: 1})


def test_pool_sums_replicates_per_strand(tmp_path):
    out = tmp_path / "out"
    reps = {
        1: {("chr1", "+"): Counter({5: 2, 9: 1}), ("chr1", "-"): Counter({29: 1})},
        2: {("chr1", "+"): Counter({5: 3}), ("chr2", "-"): Counter({7: 4})},
    }
    for rep, counts in reps.items():
        write_bigwigs(counts, SIZES, track_prefix(out, "c", "rpf", rep))
    assert pool(out, "c", "rpf", SIZES) == 11
    prefix = track_prefix(out, "c", "rpf", "pooled")
    plus = read_bigwig_counts(f"{prefix}.plus.bw")
    minus = read_bigwig_counts(f"{prefix}.minus.bw")
    assert plus["chr1"] == Counter({5: 5, 9: 1}) and not plus["chr2"]
    assert minus["chr1"] == Counter({29: 1}) and minus["chr2"] == Counter({7: 4})


def test_header_only_strand_reads_and_pools(tmp_path):
    out = tmp_path / "out"
    for rep in (1, 2):
        write_bigwigs({("chr1", "+"): Counter({3: rep})}, SIZES, track_prefix(out, "c", "rpf", rep))
    minus = read_bigwig_counts(track_prefix(out, "c", "rpf", 1).with_name("rpf_rep1.minus.bw"))
    assert set(minus) == {"chr1", "chr2"} and not any(minus.values())
    assert pool(out, "c", "rpf", SIZES) == 3
    prefix = track_prefix(out, "c", "rpf", "pooled")
    assert read_bigwig_counts(f"{prefix}.plus.bw")["chr1"] == Counter({3: 3})
    pooled_minus = read_bigwig_counts(f"{prefix}.minus.bw")
    assert not any(pooled_minus.values())
    assert write_bigwigs({}, SIZES, tmp_path / "empty") == 0
    assert not any(read_bigwig_counts(tmp_path / "empty.plus.bw").values())
