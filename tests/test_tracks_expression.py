"""Tests for the per-condition expressed-gene table."""

import pandas as pd
import pytest

from tisiago.tracks.expression import counts_path, expression_table, read_counts, read_gene_spans


def _gtf(tmp_path):
    path = tmp_path / "g.gtf"
    path.write_text(
        'chr1\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id "G1.1"; gene_type "x";\n'
        'chr1\tHAVANA\texon\t11\t20\t.\t+\t.\tgene_id "G1.1";\n'
        'chr2\tHAVANA\tgene\t5\t9\t.\t-\t.\tgene_id "G2.3";\n'
    )
    return path


def test_read_gene_spans_zero_based(tmp_path):
    spans = read_gene_spans(_gtf(tmp_path))
    assert spans.to_dict("records") == [
        {"gene_id": "G1.1", "chrom": "chr1", "start": 10, "end": 20, "strand": "+"},
        {"gene_id": "G2.3", "chrom": "chr2", "start": 4, "end": 9, "strand": "-"},
    ]


def test_read_counts_drops_htseq_summary(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("G1.1\t3\nG2.3\t0\n__no_feature\t99\n")
    assert read_counts(path).to_dict() == {"G1.1": 3, "G2.3": 0}


def test_read_counts_raises_on_missing_file(tmp_path):
    path = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        read_counts(path)


def test_counts_path_exact_match(tmp_path):
    counts = tmp_path / "counts"
    counts.mkdir()
    (counts / "sample1_htseqcount.txt").write_text("G1\t10\n")
    assert counts_path(tmp_path, "sample1") == counts / "sample1_htseqcount.txt"


def test_counts_path_lane_fallback(tmp_path):
    counts = tmp_path / "counts"
    counts.mkdir()
    (counts / "713_S10_L002_htseqcount.txt").write_text("G1\t10\n")
    # Try to find by lane-stripped name (L001 missing, should fall back to L002)
    assert counts_path(tmp_path, "713_S10_L001") == counts / "713_S10_L002_htseqcount.txt"


def test_counts_path_missing_raises_file_not_found(tmp_path):
    counts = tmp_path / "counts"
    counts.mkdir()
    with pytest.raises(FileNotFoundError):
        counts_path(tmp_path, "nonexistent_S1_L001")


def test_counts_path_multiple_lanes_raises_value_error(tmp_path):
    counts = tmp_path / "counts"
    counts.mkdir()
    (counts / "sample_S1_L001_htseqcount.txt").write_text("G1\t10\n")
    (counts / "sample_S1_L002_htseqcount.txt").write_text("G1\t20\n")
    with pytest.raises(ValueError, match="multiple lane variants"):
        counts_path(tmp_path, "sample_S1_L003")


def test_expression_table_sums_replicates_and_thresholds(tmp_path, monkeypatch):
    samples = pd.DataFrame(
        [
            {"condition": "HeLa", "library": "rna", "rep": 1, "sample_id": "a"},
            {"condition": "HeLa", "library": "rna", "rep": 2, "sample_id": "b"},
            {"condition": "HeLa", "library": "tis", "rep": 1, "sample_id": "t"},
        ]
    )
    monkeypatch.setattr("tisiago.tracks.expression.load_samples", lambda root: samples)
    counts = tmp_path / "counts"
    counts.mkdir()
    (counts / "a_htseqcount.txt").write_text("G1.1\t1\nG2.3\t0\n")
    (counts / "b_htseqcount.txt").write_text("G1.1\t1\n")
    table = expression_table(tmp_path, _gtf(tmp_path))
    row = table.set_index("gene_id")
    assert row.loc["G1.1", "rna_count"] == 2 and bool(row.loc["G1.1", "expressed"])
    assert row.loc["G2.3", "rna_count"] == 0 and not bool(row.loc["G2.3", "expressed"])
    assert set(table.condition) == {"HeLa"}


def test_expression_table_reads_counts_dir(tmp_path, monkeypatch):
    samples = pd.DataFrame([{"condition": "K562", "library": "rna", "rep": 1, "sample_id": "a"}])
    monkeypatch.setattr("tisiago.tracks.expression.load_samples", lambda root: samples)
    other = tmp_path / "rebuilt"
    other.mkdir()
    (other / "a_htseqcount.txt").write_text("G1.1\t3\n")
    table = expression_table(tmp_path, _gtf(tmp_path), counts_dir=other)
    assert table.set_index("gene_id").loc["G1.1", "rna_count"] == 3
