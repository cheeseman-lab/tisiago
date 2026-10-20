"""Tests for the transcript-space embedding cache of the P6 fast loop."""

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from tisiago.seq2func import txcache
from tisiago.seq2func.predict_tracks import owned_spans
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT
from tisiago.seq2func.scorecard import BigWigs, load_units, transcript_vector
from tisiago.seq2func.tracks import track_prefix

SIZES = [("chr7", 3000), ("chr1", 3000), ("chr8", 3000)]
CONDITIONS = ("HeLa", "K562")
MANE = '; tag "MANE_Select"'
# 1-based GTF. T1 (+, chr7): exons [100,160) [200,290), CDS [130,160) [200,260).
# T2 (-, chr7): exons [500,560) [600,690), CDS [520,560) [600,650). T4: chr1 (train).
# T5: chr8 (test, never cached). T6: chr7, gene not expressed anywhere.
FEATURES = [
    ("chr7", "exon", 101, 160, "+", "G1", "T1", MANE),
    ("chr7", "exon", 201, 290, "+", "G1", "T1", MANE),
    ("chr7", "CDS", 131, 160, "+", "G1", "T1", MANE),
    ("chr7", "CDS", 201, 260, "+", "G1", "T1", MANE),
    ("chr7", "exon", 501, 560, "-", "G2", "T2", MANE),
    ("chr7", "exon", 601, 690, "-", "G2", "T2", MANE),
    ("chr7", "CDS", 521, 560, "-", "G2", "T2", MANE),
    ("chr7", "CDS", 601, 650, "-", "G2", "T2", MANE),
    ("chr1", "exon", 101, 190, "+", "G4", "T4", MANE),
    ("chr1", "CDS", 101, 190, "+", "G4", "T4", MANE),
    ("chr8", "exon", 101, 190, "+", "G5", "T5", MANE),
    ("chr8", "CDS", 101, 190, "+", "G5", "T5", MANE),
    ("chr7", "exon", 1101, 1190, "+", "G6", "T6", MANE),
    ("chr7", "CDS", 1101, 1190, "+", "G6", "T6", MANE),
]


def _gtf(tmp_path):
    lines = ["#!header"]
    for chrom, kind, s, e, strand, gene, tid, tag in FEATURES:
        attrs = f'gene_id "{gene}"; transcript_id "{tid}"; gene_name "N{gene}"{tag};'
        lines.append("\t".join([chrom, "X", kind, str(s), str(e), ".", strand, ".", attrs]))
    path = tmp_path / "mane.gtf"
    path.write_text("\n".join(lines) + "\n")
    return path


def _genes():
    rows = []
    for gene, chrom, expressed in (
        ("G1", "chr7", {"HeLa"}),
        ("G2", "chr7", {"HeLa", "K562"}),
        ("G4", "chr1", {"K562"}),
        ("G5", "chr8", {"HeLa", "K562"}),
        ("G6", "chr7", set()),
    ):
        for c in CONDITIONS:
            rows.append(
                {"condition": c, "gene_id": gene, "chrom": chrom, "expressed": c in expressed}
            )
    return pd.DataFrame(rows)


def _windows():
    return {
        "train": pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1000]}),
        "val": pd.DataFrame({"chrom": ["chr7"] * 2, "start": [0, 400], "end": [1400, 1800]}),
    }


def _write(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(SIZES)
    for chrom, _ in SIZES:
        values = np.asarray(arrays.get(chrom, []), dtype=np.float64)
        keep = np.flatnonzero(values)
        if len(keep):
            bw.addEntries(chrom, keep.tolist(), values=values[keep].tolist(), span=1)
    bw.close()


def _index(tmp_path):
    units = load_units(_gtf(tmp_path))
    return txcache.select_transcripts(units, _genes(), _windows(), CONDITIONS, ["chr8"])


def test_select_keeps_expressed_units_inside_a_split_and_orders_shards(tmp_path):
    index = _index(tmp_path)
    assert index.transcript_id.tolist() == ["T4", "T1", "T2"]
    assert index.split.tolist() == ["train", "val", "val"]
    assert index.shard.tolist() == ["train_chr1", "val_chr7", "val_chr7"]
    assert index.offset.tolist() == [0, 0, 150]
    t1, t2 = index.iloc[1], index.iloc[2]
    assert (t1.length, t1.cds_start, t1.cds_end) == (150, 30, 120)
    assert (t2.length, t2.cds_start, t2.cds_end) == (150, 40, 130)
    assert list(t1.junctions) == [60] and list(t2.junctions) == [90]
    assert [bool(t1.expressed_HeLa), bool(t1.expressed_K562)] == [True, False]
    assert txcache.shard_sizes(index) == {"train_chr1": 90, "val_chr7": 300}


def test_select_refuses_test_split_and_test_chromosomes(tmp_path):
    units = load_units(_gtf(tmp_path))
    windows = {"test": pd.DataFrame({"chrom": ["chr8"], "start": [0], "end": [1000]})}
    with pytest.raises(ValueError, match="never cached"):
        txcache.select_transcripts(units, _genes(), windows, CONDITIONS)
    windows = {"val": pd.DataFrame({"chrom": ["chr8"], "start": [0], "end": [1000]})}
    with pytest.raises(ValueError, match="test chromosomes"):
        txcache.select_transcripts(units, _genes(), windows, CONDITIONS, ["chr8"])


def test_unit_of_round_trips_positions(tmp_path):
    index = _index(tmp_path)
    units = {u.transcript_id: u for u in load_units(_gtf(tmp_path))}
    for row in index.itertuples():
        unit = txcache.unit_of(row)
        assert unit == units[row.transcript_id]
        assert unit.positions.tolist() == units[row.transcript_id].positions.tolist()


@pytest.fixture
def tracks(tmp_path):
    """Tracks whose value encodes canonical position, strand, library, replicate, condition."""
    root = tmp_path / "tracks"
    n = 3000
    canonical = np.arange(n, dtype=np.float64)
    for k, condition in enumerate(CONDITIONS):
        for li, library in enumerate(("rna", "tis", "chx")):
            for rep in (1, 2, "pooled"):
                r = 3 if rep == "pooled" else rep
                base = 1e5 * (k + 1) + 1e4 * li + 1e3 * r
                plus = base + canonical + 1
                minus_canonical = base + 0.5 + canonical + 1
                shift = 0 if library == "rna" else RIBOTISH_MINUS_SHIFT
                minus = np.zeros(n)
                minus[shift:] = minus_canonical[: n - shift]
                prefix = track_prefix(root, condition, library, rep)
                _write(prefix.with_name(prefix.name + ".plus.bw"), {"chr7": plus, "chr1": plus})
                _write(prefix.with_name(prefix.name + ".minus.bw"), {"chr7": minus, "chr1": minus})
    return root


def _expected(condition, library, rep, positions, strand):
    k, li = CONDITIONS.index(condition), ("rna", "tis", "chx").index(library)
    r = 3 if rep == "pooled" else rep
    base = 1e5 * (k + 1) + 1e4 * li + 1e3 * r + (0.5 if strand == "-" else 0.0)
    return base + positions + 1


def test_transcript_tracks_both_strands_sense_antisense_and_canonical(tmp_path, tracks):
    index = _index(tmp_path)
    reader = BigWigs()
    try:
        for row in index.itertuples():
            unit = txcache.unit_of(row)
            out = txcache.transcript_tracks(reader, tracks, "K562", unit, 2.0)
            other = "-" if unit.strand == "+" else "+"
            pos = unit.positions
            assert out["rna"].shape == (unit.length, 2)
            np.testing.assert_allclose(
                out["rna"][:, 0], 2.0 * _expected("K562", "rna", "pooled", pos, unit.strand)
            )
            np.testing.assert_allclose(
                out["rna"][:, 1], 2.0 * _expected("K562", "rna", "pooled", pos, other)
            )
            for library in ("tis", "chx"):
                for j, rep in enumerate((1, 2, "pooled")):
                    np.testing.assert_allclose(
                        out[library][:, j], _expected("K562", library, rep, pos, unit.strand)
                    )
    finally:
        reader.close()


def test_spliced_values_equal_scorecard_transcript_vector(tmp_path, tracks):
    index = _index(tmp_path)
    reader = BigWigs()
    try:
        for row in index.itertuples():
            unit = txcache.unit_of(row)
            for name, shift in (("plus", 0), ("minus", RIBOTISH_MINUS_SHIFT)):
                path = f"{track_prefix(tracks, 'HeLa', 'chx', 2)}.{name}.bw"
                np.testing.assert_array_equal(
                    txcache.spliced_values(reader, path, unit, shift),
                    transcript_vector(reader, path, unit, shift),
                )
    finally:
        reader.close()


def test_build_shard_tracks_writes_condition_major_arrays(tmp_path, tracks):
    index = _index(tmp_path)
    cache = tmp_path / "cache"
    scales = {"HeLa": 1.0, "K562": 0.5}
    txcache.build_shard_tracks(index, "val_chr7", cache, tracks, CONDITIONS, scales)
    tis = np.load(cache / "shards" / "val_chr7" / "tis.npy", mmap_mode="r")
    rna = np.load(cache / "shards" / "val_chr7" / "rna.npy", mmap_mode="r")
    assert tis.shape == (2, 300, 3) and rna.shape == (2, 300, 2) and tis.dtype == np.float32
    t2 = txcache.unit_of(index.iloc[2])
    np.testing.assert_allclose(tis[1, 150:300, 1], _expected("K562", "tis", 2, t2.positions, "-"))
    np.testing.assert_allclose(
        rna[1, 150:300, 0], 0.5 * _expected("K562", "rna", "pooled", t2.positions, "-")
    )


def test_window_rows_follow_nearest_centre_and_fill_rows_orients_values(tmp_path):
    index = _index(tmp_path)
    windows = _windows()["val"]
    positions, strands = txcache.shard_positions(index, "val_chr7")
    assert strands.tolist() == [1] * 150 + [-1] * 150
    plan = txcache.window_rows(windows, "chr7", positions)
    spans = owned_spans(windows)
    for start, end, rows in plan:
        k = windows.index[windows.start == start][0]
        lo, hi = spans[k]
        assert ((positions[rows] >= lo) & (positions[rows] < hi)).all()
    assert sorted(np.concatenate([r for _, _, r in plan]).tolist()) == list(range(300))

    def compute(start, end):  # value = genomic position, column 2 = window start
        pos = np.arange(start, end, dtype=np.float32)
        return np.stack([pos, np.full_like(pos, start)], axis=1)

    out = txcache.fill_rows(positions, plan, compute, (2,))
    np.testing.assert_array_equal(out[:, 0], positions)
    # T1 [100, 290) is nearest window 0's centre (700); T2's 5' end (689) too, 3' end as well
    centres = (windows.start + windows.end).to_numpy() / 2
    nearest = windows.start.to_numpy()[np.abs(positions[:, None] - centres).argmin(axis=1)]
    np.testing.assert_array_equal(out[:, 1], nearest)


def test_window_rows_raises_when_a_position_is_not_owned():
    windows = pd.DataFrame({"chrom": ["chr7"], "start": [0], "end": [100]})
    with pytest.raises(ValueError, match="owned by no window"):
        txcache.window_rows(windows, "chr7", np.array([5, 150]))


def test_select_strand_picks_sense_channels():
    pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = txcache.select_strand(pred, np.array([1, -1, 1], np.int8))
    np.testing.assert_array_equal(out[:, 0], pred[:, 0][:, [0, 2]])
    np.testing.assert_array_equal(out[:, 1], pred[:, 1][:, [1, 3]])


def test_pca_round_trip_on_low_rank_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(500, 6))
    x = latent @ rng.normal(size=(6, 40)) + rng.normal(size=40)
    pca = txcache.fit_pca(x, dim=6)
    assert pca["components"].shape == (6, 40) and pca["mean"].shape == (40,)
    np.testing.assert_allclose(pca["components"] @ pca["components"].T, np.eye(6), atol=1e-5)
    z = txcache.pca_project(x, pca)
    np.testing.assert_allclose(txcache.pca_reconstruct(z, pca), x, atol=1e-3)
    assert pca["explained_variance"].sum() == pytest.approx(pca["total_variance"], rel=1e-6)
    assert np.all(np.diff(pca["explained_variance"]) <= 0)
    again = txcache.fit_pca(x, dim=6)
    np.testing.assert_array_equal(again["components"], pca["components"])


def test_ribotish_sites_map_anchors_to_mrna_coordinates(tmp_path):
    index = _index(tmp_path)
    val = index[index.split == "val"]
    calls = pd.DataFrame(
        {
            # T1 + start at genomic 130 (mRNA 30); T2 - start: hi=650 -> canonical 649 (mRNA 40);
            # duplicate; an intronic anchor; another strand
            "GenomePos": [
                "chr7:130-260:+",
                "chr7:520-650:-",
                "chr7:130-999:+",
                "chr7:170-260:+",
                "chr7:130-260:-",
            ],
            "StartCodon": ["ATG", "CTG", "ATG", "ATG", "ATG"],
            "TisType": ["Annotated", "Novel", "Annotated", "Novel", "Novel"],
        }
    )
    sites = txcache.ribotish_sites(calls, "K562", val)
    assert sites.values.tolist() == [
        ["K562", "T1", 30, "ATG", "Annotated"],
        ["K562", "T2", 40, "CTG", "Novel"],
    ]


def test_tx_cache_reader_memory_maps_shards(tmp_path, tracks):
    import json

    index = _index(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    index.to_parquet(cache / txcache.INDEX, index=False)
    (cache / txcache.CONFIG).write_text(json.dumps({"conditions": list(CONDITIONS)}))
    txcache.build_shard_tracks(index, "val_chr7", cache, tracks, CONDITIONS, {"HeLa": 1, "K562": 1})
    view = txcache.TxCache(cache)
    assert view.conditions == CONDITIONS and view.sites is None
    assert isinstance(view.array("val_chr7", "tis"), np.memmap)
    assert view.has("val_chr7", "tis") and not view.has("val_chr7", "emb")
    assert view.rows("val").transcript_id.tolist() == ["T1", "T2"]
    view.preload("val", ["tis"])
    tis = view.array("val_chr7", "tis")
    assert not isinstance(tis, np.memmap) and tis.shape == (2, 300, 3)
