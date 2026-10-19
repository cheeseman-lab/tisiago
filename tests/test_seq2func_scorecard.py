"""Tests for the P6 concordance scorecard (prediction vs replicate agreement, plain units)."""

import json
import sys

import numpy as np
import pandas as pd
import pyBigWig
import pytest

pytest.importorskip("sklearn")  # CPU stage: the conda tisiago env
from tisiago.seq2func import scorecard  # noqa: E402
from tisiago.seq2func.predict_tracks import pooled_prefix  # noqa: E402
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT  # noqa: E402
from tisiago.seq2func.scorecard import (  # noqa: E402
    Unit,
    bin_sums,
    example_genes,
    fit_rna_scalars,
    library_rows,
    load_units,
    parse_sources,
    pearson,
    select_units,
    transcript_vector,
)
from tisiago.seq2func.tracks import track_prefix  # noqa: E402

SIZES = [("chr7", 3000), ("chr1", 3000)]
CONDITIONS = ("HeLa", "K562")

# 1-based GTF coordinates. T1 (+): exons [100,160) [200,290), CDS [130,160) [200,260).
# T2 (-): exons [500,560) [600,690), CDS [520,560) [600,650). T3: not MANE. T4: chr1 (train).
MANE = '; tag "MANE_Select"'
FEATURES = [
    ("chr7", "gene", 101, 290, "+", "G1", None, "ONE", ""),
    ("chr7", "exon", 101, 160, "+", "G1", "T1", "ONE", MANE),
    ("chr7", "exon", 201, 290, "+", "G1", "T1", "ONE", MANE),
    ("chr7", "CDS", 131, 160, "+", "G1", "T1", "ONE", MANE),
    ("chr7", "CDS", 201, 260, "+", "G1", "T1", "ONE", MANE),
    ("chr7", "exon", 101, 290, "+", "G1", "T1b", "ONE", ""),
    ("chr7", "exon", 501, 560, "-", "G2", "T2", "TWO", MANE),
    ("chr7", "exon", 601, 690, "-", "G2", "T2", "TWO", MANE),
    ("chr7", "CDS", 521, 560, "-", "G2", "T2", "TWO", MANE),
    ("chr7", "CDS", 601, 650, "-", "G2", "T2", "TWO", MANE),
    ("chr7", "exon", 1001, 1100, "+", "G3", "T3", "THREE", ""),
    ("chr1", "exon", 101, 190, "+", "G4", "T4", "FOUR", MANE),
    ("chr1", "CDS", 101, 190, "+", "G4", "T4", "FOUR", MANE),
]


def _gtf_text():
    lines = ["#!header"]
    for chrom, kind, s, e, strand, gene, tid, name, tag in FEATURES:
        attrs = f'gene_id "{gene}";' + (f' transcript_id "{tid}";' if tid else "")
        attrs += f' gene_name "{name}"{tag};'
        lines.append("\t".join([chrom, "X", kind, str(s), str(e), ".", strand, ".", attrs]))
    return "\n".join(lines) + "\n"


def _gtf(tmp_path):
    path = tmp_path / "mane.gtf"
    path.write_text(_gtf_text())
    return path


def _write_dense(path, arrays):
    """1-bp bigWig from ``{chrom: dense array over file positions}`` (zeros skipped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(SIZES)
    for chrom, _ in SIZES:
        values = np.asarray(arrays.get(chrom, []), dtype=np.float64)
        keep = np.flatnonzero(values)
        if len(keep):
            bw.addEntries(chrom, keep.tolist(), values=values[keep].tolist(), span=1)
    bw.close()


def _track(prefix, plus=None, minus=None):
    for name, arrays in (("plus", plus or {}), ("minus", minus or {})):
        _write_dense(prefix.with_name(f"{prefix.name}.{name}.bw"), arrays)


def _units(tmp_path):
    return {u.transcript_id: u for u in load_units(_gtf(tmp_path))}


def test_load_units_parses_mane_exons_and_cds_in_transcript_order(tmp_path):
    units = _units(tmp_path)
    assert sorted(units) == ["T1", "T2", "T4"]
    t1, t2 = units["T1"], units["T2"]
    assert (t1.gene_id, t1.gene_name, t1.strand) == ("G1", "ONE", "+")
    assert t1.exons == ((100, 160), (200, 290))
    assert t1.length == 150 and t1.cds_range == (30, 120)
    assert t1.positions[[0, 59, 60, -1]].tolist() == [100, 159, 200, 289]
    # minus: mRNA 5' end is the highest exonic base; CDS starts at 649 (mRNA index 40)
    assert t2.positions[[0, 89, 90, -1]].tolist() == [689, 600, 559, 500]
    assert t2.cds_range == (40, 130)
    assert (t2.start, t2.end) == (500, 690)


def test_transcript_vector_reads_both_strands_spliced_in_transcript_order(tmp_path):
    units = _units(tmp_path)
    canonical = np.arange(3000, dtype=np.float64) + 1  # value = canonical position + 1
    shifted = np.zeros(3000)
    shifted[RIBOTISH_MINUS_SHIFT:] = canonical[:-RIBOTISH_MINUS_SHIFT]
    prefix = track_prefix(tmp_path / "tracks", "HeLa", "chx", 2)
    _track(prefix, plus={"chr7": canonical}, minus={"chr7": shifted})
    reader = scorecard.BigWigs()
    try:
        for tid in ("T1", "T2"):
            unit = units[tid]
            name = "plus" if unit.strand == "+" else "minus"
            shift = RIBOTISH_MINUS_SHIFT if unit.strand == "-" else 0
            vec = transcript_vector(reader, f"{prefix}.{name}.bw", unit, shift)
            assert vec.tolist() == (unit.positions + 1).tolist()
        # RNA convention: no minus-strand shift, so the same file reads one base off
        vec = transcript_vector(reader, f"{prefix}.minus.bw", units["T2"], 0)
        assert vec.tolist() == units["T2"].positions.tolist()
    finally:
        reader.close()


def test_pearson_bins_and_constant_vectors():
    assert pearson(np.array([1.0, 2, 3]), np.array([2.0, 4, 7])) == pytest.approx(
        np.corrcoef([1, 2, 3], [2, 4, 7])[0, 1]
    )
    assert pearson(np.zeros(4), np.arange(4.0)) == 0.0
    assert np.isnan(pearson(np.array([1.0]), np.array([2.0])))
    v = np.arange(10.0)
    assert bin_sums(v, 3).tolist() == [3.0, 12.0, 21.0]
    assert bin_sums(v, 3, offset=1).tolist() == [6.0, 15.0, 24.0]
    assert bin_sums(v, 30).size == 0


def _unit(tid, length, cds):
    return Unit(tid, f"G{tid}", tid, "chr7", "+", ((0, length),), ((cds[0], cds[1]),))


def test_library_rows_metric_math_and_yardstick():
    rng = np.random.default_rng(0)
    units = [_unit(str(i), 90, (3, 63)) for i in range(5)]
    rep1 = [rng.poisson(2.0, 90).astype(float) for _ in units]
    rep2 = [rng.poisson(2.0, 90).astype(float) for _ in units]
    rep2[4] = np.zeros(90)
    rep2[4][[5, 40]] = 1.0  # 2 reads: below min_reads at within-transcript scales
    pred = [r + rng.random(90) for r in rep2]
    rows = library_rows(units, {"rep1": rep1, "rep2": rep2, "m": pred}, "chx", min_reads=50)
    by = {(r["source"], r["scale"]): r for r in rows}
    # gene: log1p CDS totals for CHX, across all 5 transcripts
    tot = lambda vs: np.log1p([v[3:63].sum() for v in vs])  # noqa: E731
    gene = by[("m", "gene")]
    assert gene["n"] == 5
    assert gene["value"] == pytest.approx(np.corrcoef(tot(pred), tot(rep2))[0, 1])
    assert gene["yardstick"] == pytest.approx(np.corrcoef(tot(rep1), tot(rep2))[0, 1])
    assert gene["percent"] == pytest.approx(100 * gene["value"] / gene["yardstick"])
    # codon: 3-nt bins in CDS frame (offset 3 % 3 = 0), median over eligible transcripts
    codon = by[("m", "codon")]
    keep = [i for i in range(5) if rep2[i].sum() >= 50]
    assert codon["n"] == len(keep) == 4
    expected = np.median([pearson(bin_sums(pred[i], 3), bin_sums(rep2[i], 3)) for i in keep])
    assert codon["value"] == pytest.approx(expected)
    yard = np.median([pearson(bin_sums(rep1[i], 3), bin_sums(rep2[i], 3)) for i in keep])
    assert codon["yardstick"] == pytest.approx(yard)
    nt = by[("m", "nt")]
    assert nt["value"] == pytest.approx(np.median([pearson(pred[i], rep2[i]) for i in keep]))
    assert by[("m", "bin30")]["n"] == 4
    assert {r["scale"] for r in rows} == {"gene", "bin30", "codon", "nt", "frame"}
    # rows are per prediction source; the replicates only enter as value / yardstick
    assert {r["source"] for r in rows} == {"m"}


def test_library_rows_tis_uses_whole_mrna_totals_and_has_no_frame_row():
    units = [_unit(str(i), 60, (30, 45)) for i in range(3)]
    rep2 = [np.r_[np.full(30, float(i + 1)), np.zeros(30)] for i in range(3)]
    rep1 = [v.copy() for v in rep2]
    rows = library_rows(units, {"rep1": rep1, "rep2": rep2, "m": rep2}, "tis", min_reads=1)
    by = {r["scale"]: r for r in rows}
    assert "frame" not in by
    # CDS totals are all zero (constant -> 0); whole-mRNA totals correlate perfectly
    assert by["gene"]["value"] == pytest.approx(1.0)
    assert by["gene"]["percent"] == pytest.approx(100.0)


def test_frame_agreement_counts_codons_with_enough_reads():
    unit = _unit("a", 30, (0, 30))
    rep2 = np.zeros(30)
    rep2[0::3] = 6.0  # frame 0 dominant, 10 codons with 6 reads
    rep2[4] = 1.0  # codon 1 now has 7 reads, still frame 0
    rep1 = np.zeros(30)
    rep1[0:15:3] = 5.0  # agrees on codons 0-4
    rep1[16:30:3] = 5.0  # frame 1 on codons 5-9
    pred = np.zeros(30)
    pred[0::3] = 1.0
    rows = library_rows(
        [unit], {"rep1": [rep1], "rep2": [rep2], "m": [pred]}, "chx", min_frame_reads=5
    )
    frame = next(r for r in rows if r["scale"] == "frame")
    assert frame["n"] == 10
    assert frame["value"] == pytest.approx(1.0)
    assert frame["yardstick"] == pytest.approx(0.5)
    assert frame["percent"] == pytest.approx(200.0)


def _genes(rows):
    return pd.DataFrame(
        rows, columns=["condition", "gene_id", "chrom", "start", "end", "strand", "rna_count"]
    ).assign(expressed=lambda d: d.rna_count >= 2)


def test_select_units_needs_expression_and_a_span_inside_the_windows(tmp_path):
    units = list(_units(tmp_path).values())
    genes = _genes(
        [
            ("HeLa", "G1", "chr7", 100, 290, "+", 10),
            ("HeLa", "G2", "chr7", 500, 690, "-", 1),
            ("K562", "G1", "chr7", 100, 290, "+", 10),
            ("K562", "G2", "chr7", 500, 690, "-", 9),
        ]
    )
    windows = pd.DataFrame({"chrom": ["chr7", "chr7"], "start": [0, 400], "end": [600, 1000]})
    assert [u.transcript_id for u in select_units(units, genes, "HeLa", windows)] == ["T1"]
    assert [u.transcript_id for u in select_units(units, genes, "K562", windows)] == ["T1", "T2"]
    short = pd.DataFrame({"chrom": ["chr7"], "start": [0], "end": [650]})
    assert [u.transcript_id for u in select_units(units, genes, "K562", short)] == ["T1"]


def _rna_and_ribo(tracks, condition, ribo, rna, chrom="chr1"):
    """Flat tracks on [100, 200): ``ribo`` per base for TIS (twice that for CHX), ``rna`` RNA.

    Ribo-seq also has stray counts on [2000, 2100), outside every training window used here.
    """
    dense = np.zeros(3000)
    dense[100:200] = 1.0
    stray = np.zeros(3000)
    stray[2000:2100] = 1e4
    for library, value in (("tis", ribo), ("chx", 2 * ribo), ("rna", rna)):
        extra = 0 if library == "rna" else stray
        _track(
            track_prefix(tracks, condition, library, "pooled"),
            plus={chrom: dense * value + extra},
            minus={chrom: dense * value + extra},
        )


def test_fit_rna_scalars_uses_only_the_fit_conditions_on_training_windows(tmp_path):
    tracks = tmp_path / "tracks"
    _rna_and_ribo(tracks, "HeLa", ribo=1.0, rna=2.0)
    _rna_and_ribo(tracks, "K562", ribo=1.0, rna=4.0)
    _rna_and_ribo(tracks, "U2OS", ribo=1000.0, rna=1.0)  # evaluated / held out: must not leak
    train = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [300]})
    scalars = fit_rna_scalars(tracks, ("HeLa", "K562"), train)
    # every condition's RNA in CPM of its own total: 1e6 over [100, 200) for each condition
    assert scalars["tis"] == pytest.approx((200 + 200) / 2e6)
    assert scalars["chx"] == pytest.approx(2 * scalars["tis"])
    other = fit_rna_scalars(tracks, ("HeLa", "K562", "U2OS"), train)
    assert other["tis"] > 100 * scalars["tis"]
    # no overlap with the training windows: refuse
    with pytest.raises(ValueError, match="no RNA"):
        empty = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [600]})
        fit_rna_scalars(tracks, ("HeLa",), empty)


def test_rna_fit_conditions_exclude_heldout_and_evaluated_condition():
    assert scorecard.fit_conditions_for("HeLa", ("HeLa", "K562", "U2OS")) == ("K562", "U2OS")
    with pytest.raises(ValueError, match="no fit"):
        scorecard.fit_conditions_for("HeLa", ("HeLa",))
    assert scorecard.default_fit_conditions(("RPE1_Async", "RPE1_Que", "RPE1_Sen")) == (
        "HeLa",
        "K562",
        "U2OS",
    )


def test_parse_sources_per_condition_and_shared_dirs():
    out = parse_sources(["base:HeLa=a/{split}", "base:K562=b", "fold=c"], ("HeLa", "K562"))
    assert out == {"base": {"HeLa": "a/{split}", "K562": "b"}, "fold": {"HeLa": "c", "K562": "c"}}
    for bad in ("x", "rna_scaled=d", "rep2=d", "m:Nope=d", "m=d m=e"):
        with pytest.raises(ValueError):
            parse_sources(bad.split(), ("HeLa", "K562"))


def test_example_genes_top_expressed_then_gene_id(tmp_path):
    units = list(_units(tmp_path).values())
    genes = _genes(
        [
            ("HeLa", "G1", "chr7", 100, 290, "+", 10),
            ("HeLa", "G2", "chr7", 500, 690, "-", 10),
            ("HeLa", "G4", "chr1", 100, 190, "+", 50),
        ]
    )
    chosen = example_genes(units, genes, "HeLa", 2)
    assert [u.gene_id for u in chosen] == ["G4", "G1"]


def _scorecard_fixture(tmp_path):
    """Real replicate tracks, RNA, one float prediction (== rep2 for HeLa), windows, GTF."""
    rng = np.random.default_rng(1)
    tracks = tmp_path / "tracks"
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    pd.DataFrame({"chrom": ["chr7"], "start": [0], "end": [1000]}).to_parquet(
        windows_dir / "val.parquet"
    )
    pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1000]}).to_parquet(
        windows_dir / "train.parquet"
    )
    pred_dir = tmp_path / "pred_val"
    for condition in CONDITIONS:
        pooled = {}
        for library in ("tis", "chx", "rna"):
            per_rep = []
            for rep in (1, 2):
                plus = {c: rng.poisson(3.0, 3000).astype(float) for c in ("chr7", "chr1")}
                minus = {c: rng.poisson(3.0, 3000).astype(float) for c in ("chr7", "chr1")}
                _track(track_prefix(tracks, condition, library, rep), plus, minus)
                per_rep.append((plus, minus))
            pooled[library] = per_rep
            _track(
                track_prefix(tracks, condition, library, "pooled"),
                {c: per_rep[0][0][c] + per_rep[1][0][c] for c in ("chr7", "chr1")},
                {c: per_rep[0][1][c] + per_rep[1][1][c] for c in ("chr7", "chr1")},
            )
            if library != "rna":
                plus2, minus2 = per_rep[1]
                prefix = pooled_prefix(pred_dir, condition, library)
                _track(prefix, {"chr7": plus2["chr7"]}, {"chr7": minus2["chr7"]})
        (pred_dir / condition / "predict_tracks.json").write_text(
            json.dumps(
                {
                    "mode": "float",
                    "split": "val",
                    "windows_dir": str(windows_dir),
                    "match_depth": False,
                    "mask_unexpressed": False,
                }
            )
        )
    genes = _genes(
        [
            (c, g, "chr7", s, e, st, 10)
            for c in CONDITIONS
            for g, s, e, st in (("G1", 100, 290, "+"), ("G2", 500, 690, "-"))
        ]
    )
    expression = tmp_path / "expression.parquet"
    genes.to_parquet(expression)
    return tracks, windows_dir, expression, pred_dir


def test_run_scorecard_writes_tsv_markdown_and_plots(tmp_path):
    tracks, windows_dir, expression, pred_dir = _scorecard_fixture(tmp_path)
    out = tmp_path / "out"
    table = scorecard.run_scorecard(
        conditions=CONDITIONS,
        sources={"model": {"HeLa": str(pred_dir), "K562": str(pred_dir)}},
        split="val",
        windows_dir=windows_dir,
        expression=expression,
        tracks_dir=tracks,
        gtf=_gtf(tmp_path),
        out_dir=out,
        heldout=("HeLa",),
        fit_conditions=("K562",),
        min_reads=10,
        example_genes_n=2,
    )
    assert list(table.columns) == list(scorecard.COLUMNS)
    written = pd.read_csv(out / "scorecard.tsv", sep="\t")
    assert list(written.columns) == list(scorecard.COLUMNS)
    assert set(written.source) == {"model", "rna_scaled"}
    assert set(written.scale) == {"gene", "bin30", "codon", "nt", "frame"}
    model = written[(written.source == "model") & (written.scale != "frame")]
    # the prediction is replicate 2 itself
    assert model.value.to_numpy() == pytest.approx(1.0)
    assert (written.loc[written.scale == "frame", "library"] == "chx").all()
    md = (out / "scorecard.md").read_text()
    for scale in ("gene", "bin30", "codon", "nt", "frame"):
        assert f"## {scale}" in md
    assert "rna_scaled" in md and "%)" in md
    meta = json.loads((out / "scorecard.json").read_text())
    # HeLa held out: its scalar is fit on K562 only; K562 (evaluated) cannot fit on itself
    assert meta["rna_scaled"]["HeLa"]["fit_conditions"] == ["K562"]
    assert meta["rna_scaled"]["K562"] is None
    assert not ((written.condition == "K562") & (written.source == "rna_scaled")).any()
    plots = sorted(p.name for p in (out / "plots").glob("*.png"))
    assert len(plots) == 4 and all(p.startswith(("HeLa_", "K562_")) for p in plots)


def test_cli_refuses_test_split_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scorecard", "--split", "test", "--gtf", "x"])
    with pytest.raises(SystemExit):
        scorecard.main()
