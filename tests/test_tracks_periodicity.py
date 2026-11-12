"""Tests for the CHX periodicity diagnostic (synthetic arrays, no model)."""

import numpy as np
import pandas as pd
import pytest

from tests.test_tracks_evaluate import WINDOWS, world  # noqa: F401
from tisiago.tracks import periodicity
from tisiago.tracks.periodicity import (
    PeriodicityAccumulator,
    block_phase_kl,
    block_ranks,
    dominant_phases,
    expression_tertiles,
    high_freq_power,
    load_mane_utr5,
    markdown_table,
    period3_power,
    phase_counts,
    region_positions,
    region_table,
    segment_metrics,
    summarize,
    summary_markdown,
    switch_counts,
)
from tisiago.tracks.targets import channel_layout
from tisiago.tracks.track_io import track_prefix, write_bigwigs


def _periodic(length=300, phase=0, weights=(6.0, 1.0, 1.0)):
    out = np.zeros(length)
    for k, w in enumerate(weights):
        out[(phase + k) % 3 :: 3] = w
    return out


def test_region_positions_plus_and_minus():
    pos, ranks = region_positions("+", [(20, 23), (10, 12)])
    assert pos.tolist() == [10, 11, 20, 21, 22]
    assert ranks.tolist() == [0, 0, 1, 1, 1]
    pos, ranks = region_positions("-", [(10, 12), (20, 23)])
    assert pos.tolist() == [22, 21, 20, 11, 10]
    assert ranks.tolist() == [0, 0, 0, 1, 1]


def test_phase_counts_trims_and_pools():
    counts = phase_counts(np.arange(14), block=2)
    assert counts.tolist() == [[0 + 3, 1 + 4, 2 + 5], [6 + 9, 7 + 10, 8 + 11]]


def test_block_phase_kl_zero_when_composition_matches_and_positive_when_shifted():
    obs = _periodic()
    num, den = block_phase_kl(3 * obs, obs, 10)
    assert num == pytest.approx(0.0, abs=1e-9) and den == obs.sum()
    num, _ = block_phase_kl(_periodic(phase=1), obs, 10)
    assert num > 0
    assert block_phase_kl(obs, np.zeros_like(obs), 10) == (0.0, 0.0)


def test_period3_power_periodic_vs_flat():
    assert period3_power(_periodic(weights=(1.0, 0.0, 0.0))) == pytest.approx(1.0)
    assert period3_power(_periodic()) == pytest.approx(1.0)
    assert period3_power(np.ones(300)) == 0.0
    rng = np.random.default_rng(0)
    assert period3_power(1.0 + 0.01 * rng.standard_normal(300)) < 0.05
    assert np.isnan(period3_power(np.ones(4)))


def test_high_freq_power_separates_smooth_from_periodic():
    x = np.arange(300)
    smooth = np.exp(-0.5 * ((x - 150) / 40) ** 2)
    assert high_freq_power(smooth) < 1e-6
    assert high_freq_power(_periodic()) > 100 * high_freq_power(smooth)
    assert np.isnan(high_freq_power(np.zeros(300)))


def test_dominant_phases_marks_ties_and_empty_blocks():
    values = np.concatenate([_periodic(30, phase=2), np.zeros(30), np.ones(30)])
    assert dominant_phases(values, 10).tolist() == [2, -1, -1]


def test_switch_counts_split_within_and_across_junctions():
    # blocks of one codon; exon ranks: blocks 0-1 exon 0, block 2 spans 0->1, blocks 3-4 exon 1
    ranks = np.array([0] * 7 + [1] * 8)
    first, last = block_ranks(ranks, 1)
    assert first.tolist() == [0, 0, 0, 1, 1] and last.tolist() == [0, 0, 1, 1, 1]
    dominant = np.array([0, 0, 1, -1, 2])
    # pairs (0,1) within no switch; (1,2) across switch; (2,4) across switch
    assert switch_counts(dominant, first, last) == (0, 1, 2, 2)


def test_segment_metrics_periodic_prediction_matches():
    obs = _periodic()
    row = segment_metrics(2 * obs, obs, np.zeros(300, int), block=10, cds_offset=0)
    assert row["block_kl_num"] == pytest.approx(0.0, abs=1e-9)
    assert row["dom_agree"] == row["dom_blocks"] == 10
    assert row["p3_obs"] == pytest.approx(row["p3_pred"])
    frame0 = row["annot_f0_obs"] / sum(row[f"annot_f{f}_obs"] for f in range(3))
    assert frame0 == pytest.approx(0.75)
    assert row["switch_obs_within"] == 0 and row["pairs_obs_within"] == 9


def test_segment_metrics_flat_prediction_has_no_period3_and_chance_frame():
    obs = _periodic()
    row = segment_metrics(np.ones(300), obs, np.zeros(300, int), block=10, cds_offset=0)
    assert row["p3_pred"] == 0.0
    assert row["dom_blocks"] == 0  # every flat block is a three-way tie
    frame0 = row["annot_f0_pred"] / sum(row[f"annot_f{f}_pred"] for f in range(3))
    assert frame0 == pytest.approx(1 / 3)
    assert row["block_kl_num"] > 0


def test_segment_metrics_annotation_offset_and_utr():
    obs = _periodic()
    row = segment_metrics(obs, obs, np.zeros(300, int), cds_offset=1)
    assert row["annot_f1_obs"] == pytest.approx(6.0 * 100)  # segment index 0 = CDS frame 1
    utr = segment_metrics(obs, obs, np.zeros(300, int), cds_offset=None)
    assert np.isnan(utr["annot_f0_obs"])


GTF_LINES = [
    # + strand: two 5'UTR pieces, CDS in two exons, a 3'UTR
    ("chr7", "UTR", 11, 20, "+", "T1", "G1"),
    ("chr7", "UTR", 31, 35, "+", "T1", "G1"),
    ("chr7", "CDS", 36, 50, "+", "T1", "G1"),
    ("chr7", "CDS", 61, 75, "+", "T1", "G1"),
    ("chr7", "UTR", 76, 90, "+", "T1", "G1"),
    # - strand: 5'UTR is downstream in genome coordinates
    ("chr7", "UTR", 101, 110, "-", "T2", "G2"),
    ("chr7", "CDS", 111, 140, "-", "T2", "G2"),
    ("chr7", "UTR", 141, 150, "-", "T2", "G2"),
]


def _gtf(tmp_path):
    path = tmp_path / "mane.gtf"
    lines = ["#header"]
    for chrom, feature, start, end, strand, tid, gid in GTF_LINES:
        attrs = f'gene_id "{gid}"; transcript_id "{tid}"; tag "MANE_Select";'
        fields = [chrom, "X", feature, str(start), str(end), ".", strand, ".", attrs]
        lines.append("\t".join(fields))
    untagged = 'gene_id "G9"; transcript_id "T9";'
    lines.append("\t".join(["chr7", "X", "UTR", "1", "5", ".", "+", ".", untagged]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_mane_utr5_keeps_upstream_pieces_per_strand(tmp_path):
    utr, genes = load_mane_utr5(_gtf(tmp_path))
    assert genes == ["G1", "G2"]
    assert utr[0] == ("chr7", "+", [(10, 20), (30, 35)])
    assert utr[1] == ("chr7", "-", [(140, 150)])


def _genes():
    return pd.DataFrame(
        {
            "condition": "HeLa",
            "gene_id": ["G1", "G2", "G3"],
            "chrom": "chr7",
            "start": [0, 100, 200],
            "end": [90, 150, 260],
            "strand": ["+", "-", "+"],
            "rna_count": [50, 10, 0],
            "expressed": [True, True, False],
        }
    )


def _regions():
    cds = [
        ("chr7", "+", [(30, 60), (90, 120)]),  # two exons
        ("chr7", "-", [(150, 180)]),  # single exon
        ("chr7", "+", [(210, 240)]),  # unexpressed gene
        ("chr7", "+", [(290, 330)]),  # straddles both windows -> skipped
    ]
    utr = [("chr7", "+", [(0, 30)]), ("chr7", "-", [(180, 210)])]
    return pd.concat(
        [
            region_table(cds, "cds", ["G1", "G2", "G3", "G1"]),
            region_table(utr, "utr5", ["G1", "G2"]),
        ],
        ignore_index=True,
    )


def _tracks(layout, size=300):
    obs = np.zeros((size, len(layout)))
    plus = layout.index(("HeLa", "chx", "+"))
    minus = layout.index(("HeLa", "chx", "-"))
    obs[30:60, plus] = _periodic(30)
    obs[90:120, plus] = _periodic(30, phase=0)
    obs[150:180, minus] = _periodic(30)[::-1]
    obs[210:240, plus] = _periodic(30)
    return obs, plus, minus


def test_accumulator_rows_segments_and_filters():
    layout = channel_layout(["HeLa"])
    windows = pd.DataFrame({"chrom": "chr7", "start": [0, 300], "end": [300, 600]})
    acc = PeriodicityAccumulator(layout, _genes(), windows, _regions(), block=2)
    obs, plus, minus = _tracks(layout)
    acc.add(0, obs * 2, obs)
    acc.add(1, np.zeros((300, len(layout))), np.zeros((300, len(layout))))
    table = acc.table()
    assert (acc.not_in_one_window, acc.outside_windows) == (1, 0)
    assert set(table.gene_id) == {"G1", "G2"}  # G3 unexpressed
    cds1 = table[(table.region == "cds") & (table.gene_id == "G1")]
    assert cds1.segment.tolist() == ["all", "first", "rest"]
    assert (cds1.channel == plus).all()
    assert cds1.set_index("segment").length.to_dict() == {"all": 60, "first": 30, "rest": 30}
    cds2 = table[(table.region == "cds") & (table.gene_id == "G2")]
    assert cds2.segment.tolist() == ["all", "single"] and (cds2.channel == minus).all()
    assert (table.region == "utr5").sum() == 0  # no observed UTR counts
    assert table.block_kl_num.abs().max() == pytest.approx(0.0, abs=1e-9)
    all_rows = table[table.segment == "all"]
    frame0 = (
        all_rows.annot_f0_obs.sum()
        / all_rows[[f"annot_f{f}_obs" for f in range(3)]].sum(axis=1).sum()
    )
    assert frame0 == pytest.approx(0.75)


def test_accumulator_keeps_utr_with_counts():
    layout = channel_layout(["HeLa"])
    windows = pd.DataFrame({"chrom": "chr7", "start": [0], "end": [300]})
    acc = PeriodicityAccumulator(layout, _genes(), windows, _regions(), block=2, min_utr_counts=5)
    obs, plus, _ = _tracks(layout)
    obs[0:30, plus] = _periodic(30, phase=1)
    acc.add(0, np.ones_like(obs), obs)
    utr = acc.table().query("region == 'utr5'")
    assert utr.gene_id.tolist() == ["G1"] and utr.segment.tolist() == ["all"]
    assert np.isnan(utr.annot_f0_obs.iloc[0])
    assert utr.p3_pred.iloc[0] == 0.0 and utr.p3_obs.iloc[0] == pytest.approx(1.0)


def test_expression_tertiles_and_summary():
    table = pd.DataFrame(
        {
            "condition": "HeLa",
            "gene_id": [f"G{i}" for i in range(6)] * 2,
            "rna_count": list(range(6)) * 2,
            "region": "cds",
            "segment": ["all"] * 6 + ["single"] * 6,
        }
    )
    tertile = expression_tertiles(table)
    assert tertile.iloc[:6].tolist() == [1, 1, 2, 2, 3, 3]
    obs = _periodic(60)
    metrics = segment_metrics(np.ones(60), obs, np.zeros(60, int), block=2, cds_offset=0)
    full = table.assign(expr_tertile=tertile, **{k: v for k, v in metrics.items()})
    summary = summarize(full)
    assert set(summary.stratum) == {"all", "single", *(f"expr_tertile_{t}" for t in (1, 2, 3))}
    row = summary.set_index("stratum").loc["all"]
    assert row.annot_frame0_pred == pytest.approx(1 / 3)
    assert row.annot_frame0_obs == pytest.approx(0.75)
    assert row.p3_pred_median == 0.0
    text = summary_markdown(summary, {"run_name": "x"})
    assert "| condition |" in text and "expr_tertile_3" in text
    assert markdown_table(pd.DataFrame()) == "(no regions)"


def test_cli_requires_arm_without_run_json(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["periodicity", "--checkpoint", str(tmp_path / "best")])
    with pytest.raises(SystemExit) as exc:
        periodicity.main()
    assert exc.value.code != 0


def test_run_periodicity_oracle_and_flat_models(world):  # noqa: F811
    regions = region_table(world["cds"], "cds", world["cds_genes"])
    kwargs = dict(
        fasta=world["fasta"],
        tracks_dir=world["tracks_dir"],
        genes=world["genes"],
        regions=regions,
        block=2,
    )
    oracle, skipped = periodicity.run_periodicity(
        lambda b: b["targets"], "condition:HeLa", WINDOWS, **kwargs
    )
    assert skipped == {"not_in_one_window": 1, "outside_windows": 0}  # + CDS [40, 160)
    full = oracle[oracle.segment == "all"].set_index("strand")
    assert full.loc["+", "annot_f0_obs"] == full.loc["+", "annot_f0_pred"] == 30
    assert full.loc["+", "p3_obs"] == pytest.approx(1.0)
    assert oracle.block_kl_num.abs().max() == pytest.approx(0.0, abs=1e-9)
    flat, _ = periodicity.run_periodicity(
        lambda b: np.ones_like(b["targets"]), "condition:HeLa", WINDOWS, **kwargs
    )
    assert (flat.p3_pred == 0.0).all() and (flat.block_kl_num > 0).all()


def test_dominant_phases_count_floor_applies_to_block_total():
    values = np.concatenate([_periodic(30, weights=(2.0, 1.0, 0.0)), _periodic(30)])
    # block totals 30 and 80
    assert dominant_phases(values, 10, min_counts=30).tolist() == [0, 0]
    assert dominant_phases(values, 10, min_counts=31).tolist() == [-1, 0]


def test_segment_metrics_min_block_counts_gates_dominant_metrics():
    obs = np.concatenate([_periodic(150, weights=(0.2, 0.0, 0.0)), _periodic(150)])
    pred = np.concatenate([_periodic(150, phase=1), _periodic(150)])
    ranks = np.zeros(300, int)
    loose = segment_metrics(pred, obs, ranks, block=10, min_block_counts=1)
    strict = segment_metrics(pred, obs, ranks, block=10, min_block_counts=10)
    # first 5 blocks hold 2 counts each; the last 5 hold 80
    assert (loose["dom_obs_blocks"], loose["dom_blocks"], loose["dom_agree"]) == (10, 10, 5)
    assert (strict["dom_obs_blocks"], strict["dom_blocks"], strict["dom_agree"]) == (5, 5, 5)
    assert strict["pairs_obs_within"] == 4 and strict["pairs_pred_within"] == 4
    assert strict["min_block_counts"] == 10
    # the block KL is not gated (it is counts-weighted)
    assert strict["block_kl_num"] == loose["block_kl_num"]


def test_accumulator_counts_excluded_and_outside_separately():
    layout = channel_layout(["HeLa"])
    windows = pd.DataFrame({"chrom": "chr7", "start": [0], "end": [300]})
    regions = _regions()
    far = region_table([("chr7", "+", [(900, 930)])], "cds", ["G1"])
    regions = pd.concat([regions, far], ignore_index=True)
    acc = PeriodicityAccumulator(layout, _genes(), windows, regions)
    assert acc.not_in_one_window == 1  # [290, 330) straddles the window end
    assert acc.outside_windows == 1
    assert acc.min_utr_counts == 10


def _write_reps(tracks, rep1, rep2):
    from collections import Counter

    filler = {("chr7", s): Counter({0: 1}) for s in "+-"}
    for rep, chx in ((1, rep1), (2, rep2)):
        for library in ("tis", "chx", "rna"):
            counts = {**filler, **chx} if library == "chx" else filler
            write_bigwigs(counts, [("chr7", 200)], track_prefix(tracks, "HeLa", library, rep))


def test_run_reference_replicate_scores_rep2_against_rep1(world, tmp_path):  # noqa: F811
    from collections import Counter

    tracks = tmp_path / "rep_tracks"
    plus = Counter({p: 12 for p in range(60, 90, 3)})
    _write_reps(tracks, {("chr7", "+"): plus}, {("chr7", "+"): plus})
    regions = region_table(world["cds"], "cds", world["cds_genes"])
    table, counts = periodicity.run_reference(
        "condition:HeLa",
        WINDOWS,
        tracks_dir=tracks,
        genes=world["genes"],
        regions=regions,
        block=2,
    )
    assert counts == {"not_in_one_window": 1, "outside_windows": 0}
    row = table[(table.segment == "all") & (table.strand == "+")].iloc[0]
    assert row.block_kl_num == pytest.approx(0.0, abs=1e-9)
    assert row.dom_agree == row.dom_blocks == 5
    assert row.p3_obs == pytest.approx(row.p3_pred)

    shifted = Counter({p + 1: 12 for p in range(60, 90, 3)})
    _write_reps(tracks, {("chr7", "+"): plus}, {("chr7", "+"): shifted})
    table, _ = periodicity.run_reference(
        "condition:HeLa",
        WINDOWS,
        tracks_dir=tracks,
        genes=world["genes"],
        regions=regions,
        block=2,
    )
    row = table[(table.segment == "all") & (table.strand == "+")].iloc[0]
    assert row.dom_agree == 0 and row.block_kl_num > 0


def test_output_paths_separate_reference_rows():
    from pathlib import Path

    model = periodicity.output_paths(Path("o"), "run", "val", None, None)
    ref = periodicity.output_paths(Path("o"), "run", "val", 100, "replicate")
    assert model == (Path("o/run/regions_val.tsv"), Path("o/run/summary_val.md"))
    assert ref == (
        Path("o/run/regions_val_max100_replicate.tsv"),
        Path("o/run/summary_val_max100_replicate.md"),
    )


def test_cli_reference_needs_no_checkpoint_but_needs_arm(monkeypatch):
    monkeypatch.setattr("sys.argv", ["periodicity", "--reference", "replicate"])
    with pytest.raises(SystemExit) as exc:
        periodicity.main()
    assert exc.value.code != 0
    args = periodicity.build_parser().parse_args(["--reference", "replicate", "--arm", "x"])
    assert args.checkpoint is None and args.min_block_counts == 10 and args.min_utr_counts == 10
