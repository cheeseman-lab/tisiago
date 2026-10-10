import json
import math

import pandas as pd
import pytest

from tisiago.seq2func import trial_score
from tisiago.seq2func.trial_score import (
    COLUMNS,
    append_row,
    fit_metrics,
    gate_fit,
    periodicity_metrics,
    reference_dom_agree,
    score_run,
)

CHANNELS = [(lib, s) for lib in ("tis", "chx", "rna") for s in ("+", "-")]


def eval_table(kl, r, condition="K562"):
    return pd.DataFrame(
        [
            {
                "condition": condition,
                "library": lib,
                "strand": s,
                "channel": i,
                "profile_kl": kl[i],
                "log_pearson": r[i],
                "split": "val",
            }
            for i, (lib, s) in enumerate(CHANNELS)
        ]
    )


def ceiling_table():
    rows = [eval_table([1.0] * 6, [0.8] * 6, "K562"), eval_table([9.0] * 6, [0.1] * 6, "HeLa")]
    return pd.concat(rows, ignore_index=True)


def region_row(region, segment, dom=(8, 10), sw_pred=(2, 10), sw_obs=(1, 10), f0=(6, 7)):
    """One regions.tsv row; annot frame-0 pred = f0[0]/10, obs = f0[1]/10."""
    return {
        "region": region,
        "segment": segment,
        "obs_total": 100.0,
        "p3_obs": 0.4,
        "p3_pred": 0.2,
        "hf_obs": 1.0,
        "hf_pred": 0.5,
        "block_kl_num": 1.0,
        "block_kl_den": 10.0,
        "dom_agree": dom[0],
        "dom_blocks": dom[1],
        "switch_pred_within": sw_pred[0],
        "pairs_pred_within": sw_pred[1],
        "switch_obs_within": sw_obs[0],
        "pairs_obs_within": sw_obs[1],
        "switch_pred_junction": 0,
        "pairs_pred_junction": 0,
        "switch_obs_junction": 0,
        "pairs_obs_junction": 0,
        "annot_f0_pred": f0[0],
        "annot_f1_pred": 10 - f0[0],
        "annot_f2_pred": 0.0,
        "annot_f0_obs": f0[1],
        "annot_f1_obs": 10 - f0[1],
        "annot_f2_obs": 0.0,
    }


def regions_table():
    return pd.DataFrame(
        [
            region_row("cds", "all"),
            region_row("cds", "first", dom=(0, 10)),  # not the pooled 'all' segment
            region_row("utr5", "all", dom=(3, 6)),
        ]
    )


def test_fit_metrics_relative_to_matching_ceiling_channels():
    out = fit_metrics(eval_table([2.0] * 6, [0.4] * 5 + [0.7]), ceiling_table())
    assert out["val_profile_kl"] == pytest.approx(2.0)
    assert out["ceiling_profile_kl"] == pytest.approx(1.0)
    assert out["profile_kl_rel"] == pytest.approx(0.5)
    assert out["gene_log_pearson"] == pytest.approx(0.45)
    assert out["log_pearson_rel"] == pytest.approx(0.45 / 0.8)


def test_fit_metrics_requires_ceiling_channels():
    with pytest.raises(ValueError, match="lacks the run's channels"):
        fit_metrics(eval_table([1.0] * 6, [0.5] * 6, "U2OS"), ceiling_table())


def test_periodicity_metrics_and_score():
    out = periodicity_metrics(regions_table(), ref_dom_agree=0.9)
    assert out["dom_agree"] == pytest.approx(0.8)
    assert out["switch_pred_within"] == pytest.approx(0.2)
    assert out["switch_obs_within"] == pytest.approx(0.1)
    assert out["annot_frame0_pred"] == pytest.approx(0.6)
    assert out["annot_frame0_obs"] == pytest.approx(0.7)
    assert out["p3_pred_median"] == pytest.approx(0.2)
    assert out["p3_obs_median"] == pytest.approx(0.4)
    assert out["utr5_dom_agree"] == pytest.approx(0.5)
    expected = (0.8 / 0.9 + (1 - 0.1) + 0.6 / 0.7) / 3
    assert out["periodicity_score"] == pytest.approx(expected)


def test_periodicity_score_terms_are_clipped():
    rows = pd.DataFrame([region_row("cds", "all", dom=(10, 10), f0=(9, 7))])
    out = periodicity_metrics(rows, ref_dom_agree=0.5)  # ratios 2.0 and 9/7 clip to 1
    assert out["periodicity_score"] == pytest.approx((1 + 0.9 + 1) / 3)
    assert math.isnan(out["utr5_dom_agree"])


def test_periodicity_score_nan_without_reference():
    assert math.isnan(periodicity_metrics(regions_table(), math.nan)["periodicity_score"])


@pytest.mark.parametrize(
    ("kl", "r", "expected"),
    [(1.02, 0.495, True), (1.021, 0.5, False), (1.0, 0.4949, False), (0.5, 0.9, True)],
)
def test_gate_fit_thresholds(kl, r, expected):
    base = {"val_profile_kl": 1.0, "gene_log_pearson": 0.5}
    assert gate_fit({"val_profile_kl": kl, "gene_log_pearson": r}, base) is expected
    assert gate_fit({"val_profile_kl": kl, "gene_log_pearson": r}, None) is None


def test_reference_dom_agree_from_markdown_and_tsv(tmp_path):
    md = tmp_path / "summary_val_max100_replicate.md"
    md.write_text(
        "# CHX periodicity\n\n- arm: condition:K562\n\n"
        "| condition | region | stratum | n | dom_agree | dom_blocks |\n"
        "|---|---|---|---|---|---|\n"
        "| K562 | cds | all | 66 | 0.867 | 848 |\n"
        "| K562 | cds | first | 60 | 0.928 | 166 |\n"
        "| K562 | utr5 | all | 5 | nan | 0 |\n"
    )
    assert reference_dom_agree(md) == pytest.approx(0.867)
    tsv = tmp_path / "regions_val_max100_replicate.tsv"
    regions_table().to_csv(tsv, sep="\t", index=False)
    assert reference_dom_agree(tsv) == pytest.approx(0.8)


def make_run(tmp_path, name="ar_x", kl=2.0):
    run = tmp_path / name
    (run / "eval").mkdir(parents=True)
    (run / "periodicity").mkdir()
    args = {
        "arm": "condition:K562",
        "head": "seqconv",
        "frame_weight": 50.0,
        "init_from": "data/p6/runs/k562_base/best",
        "windows_dir": "data/p6/windows_new",
        "out": str(run),
    }
    (run / "run.json").write_text(json.dumps({"args": args, "best": {"step": 7000}}))
    eval_table([kl] * 6, [0.5] * 6).to_csv(run / "eval/condition_k562_val.tsv", sep="\t")
    regions_table().to_csv(run / "periodicity/regions_val_max100.tsv", sep="\t", index=False)
    ref = tmp_path / "ref_regions_replicate.tsv"
    regions_table().to_csv(ref, sep="\t", index=False)
    ceiling = tmp_path / "ceiling_val.tsv"
    ceiling_table().to_csv(ceiling, sep="\t", index=False)
    return run, ceiling, ref


def test_score_run_reads_default_paths(tmp_path):
    run, ceiling, ref = make_run(tmp_path)
    base = {"val_profile_kl": 1.99, "gene_log_pearson": 0.5}
    row = score_run(run, ceiling=ceiling, reference_periodicity=ref, base=base)
    assert list(row) == COLUMNS
    assert row["run"] == "ar_x" and row["best_step"] == 7000
    config = json.loads(row["config"])
    assert config == {
        "arm": "condition:K562",
        "frame_weight": 50.0,
        "head": "seqconv",
        "init_from": "data/p6/runs/k562_base/best",
    }
    assert row["gate_fit"] is True
    assert row["ref_dom_agree"] == pytest.approx(0.8)


def test_score_run_refuses_test_split(tmp_path):
    run, ceiling, ref = make_run(tmp_path)
    eval_table([1.0] * 6, [0.5] * 6).assign(split="test").to_csv(
        run / "eval/condition_k562_val.tsv", sep="\t"
    )
    with pytest.raises(ValueError, match="validation"):
        score_run(run, ceiling=ceiling, reference_periodicity=ref)


def test_append_row_writes_header_once_and_refuses_other_header(tmp_path):
    out = tmp_path / "results.tsv"
    row = dict.fromkeys(COLUMNS, 1.0) | {"run": "a", "gate_fit": None}
    append_row(row, out)
    append_row(row | {"run": "b"}, out)
    table = pd.read_csv(out, sep="\t")
    assert list(table.columns) == COLUMNS
    assert table.run.tolist() == ["a", "b"]
    assert table.gate_fit.isna().all()
    other = tmp_path / "other.tsv"
    other.write_text("run\tscore\n")
    with pytest.raises(ValueError, match="another header"):
        append_row(row, other)


def test_cli_writes_score_json_and_appends(tmp_path, monkeypatch):
    run, ceiling, ref = make_run(tmp_path)
    base_run, *_ = make_run(tmp_path, "k562_base", kl=1.0)
    out = tmp_path / "results.tsv"
    common = ["--ceiling", str(ceiling), "--reference-periodicity", str(ref), "--out", str(out)]
    monkeypatch.setattr("sys.argv", ["trial_score", "--run", str(base_run), *common])
    trial_score.main()
    base_json = base_run / "trial_score.json"
    assert json.loads(base_json.read_text())["gate_fit"] is None
    argv = ["trial_score", "--run", str(run), "--base", str(base_json), *common]
    monkeypatch.setattr("sys.argv", argv)
    trial_score.main()
    table = pd.read_csv(out, sep="\t")
    assert table.run.tolist() == ["k562_base", "ar_x"]
    assert json.loads((run / "trial_score.json").read_text())["gate_fit"] is False


@pytest.mark.parametrize("key", ["val_profile_kl", "gene_log_pearson"])
def test_gate_fit_refuses_null_base_values(key):
    base = {"val_profile_kl": 1.0, "gene_log_pearson": 0.5, key: None}
    with pytest.raises(ValueError, match=f"base score has no finite {key}"):
        gate_fit({"val_profile_kl": 1.0, "gene_log_pearson": 0.5}, base)
