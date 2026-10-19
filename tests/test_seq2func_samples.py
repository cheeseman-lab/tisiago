"""Tests for the aTIS sample table and Ribo-TISH offset files."""

import pandas as pd
import pytest

from tisiago.seq2func.samples import (
    CELL_LINE_GROUPS,
    CONDITIONS,
    condition_name,
    load_samples,
    parse_para,
)


def test_condition_name_maps_rpe1_states():
    assert condition_name("RPE1", "Quiescent") == "RPE1_Que"
    assert condition_name("RPE1", "Senescent") == "RPE1_Sen"
    assert condition_name("RPE1", "Async") == "RPE1_Async"
    assert condition_name("HeLa", "") == "HeLa"


def test_cell_line_groups_partition_the_conditions():
    assert CELL_LINE_GROUPS == {
        "HeLa": ("HeLa",),
        "K562": ("K562",),
        "U2OS": ("U2OS",),
        "RPE1": ("RPE1_Async", "RPE1_Que", "RPE1_Sen"),
    }
    flat = [c for group in CELL_LINE_GROUPS.values() for c in group]
    assert sorted(flat) == sorted(CONDITIONS) and len(flat) == len(set(flat))


def test_parse_para_splits_m0_table(tmp_path):
    path = tmp_path / "x.bam.para.py"
    path.write_text("offdict = {32: 13, 33: 13, 'm0': {}}\n")
    offsets, m0 = parse_para(path)
    assert offsets == {32: 13, 33: 13}
    assert m0 == {}


def test_parse_para_without_m0(tmp_path):
    path = tmp_path / "x.bam.para.py"
    path.write_text("offdict = {2: 1}\n")
    assert parse_para(path) == ({2: 1}, None)


def _sheet(tmp_path):
    rows = []
    lines = [
        ("HeLa", ""),
        ("K562", ""),
        ("U2OS", ""),
        ("RPE1", "Async"),
        ("RPE1", "Quiescent"),
        ("RPE1", "Senescent"),
    ]
    for cell_line, condition in lines:
        for treatment in ("TIS", "CHX"):
            for kind in ("RPF", "Input"):
                for rep in (1, 2):
                    sid = f"{cell_line}{condition}_{treatment}_{kind}_{rep}"
                    rows.append(
                        {
                            "sample_name": sid,
                            "cell_line": cell_line,
                            "condition": condition,
                            "treatment": treatment,
                            "rep": rep,
                            "type": kind,
                            "sample_id": sid,
                            "bam_path": f"ribosome_profiling/bam/{sid}.bam",
                            "bedgraph_path": "",
                        }
                    )
    rows.append(
        {
            "sample_name": "Harr_iPSC",
            "cell_line": "iPSC",
            "condition": "",
            "treatment": "Harringtonine",
            "rep": "",
            "type": "RPF",
            "sample_id": "HarrIPSC",
            "bam_path": "",
            "bedgraph_path": "x.bedgraph",
        }
    )
    sheet_dir = tmp_path / "ribosome_profiling"
    sheet_dir.mkdir()
    pd.DataFrame(rows).to_csv(sheet_dir / "manifest.csv", index=False)
    return tmp_path


def test_load_samples_keeps_tis_chx_and_tis_input(tmp_path):
    samples = load_samples(_sheet(tmp_path))
    assert len(samples) == 36
    assert set(samples.condition) == set(CONDITIONS)
    assert set(samples.library) == {"tis", "chx", "rna"}
    rna = samples[samples.library == "rna"]
    assert rna.para.isna().all()
    assert all(str(p).endswith(".bam.para.py") for p in samples[samples.library != "rna"].para)


def test_load_samples_rejects_incomplete_sheet(tmp_path):
    root = _sheet(tmp_path)
    sheet = pd.read_csv(root / "ribosome_profiling" / "manifest.csv")
    sheet.iloc[1:].to_csv(root / "ribosome_profiling" / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="expected 36"):
        load_samples(root)


def test_load_samples_bam_overrides_replace_bam_keep_offsets(tmp_path):
    root = _sheet(tmp_path)
    table = tmp_path / "overrides.tsv"
    table.write_text(
        "sample_id\tbam\nK562_CHX_RPF_1\t/new/718.bam\nK562_TIS_Input_1\t/new/724.bam\n"
    )
    samples = load_samples(root, bam_overrides=table).set_index("sample_id")
    assert str(samples.loc["K562_CHX_RPF_1", "bam"]) == "/new/718.bam"
    assert str(samples.loc["K562_CHX_RPF_1", "para"]).startswith(str(root))
    assert str(samples.loc["K562_TIS_Input_1", "bam"]) == "/new/724.bam"
    assert pd.isna(samples.loc["K562_TIS_Input_1", "para"])
    assert str(samples.loc["HeLa_CHX_RPF_1", "bam"]).startswith(str(root))


def test_load_samples_rejects_unknown_override(tmp_path):
    root = _sheet(tmp_path)
    table = tmp_path / "overrides.tsv"
    table.write_text("sample_id\tbam\nnot_a_sample\t/new/x.bam\n")
    with pytest.raises(ValueError, match="not_a_sample"):
        load_samples(root, bam_overrides=table)
