import io

import pandas as pd
import pytest

from tisiago.truth import load_truth, summarize_truth, validate_truth
from tisiago.truth.schema import CODON_CLASSES, REGIONS, TRUTH_COLUMNS


def _transcripts() -> pd.DataFrame:
    """Release ``transcripts`` columns: one + and one - transcript, exons [100,110),[200,220)."""
    return pd.DataFrame(
        {
            "Tid": ["TP", "TM"],
            "chrom": ["chr7", "chr1"],
            "strand": ["+", "-"],
            "exon_starts": ["100,200", "100,200"],
            "exon_ends": ["110,220", "110,220"],
            "cds_start": [105, 105],
            "cds_end": [215, 215],
        }
    )


def _starts() -> pd.DataFrame:
    """Release ``starts`` rows; ``Locus`` is the half-open end on -, ``Start`` is tx_pos."""
    return pd.DataFrame(
        {
            "line": ["K562", "K562", "K562", "HeLa", "RPE1_Que", "K562"],
            "Tid": ["TP", "TM", "TM", "TP", "TP", "TP"],
            "Symbol": ["GP", "GM", "GM", "GP", "GP", "GP"],
            "Chromosome": ["chr7", "chr1", "chr1", "chr7", "chr7", "chr7"],
            "Strand": ["+", "-", "-", "+", "+", "+"],
            # +105 annotated; -214 annotated (Locus 215); -219 utr5 (Locus 220);
            # +101 utr5; +201 in-frame; imputed row
            "Locus": [105, 215, 220, 101, 201, 102],
            "Start": [5, 5, 0, 1, 11, 2],
            "StartCodon": ["ATG", "ATG", "CTG", "AGG", "AAG", "GTG"],
            "rep1_called": [True, True, False, True, pd.NA, False],
            "rep2_called": [True, True, True, True, pd.NA, False],
            "Imputed": [False, False, False, False, False, True],
            "TisType": ["Annotated"] * 6,
            "region": ["junk"] * 6,
            "codon_class": ["junk"] * 6,
        }
    )


def test_load_truth_from_release_columns():
    buf = io.StringIO(_starts().to_csv(sep="\t", index=False))
    truth = load_truth(buf, _transcripts())
    assert list(truth.columns) == list(TRUTH_COLUMNS)
    assert len(truth) == 5  # the imputed row is dropped
    assert truth.gpos.tolist() == [105, 214, 219, 101, 201]
    assert truth.tx_pos.tolist() == [5, 5, 0, 1, 11]
    assert truth.codon_class.tolist() == ["AUG", "AUG", "other_near_cognate", "AGG", "AAG"]
    assert truth.region.tolist() == ["annotated", "annotated", "utr5", "utr5", "cds_inframe"]
    assert truth.reproducible.tolist() == [True, True, False, True, False]
    assert truth.rep1.isna().tolist() == [False, False, False, False, True]
    assert truth.pause.isna().all()
    assert truth.line.tolist() == ["K562", "K562", "K562", "HeLa", "RPE1"]
    assert truth.split.tolist() == ["val", "train", "train", "val", "val"]
    assert truth.gene.tolist() == ["GP", "GM", "GM", "GP", "GP"]


def test_load_truth_custom_column_map_with_gpos_and_pause():
    df = pd.DataFrame(
        {
            "cond": ["K562"],
            "tid": ["TM"],
            "chr": ["chr1"],
            "str": ["-"],
            "pos0": [214],
            "codon": ["AUG"],
            "r1": [1],
            "r2": [0],
            "is_pause": ["true"],
        }
    )
    cmap = {
        "cond": "condition",
        "tid": "transcript_id",
        "chr": "chrom",
        "str": "strand",
        "pos0": "gpos",
        "codon": "codon",
        "r1": "rep1",
        "r2": "rep2",
        "is_pause": "pause",
    }
    ann = _transcripts().rename(columns={"Tid": "transcript_id"})
    amap = {c: c for c in ann.columns}
    truth = load_truth(df, ann, column_map=cmap, annotation_column_map=amap)
    assert truth.loc[0, "codon"] == "ATG"
    assert truth.loc[0, "region"] == "annotated"
    assert bool(truth.loc[0, "pause"]) is True
    assert not truth.loc[0, "reproducible"]
    assert truth.loc[0, "gene"] == "TM"  # no gene column: falls back to transcript


def test_load_truth_rejects_tx_pos_disagreeing_with_annotation():
    starts = _starts()
    starts.loc[1, "Start"] = 6  # one-off on the minus strand
    with pytest.raises(ValueError, match="tx_pos"):
        load_truth(starts, _transcripts())


def test_load_truth_rejects_duplicate_starts():
    starts = pd.concat([_starts().iloc[[0]], _starts().iloc[[0]]])
    with pytest.raises(ValueError, match="share a"):
        load_truth(starts, _transcripts())


def test_load_truth_rejects_bad_codon_and_missing_columns():
    starts = _starts()
    starts.loc[0, "StartCodon"] = "ANG"
    with pytest.raises(ValueError, match="invalid codon"):
        load_truth(starts, _transcripts())
    with pytest.raises(ValueError, match="missing columns"):
        load_truth(_starts().drop(columns=["rep2_called"]), _transcripts())
    with pytest.raises(ValueError, match="gpos or locus"):
        load_truth(_starts().drop(columns=["Locus"]), _transcripts())


def test_validate_truth_catches_tampering():
    truth = load_truth(_starts(), _transcripts())
    validate_truth(truth)
    with pytest.raises(ValueError, match="missing"):
        validate_truth(truth.drop(columns=["region"]))
    with pytest.raises(ValueError, match="codon_class"):
        validate_truth(truth.assign(codon_class="AUG"))
    with pytest.raises(ValueError, match="reproducible"):
        validate_truth(truth.assign(reproducible=True))
    with pytest.raises(ValueError, match="region"):
        validate_truth(truth.assign(region="cds"))
    with pytest.raises(ValueError, match="strands"):
        validate_truth(truth.assign(strand="."))


def test_summarize_truth_counts():
    truth = load_truth(_starts(), _transcripts())
    summary = summarize_truth(truth)
    assert len(summary) == 3 * len(CODON_CLASSES) * len(REGIONS) * 2
    assert summary.n.sum() == len(truth)
    k562 = summary[(summary.condition == "K562") & (summary.n > 0)]
    got = set(zip(k562.codon_class, k562.region, k562.reproducible, k562.n, strict=True))
    assert got == {("AUG", "annotated", True, 2), ("other_near_cognate", "utr5", False, 1)}
