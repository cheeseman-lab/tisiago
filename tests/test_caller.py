import numpy as np

from tisiago.caller import recall_at_fp_budget


def test_recall_at_fp_budget_perfect_separation():
    # 2 transcripts, perfectly separable: all positives score above all negatives.
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.85, 0.05, 0.15])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    tx = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    res = recall_at_fp_budget(p, y, tx, budget=1.0)
    # A threshold exists that admits all 4 positives and 0 false positives.
    assert res["recall"] == 1.0
    assert res["fp_per_transcript"] <= 1.0
    assert 0.2 < res["threshold"] <= 0.8


def test_recall_at_fp_budget_respects_budget():
    # One transcript, 1 positive and 4 negatives just below it.
    # Budget 0 forbids any FP -> threshold must sit above the top negative,
    # so recall can still be 1.0 only if the positive outscores every negative.
    p = np.array([0.6, 0.5, 0.4, 0.3, 0.2])
    y = np.array([1, 0, 0, 0, 0])
    tx = np.array(["A", "A", "A", "A", "A"])
    res = recall_at_fp_budget(p, y, tx, budget=0.0)
    assert res["recall"] == 1.0
    assert res["fp_per_transcript"] == 0.0
    assert res["threshold"] > 0.5  # above the highest negative


def test_recall_at_fp_budget_returns_keys():
    p = np.array([0.7, 0.3])
    y = np.array([1, 0])
    tx = np.array(["A", "A"])
    res = recall_at_fp_budget(p, y, tx, budget=1.0)
    assert set(res) == {"recall", "threshold", "fp_per_transcript", "budget"}
