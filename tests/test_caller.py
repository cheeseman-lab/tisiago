import numpy as np

from tisiago.caller import fit_calibrated_head, recall_at_fp_budget, reliability


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


def test_reliability_perfectly_calibrated():
    # p exactly equals empirical frequency in each bin -> low Brier, small gap.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=20000) < p).astype(int)  # P(y=1) = p by construction
    res = reliability(p, y, n_bins=10)
    assert res["brier"] < 0.20
    assert res["max_gap"] < 0.05  # |confidence - accuracy| per bin
    assert len(res["bin_confidence"]) == len(res["bin_accuracy"]) == 10


def test_reliability_overconfident_has_large_gap():
    # Always predict 0.99 but only half are positive -> big calibration gap.
    p = np.full(1000, 0.99)
    y = np.array([1, 0] * 500)
    res = reliability(p, y, n_bins=10)
    assert res["brier"] > 0.4
    assert res["max_gap"] > 0.4


def test_fit_calibrated_head_improves_brier_on_separable_data():
    # Two well-separated Gaussian blobs in 5-D. Raw logistic is already decent;
    # isotonic calibration should not worsen Brier on a held-out calibration split.
    rng = np.random.default_rng(0)
    d = 5

    def blob(center, n):
        return rng.normal(center, 1.0, size=(n, d))

    Xtr = np.vstack([blob(0, 400), blob(3, 400)])
    ytr = np.array([0] * 400 + [1] * 400)
    Xcal = np.vstack([blob(0, 200), blob(3, 200)])
    ycal = np.array([0] * 200 + [1] * 200)
    Xte = np.vstack([blob(0, 200), blob(3, 200)])
    yte = np.array([0] * 200 + [1] * 200)

    head = fit_calibrated_head(Xtr, ytr, Xcal, ycal)
    p_raw = head["predict_raw"](Xte)
    p_cal = head["predict"](Xte)
    assert p_raw.shape == (400,)
    assert p_cal.shape == (400,)
    assert ((p_cal >= 0) & (p_cal <= 1)).all()
    # On separable data both are good; calibrated Brier should be close or better.
    from sklearn.metrics import brier_score_loss

    assert brier_score_loss(yte, p_cal) <= brier_score_loss(yte, p_raw) + 0.05
