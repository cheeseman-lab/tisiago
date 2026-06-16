import numpy as np

from tisiago.scan_eval import grounding_stats


def test_grounding_stats_low_for_confident_negatives():
    p = np.full(1000, 0.01)
    res = grounding_stats(p, threshold=0.38)
    assert res["mean_p"] < 0.05
    assert res["fpr_at_threshold"] == 0.0
    assert res["n"] == 1000


def test_grounding_stats_flags_high_scores():
    p = np.concatenate([np.full(900, 0.01), np.full(100, 0.9)])
    res = grounding_stats(p, threshold=0.38)
    assert res["fpr_at_threshold"] == 0.1  # 100/1000 above threshold
    assert res["p95"] >= 0.9
