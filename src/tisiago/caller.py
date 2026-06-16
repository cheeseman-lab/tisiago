"""Calibrated, caller-shaped evaluation of the TIS head (deliverable track).

Where ``eval.py`` / ``resolution.py`` measure *ranking* (AUROC, win-rate), this
module measures whether the head emits a *calibrated probability* and *rejects*
non-starts: reliability + Brier, and recall at a false-positives-per-transcript
budget. Metric functions take plain arrays so the Phase-2 dense scan reuses them
unchanged.

Train on the ``train`` split, calibrate on the held-out ``val`` split (chr7),
evaluate on ``test`` (chr8/chr9). Pure CPU over an assembled store.
"""

from __future__ import annotations

import numpy as np


def recall_at_fp_budget(p, y, transcript_id, budget: float = 1.0) -> dict:
    """Recall at the threshold whose mean false-positives-per-transcript ≤ budget.

    A false positive is a negative codon scored ≥ threshold. The budget is the mean,
    over distinct transcripts, of such false positives. We pick the *lowest* threshold
    (highest recall) that still satisfies the budget.

    Args:
        p: calibrated probabilities, shape [N].
        y: binary labels (1 = called TIS), shape [N].
        transcript_id: per-row transcript id, shape [N].
        budget: maximum allowed mean false positives per transcript.

    Returns:
        dict with ``recall``, ``threshold``, ``fp_per_transcript``, ``budget``.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    tx = np.asarray(transcript_id)
    n_tx = len(np.unique(tx))
    n_pos = int(y.sum())

    # Candidate thresholds: all unique scores plus an +inf sentinel (admit nothing).
    # Sweep from strict (high) to loose (low); each score used as a ">=" cutoff so
    # every distinct operating point is visited.
    candidates = np.concatenate([np.unique(p), [np.inf]])
    candidates.sort()  # ascending
    candidates = candidates[::-1]  # descending: strict -> loose

    best = {
        "recall": 0.0,
        "threshold": float("inf"),
        "fp_per_transcript": 0.0,
        "budget": float(budget),
    }
    for thr in candidates:
        admitted = p >= thr
        fp = admitted & ~y
        # mean FP per transcript across ALL transcripts present (not just those with FP)
        fp_per_tx = fp.sum() / max(1, n_tx)
        if fp_per_tx <= budget:
            recall = (admitted & y).sum() / max(1, n_pos)
            if recall > best["recall"]:  # strict: keep the tightest threshold at each recall level
                best = {
                    "recall": float(recall),
                    "threshold": float(thr),
                    "fp_per_transcript": float(fp_per_tx),
                    "budget": float(budget),
                }
        else:
            break  # thresholds only get looser -> FP only grows
    return best
