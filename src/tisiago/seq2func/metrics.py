"""Numpy track metrics: profile KL and gene-level log-count Pearson."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-7


def _xlogy(x, y):
    return np.where(x > 0, x * np.log(np.maximum(y, EPS)), 0.0)


def profile_kl(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel KL of the target profile vs the predicted profile over unmasked positions."""
    pred = np.where(mask, pred.astype(np.float64), 0.0)
    target = np.where(mask, np.maximum(target.astype(np.float64), 0.0), 0.0)
    total_pred = pred.sum(axis=0)
    total_target = target.sum(axis=0)
    p_target = target / np.maximum(total_target, EPS)
    p_pred = pred / np.maximum(total_pred, EPS)
    kl = (_xlogy(p_target, p_target) - _xlogy(p_target, p_pred)).sum(axis=0)
    return np.where(total_target > 0, kl, np.nan)


def gene_totals(track: np.ndarray, genes_in_window: pd.DataFrame, window_start: int) -> np.ndarray:
    """Sum each channel over each gene span (genomic half-open) clipped to the window."""
    S, C = track.shape
    out = np.zeros((len(genes_in_window), C), dtype=np.float64)
    starts = genes_in_window["start"].to_numpy() - window_start
    ends = genes_in_window["end"].to_numpy() - window_start
    for i, (s, e) in enumerate(zip(starts, ends)):
        s, e = max(int(s), 0), min(int(e), S)
        if e > s:
            out[i] = track[s:e].sum(axis=0)
    return out


def log_pearson(pred_totals: np.ndarray, obs_totals: np.ndarray) -> np.ndarray:
    """Per-channel Pearson of log1p totals; NaN for < 3 rows or zero variance."""
    C = pred_totals.shape[1]
    out = np.full(C, np.nan)
    if pred_totals.shape[0] < 3:
        return out
    a, b = np.log1p(pred_totals), np.log1p(obs_totals)
    with np.errstate(all="ignore"):
        for c in range(C):
            x, y = a[:, c] - a[:, c].mean(), b[:, c] - b[:, c].mean()
            denom = np.sqrt((x * x).sum() * (y * y).sum())
            if np.isfinite(denom) and (x * x).sum() > 0 and (y * y).sum() > 0:
                out[c] = (x * y).sum() / denom
    return out
