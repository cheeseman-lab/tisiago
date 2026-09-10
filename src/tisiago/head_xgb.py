# src/tisiago/head_xgb.py
"""Gradient-boosted tree (and random-forest) heads for dense TIS classification.

The dense-trained logistic head (FINDINGS §7) tops out at 0.300 recall @ ≤1 FP/tx. Trees can
capture interactions between the AG regional embedding and the Evo2 nucleotide embedding that a
linear boundary misses, and they need no feature scaling. Each fit mirrors
``caller.fit_calibrated_head``: train, then isotonic-calibrate on the held-out val split, so the
returned ``predict`` plugs straight into ``dense_caller.evaluate`` with probabilities in [0, 1].
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression


def _auto_spw(y_train) -> float:
    """scale_pos_weight = negatives / positives (XGBoost/LightGBM imbalance lever)."""
    y = np.asarray(y_train)
    return float((y == 0).sum()) / max(1, int((y == 1).sum()))


def _calibrate(predict_raw, X_val, y_val, model):
    """Isotonic-calibrate raw scores on val; return the standard head dict (scaler=None)."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(predict_raw(X_val), y_val)

    def predict(X):
        return iso.predict(predict_raw(X))

    return {"predict": predict, "predict_raw": predict_raw, "model": model, "scaler": None}


def fit_xgb_head(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=None,
    subsample=0.8,
    colsample_bytree=0.5,
    seed=0,
) -> dict:
    """Train an XGBoost classifier (hist), then isotonic-calibrate on independent val rows."""
    import xgboost as xgb

    if scale_pos_weight is None:
        scale_pos_weight = _auto_spw(y_train)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    # Keep the calibration rows statistically independent from model fitting.
    # If early stopping is needed, its data must be supplied as a separate
    # tuning split rather than reusing X_val here.
    clf.fit(X_train, y_train, verbose=False)

    def predict_raw(X):
        return clf.predict_proba(X)[:, 1]

    return _calibrate(predict_raw, X_val, y_val, clf)


def fit_lgb_head(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=None,
    subsample=0.8,
    colsample_bytree=0.5,
    seed=0,
) -> dict:
    """Train a LightGBM classifier, isotonic-calibrate on val."""
    import lightgbm as lgb

    if scale_pos_weight is None:
        scale_pos_weight = _auto_spw(y_train)
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    def predict_raw(X):
        # Booster.predict avoids sklearn's synthetic feature-name warning for
        # ndarray input and returns the same positive-class probabilities.
        return clf.booster_.predict(X)

    return _calibrate(predict_raw, X_val, y_val, clf)


def fit_rf_head(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_estimators=500,
    max_depth=12,
    seed=0,
) -> dict:
    """Train a balanced random forest, isotonic-calibrate on val.

    Diagnostic only: RF at the full 2M×19.6k scale is memory-prohibitive — fit it on a train
    subsample (see the runbook). If RF beats logistic but XGBoost doesn't, the gain is ensemble
    averaging, not interaction learning.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    def predict_raw(X):
        return clf.predict_proba(X)[:, 1]

    return _calibrate(predict_raw, X_val, y_val, clf)
