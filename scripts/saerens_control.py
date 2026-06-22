"""D1 negative control: Saerens prior-correction of the curated head.

Operon's literature review predicted that recalibrating the curated-trained head to the true
prior recovers *calibration* (Brier) but not *ranking* (AUPRC), because prior-correction is a
monotonic transform — proving the §7 dense-training gain is a negative-*distribution* effect,
not a prior shift. This reads the saved per-head test predictions and confirms it.

Saerens, Latinne & Decaestecker (Neural Computation 14, 2002): given a calibrated posterior p
at training prior pi_tr, the posterior at deployment prior pi_de is the monotonic map below.

Run: python scripts/saerens_control.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

PREDS = {
    "ag (1M)": "data/scan_parts_allsplits/dense_ag_matched_preds.npz",
    "ag7 (1M)": "data/scan_parts_allsplits/dense_ag7_preds.npz",
}
PI_TRAIN = 0.25  # curated set is 3:1 neg:pos -> positive rate 1/4


def saerens(p: np.ndarray, pi_tr: float, pi_de: float) -> np.ndarray:
    """Adjust posteriors from training prior ``pi_tr`` to deployment prior ``pi_de``."""
    a, b = pi_de / pi_tr, (1 - pi_de) / (1 - pi_tr)
    return (a * p) / (a * p + b * (1 - p))


def main() -> None:
    for tag, path in PREDS.items():
        z = np.load(path, allow_pickle=True)
        cog, ncog = z["cognate"], z["noncog"]
        y = z["y"][cog].astype(int)
        pc = z["p::curated-C"][cog]
        pd_none = z["p::Dense(None)"][cog]
        pi_de = float(y.mean())
        pc_s = saerens(pc, PI_TRAIN, pi_de)
        nc, nc_s = z["p::curated-C"][ncog], saerens(z["p::curated-C"][ncog], PI_TRAIN, pi_de)
        print(f"=== {tag}  (true cognate prior {pi_de:.5f}) ===")
        print(f"  curated-C  AUPRC raw={average_precision_score(y, pc):.4f}  "
              f"+Saerens={average_precision_score(y, pc_s):.4f}  (rank-invariant)")
        print(f"  curated-C  Brier raw={brier_score_loss(y, pc):.4f}  "
              f"+Saerens={brier_score_loss(y, pc_s):.4f}  (recovers)")
        print(f"  curated-C  non-cog mean p raw={nc.mean():.4f}  +Saerens={nc_s.mean():.4f}")
        print(f"  Dense(None) AUPRC={average_precision_score(y, pd_none):.4f}  "
              f"Brier={brier_score_loss(y, pd_none):.4f}  <- target\n")


if __name__ == "__main__":
    main()
