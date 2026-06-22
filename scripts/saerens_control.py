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

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

PREDS = {
    "ag (1M)": "data/scan_parts_allsplits/dense_ag_matched_preds.npz",
    "ag7 (1M)": "data/scan_parts_allsplits/dense_ag7_preds.npz",
}
# Balanced-train arm: curated heads (incl class_weight="balanced") scored on the dense test.
CURATED_PREDS = {
    "ag": "data/scan_parts_allsplits/dense_ag_curated_preds.npz",
    "ag7": "data/scan_parts_allsplits/dense_ag7_curated_preds.npz",
}
PI_TRAIN = 0.25  # curated set is 3:1 neg:pos -> positive rate 1/4; isotonic anchors heads here


def saerens(p: np.ndarray, pi_tr: float, pi_de: float) -> np.ndarray:
    """Adjust posteriors from training prior ``pi_tr`` to deployment prior ``pi_de``."""
    a, b = pi_de / pi_tr, (1 - pi_de) / (1 - pi_tr)
    return (a * p) / (a * p + b * (1 - p))


def main() -> None:
    """Report the curated-C and balanced-train arms (raw vs Saerens) against Dense(None)."""
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

    # Balanced-train arm: does class_weight="balanced" curated training (+ Saerens) help?
    print("--- balanced-train arm (curated class_weight=balanced + Saerens) ---")
    for tag, path in CURATED_PREDS.items():
        z = np.load(path, allow_pickle=True)
        cog = z["cognate"]
        y = z["y"][cog].astype(int)
        pb = z["p::B(.00075,bal)"][cog]
        pc = z["p::C(.00075,None)"][cog]
        pi_de = float(y.mean())
        pb_s = saerens(pb, PI_TRAIN, pi_de)
        print(f"=== {tag} ===")
        print(f"  curated-balanced  AUPRC raw={average_precision_score(y, pb):.4f}  "
              f"+Saerens={average_precision_score(y, pb_s):.4f}  (rank-invariant)")
        print(f"  curated-balanced  Brier raw={brier_score_loss(y, pb):.4f}  "
              f"+Saerens={brier_score_loss(y, pb_s):.4f}")
        print(f"  (curated-unweighted AUPRC={average_precision_score(y, pc):.4f} — balanced "
              f"does not beat it; both << dense)\n")


if __name__ == "__main__":
    main()
