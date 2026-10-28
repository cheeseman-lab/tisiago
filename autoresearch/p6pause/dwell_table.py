"""Learned codon dwell of a fast-loop export with ``chx_flux.dwell_codon``.

Log dwell per P-site codon (mean over A-site codons) and per A-site codon (mean over P-site
codons), centred.

Usage: python autoresearch/p6pause/dwell_table.py data/p6/fastloop/export_pause/p1_dwell
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

BASES = "ACGT"


def main() -> None:
    params = np.load(Path(sys.argv[1]) / "params.npz")
    w = params[next(k for k in params.files if k.endswith("chx_flux_dwell_codon/w"))]
    b = params[next(k for k in params.files if k.endswith("chx_flux_dwell_codon/b"))]
    wo = params[next(k for k in params.files if k.endswith("chx_flux_dwell_codon_out/w"))]
    bo = params[next(k for k in params.files if k.endswith("chx_flux_dwell_codon_out/b"))]
    codons = ["".join(c) for c in itertools.product(BASES, repeat=3)]
    table = np.zeros((64, 64))
    for i, p in enumerate(codons):
        for j, a in enumerate(codons):
            onehot = np.eye(4)[[BASES.index(x) for x in p + a]]  # (6, 4)
            h = np.maximum(np.einsum("kc,kcn->n", onehot, w) + b, 0.0)
            table[i, j] = float(h @ wo[:, 0] + bo[0])
    table -= table.mean()
    for label, values in (("P site", table.mean(1)), ("A site", table.mean(0))):
        order = np.argsort(values)[::-1]
        print(f"{label}: log dwell (centred) top / bottom 8; sd {values.std():.3f}")
        print("  " + " ".join(f"{codons[k]} {values[k]:+.2f}" for k in order[:8]))
        print("  " + " ".join(f"{codons[k]} {values[k]:+.2f}" for k in order[-8:]))
    stops = [codons.index(s) for s in ("TAA", "TAG", "TGA")]
    print(
        "near-cognate P site:",
        " ".join(
            f"{c} {table.mean(1)[codons.index(c)]:+.2f}"
            for c in ("AGG", "AAG", "CTG", "GTG", "TTG", "ACG", "ATG", "GAT", "GAC", "CCG", "CCA")
        ),
        "| stops",
        " ".join(f"{table.mean(1)[s]:+.2f}" for s in stops),
    )


if __name__ == "__main__":
    main()
