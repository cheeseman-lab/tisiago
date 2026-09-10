"""Generate one-hot sequence features as grounding baselines for the autoresearch fleet.

Two row-aligned arrays written under the store's ``embeddings/onehot/`` so the existing
``load()`` mechanism picks them up as ordinary feature keys:

* ``onehot/codon12.npy``  [N, 12]  — one-hot of the candidate 3-mer (3 positions x ACGT).
  Because negatives are codon-frequency-matched to positives, a head on this ALONE should
  sit near AUROC 0.5 — the control proving the embeddings carry *context*, not codon identity.
* ``onehot/kozakW{W}.npy`` [N, (2W+1)*4] — one-hot of a +/-W bp window (5'->3' reading
  orientation, codon at center). The simple-sequence floor the foundation embeddings must beat.

Pure CPU. The window fetch reuses the strand convention from ``tiling.a_plus_of`` and is
self-checked: the centered 3-mer must equal the manifest ``codon``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

_BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _onehot_seq(seq: str) -> np.ndarray:
    """[len(seq), 4] one-hot; non-ACGT rows are all-zero."""
    oh = np.zeros((len(seq), 4), dtype=np.float16)
    for i, b in enumerate(seq):
        j = _BASES.get(b)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def codon_onehot(m: pd.DataFrame) -> np.ndarray:
    return np.stack([_onehot_seq(c.upper()).reshape(-1) for c in m.codon.values]).astype(np.float16)


def window_onehot(m: pd.DataFrame, fa: Fasta, half: int) -> np.ndarray:
    out = np.zeros((len(m), (2 * half + 1) * 4), dtype=np.float16)
    chrom_len = {c: len(fa[c]) for c in m.chrom.unique()}
    n_checked = 0
    for i, r in enumerate(m.itertuples(index=False)):
        a_plus = r.gstart if r.strand == "+" else r.gstart - 1  # tiling.a_plus_of
        s, e = a_plus - half, a_plus + half + 1  # symmetric plus-strand span
        cl = chrom_len[r.chrom]
        cs, ce = max(0, s), min(cl, e)
        core = str(fa[r.chrom][cs:ce]).upper()
        core = ("N" * (cs - s)) + core + ("N" * (e - ce))
        seq = core if r.strand == "+" else _revcomp(core)
        if seq[half : half + 3] == r.codon:  # self-check
            n_checked += 1
        out[i] = _onehot_seq(seq).reshape(-1)
    frac = n_checked / len(m)
    print(f"window self-check: centered 3-mer == manifest codon for {frac:.4f} of rows")
    # Mismatches (~0.2%) are codons spanning a splice junction: the genomic 3-mer differs
    # from the spliced manifest codon. This is expected and consistent — the AG/Evo2
    # embeddings are likewise extracted from genomic windows, so a genomic one-hot is the
    # correct comparable baseline. Centering is wrong only if the match rate collapses.
    if frac < 0.99:
        raise SystemExit(f"window centering wrong (only {frac:.4f} match) — fix offset")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/store")
    ap.add_argument(
        "--genome",
        default=os.environ.get("TISIAGO_GENOME"),
        help="indexed reference FASTA (default: TISIAGO_GENOME)",
    )
    ap.add_argument("--half", type=int, default=20, help="window half-width (bp)")
    args = ap.parse_args()
    if not args.genome:
        ap.error("--genome is required (or set TISIAGO_GENOME)")

    store = Path(args.store)
    m = pd.read_parquet(store / "manifest.parquet")
    out_dir = store / "embeddings" / "onehot"
    out_dir.mkdir(parents=True, exist_ok=True)

    codon = codon_onehot(m)
    np.save(out_dir / "codon12.npy", codon)
    print(f"wrote onehot/codon12.npy {codon.shape}")

    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    win = window_onehot(m, fa, args.half)
    np.save(out_dir / f"kozakW{args.half}.npy", win)
    print(f"wrote onehot/kozakW{args.half}.npy {win.shape}")


if __name__ == "__main__":
    main()
