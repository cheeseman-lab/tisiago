"""Kozak ±Wbp one-hot for a (large) scan manifest — vectorized by chromosome.

Reproduces ``autoresearch/make_onehot.py:window_onehot`` EXACTLY (same strand/centering
convention: ``a_plus = gstart if "+" else gstart-1``; symmetric plus-strand span; reverse-
complement for minus; codon at window center) but groups rows by chromosome and uses numpy
fancy-indexing instead of a per-row pyfaidx fetch, so it scales to ~63M positions in minutes.

``--validate`` runs against the curated store and asserts the output is bit-for-bit identical
to the existing ``onehot/kozakW20.npy`` — proving the convention matches what the head trains
on before we apply it to the scan manifest.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

# base -> index 0..3 (A,C,G,T), everything else -> 4 (ignored / all-zero one-hot row)
_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _i in zip(b"ACGT", (0, 1, 2, 3)):
    _LUT[_b] = _i
    _LUT[ord(chr(_b).lower())] = _i
# complement in index space: A<->T (0<->3), C<->G (1<->2), N stays 4
_COMP_IDX = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def window_onehot_fast(m: pd.DataFrame, fa: Fasta, half: int) -> np.ndarray:
    """Vectorized ±half one-hot, grouped by chromosome (see module docstring)."""
    w = 2 * half + 1
    out = np.zeros((len(m), w * 4), dtype=np.float16)
    a_plus = np.where(m.strand.values == "+", m.gstart.values, m.gstart.values - 1)
    is_minus = m.strand.values == "-"
    offs = np.arange(-half, half + 1)
    by_chrom = m.groupby("chrom").indices  # chrom -> array of positional row indices

    for chrom, idx in by_chrom.items():
        codes = _LUT[np.frombuffer(str(fa[chrom]).upper().encode(), dtype=np.uint8)]
        cl = codes.shape[0]
        pos = a_plus[idx][:, None] + offs[None, :]  # [n, w] plus-strand genomic positions
        valid = (pos >= 0) & (pos < cl)
        win = codes[np.clip(pos, 0, cl - 1)]  # [n, w] base indices
        win[~valid] = 4  # off-chromosome -> ignored
        mi = is_minus[idx]
        if mi.any():
            win[mi] = _COMP_IDX[win[mi][:, ::-1]]  # reverse-complement in index space
        oh = np.zeros((len(idx), w, 4), dtype=np.float16)
        rr, cc = np.where(win < 4)
        oh[rr, cc, win[rr, cc]] = 1.0
        out[idx] = oh.reshape(len(idx), w * 4)
    return out


def _self_check(m: pd.DataFrame, out: np.ndarray, half: int) -> float:
    """Fraction of rows whose centered 3-mer one-hot decodes back to the manifest codon."""
    w = 2 * half + 1
    oh = out.reshape(len(m), w, 4)
    center = oh[:, half : half + 3, :]  # [n, 3, 4]
    idx = center.argmax(axis=2)  # [n, 3]
    has = center.max(axis=2) > 0  # zero rows (N / padding) don't decode
    bases = np.array(list("ACGT"))
    dec = np.where(has, bases[idx], "N")
    decoded = ["".join(r) for r in dec]
    cod = m.codon.str.upper().values
    return float(np.mean([d == c for d, c in zip(decoded, cod)]))


def main() -> None:
    """Build (or validate) the Kozak one-hot for a scan manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument(
        "--genome",
        default=os.environ.get("TISIAGO_GENOME"),
        help="indexed reference FASTA (default: TISIAGO_GENOME)",
    )
    ap.add_argument("--half", type=int, default=20)
    ap.add_argument("--validate", default=None,
                    help="path to a reference kozak npy to assert bit-for-bit equality")
    args = ap.parse_args()
    if not args.genome:
        ap.error("--genome is required (or set TISIAGO_GENOME)")

    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    m = pd.read_parquet(args.manifest)
    print(f"{len(m):,} positions, {m.chrom.nunique()} chromosomes", flush=True)
    out = window_onehot_fast(m, fa, args.half)

    frac = _self_check(m, out, args.half)
    print(f"self-check: centered 3-mer == manifest codon for {frac:.4f} of rows", flush=True)
    if frac < 0.99:
        raise SystemExit(f"centering wrong (only {frac:.4f} match)")

    if args.validate:
        ref = np.load(args.validate)
        ok = ref.shape == out.shape and np.array_equal(ref, out)
        print(f"validate vs {args.validate}: shapes {ref.shape} vs {out.shape}, equal={ok}")
        if not ok:
            raise SystemExit("VALIDATION FAILED — convention does not match curated kozak")
        print("VALIDATION PASSED — bit-for-bit identical to curated kozakW20")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"wrote {args.out} {out.shape} ({out.nbytes / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
