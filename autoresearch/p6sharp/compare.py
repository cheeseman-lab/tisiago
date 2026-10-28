"""Sharpness, track metrics and matched-call Ribo-TISH of step-2 runs, per seed + mean.

Usage:
    python autoresearch/p6sharp/compare.py --condition K562 --models sbflux shh1 shh2 \
        [--calls data/p6sharp/matched_calls_k562/matched_calls.tsv]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TRACK = {
    "tis_codon": "{c}_tis_codon",
    "tis_nt": "{c}_tis_nt",
    "chx_codon": "{c}_chx_codon",
    "chx_nt": "{c}_chx_nt",
    "nc_ratio": "{c}_tis_rep_ratio_nc",
    "utr5_ratio": "{c}_utr5_tis_rep_ratio",
    "utr5_framed": "{c}_utr5_orf_framed",
}
SHARP = {
    **{f"sh15_{x}": f"{{c}}_tis_sharp_share15_{x}" for x in ("atg", "nc", "utr5")},
    **{f"am15_{x}": f"{{c}}_tis_sharp_argmax15_{x}" for x in ("atg", "nc", "utr5")},
    "top1": "{c}_tis_top1_resampled",
}
REAL = {
    **{f"sh15_{x}": f"{{c}}_tis_sharp_share15_{x}_real" for x in ("atg", "nc", "utr5")},
    **{f"am15_{x}": f"{{c}}_tis_sharp_argmax15_{x}_real" for x in ("atg", "nc", "utr5")},
    "top1": "{c}_tis_top1_real",
}
CALLS = {
    "N": "N",
    "J": "overall_J",
    "R": "overall_R",
    "P": "overall_P",
    "NC_J": "near_cognate_J",
    "utr5_R": "5'UTR_R",
    "utr5_P": "5'UTR_P",
    "utr5_n": "5'UTR_n",
}


def model_seed(name: str) -> tuple[str, int]:
    m = re.match(r"(.+)_[a-z0-9]+_s(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", default="K562")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sharpness", type=Path, default=None, help="default sharpness_<c>.tsv")
    ap.add_argument("--calls", type=Path, default=None)
    args = ap.parse_args()
    c = args.condition
    pd.set_option("display.width", 250)
    path = args.sharpness or Path(__file__).parent / f"sharpness_{c.lower()}.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df[df.heldout == c].drop_duplicates("name", keep="last")
    df[["model", "seed"]] = df.name.apply(lambda n: pd.Series(model_seed(n)))
    df = df[df.model.isin(args.models)]
    cols = {**TRACK, **SHARP}
    out = pd.DataFrame({k: df[v.format(c=c)] for k, v in cols.items()})
    out.insert(0, "seed", df.seed)
    out.insert(0, "model", df.model)
    print(out.round(3).to_string(index=False))
    g = out.drop(columns="seed").groupby("model", sort=False)
    print("\nmean\n" + g.mean().reindex(args.models).round(3).to_string())
    print("\nsd\n" + g.std().reindex(args.models).round(3).to_string())
    real = {k: df[v.format(c=c)].iloc[0] for k, v in REAL.items()}
    print("\nreal " + "  ".join(f"{k} {v:.3f}" for k, v in real.items()))
    if args.calls is not None:
        calls = pd.read_csv(args.calls, sep="\t")
        for setting in ("report", "tis/filtered/N_at_P", "tis/filtered/N_real"):
            part = calls[(calls.setting == setting) & calls.model.isin(args.models)]
            t = pd.DataFrame({k: part[v] for k, v in CALLS.items() if v in part})
            t.insert(0, "model", part.model)
            m = t.groupby("model", sort=False).mean().reindex(args.models)
            m["J_sd"] = t.groupby("model").J.std()
            print(f"\n== Ribo-TISH {setting} (seed means)\n" + m.round(3).to_string())


if __name__ == "__main__":
    main()
