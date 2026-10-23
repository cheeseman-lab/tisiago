"""Write phase-3 configs: the phase-2 leader (p2r04_tisdil_l32) on the fc2d_k562 cache + overrides.

Usage: python autoresearch/p6fast/mkcfg.py <round> <name> '<yaml override>' [--seeds 0,1,2]
``losses_extra`` in the override is appended to the leader's losses; other keys deep-merge.
"""

import argparse
import copy
from pathlib import Path

import yaml

from tisiago.seq2func.fastloop import deep_merge

HERE = Path(__file__).parent
BASE = yaml.safe_load((HERE / "configs/p2r04/p2r04_tisdil_l32.yaml").read_text())
BASE["cache"] = "data/p6/txcache/fc2d_k562"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round")
    ap.add_argument("name")
    ap.add_argument("override", nargs="?", default="{}")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    over = yaml.safe_load(args.override) or {}
    extra = over.pop("losses_extra", [])
    replace = over.pop("losses", None)
    cfg = deep_merge(copy.deepcopy(BASE), over)
    if replace is not None:
        cfg["losses"] = replace
    cfg["losses"] = cfg["losses"] + extra
    out = HERE / "configs" / args.round
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        name = f"{args.round}_{args.name}" + (f"_s{seed}" if seed else "")
        c = dict(cfg, name=name)
        if seed:
            c["seed"] = seed
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
