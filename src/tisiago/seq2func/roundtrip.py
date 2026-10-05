"""Round-trip contract: observed tracks -> synthetic BAMs -> Jimmy's Ribo-TISH -> his calls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from tisiago.seq2func.ribotish import compare_calls, filter_predictions, predict_command
from tisiago.seq2func.synthetic_bam import tracks_to_bam, write_para
from tisiago.seq2func.tracks import track_prefix

PASS_JACCARD = 0.95


def prepare_inputs(condition: str, tracks_dir: Path, work_dir: Path) -> dict[str, list[Path]]:
    """Synthetic per-replicate TIS and CHX BAMs plus matching para files."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, list[Path]] = {}
    for library in ("tis", "chx"):
        bams, paras = [], []
        for rep in (1, 2):
            prefix = track_prefix(tracks_dir, condition, library, rep)
            bam = work_dir / f"{library}_rep{rep}.bam"
            tracks_to_bam(f"{prefix}.plus.bw", f"{prefix}.minus.bw", bam)
            para = bam.with_name(bam.name + ".para.py")
            write_para(para)
            bams.append(bam)
            paras.append(para)
        inputs[f"{library}_bams"] = bams
        inputs[f"{library}_paras"] = paras
    return inputs


def run_condition(
    condition: str, tracks_dir: Path, work_dir: Path, atis_root: Path, ribotish: str
) -> dict:
    """Run Jimmy's Ribo-TISH on synthetic BAMs and compare with his pooled calls."""
    work_dir, atis_root = Path(work_dir), Path(atis_root)
    inputs = prepare_inputs(condition, tracks_dir, work_dir)
    output = work_dir / f"{condition}_TIS_predict.txt"
    argv = predict_command(
        ribotish, inputs["tis_bams"], inputs["chx_bams"], inputs["tis_paras"],
        inputs["chx_paras"],
        atis_root / "reference" / "gencode.v49.primary_assembly.annotation.gtf",
        atis_root / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa",
        output, work_dir / "tisBackground.txt",
    )
    subprocess.run(argv, check=True, cwd=work_dir)
    ours = pd.read_csv(work_dir / f"{condition}_TIS_predict_all.txt", sep="\t")
    jimmy = pd.read_csv(
        atis_root / "ribotish" / "combined" / f"{condition}_TIS_predict_all.txt", sep="\t"
    )
    report = {"condition": condition, **compare_calls(filter_predictions(ours),
                                                      filter_predictions(jimmy))}
    report["passed"] = report["jaccard"] >= PASS_JACCARD
    return report


def main() -> None:
    """CLI: one condition per call (a Slurm array task)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--tracks-dir", default="data/p6/tracks")
    ap.add_argument("--out-dir", default="data/p6/roundtrip")
    ap.add_argument("--atis-root", default=os.environ.get("TISIAGO_ATIS_DATA"))
    ap.add_argument("--ribotish", default=os.environ.get("TISIAGO_RIBOTISH", "ribotish"))
    args = ap.parse_args()
    work_dir = Path(args.out_dir) / args.condition
    report = run_condition(
        args.condition, Path(args.tracks_dir), work_dir, Path(args.atis_root), args.ribotish
    )
    (work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
