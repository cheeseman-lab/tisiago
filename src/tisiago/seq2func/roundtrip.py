"""Round-trip contract: observed tracks -> synthetic BAMs -> Ribo-TISH fork -> real-BAM calls."""

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


def ribotish_version(ribotish: str) -> str:
    """First line of ``<ribotish> --version`` (stdout or stderr)."""
    done = subprocess.run([str(ribotish), "--version"], capture_output=True, text=True)
    lines = (done.stdout + done.stderr).strip().splitlines()
    return lines[0] if lines else "unknown"


def run_ribotish(
    condition: str,
    tracks_dir: Path,
    work_dir: Path,
    gtf: Path,
    fasta: Path,
    ribotish: str,
    nproc: int = 1,
) -> pd.DataFrame:
    """Synthetic BAMs from ``tracks_dir`` -> Ribo-TISH predict in ``work_dir``; all candidates."""
    gtf, fasta = Path(gtf).resolve(), Path(fasta).resolve()
    work_dir = Path(work_dir).resolve()
    inputs = prepare_inputs(condition, Path(tracks_dir).resolve(), work_dir)
    output = work_dir / f"{condition}_TIS_predict.txt"
    argv = predict_command(
        ribotish,
        inputs["tis_bams"],
        inputs["chx_bams"],
        inputs["tis_paras"],
        inputs["chx_paras"],
        gtf,
        fasta,
        output,
        work_dir / "tisBackground.txt",
        nproc=nproc,
    )
    subprocess.run(argv, check=True, cwd=work_dir)
    return pd.read_csv(work_dir / f"{condition}_TIS_predict_all.txt", sep="\t")


def run_condition(
    condition: str,
    tracks_dir: Path,
    work_dir: Path,
    reference_all: Path,
    ribotish: str,
    nproc: int = 1,
    gtf: Path | None = None,
    fasta: Path | None = None,
) -> dict:
    """Run Ribo-TISH on synthetic BAMs and compare with the fork's real-BAM pooled calls."""
    reference_all = Path(reference_all).resolve()
    if not reference_all.is_file():
        raise FileNotFoundError(f"reference candidates table not found: {reference_all}")
    gtf = gtf or os.environ.get("TISIAGO_GTF")
    fasta = fasta or os.environ.get("TISIAGO_GENOME")
    if not gtf or not fasta:
        raise ValueError("GTF and FASTA are required (or set TISIAGO_GTF / TISIAGO_GENOME)")
    ours = run_ribotish(condition, tracks_dir, work_dir, gtf, fasta, ribotish, nproc=nproc)
    reference = pd.read_csv(reference_all, sep="\t")
    report = {
        "condition": condition,
        "ribotish_version": ribotish_version(ribotish),
        "reference": str(reference_all),
        **compare_calls(filter_predictions(ours), filter_predictions(reference)),
    }
    report["passed"] = report["jaccard"] >= PASS_JACCARD
    return report


def main() -> None:
    """CLI: one condition per call (a Slurm array task)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--tracks-dir", default="data/p6/tracks")
    ap.add_argument("--out-dir", default="data/p6/roundtrip_fork")
    ap.add_argument("--reference-run", default=os.environ.get("TISIAGO_RIBOTISH_RUN"))
    ap.add_argument("--ribotish", default=os.environ.get("TISIAGO_RIBOTISH", "ribotish"))
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if not args.reference_run:
        ap.error("--reference-run is required (or set TISIAGO_RIBOTISH_RUN)")
    c = args.condition
    reference = Path(args.reference_run) / c / "pooled" / f"{c}_pooled.candidates.tsv"
    if not reference.is_file():
        ap.error(f"reference candidates table not found: {reference}")
    work_dir = Path(args.out_dir) / c
    report = run_condition(
        c, Path(args.tracks_dir), work_dir, reference, args.ribotish, nproc=args.nproc
    )
    (work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
