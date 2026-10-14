"""Same-pipeline comparison: Ribo-TISH on predicted tracks vs on real tracks (P6 round trip).

Usage::

    python -m tisiago.seq2func.predicted_calls --condition K562 \
        --predicted-dir data/p6/runs/<run>/predicted_val

Both track sets are restricted to the union of the split's windows (minus strand in Ribo-TISH
coordinates, canonical + ``RIBOTISH_MINUS_SHIFT``), turned into synthetic BAMs the same way
(:func:`tisiago.seq2func.roundtrip.prepare_inputs`) and run through the Ribo-TISH fork with
identical flags against a GTF restricted to the split's chromosomes. Calls pass
:func:`~tisiago.seq2func.ribotish.filter_predictions` and are compared by (Tid, Start): overall,
by start codon (ATG vs near-cognate) and by TisType. ``report.json`` holds the counts, Jaccard,
recall and precision of the predicted-track calls against the real-track calls, and pass/fail
at ``PASS_JACCARD``. Split, windows and real tracks default to the prediction's manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tisiago.seq2func.predict_tracks import (
    LIBRARIES,
    MANIFEST,
    REPS,
    SPLIT_FILES,
    chrom_sizes_of,
    region_counts,
    region_of,
    strand_shift,
)
from tisiago.seq2func.ribotish import compare_calls, filter_predictions
from tisiago.seq2func.roundtrip import PASS_JACCARD, ribotish_version, run_ribotish
from tisiago.seq2func.targets import STRAND_FILES
from tisiago.seq2func.tracks import track_prefix, write_bigwigs

ATG = "ATG"


def restrict_tracks(
    tracks_dir: str | Path,
    out_dir: str | Path,
    condition: str,
    region: dict[str, list[tuple[int, int]]],
) -> dict[str, list[int]]:
    """Copy the TIS/CHX replicate tracks keeping only ``region``; return totals per library."""
    totals = {}
    for library in LIBRARIES:
        totals[library] = []
        for rep in REPS:
            src = track_prefix(tracks_dir, condition, library, rep)
            sizes = chrom_sizes_of(f"{src}.plus.bw")
            counts = {}
            for strand, name in STRAND_FILES.items():
                kept = region_counts(f"{src}.{name}.bw", region, strand_shift(strand))
                for chrom, counter in kept.items():
                    counts[(chrom, strand)] = counter
            dest = track_prefix(out_dir, condition, library, rep)
            totals[library].append(write_bigwigs(counts, sizes, dest))
    return totals


def restrict_gtf(gtf: str | Path, chroms: Sequence[str], out: str | Path) -> Path:
    """Write the header and the ``chroms`` lines of ``gtf`` to ``out``."""
    keep = set(chroms)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(gtf) as src, open(out, "w") as dst:
        for line in src:
            if line.startswith("#") or line.split("\t", 1)[0] in keep:
                dst.write(line)
    return out


def overlap(predicted: pd.DataFrame, real: pd.DataFrame) -> dict:
    """:func:`compare_calls` plus recall and precision of ``predicted`` against ``real``."""
    out = compare_calls(predicted, real)
    shared = out["n_shared"]
    out["recall"] = shared / out["n_reference"] if out["n_reference"] else None
    out["precision"] = shared / out["n_observed"] if out["n_observed"] else None
    return out


def call_report(predicted_all: pd.DataFrame, real_all: pd.DataFrame) -> dict:
    """Filtered-call overlap overall, by start codon and by TisType (``None`` if absent)."""
    predicted, real = filter_predictions(predicted_all), filter_predictions(real_all)

    def is_atg(df):
        return df.StartCodon.astype(str).str.upper() == ATG

    by_codon = {
        "ATG": overlap(predicted[is_atg(predicted)], real[is_atg(real)]),
        "near_cognate": overlap(predicted[~is_atg(predicted)], real[~is_atg(real)]),
    }
    by_type = None
    if "TisType" in predicted and "TisType" in real:
        types = sorted(set(predicted.TisType.astype(str)) | set(real.TisType.astype(str)))
        by_type = {
            t: overlap(
                predicted[predicted.TisType.astype(str) == t], real[real.TisType.astype(str) == t]
            )
            for t in types
        }
    report = {
        "overall": overlap(predicted, real),
        "by_start_codon": by_codon,
        "by_tis_type": by_type,
        "pass_jaccard": PASS_JACCARD,
    }
    report["passed"] = report["overall"]["jaccard"] >= PASS_JACCARD
    return report


def run_predicted_calls(
    condition: str,
    predicted_dir: str | Path,
    tracks_dir: str | Path,
    windows: pd.DataFrame,
    work_dir: str | Path,
    *,
    gtf: str | Path,
    fasta: str | Path,
    ribotish: str,
    nproc: int = 1,
) -> dict:
    """Restrict both track sets to ``windows``, run Ribo-TISH on each, compare the calls."""
    work_dir = Path(work_dir).resolve()
    region = region_of(windows)
    chroms = list(region)
    annotation = restrict_gtf(gtf, chroms, work_dir / f"annotation.{'_'.join(chroms)}.gtf")
    tables, totals = {}, {}
    for name, source in (("predicted", predicted_dir), ("real", tracks_dir)):
        restricted = work_dir / f"tracks_{name}"
        totals[name] = restrict_tracks(source, restricted, condition, region)
        tables[name] = run_ribotish(
            condition, restricted, work_dir / name, annotation, fasta, ribotish, nproc=nproc
        )
    return {
        "condition": condition,
        "ribotish_version": ribotish_version(ribotish),
        "predicted_dir": str(predicted_dir),
        "tracks_dir": str(tracks_dir),
        "chroms": chroms,
        "windows": len(windows),
        "track_totals": totals,
        **call_report(tables["predicted"], tables["real"]),
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--predicted-dir", type=Path, required=True, help="predict_tracks --out-dir")
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), help="default: the prediction's")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--windows-dir", type=Path, help="default: the prediction's")
    ap.add_argument("--tracks-dir", type=Path, help="real tracks; default: the prediction's")
    ap.add_argument("--out-dir", type=Path, help="default: <predicted-dir>/calls/<C>")
    ap.add_argument("--ribotish", default=os.environ.get("TISIAGO_RIBOTISH", "ribotish"))
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"))
    return ap


def main() -> None:
    """CLI: one condition's predicted-vs-real call comparison; writes ``report.json``."""
    ap = build_parser()
    args = ap.parse_args()
    c = args.condition
    manifest_path = args.predicted_dir / c / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    split = args.split or manifest.get("split", "val")
    if split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    if manifest.get("split", split) != split:
        ap.error(f"--split {split} differs from the prediction's split {manifest['split']}")
    for flag in ("genome", "gtf"):
        if not getattr(args, flag):
            ap.error(f"--{flag} is required (or set it in .env)")
    windows_dir = args.windows_dir or Path(manifest.get("windows_dir", "data/p6/windows"))
    tracks_dir = args.tracks_dir or Path(manifest.get("tracks_dir", "data/p6/tracks"))
    windows = pd.read_parquet(windows_dir / SPLIT_FILES[split])
    if manifest.get("max_windows") is not None:
        windows = windows.iloc[: manifest["max_windows"]]
    if "windows" in manifest and manifest["windows"] != len(windows):
        ap.error(f"{len(windows)} windows in {windows_dir}, prediction used {manifest['windows']}")
    out_dir = args.out_dir or args.predicted_dir / "calls" / c
    report = run_predicted_calls(
        c,
        args.predicted_dir,
        tracks_dir,
        windows,
        out_dir,
        gtf=args.gtf,
        fasta=args.genome,
        ribotish=args.ribotish,
        nproc=args.nproc,
    )
    report = {"split": split, "prediction": manifest, **report}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (out_dir / "report.json").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
