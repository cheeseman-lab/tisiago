"""Predicted TIS/CHX tracks as per-replicate bigWigs in the real-track format (P6 round trip).

Usage::

    python -m tisiago.seq2func.predict_tracks --checkpoint data/p6/runs/<run>/best \
        --arm condition:K562 --split val

Writes ``<out>/<C>/{tis,chx}_rep{1,2}.{plus,minus}.bw`` (default ``<out>`` =
``<run>/predicted_<split>``) with the chromosome sizes of the real tracks and the Ribo-TISH
minus-strand convention (canonical minus P-site + ``RIBOTISH_MINUS_SHIFT``), so
:func:`tisiago.seq2func.roundtrip.prepare_inputs` reads them like the real tracks.

The model predicts expected counts for the replicate-pooled target. Replicate ``r`` gets the
pooled prediction times ``total_r / (total_1 + total_2)``, the library's share measured on the
real tracks over the same region (both strands). ``--mode sample`` (default) draws Poisson
counts with one generator per library and replicate seeded from ``--seed`` (``--copies k``
draws ``k * Poisson(expectation / k)``, matching BAMs that carry every alignment ``k`` times);
``--mode expected`` writes the expectation rounded to the nearest integer.

Overlapping windows are stitched by nearest centre: each position takes the prediction of the
window whose centre is nearest (ties to the earlier window), i.e. the central part of every
window, and is written once. The written region is the union of the split's windows.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from tisiago.seq2func.data import iterate
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT
from tisiago.seq2func.targets import STRAND_FILES, channel_layout
from tisiago.seq2func.tracks import track_prefix
from tisiago.seq2func.windows import merge_spans

LIBRARIES = ("tis", "chx")
REPS = (1, 2)
MODES = ("sample", "expected")
SPLIT_FILES = {"val": "val.parquet", "test": "test.parquet"}
MANIFEST = "predict_tracks.json"


def strand_shift(strand: str) -> int:
    """Ribo-TISH file position minus canonical position for a TIS/CHX strand."""
    return RIBOTISH_MINUS_SHIFT if strand == "-" else 0


def owned_spans(windows: pd.DataFrame) -> list[tuple[int, int]]:
    """``[lo, hi)`` each window writes: positions whose nearest window centre is its own.

    Windows are equal-length; ties go to the earlier window. Returned in ``windows`` row order.
    """
    out: list[tuple[int, int]] = [(0, 0)] * len(windows)
    frame = windows.reset_index(drop=True)
    for _, group in frame.groupby("chrom", sort=False):
        group = group.sort_values("start")
        rows = group.index.to_list()
        starts, ends = group.start.to_numpy(int), group.end.to_numpy(int)
        for k, row in enumerate(rows):
            lo, hi = int(starts[k]), int(ends[k])
            if k > 0:  # first base nearer this centre than the previous one
                lo = max(lo, (starts[k - 1] + ends[k - 1] + starts[k] + ends[k] - 2) // 4 + 1)
            if k + 1 < len(rows):
                hi = min(hi, (starts[k] + ends[k] + starts[k + 1] + ends[k + 1] - 2) // 4 + 1)
            out[row] = (lo, max(lo, hi))
    return out


def region_of(windows: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    """Union of the windows per chromosome (canonical coordinates)."""
    return {
        chrom: merge_spans(zip(g.start.astype(int), g.end.astype(int), strict=True))
        for chrom, g in windows.groupby("chrom", sort=False)
    }


def in_region(positions: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """Boolean membership of ``positions`` in sorted, disjoint half-open ``spans``."""
    if not spans:
        return np.zeros(len(positions), dtype=bool)
    starts = np.array([s for s, _ in spans])
    ends = np.array([e for _, e in spans])
    i = np.searchsorted(starts, positions, side="right") - 1
    return (i >= 0) & (positions < ends[np.maximum(i, 0)])


def region_counts(
    bw_path: str | Path, region: dict[str, list[tuple[int, int]]], shift: int = 0
) -> dict[str, Counter]:
    """Counts of a 1-bp bigWig at file positions ``p`` with ``p - shift`` in ``region``."""
    result: dict[str, Counter] = {}
    bw = pyBigWig.open(str(bw_path))
    if bw is None:
        raise FileNotFoundError(bw_path)
    try:
        sizes = bw.chroms()
        for chrom, spans in region.items():
            intervals = (bw.intervals(chrom) or ()) if chrom in sizes else ()
            if not intervals:
                result[chrom] = Counter()
                continue
            starts, ends, values = (np.asarray(x) for x in zip(*intervals, strict=True))
            lengths = (ends - starts).astype(np.int64)
            first = np.repeat(np.cumsum(lengths) - lengths, lengths)
            positions = np.repeat(starts, lengths) + np.arange(lengths.sum()) - first
            values = np.repeat(values, lengths)
            keep = in_region(positions - shift, spans)
            result[chrom] = Counter(
                dict(zip(positions[keep].tolist(), values[keep].tolist(), strict=True))
            )
    finally:
        bw.close()
    return result


def replicate_fractions(
    tracks_dir: str | Path, condition: str, region: dict[str, list[tuple[int, int]]]
) -> dict[str, dict[str, list[float]]]:
    """Per library: real replicate totals in ``region`` (both strands) and their shares."""
    out = {}
    for library in LIBRARIES:
        totals = []
        for rep in REPS:
            prefix = track_prefix(tracks_dir, condition, library, rep)
            total = 0.0
            for strand, name in STRAND_FILES.items():
                counts = region_counts(f"{prefix}.{name}.bw", region, strand_shift(strand))
                total += sum(sum(c.values()) for c in counts.values())
            totals.append(total)
        pooled = sum(totals)
        if pooled <= 0:
            raise ValueError(f"{condition} {library}: no real counts in the region")
        out[library] = {"totals": totals, "fractions": [t / pooled for t in totals]}
    return out


def chrom_sizes_of(bw_path: str | Path) -> list[tuple[str, int]]:
    """Header chromosome sizes of a bigWig, in header order."""
    bw = pyBigWig.open(str(bw_path))
    if bw is None:
        raise FileNotFoundError(bw_path)
    try:
        return list(bw.chroms().items())
    finally:
        bw.close()


def _draw(expectation: np.ndarray, mode: str, rng, copies: int) -> np.ndarray:
    if mode == "expected":
        return np.rint(expectation)
    return copies * rng.poisson(np.maximum(expectation, 0.0) / copies).astype(np.float64)


def predict_condition(
    predict_fn: Callable[[dict], np.ndarray],
    windows: pd.DataFrame,
    *,
    fasta: str | Path,
    tracks_dir: str | Path,
    genes: pd.DataFrame,
    condition: str,
    conditions: Sequence[str],
    out_dir: str | Path,
    mode: str = "sample",
    seed: int = 0,
    copies: int = 1,
    mask_unexpressed: bool = False,
    transcripts=None,
    prefetch: int = 4,
    meta: dict | None = None,
) -> dict:
    """Write ``condition``'s predicted per-replicate TIS/CHX bigWigs; return the manifest.

    ``predict_fn(batch)`` maps an ``iterate`` batch to ``(1, S, C)`` pooled predictions in
    :func:`channel_layout` order of ``conditions`` (canonical P-sites). ``mask_unexpressed``
    zeroes positions outside the loss mask (unexpressed genes) before scaling.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if copies < 1:
        raise ValueError("copies must be >= 1")
    layout = channel_layout(conditions)
    channel = {(lib, s): i for i, (c, lib, s) in enumerate(layout) if c == condition}
    if not channel:
        raise ValueError(f"{condition} is not among {tuple(conditions)}")
    sizes = chrom_sizes_of(f"{track_prefix(tracks_dir, condition, 'tis', 1)}.plus.bw")
    order = {chrom: i for i, (chrom, _) in enumerate(sizes)}
    unknown = set(windows.chrom) - set(order)
    if unknown:
        raise ValueError(f"window chromosomes {sorted(unknown)} not in the track header")
    windows = windows.assign(_order=windows.chrom.map(order)).sort_values(["_order", "start"])
    windows = windows.drop(columns="_order").reset_index(drop=True)
    spans = owned_spans(windows)
    region = region_of(windows)
    fractions = replicate_fractions(tracks_dir, condition, region)
    size_of = dict(sizes)

    out_dir = Path(out_dir)
    writers = {}
    rngs = {}
    for k, library in enumerate(LIBRARIES):
        for rep in REPS:
            prefix = track_prefix(out_dir, condition, library, rep)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            rngs[(library, rep)] = np.random.default_rng([seed, k, rep])
            for strand, name in STRAND_FILES.items():
                bw = pyBigWig.open(str(prefix.with_name(f"{prefix.name}.{name}.bw")), "w")
                bw.addHeader(sizes)
                writers[(library, rep, strand)] = bw
    expected = dict.fromkeys(LIBRARIES, 0.0)
    written = {lib: [0.0] * len(REPS) for lib in LIBRARIES}
    batches = iterate(
        windows,
        fasta=fasta,
        tracks_dir=tracks_dir,
        conditions=conditions,
        genes=genes,
        seed=0,
        augment_examples=False,
        loop=False,
        prefetch=prefetch,
        transcripts=transcripts,
    )
    try:
        for i, batch in enumerate(batches):
            row = windows.iloc[i]
            start, (lo, hi) = int(row.start), spans[i]
            pred = np.asarray(predict_fn(batch), dtype=np.float64)[0]
            if mask_unexpressed:
                pred = np.where(batch["mask"][0], pred, 0.0)
            pred = pred[lo - start : hi - start]
            for library in LIBRARIES:
                for strand in STRAND_FILES:
                    pooled = pred[:, channel[(library, strand)]]
                    expected[library] += float(pooled.sum())
                    positions = np.arange(lo, hi) + strand_shift(strand)
                    inside = (positions >= 0) & (positions < size_of[row.chrom])
                    for r, rep in enumerate(REPS):
                        share = fractions[library]["fractions"][r]
                        values = _draw(pooled * share, mode, rngs[(library, rep)], copies)
                        keep = inside & (values > 0)
                        if not keep.any():
                            continue
                        writers[(library, rep, strand)].addEntries(
                            row.chrom,
                            positions[keep].tolist(),
                            values=values[keep].tolist(),
                            span=1,
                        )
                        written[library][r] += float(values[keep].sum())
    finally:
        batches.close()
        for bw in writers.values():
            bw.close()

    manifest = {
        **(meta or {}),
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "copies": copies,
        "mask_unexpressed": mask_unexpressed,
        "stitching": "nearest_centre",
        "windows": len(windows),
        "chroms": list(region),
        "fractions": fractions,
        "expected_totals": expected,
        "written_totals": written,
    }
    path = out_dir / condition / MANIFEST
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _run_arg(checkpoint: Path, key: str, default):
    from tisiago.seq2func.evaluate import run_args

    value = run_args(checkpoint).get(key)
    return Path(value) if value else Path(default)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="run's best/ or last/")
    ap.add_argument("--arm", required=True, help="'shared' or 'condition:<C>'")
    ap.add_argument("--condition", help="condition to write (default: the arm's only one)")
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--windows-dir", type=Path, help="default: run.json, else data/p6/windows")
    ap.add_argument("--expression", type=Path, help="default: run.json, else data/p6/expression")
    ap.add_argument("--tracks-dir", type=Path, help="real tracks; default: run.json")
    ap.add_argument("--out-dir", type=Path, help="default: <run>/predicted_<split>")
    ap.add_argument("--mode", choices=MODES, default="sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copies", type=int, default=1, help="draw k * Poisson(expectation / k)")
    ap.add_argument("--mask-unexpressed", action="store_true")
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows (smoke)")
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"))
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--head", default=None, help="head kind; default: run.json")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    """CLI: predict one condition's tracks over a split."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    from tisiago.seq2func.evaluate import (
        head_kind_of,
        load_predict_fn,
        load_transcripts,
        tracks_dir_of,
    )
    from tisiago.seq2func.train import parse_arm

    try:
        _, conditions = parse_arm(args.arm)
    except ValueError as exc:
        ap.error(str(exc))
    condition = args.condition or (conditions[0] if len(conditions) == 1 else None)
    if condition not in conditions:
        ap.error(f"--condition must be one of {conditions}")
    for flag in ("genome", "base_checkpoint"):
        if not getattr(args, flag):
            ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")
    out_dir = args.out_dir or args.checkpoint.parent / f"predicted_{args.split}"
    if (out_dir / condition / MANIFEST).exists() and not args.overwrite:
        ap.error(f"{out_dir / condition / MANIFEST} exists; pass --overwrite to replace")
    windows_dir = args.windows_dir or _run_arg(args.checkpoint, "windows_dir", "data/p6/windows")
    expression = args.expression or _run_arg(
        args.checkpoint, "expression", "data/p6/expression/expression.parquet"
    )
    tracks_dir = args.tracks_dir or tracks_dir_of(args.checkpoint)
    windows = pd.read_parquet(windows_dir / SPLIT_FILES[args.split])
    if args.max_windows is not None:
        windows = windows.iloc[: args.max_windows]
    genes = pd.read_parquet(expression).reset_index(drop=True)
    genes = genes[genes.condition.isin(conditions)].reset_index(drop=True)
    head_kind = args.head or head_kind_of(args.checkpoint)
    predict_fn = load_predict_fn(
        args.checkpoint, args.arm, base_checkpoint=args.base_checkpoint, head_kind=head_kind
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "arm": args.arm,
        "head": head_kind,
        "split": args.split,
        "max_windows": args.max_windows,
        "windows_dir": str(windows_dir),
        "expression": str(expression),
        "tracks_dir": str(tracks_dir),
    }
    manifest = predict_condition(
        predict_fn,
        windows,
        fasta=args.genome,
        tracks_dir=tracks_dir,
        genes=genes,
        condition=condition,
        conditions=conditions,
        out_dir=out_dir,
        mode=args.mode,
        seed=args.seed,
        copies=args.copies,
        mask_unexpressed=args.mask_unexpressed,
        transcripts=load_transcripts(head_kind, args.gtf),
        prefetch=args.prefetch,
        meta=meta,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
