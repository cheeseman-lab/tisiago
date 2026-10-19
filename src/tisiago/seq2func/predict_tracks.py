"""Predicted TIS/CHX tracks as per-replicate bigWigs in the real-track format (P6 round trip).

Usage::

    python -m tisiago.seq2func.predict_tracks --checkpoint data/p6/runs/<run>/best \
        --arm condition:K562 --split val
    python -m tisiago.seq2func.predict_tracks --checkpoint data/p6/runs/rnacond_k562/best \
        --arm rnacond:K562 --condition K562 --split val --mode float

Writes ``<out>/<C>/{tis,chx}_rep{1,2}.{plus,minus}.bw`` (default ``<out>`` =
``<run>/predicted_<split>``) with the chromosome sizes of the real tracks and the Ribo-TISH
minus-strand convention (canonical minus P-site + ``RIBOTISH_MINUS_SHIFT``), so
:func:`tisiago.seq2func.roundtrip.prepare_inputs` reads them like the real tracks.

The model predicts expected counts for the replicate-pooled target. Replicate ``r`` gets the
pooled prediction times ``total_r / (total_1 + total_2)``, the library's share measured on the
real tracks over the same region (both strands). ``--mode sample`` (default) draws Poisson
counts with one generator per library and replicate seeded from ``--seed`` (``--copies k``
draws ``k * Poisson(expectation / k)``, matching BAMs that carry every alignment ``k`` times);
``--mode expected`` writes the expectation rounded to the nearest integer. ``--mode float``
writes the pooled expectation itself, unsplit, unsampled and unrounded, to
``<out>/<C>/{tis,chx}_pooled.{plus,minus}.bw`` (default ``<out>`` =
``<run>/predicted_<split>_float``), every position of the written region included.

``--match-depth`` first rescales each library's pooled prediction so its total over the written
region equals the real pooled total there (factors in the manifest). ``--source real-resample``
replaces the model by the real pooled tracks (already in Ribo-TISH convention, read back to
canonical coordinates and re-shifted once on writing) and draws replicates the same way: the
same-pipeline noise ceiling for the comparison; it needs no checkpoint
(``--condition --tracks-dir --windows-dir --out-dir``).

Overlapping windows are stitched by nearest centre: each position takes the prediction of the
window whose centre is nearest (ties to the earlier window), i.e. the central part of every
window, and is written once. The written region is the union of the split's windows.

An ``rnacond:<group>`` checkpoint predicts any ``--condition`` (held out or not) from that
condition's pooled RNA; the manifest records ``heldout``.
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

from tisiago.seq2func.data import iterate, rna_depth_scale
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT
from tisiago.seq2func.targets import STRAND_FILES, channel_layout, rnacond_layout
from tisiago.seq2func.tracks import track_prefix
from tisiago.seq2func.windows import merge_spans

LIBRARIES = ("tis", "chx")
REPS = (1, 2)
MODES = ("sample", "expected", "float")
SOURCES = ("model", "real-resample")
SPLIT_FILES = {"val": "val.parquet", "test": "test.parquet"}
MANIFEST = "predict_tracks.json"


def pooled_prefix(out_dir: str | Path, condition: str, library: str) -> Path:
    """Path prefix of a ``--mode float`` pooled track."""
    return Path(out_dir) / condition / f"{library}_pooled"


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


def pooled_real_arrays(
    tracks_dir: str | Path, condition: str, region: dict[str, list[tuple[int, int]]]
) -> dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Real replicate-pooled counts in ``region`` per (library, strand) and chromosome.

    Values are ``(canonical positions, counts)`` sorted by position: the sum of the replicate
    tracks with the minus strand shifted back out of the Ribo-TISH convention.
    """
    out = {}
    for library in LIBRARIES:
        for strand, name in STRAND_FILES.items():
            total: dict[str, Counter] = {}
            for rep in REPS:
                prefix = track_prefix(tracks_dir, condition, library, rep)
                counts = region_counts(f"{prefix}.{name}.bw", region, strand_shift(strand))
                for chrom, counter in counts.items():
                    total.setdefault(chrom, Counter()).update(counter)
            per_chrom = {}
            for chrom, counter in total.items():
                positions = np.array(sorted(counter), dtype=np.int64)
                values = np.array([counter[p] for p in positions.tolist()], dtype=np.float64)
                per_chrom[chrom] = (positions - strand_shift(strand), values)
            out[(library, strand)] = per_chrom
    return out


def _real_span(arrays: dict, chrom: str, lo: int, hi: int) -> np.ndarray:
    """Dense counts over canonical ``[lo, hi)`` from :func:`pooled_real_arrays` entries."""
    dense = np.zeros(max(hi - lo, 0))
    positions, values = arrays.get(chrom, (np.empty(0, np.int64), np.empty(0)))
    i, j = np.searchsorted(positions, [lo, hi])
    dense[positions[i:j] - lo] = values[i:j]
    return dense


def predict_condition(
    predict_fn: Callable[[dict], np.ndarray] | None,
    windows: pd.DataFrame,
    *,
    fasta: str | Path | None = None,
    tracks_dir: str | Path,
    genes: pd.DataFrame | None = None,
    condition: str,
    conditions: Sequence[str] = (),
    out_dir: str | Path,
    mode: str = "sample",
    seed: int = 0,
    copies: int = 1,
    mask_unexpressed: bool = False,
    match_depth: bool = False,
    source: str = "model",
    transcripts=None,
    prefetch: int = 4,
    meta: dict | None = None,
    rna_conditioned: bool = False,
) -> dict:
    """Write ``condition``'s predicted per-replicate TIS/CHX bigWigs; return the manifest.

    ``predict_fn(batch)`` maps an ``iterate`` batch to ``(1, S, C)`` pooled predictions in
    :func:`channel_layout` order of ``conditions`` (canonical P-sites). ``mask_unexpressed``
    zeroes positions outside the loss mask (unexpressed genes) before scaling. ``match_depth``
    rescales each library's pooled prediction so its total over the written region equals the
    real pooled total there (one factor per library; the whole prediction is held in memory).
    ``source="real-resample"`` replaces the model by the real pooled tracks as the expectation
    (the same-pipeline noise ceiling); ``predict_fn``, ``fasta``, ``genes`` are then unused.
    ``rna_conditioned`` (``rnacond`` models): batches carry ``condition``'s RNA and predictions
    are in :func:`tisiago.seq2func.targets.rnacond_layout` order; ``conditions`` is unused.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    if copies < 1:
        raise ValueError("copies must be >= 1")
    model = source == "model"
    channel = {}
    if model:
        if rna_conditioned:
            conditions = (condition,)
            layout = rnacond_layout(condition)
        else:
            layout = channel_layout(conditions)
        channel = {
            (lib, s): i
            for i, (c, lib, s) in enumerate(layout)
            if c == condition and lib in LIBRARIES
        }
        if not channel:
            raise ValueError(f"{condition} is not among {tuple(conditions)}")
    pooled_only = mode == "float"
    sizes = chrom_sizes_of(f"{track_prefix(tracks_dir, condition, 'tis', 1)}.plus.bw")
    order = {chrom: i for i, (chrom, _) in enumerate(sizes)}
    unknown = set(windows.chrom) - set(order)
    if unknown:
        raise ValueError(f"window chromosomes {sorted(unknown)} not in the track header")
    windows = windows.assign(_order=windows.chrom.map(order)).sort_values(["_order", "start"])
    windows = windows.drop(columns="_order").reset_index(drop=True)
    spans = owned_spans(windows)
    region = region_of(windows)
    fractions = (
        None
        if pooled_only and not match_depth
        else replicate_fractions(tracks_dir, condition, region)
    )
    size_of = dict(sizes)

    def expectations():
        """Yield ``(chrom, lo, hi, {(library, strand): pooled expectation})`` per window."""
        if not model:
            real = pooled_real_arrays(tracks_dir, condition, region)
            for i, row in enumerate(windows.itertuples()):
                lo, hi = spans[i]
                yield (
                    row.chrom,
                    lo,
                    hi,
                    {key: _real_span(arrays, row.chrom, lo, hi) for key, arrays in real.items()},
                )
            return
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
            rna_conditioned=rna_conditioned,
        )
        try:
            for i, batch in enumerate(batches):
                row = windows.iloc[i]
                start, (lo, hi) = int(row.start), spans[i]
                pred = np.asarray(predict_fn(batch), dtype=np.float64)[0]
                if mask_unexpressed:
                    pred = np.where(batch["mask"][0], pred, 0.0)
                pred = pred[lo - start : hi - start]
                yield row.chrom, lo, hi, {key: pred[:, c] for key, c in channel.items()}
        finally:
            batches.close()

    stream = expectations()
    expected = dict.fromkeys(LIBRARIES, 0.0)
    factors = dict.fromkeys(LIBRARIES, 1.0)
    unscaled = None
    if match_depth:
        stream = list(stream)
        for _, _, _, pooled_by in stream:
            for (library, _), pooled in pooled_by.items():
                expected[library] += float(pooled.sum())
        for library in LIBRARIES:
            real_total = sum(fractions[library]["totals"])
            if expected[library] <= 0:
                raise ValueError(f"{condition} {library}: no predicted counts to depth-match")
            factors[library] = real_total / expected[library]
        unscaled, expected = expected, dict.fromkeys(LIBRARIES, 0.0)

    out_dir = Path(out_dir)
    writers = {}
    rngs = {}
    files = {}
    for k, library in enumerate(LIBRARIES):
        for rep in ("pooled",) if pooled_only else REPS:
            if pooled_only:
                prefix = pooled_prefix(out_dir, condition, library)
            else:
                prefix = track_prefix(out_dir, condition, library, rep)
                rngs[(library, rep)] = np.random.default_rng([seed, k, rep])
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for strand, name in STRAND_FILES.items():
                path = prefix.with_name(f"{prefix.name}.{name}.bw")
                files.setdefault(library, {})[strand] = path.name
                bw = pyBigWig.open(str(path), "w")
                bw.addHeader(sizes)
                writers[(library, rep, strand)] = bw
    written = {lib: [0.0] * (1 if pooled_only else len(REPS)) for lib in LIBRARIES}
    try:
        for chrom, lo, hi, pooled_by in stream:
            for library in LIBRARIES:
                for strand in STRAND_FILES:
                    pooled = pooled_by[(library, strand)] * factors[library]
                    expected[library] += float(pooled.sum())
                    positions = np.arange(lo, hi) + strand_shift(strand)
                    inside = (positions >= 0) & (positions < size_of[chrom])
                    if pooled_only:
                        if inside.any():  # contiguous: one fixed-step block per window
                            writers[(library, "pooled", strand)].addEntries(
                                chrom,
                                int(positions[inside][0]),
                                values=pooled[inside].astype(np.float64).tolist(),
                                span=1,
                                step=1,
                            )
                            written[library][0] += float(pooled[inside].sum())
                        continue
                    for r, rep in enumerate(REPS):
                        share = fractions[library]["fractions"][r]
                        values = _draw(pooled * share, mode, rngs[(library, rep)], copies)
                        keep = inside & (values > 0)
                        if not keep.any():
                            continue
                        writers[(library, rep, strand)].addEntries(
                            chrom,
                            positions[keep].tolist(),
                            values=values[keep].tolist(),
                            span=1,
                        )
                        written[library][r] += float(values[keep].sum())
    finally:
        if not isinstance(stream, list):
            stream.close()
        for bw in writers.values():
            bw.close()

    manifest = {
        **(meta or {}),
        "condition": condition,
        "source": source,
        "mode": mode,
        "layout": "pooled" if pooled_only else "replicates",
        "files": files,
        "seed": seed,
        "copies": copies,
        "mask_unexpressed": mask_unexpressed,
        "match_depth": match_depth,
        "rna_conditioned": rna_conditioned,
        "rna_depth_scale": rna_depth_scale(tracks_dir, condition) if rna_conditioned else None,
        "depth_factors": factors if match_depth else None,
        "unscaled_expected_totals": unscaled,
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
    ap.add_argument(
        "--checkpoint", type=Path, help="run's best/ or last/ (required for --source model)"
    )
    ap.add_argument(
        "--arm",
        help="'shared', 'condition:<C>' or 'rnacond:<group>' (required for --source model)",
    )
    ap.add_argument(
        "--condition",
        help="condition to write (default: the arm's only one; required for rnacond arms)",
    )
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--windows-dir", type=Path, help="default: run.json, else data/p6/windows")
    ap.add_argument("--expression", type=Path, help="default: run.json, else data/p6/expression")
    ap.add_argument("--tracks-dir", type=Path, help="real tracks; default: run.json")
    ap.add_argument(
        "--out-dir", type=Path, help="default: <run>/predicted_<split>[_float for --mode float]"
    )
    ap.add_argument(
        "--source",
        choices=SOURCES,
        default="model",
        help="model: checkpoint predictions; real-resample: Poisson resample of the real pooled\n"
        "tracks (same-pipeline noise ceiling; no checkpoint needed)",
    )
    ap.add_argument(
        "--match-depth",
        action="store_true",
        help="scale each library's pooled prediction to the real pooled total over the region",
    )
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
    model = args.source == "model"
    if model and not (args.checkpoint and args.arm):
        ap.error("--checkpoint and --arm are required for --source model")
    if not model and not (args.condition or args.arm):
        ap.error("--source real-resample needs --condition (or --arm)")
    from tisiago.seq2func.samples import CONDITIONS
    from tisiago.seq2func.train import is_rnacond_arm, parse_arm, parse_rnacond_arm

    conditions: tuple[str, ...] = ()
    rnacond = bool(args.arm) and is_rnacond_arm(args.arm)
    trained: tuple[str, ...] = ()
    if args.arm:
        try:
            if rnacond:
                _, _, trained = parse_rnacond_arm(args.arm)
            else:
                _, conditions = parse_arm(args.arm)
        except ValueError as exc:
            ap.error(str(exc))
    if rnacond:
        if args.condition not in CONDITIONS:
            ap.error(f"rnacond arms need --condition, one of {CONDITIONS}")
        conditions = (args.condition,)
    condition = args.condition or (conditions[0] if len(conditions) == 1 else None)
    if condition is None or (conditions and condition not in conditions):
        ap.error(f"--condition must be one of {conditions}" if conditions else "--condition needed")
    if model:
        for flag in ("genome", "base_checkpoint"):
            if not getattr(args, flag):
                ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")
    if args.out_dir:
        out_dir = args.out_dir
    elif args.checkpoint:
        suffix = "_float" if args.mode == "float" else ""
        out_dir = args.checkpoint.parent / f"predicted_{args.split}{suffix}"
    else:
        ap.error("--out-dir is required without --checkpoint")
    if (out_dir / condition / MANIFEST).exists() and not args.overwrite:
        ap.error(f"{out_dir / condition / MANIFEST} exists; pass --overwrite to replace")
    if args.checkpoint:
        windows_dir = args.windows_dir or _run_arg(
            args.checkpoint, "windows_dir", "data/p6/windows"
        )
    else:
        windows_dir = args.windows_dir or Path("data/p6/windows")
    if args.tracks_dir:
        tracks_dir = args.tracks_dir
    elif args.checkpoint:
        from tisiago.seq2func.evaluate import tracks_dir_of

        tracks_dir = tracks_dir_of(args.checkpoint)
    else:
        ap.error("--tracks-dir is required without --checkpoint")
    windows = pd.read_parquet(windows_dir / SPLIT_FILES[args.split])
    if args.max_windows is not None:
        windows = windows.iloc[: args.max_windows]
    meta = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "arm": args.arm,
        "split": args.split,
        "max_windows": args.max_windows,
        "windows_dir": str(windows_dir),
        "tracks_dir": str(tracks_dir),
    }
    if rnacond:
        meta["heldout"] = condition not in trained
        if args.checkpoint:
            from tisiago.seq2func.evaluate import check_rna_scales

            try:
                check_rna_scales(args.checkpoint, tracks_dir, [condition])
            except ValueError as exc:
                ap.error(str(exc))
    kwargs = {}
    if model:
        from tisiago.seq2func.evaluate import head_kind_of, load_predict_fn, load_transcripts

        expression = args.expression or _run_arg(
            args.checkpoint, "expression", "data/p6/expression/expression.parquet"
        )
        genes = pd.read_parquet(expression).reset_index(drop=True)
        genes = genes[genes.condition.isin(conditions)].reset_index(drop=True)
        head_kind = args.head or head_kind_of(args.checkpoint)
        kwargs = dict(
            predict_fn=load_predict_fn(
                args.checkpoint, args.arm, base_checkpoint=args.base_checkpoint, head_kind=head_kind
            ),
            fasta=args.genome,
            genes=genes,
            conditions=conditions,
            transcripts=load_transcripts(head_kind, args.gtf),
            rna_conditioned=rnacond,
        )
        meta |= {"head": head_kind, "expression": str(expression)}
    manifest = predict_condition(
        kwargs.pop("predict_fn", None),
        windows,
        tracks_dir=tracks_dir,
        condition=condition,
        out_dir=out_dir,
        mode=args.mode,
        seed=args.seed,
        copies=args.copies,
        mask_unexpressed=args.mask_unexpressed,
        match_depth=args.match_depth,
        source=args.source,
        prefetch=args.prefetch,
        meta=meta,
        **kwargs,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
