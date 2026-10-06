"""Held-out evaluation of fine-tuned AlphaGenome checkpoints with a replicate ceiling (P6.2).

Usage::

    python -m tisiago.seq2func.evaluate --ceiling-only --split val
    python -m tisiago.seq2func.evaluate --checkpoint data/p6/runs/shared/best --arm shared \
        --split val
    python -m tisiago.seq2func.evaluate --checkpoint data/p6/runs/shared/best --arm shared \
        --split test --i-am-final

Writes one row per channel to ``data/p6/eval/<arm>_<split>.tsv`` (model prediction vs pooled
targets) and ``data/p6/eval/ceiling_<split>.tsv`` (replicate 2 used as the prediction of
replicate 1). Metrics: mean masked ``profile_kl`` over windows (NaN windows skipped and
counted); gene-level ``log_pearson`` over the channel's expressed same-strand genes, each gene
totalled once in the window that contains it whose centre is nearest the gene centre (genes
never fully inside a window are excluded and counted); and, for CHX channels, the frame-0
fraction over MANE CDS assigned to windows the same way (canonical P-sites, no shift).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.data import TrackReader, iterate
from tisiago.seq2func.masking import expression_mask
from tisiago.seq2func.metrics import gene_totals, log_pearson, profile_kl
from tisiago.seq2func.qc import _frame_counts, load_mane
from tisiago.seq2func.samples import CONDITIONS
from tisiago.seq2func.targets import channel_layout
from tisiago.seq2func.train import parse_arm

EXCLUDED = -1
OUTSIDE = -2
SPLIT_FILES = {"val": "val.parquet", "test": "test.parquet"}


def assign_spans(spans: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """Window position per span: the containing window with the nearest centre.

    Returns ``EXCLUDED`` for spans that overlap a window but lie fully inside none, and
    ``OUTSIDE`` for spans that overlap no window. Ties go to the earlier window.
    """
    out = np.full(len(spans), OUTSIDE, dtype=np.int64)
    s_chrom = spans["chrom"].to_numpy()
    w_chrom = windows["chrom"].to_numpy()
    for chrom in np.unique(s_chrom):
        rows = np.flatnonzero(s_chrom == chrom)
        cols = np.flatnonzero(w_chrom == chrom)
        if not len(cols):
            continue
        s0 = spans["start"].to_numpy()[rows, None]
        s1 = spans["end"].to_numpy()[rows, None]
        w0 = windows["start"].to_numpy()[None, cols]
        w1 = windows["end"].to_numpy()[None, cols]
        overlaps = ((s1 > w0) & (s0 < w1)).any(axis=1)
        inside = (s0 >= w0) & (s1 <= w1)
        distance = np.where(inside, np.abs((s0 + s1) - (w0 + w1)), np.iinfo(np.int64).max)
        best = distance.argmin(axis=1)
        out[rows] = np.where(inside.any(axis=1), cols[best], np.where(overlaps, EXCLUDED, OUTSIDE))
    return out


class _Column:
    """Read-only ``{genomic position: value}`` view of one window channel for ``_frame_counts``."""

    def __init__(self, values: np.ndarray, start: int):
        self.values, self.start = values, start

    def get(self, position: int, default=0):
        i = position - self.start
        return float(self.values[i]) if 0 <= i < len(self.values) else default


def channel_kl(kls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel NaN-skipping mean over windows of ``(W, C)`` KLs, and finite counts."""
    finite = np.isfinite(kls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.where(finite, kls, np.nan), axis=0)
    return mean, finite.sum(axis=0)


class ChannelMetrics:
    """Accumulates per-channel KL, gene totals and CHX frame counts over evaluation windows."""

    def __init__(
        self,
        layout: list[tuple[str, str, str]],
        genes: pd.DataFrame,
        windows: pd.DataFrame,
        cds: Sequence[tuple[str, str, list[tuple[int, int]]]] = (),
    ):
        """Assign expressed genes of the layout's conditions and MANE CDS to windows."""
        self.layout = layout
        self.windows = windows.reset_index(drop=True)
        conditions = {c for c, _, _ in layout}
        keep = genes.condition.isin(conditions) & genes.expressed.astype(bool)
        self.genes = genes[keep].reset_index(drop=True)
        self.where = assign_spans(self.genes, self.windows)
        self._genes_by_window = _group(self.where)
        n_channels = len(layout)
        self.obs_totals = np.zeros((len(self.genes), n_channels))
        self.pred_totals = np.zeros((len(self.genes), n_channels))
        self.kls: list[np.ndarray] = []
        self.cds = list(cds)
        spans = pd.DataFrame(
            [(c, min(s for s, _ in iv), max(e for _, e in iv)) for c, _, iv in self.cds],
            columns=["chrom", "start", "end"],
        )
        self._cds_by_window = _group(assign_spans(spans, self.windows))
        self.frame = np.zeros((n_channels, 4))  # pred in-frame, pred total, obs in, obs total
        self.frame_cds = np.zeros(n_channels, dtype=np.int64)

    def add(self, index: int, pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> None:
        """Add window ``index`` (position in ``windows``): ``(S, C)`` prediction, target, mask."""
        self.kls.append(profile_kl(pred, obs, mask))
        start = int(self.windows.start.iloc[index])
        rows = self._genes_by_window.get(index)
        if rows is not None:
            sub = self.genes.iloc[rows]
            self.obs_totals[rows] = gene_totals(obs, sub, start)
            self.pred_totals[rows] = gene_totals(pred, sub, start)
        cds_rows = self._cds_by_window.get(index)
        if cds_rows is None:
            return
        chrom = self.windows.chrom.iloc[index]
        for c, (_, library, strand) in enumerate(self.layout):
            if library != "chx":
                continue
            cds = [self.cds[i] for i in cds_rows if self.cds[i][1] == strand]
            if not cds:
                continue
            for k, track in enumerate((pred, obs)):
                counts = {chrom: _Column(track[:, c], start)}
                self.frame[c, 2 * k : 2 * k + 2] += _frame_counts(counts, cds, shift=0)
            self.frame_cds[c] += len(cds)

    def table(self) -> pd.DataFrame:
        """One row per channel: KL, gene-level log Pearson, CHX frame-0 fractions."""
        n_windows = len(self.kls)
        kls = np.stack(self.kls) if self.kls else np.full((0, len(self.layout)), np.nan)
        mean_kl, finite = channel_kl(kls)
        rows = []
        for c, (condition, library, strand) in enumerate(self.layout):
            same = (self.genes.condition == condition).to_numpy() & (
                self.genes.strand == strand
            ).to_numpy()
            sel = same & (self.where >= 0)
            pearson = log_pearson(self.pred_totals[sel][:, [c]], self.obs_totals[sel][:, [c]])
            chx = library == "chx"
            pred_in, pred_tot, obs_in, obs_tot = self.frame[c]
            rows.append(
                {
                    "condition": condition,
                    "library": library,
                    "strand": strand,
                    "channel": c,
                    "profile_kl": float(mean_kl[c]),
                    "kl_windows": int(finite[c]),
                    "kl_skipped": int(n_windows - finite[c]),
                    "log_pearson": float(pearson[0]),
                    "n_genes": int(sel.sum()),
                    "genes_excluded": int((same & (self.where == EXCLUDED)).sum()),
                    "frame0_pred": pred_in / pred_tot if chx and pred_tot else np.nan,
                    "frame0_obs": obs_in / obs_tot if chx and obs_tot else np.nan,
                    "frame0_cds": int(self.frame_cds[c]) if chx else 0,
                }
            )
        return pd.DataFrame(rows)


def _group(where: np.ndarray) -> dict[int, np.ndarray]:
    return {int(w): np.flatnonzero(where == w) for w in np.unique(where[where >= 0])}


def ceiling_metrics(
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Replicate ceiling: rep1 targets as observed, rep2 targets as the prediction."""
    layout = channel_layout(conditions)
    windows = windows.reset_index(drop=True)
    metrics = ChannelMetrics(layout, genes, windows)
    rep1 = TrackReader(tracks_dir, conditions, rep=1)
    rep2 = TrackReader(tracks_dir, conditions, rep=2)
    try:
        for i, row in enumerate(windows.itertuples(index=False)):
            start, end = int(row.start), int(row.end)
            obs = rep1.read(row.chrom, start, end)
            pred = rep2.read(row.chrom, start, end)
            metrics.add(i, pred, obs, expression_mask(layout, genes, row.chrom, start, end))
    finally:
        rep1.close()
        rep2.close()
    return metrics.table()


def load_predict_fn(
    checkpoint: str | Path, arm: str, *, base_checkpoint: str | Path, head_kind: str = "mlp"
) -> Callable[[dict], np.ndarray]:
    """``predict(batch) -> (1, S, C)`` from a ``save_checkpoint(save_full_model=True)`` dir."""
    import jax.numpy as jnp
    from alphagenome_ft import load_checkpoint
    from alphagenome_research.model import dna_model

    from tisiago.seq2func.model import ORGANISM, make_predict, register_track_head

    head, conditions = parse_arm(arm)
    register_track_head(head, len(channel_layout(conditions)), kind=head_kind)
    model = load_checkpoint(
        checkpoint,
        base_checkpoint_path=base_checkpoint,
        organism_settings={ORGANISM: dna_model.OrganismSettings()},
    )
    predict = make_predict(model, head)
    params, state = model._params, model._state
    return lambda batch: np.asarray(predict(params, state, jnp.asarray(batch["sequences"])))


def evaluate_checkpoint(
    checkpoint: str | Path | None,
    arm: str,
    windows: pd.DataFrame,
    *,
    fasta: str | Path,
    tracks_dir: str | Path,
    genes: pd.DataFrame,
    cds: Sequence[tuple[str, str, list[tuple[int, int]]]] = (),
    predict_fn: Callable[[dict], np.ndarray] | None = None,
    base_checkpoint: str | Path | None = None,
    head_kind: str = "mlp",
    prefetch: int = 4,
) -> pd.DataFrame:
    """Per-channel metrics of the arm's predictions against pooled targets.

    ``predict_fn(batch)`` maps an ``iterate`` batch to ``(1, S, C)`` predictions; when omitted
    it is loaded from ``checkpoint`` with :func:`load_predict_fn`.
    """
    _, conditions = parse_arm(arm)
    if predict_fn is None:
        predict_fn = load_predict_fn(
            checkpoint, arm, base_checkpoint=base_checkpoint, head_kind=head_kind
        )
    windows = windows.reset_index(drop=True)
    genes = genes[genes.condition.isin(conditions)].reset_index(drop=True)
    metrics = ChannelMetrics(channel_layout(conditions), genes, windows, cds)
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
    )
    try:
        for i, batch in enumerate(batches):
            pred = np.asarray(predict_fn(batch), dtype=np.float64)[0]
            metrics.add(i, pred, batch["targets"][0], batch["mask"][0])
    finally:
        batches.close()
    return metrics.table()


def output_paths(out_dir: str | Path, arm: str | None, split: str) -> dict[str, Path]:
    """``{"model": <arm>_<split>.tsv, "ceiling": ceiling_<split>.tsv}`` under ``out_dir``."""
    out_dir = Path(out_dir)
    paths = {"ceiling": out_dir / f"ceiling_{split}.tsv"}
    if arm is not None:
        paths["model"] = out_dir / f"{arm.replace(':', '_').lower()}_{split}.tsv"
    return paths


def head_kind_of(checkpoint: Path) -> str:
    """Head kind recorded in the run's ``run.json`` next to ``best/``/``last/``; else mlp."""
    run = Path(checkpoint).parent / "run.json"
    if run.exists():
        return json.loads(run.read_text()).get("args", {}).get("head", "mlp")
    return "mlp"


def write_tsv(table: pd.DataFrame, path: Path) -> None:
    """Write ``table`` to ``path`` through a sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    table.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="run's best/ or last/ directory")
    ap.add_argument("--arm", help="'shared' or 'condition:<C>'")
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--ceiling-only", action="store_true", help="CPU replicate ceiling only")
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/p6/expression/expression.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/p6/tracks"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/p6/eval"))
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows (smoke)")
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"), help="MANE_Select GTF")
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--head", choices=("mlp", "conv"), default=None, help="default: run.json")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="replace existing output TSVs")
    return ap


def main() -> None:
    """CLI: evaluate one checkpoint on a split and write the replicate ceiling if missing."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    if not args.ceiling_only:
        if args.arm is None or args.checkpoint is None:
            ap.error("--arm and --checkpoint are required unless --ceiling-only")
        try:
            parse_arm(args.arm)
        except ValueError as exc:
            ap.error(str(exc))
    paths = output_paths(args.out_dir, None if args.ceiling_only else args.arm, args.split)
    target = paths["ceiling"] if args.ceiling_only else paths["model"]
    if target.exists() and not args.overwrite:
        ap.error(f"{target} exists; pass --overwrite to replace it")
    if not args.ceiling_only:
        for flag in ("genome", "gtf", "base_checkpoint"):
            if not getattr(args, flag):
                ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")

    windows = pd.read_parquet(args.windows_dir / SPLIT_FILES[args.split])
    if args.max_windows is not None:
        windows = windows.iloc[: args.max_windows]
    windows = windows.reset_index(drop=True)
    genes = pd.read_parquet(args.expression)
    meta = {"split": args.split, "windows": len(windows)}

    if args.overwrite or not paths["ceiling"].exists():
        t0 = time.time()
        ceiling = ceiling_metrics(args.tracks_dir, CONDITIONS, genes, windows)
        write_tsv(ceiling.assign(source="ceiling", **meta), paths["ceiling"])
        print(f"ceiling {time.time() - t0:.0f}s -> {paths['ceiling']}", flush=True)
    else:
        print(f"keeping existing {paths['ceiling']}", flush=True)
    if args.ceiling_only:
        return

    t0 = time.time()
    _, cds = load_mane(args.gtf)
    table = evaluate_checkpoint(
        args.checkpoint,
        args.arm,
        windows,
        fasta=args.genome,
        tracks_dir=args.tracks_dir,
        genes=genes,
        cds=cds,
        base_checkpoint=args.base_checkpoint,
        head_kind=args.head or head_kind_of(args.checkpoint),
        prefetch=args.prefetch,
    )
    table = table.assign(source="model", arm=args.arm, checkpoint=str(args.checkpoint), **meta)
    write_tsv(table, paths["model"])
    print(f"model {time.time() - t0:.0f}s -> {paths['model']}", flush=True)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
