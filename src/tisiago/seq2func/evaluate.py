"""Held-out evaluation of fine-tuned AlphaGenome checkpoints with a replicate ceiling (P6.2).

Usage::

    python -m tisiago.seq2func.evaluate --ceiling-only --split val
    python -m tisiago.seq2func.evaluate --checkpoint data/p6/runs/shared/best --arm shared \
        --split val
    python -m tisiago.seq2func.evaluate --checkpoint data/p6/runs/shared/best --arm shared \
        --split test --i-am-final
    python -m tisiago.seq2func.evaluate --checkpoint data/p6/runs/rnacond_hela/best \
        --arm rnacond:HeLa --split val --condition HeLa,K562

Writes one row per channel to ``data/p6/eval/<arm>_<split>.tsv`` (model prediction vs pooled
targets), ``data/p6/eval/ceiling_<split>.tsv`` (replicate 1 observed; the profile prediction is
replicate 2 smoothed by a centred 5-bp boxcar plus 1e-3 per position, gene totals and CHX frame
use raw replicate 2) and ``data/p6/eval/uniform_<split>.tsv`` (a flat prediction of the pooled
targets: the KL floor). Metrics: mean masked ``profile_kl`` over windows (NaN windows skipped
and counted); gene-level ``log_pearson`` over the channel's expressed same-strand genes, each
gene totalled once in the window that contains it whose centre is nearest the gene centre (genes
never fully inside a window are excluded and counted); and, for CHX channels, the frame-0
fraction over MANE CDS of genes expressed in the channel's condition, assigned to windows the
same way (canonical P-sites, no shift). ``--max-windows`` runs write ``*_max<N>.tsv`` so a
truncated reference never stands in for the full one; a reused reference must match the
``ceiling_version``, window count and tracks directory.

An ``rnacond:<group>`` checkpoint is evaluated on any condition (held out or not) by feeding
that condition's RNA: one ``<arm>_<condition>_<split>.tsv`` per ``--condition`` (default: all
six), rows for its TIS/CHX channels in the same format plus ``heldout``, ``rna_depth_scale``
and ``rna_total`` (the RNA input's normalisation; checked against the run's ``run.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.data import RNA_REFERENCE_DEPTH, TrackReader, iterate, rna_total
from tisiago.seq2func.head_kinds import CHX_KINDS, HEAD_CHOICES, RNA_KINDS, TRANSCRIPT_KINDS
from tisiago.seq2func.masking import expression_mask
from tisiago.seq2func.metrics import gene_totals, log_pearson, profile_kl
from tisiago.seq2func.qc import _frame_counts
from tisiago.seq2func.samples import CONDITIONS
from tisiago.seq2func.targets import channel_layout, rnacond_layout
from tisiago.seq2func.train import (
    check_rnacond_run,
    is_rnacond_arm,
    parse_arm,
    parse_rnacond_arm,
)

EXCLUDED = -1
OUTSIDE = -2
SPLIT_FILES = {"val": "val.parquet", "test": "test.parquet"}
DEFAULT_TRACKS = Path("data/p6/tracks")
REFERENCES = ("ceiling", "uniform")
REFERENCE_VERSION = 2
CEILING_SMOOTH_BP = 5
CEILING_PSEUDOCOUNT = 1e-3
_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')


def load_mane_cds(gtf: str | Path) -> tuple[list[tuple[str, str, list]], list[str]]:
    """MANE_Select CDS ``(chrom, strand, intervals)`` (as ``qc.load_mane``) and their gene_ids."""
    cds: dict[str, list] = {}
    genes: dict[str, str] = {}
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#") or 'tag "MANE_Select"' not in line:
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "CDS":
                continue
            tid = _TRANSCRIPT_ID.search(f[8]).group(1)
            cds.setdefault(tid, [f[0], f[6], []])[2].append((int(f[3]) - 1, int(f[4])))
            genes[tid] = _GENE_ID.search(f[8]).group(1)
    return [tuple(v) for v in cds.values()], [genes[t] for t in cds]


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
        cds_genes: Sequence[str] = (),
    ):
        """Assign expressed genes of the layout's conditions and MANE CDS to windows.

        ``cds_genes[i]`` is the gene_id of ``cds[i]``; a channel's frame counts use only CDS of
        genes expressed in its condition.
        """
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
        if len(cds_genes) != len(self.cds):
            raise ValueError(f"{len(self.cds)} CDS but {len(cds_genes)} cds_genes")
        expressed = self.genes.groupby("condition").gene_id.agg(set).to_dict()
        self._cds_ok = {
            c: np.array([g in expressed.get(c, set()) for g in cds_genes], dtype=bool)
            for c in conditions
        }
        spans = pd.DataFrame(
            [(c, min(s for s, _ in iv), max(e for _, e in iv)) for c, _, iv in self.cds],
            columns=["chrom", "start", "end"],
        )
        self._cds_by_window = _group(assign_spans(spans, self.windows))
        self.frame = np.zeros((n_channels, 4))  # pred in-frame, pred total, obs in, obs total
        self.frame_cds = np.zeros(n_channels, dtype=np.int64)

    def add(
        self,
        index: int,
        pred: np.ndarray,
        obs: np.ndarray,
        mask: np.ndarray,
        pred_totals: np.ndarray | None = None,
    ) -> None:
        """Add window ``index`` (position in ``windows``): ``(S, C)`` prediction, target, mask.

        ``pred_totals`` replaces ``pred`` for gene totals and CHX frame counts (default ``pred``).
        """
        self.kls.append(profile_kl(pred, obs, mask))
        totals_from = pred if pred_totals is None else pred_totals
        start = int(self.windows.start.iloc[index])
        rows = self._genes_by_window.get(index)
        if rows is not None:
            sub = self.genes.iloc[rows]
            self.obs_totals[rows] = gene_totals(obs, sub, start)
            self.pred_totals[rows] = gene_totals(totals_from, sub, start)
        cds_rows = self._cds_by_window.get(index)
        if cds_rows is None:
            return
        chrom = self.windows.chrom.iloc[index]
        for c, (condition, library, strand) in enumerate(self.layout):
            if library != "chx":
                continue
            ok = self._cds_ok[condition]
            cds = [self.cds[i] for i in cds_rows if self.cds[i][1] == strand and ok[i]]
            if not cds:
                continue
            for k, track in enumerate((totals_from, obs)):
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


def smooth_profile(
    track: np.ndarray, width: int = CEILING_SMOOTH_BP, pseudocount: float = CEILING_PSEUDOCOUNT
) -> np.ndarray:
    """Per-channel centred boxcar mean of ``(S, C)`` ``track`` plus ``pseudocount``."""
    kernel = np.ones(width) / width
    out = np.empty(track.shape, dtype=np.float64)
    for c in range(track.shape[1]):
        out[:, c] = np.convolve(track[:, c].astype(np.float64), kernel, mode="same")
    return out + pseudocount


def reference_metrics(
    kind: str,
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    windows: pd.DataFrame,
    cds: Sequence[tuple[str, str, list[tuple[int, int]]]] = (),
    cds_genes: Sequence[str] = (),
) -> pd.DataFrame:
    """Reference rows for ``kind``.

    ``"ceiling"``: rep1 observed; profile prediction = :func:`smooth_profile` of rep2; gene
    totals and CHX frame counts from raw rep2. ``"uniform"``: pooled observed, flat
    prediction; no gene Pearson.
    """
    if kind not in REFERENCES:
        raise ValueError(f"unknown reference {kind!r}")
    layout = channel_layout(conditions)
    windows = windows.reset_index(drop=True)
    metrics = ChannelMetrics(layout, genes, windows, cds, cds_genes)
    readers = [
        TrackReader(tracks_dir, conditions, rep=r)
        for r in ((1, 2) if kind == "ceiling" else ("pooled",))
    ]
    try:
        for i, row in enumerate(windows.itertuples(index=False)):
            start, end = int(row.start), int(row.end)
            obs = readers[0].read(row.chrom, start, end)
            mask = expression_mask(layout, genes, row.chrom, start, end)
            if kind == "ceiling":
                raw = readers[1].read(row.chrom, start, end)
                metrics.add(i, smooth_profile(raw), obs, mask, pred_totals=raw)
            else:
                metrics.add(i, np.ones_like(obs, dtype=np.float64), obs, mask)
    finally:
        for reader in readers:
            reader.close()
    table = metrics.table()
    if kind == "uniform":
        table["log_pearson"] = np.nan
    else:
        table["ceiling_smooth_bp"] = CEILING_SMOOTH_BP
        table["ceiling_pseudocount"] = CEILING_PSEUDOCOUNT
    table["ceiling_version"] = REFERENCE_VERSION
    return table


def ceiling_metrics(
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    windows: pd.DataFrame,
    cds: Sequence[tuple[str, str, list[tuple[int, int]]]] = (),
    cds_genes: Sequence[str] = (),
) -> pd.DataFrame:
    """Replicate ceiling: rep1 observed vs smoothed rep2 (profile), raw rep2 (genes, frame)."""
    return reference_metrics("ceiling", tracks_dir, conditions, genes, windows, cds, cds_genes)


def uniform_metrics(
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    windows: pd.DataFrame,
    cds: Sequence[tuple[str, str, list[tuple[int, int]]]] = (),
    cds_genes: Sequence[str] = (),
) -> pd.DataFrame:
    """KL floor: a flat prediction of the pooled targets."""
    return reference_metrics("uniform", tracks_dir, conditions, genes, windows, cds, cds_genes)


def load_predict_fn(
    checkpoint: str | Path,
    arm: str,
    *,
    base_checkpoint: str | Path,
    head_kind: str = "mlp",
    device=None,
) -> Callable[[dict], np.ndarray]:
    """``predict(batch) -> (1, S, C)`` from a ``save_checkpoint(save_full_model=True)`` dir.

    A ``resid_chx`` head is rebuilt with the channels of the run's ``mod_libraries`` and
    ``mod_scale_tis`` (as ``train.py`` derives them; runs without them: CHX, ``MOD_SCALE``),
    checked against ``mod_channels`` / ``mod_scales`` in the run's ``run.json`` when recorded.
    """
    import jax.numpy as jnp

    from tisiago.seq2func.model import MOD_SCALE, load_finetuned, make_predict, resid_chx_spec

    rnacond = is_rnacond_arm(arm)
    if rnacond != (head_kind in RNA_KINDS):
        raise ValueError(f"arm {arm} and head kind {head_kind}: rnacond arms need rnacond heads")
    if rnacond:
        head, _, conditions = parse_rnacond_arm(arm)
        layout = rnacond_layout(conditions[0])
        path = Path(checkpoint).parent / "run.json"
        if path.exists():
            check_rnacond_run(json.loads(path.read_text()), arm)
    else:
        head, conditions = parse_arm(arm)
        layout = channel_layout(conditions)
    mod_channels = mod_scales = None
    if head_kind in CHX_KINDS:
        path = Path(checkpoint).parent / "run.json"
        run = json.loads(path.read_text()) if path.exists() else {}
        mod_channels, mod_scales = resid_chx_spec(
            layout, run.get("mod_libraries") or ("chx",), run.get("mod_scale_tis", MOD_SCALE)
        )
        for key, value in (("mod_channels", mod_channels), ("mod_scales", mod_scales)):
            recorded = run.get(key)
            if recorded is not None and tuple(recorded) != value:
                raise ValueError(f"{path} records {key}={recorded}, but arm {arm} gives {value}")
    model = load_finetuned(
        str(checkpoint),
        str(base_checkpoint),
        head,
        len(layout),
        head_kind,
        device=device,
        mod_channels=mod_channels,
        mod_scales=mod_scales,
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
    cds_genes: Sequence[str] = (),
    predict_fn: Callable[[dict], np.ndarray] | None = None,
    base_checkpoint: str | Path | None = None,
    head_kind: str = "mlp",
    prefetch: int = 4,
    transcripts=None,
    condition: str | None = None,
) -> pd.DataFrame:
    """Per-channel metrics of the arm's predictions against pooled targets.

    ``predict_fn(batch)`` maps an ``iterate`` batch to ``(1, S, C)`` predictions; when omitted
    it is loaded from ``checkpoint`` with :func:`load_predict_fn`. ``transcripts`` (see
    :func:`load_transcripts`) adds the ``seqconv_tx`` input channels. ``rnacond`` arms need
    ``condition`` (any of :data:`CONDITIONS`): its RNA is the input, its TIS/CHX the targets.
    """
    rnacond = is_rnacond_arm(arm)
    if rnacond:
        parse_rnacond_arm(arm)
        if condition not in CONDITIONS:
            raise ValueError(f"{arm} needs a condition from {CONDITIONS}, not {condition!r}")
        conditions, layout = (condition,), rnacond_layout(condition)
    else:
        if condition is not None:
            raise ValueError(f"condition is only for rnacond arms, not {arm}")
        _, conditions = parse_arm(arm)
        layout = channel_layout(conditions)
    if predict_fn is None:
        predict_fn = load_predict_fn(
            checkpoint, arm, base_checkpoint=base_checkpoint, head_kind=head_kind
        )
    windows = windows.reset_index(drop=True)
    genes = genes[genes.condition.isin(conditions)].reset_index(drop=True)
    metrics = ChannelMetrics(layout, genes, windows, cds, cds_genes)
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
        rna_conditioned=rnacond,
    )
    try:
        for i, batch in enumerate(batches):
            pred = np.asarray(predict_fn(batch), dtype=np.float64)[0]
            metrics.add(i, pred, batch["targets"][0], batch["mask"][0])
    finally:
        batches.close()
    table = metrics.table()
    if rnacond:
        total = rna_total(tracks_dir, condition)
        table["rna_depth_scale"] = RNA_REFERENCE_DEPTH / total
        table["rna_total"] = total
    return table


def check_rna_scales(checkpoint: Path, tracks_dir: Path, conditions) -> None:
    """Raise ``ValueError`` if the run's recorded RNA depth scales differ from ``tracks_dir``'s."""
    run = Path(checkpoint).parent / "run.json"
    recorded = json.loads(run.read_text()).get("rna_depth_scales", {}) if run.exists() else {}
    for condition in conditions:
        if condition not in recorded:
            continue
        scale = RNA_REFERENCE_DEPTH / rna_total(tracks_dir, condition)
        if not np.isclose(recorded[condition], scale, rtol=1e-9, atol=0):
            raise ValueError(
                f"{run} records rna_depth_scales[{condition}]={recorded[condition]}, but "
                f"{tracks_dir} gives {scale}"
            )


def output_paths(
    out_dir: str | Path,
    arm: str | None,
    split: str,
    max_windows: int | None = None,
    condition: str | None = None,
) -> dict[str, Path]:
    """``{"model": <arm>_<split>.tsv, "ceiling"|"uniform": <kind>_<split>.tsv}`` in ``out_dir``.

    With ``max_windows`` both names gain ``_max<N>`` so truncated runs never take the full names.
    A ``condition`` (rnacond arms) is added to the model name: ``<arm>_<condition>_<split>``.
    """
    out_dir = Path(out_dir)
    suffix = f"{split}_max{max_windows}" if max_windows is not None else split
    paths = {kind: out_dir / f"{kind}_{suffix}.tsv" for kind in REFERENCES}
    if arm is not None:
        name = arm.replace(":", "_") + (f"_{condition}" if condition else "")
        paths["model"] = out_dir / f"{name.lower()}_{suffix}.tsv"
    return paths


def run_args(checkpoint: Path) -> dict:
    """``args`` of the run's ``run.json`` next to ``best/``/``last/``; empty if absent."""
    run = Path(checkpoint).parent / "run.json"
    return json.loads(run.read_text()).get("args", {}) if run.exists() else {}


def head_kind_of(checkpoint: Path) -> str:
    """Head kind recorded in the run's ``run.json``; else mlp."""
    return run_args(checkpoint).get("head") or "mlp"


def load_transcripts(head_kind: str, gtf: str | Path | None):
    """``TranscriptFeatures`` from ``gtf`` for a :data:`TRANSCRIPT_KINDS` head, else ``None``."""
    if head_kind not in TRANSCRIPT_KINDS:
        return None
    if not gtf:
        raise ValueError(f"head kind {head_kind} needs --gtf (or TISIAGO_GTF)")
    from tisiago.seq2func.tx_features import TranscriptFeatures

    return TranscriptFeatures.from_gtf(gtf)


def tracks_dir_of(checkpoint: Path | None) -> Path:
    """Tracks directory the run trained on (``run.json``); else ``data/p6/tracks``."""
    recorded = run_args(checkpoint).get("tracks_dir") if checkpoint is not None else None
    return Path(recorded) if recorded else DEFAULT_TRACKS


def ceiling_mismatch(path: Path, n_windows: int, tracks_dir: Path) -> str | None:
    """Why the existing reference at ``path`` cannot be reused for this run, or ``None``."""
    table = pd.read_csv(path, sep="\t")
    versions = set(table["ceiling_version"]) if "ceiling_version" in table else set()
    if versions != {REFERENCE_VERSION}:
        return (
            f"{path} has ceiling_version {sorted(versions) or 'missing'}, not {REFERENCE_VERSION}"
        )
    counts = set(table["windows"]) if "windows" in table else set()
    if counts != {n_windows}:
        return f"{path} was computed on {sorted(counts) or 'unknown'} windows, not {n_windows}"
    recorded = set(table["tracks_dir"]) if "tracks_dir" in table else set()
    if {str(Path(t).resolve()) for t in recorded} != {str(Path(tracks_dir).resolve())}:
        return f"{path} was computed from tracks {sorted(recorded)}, not {tracks_dir}"
    return None


def write_tsv(table: pd.DataFrame, path: Path) -> None:
    """Write ``table`` to ``path`` through a sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="run's best/ or last/ directory")
    ap.add_argument("--arm", help="'shared', 'condition:<C>' or 'rnacond:<held-out group>'")
    ap.add_argument(
        "--condition",
        default=None,
        help="rnacond arms: comma list of conditions to evaluate (default: all six)",
    )
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument(
        "--ceiling-only", action="store_true", help="CPU references only (ceiling + uniform)"
    )
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/p6/expression/expression.parquet")
    )
    ap.add_argument(
        "--tracks-dir",
        type=Path,
        default=None,
        help="default: the run's run.json tracks_dir, else data/p6/tracks",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/p6/eval"))
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows (smoke)")
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"), help="MANE_Select GTF")
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--head", choices=HEAD_CHOICES, default=None, help="default: run.json")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument(
        "--overwrite", action="store_true", help="replace existing output TSVs (recomputes ceiling)"
    )
    return ap


def main() -> None:
    """CLI: evaluate one checkpoint on a split; write the ceiling and uniform refs if missing."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    rnacond = args.arm is not None and is_rnacond_arm(args.arm)
    if not args.ceiling_only:
        if args.arm is None or args.checkpoint is None:
            ap.error("--arm and --checkpoint are required unless --ceiling-only")
        try:
            parse_rnacond_arm(args.arm) if rnacond else parse_arm(args.arm)
        except ValueError as exc:
            ap.error(str(exc))
    if args.condition is not None and not (rnacond and not args.ceiling_only):
        ap.error("--condition is only for rnacond arms")
    conditions: list[str | None] = [None]
    if rnacond:
        conditions = args.condition.split(",") if args.condition else list(CONDITIONS)
        unknown = sorted(set(conditions) - set(CONDITIONS))
        if unknown or len(set(conditions)) != len(conditions):
            ap.error(f"--condition must list distinct conditions from {CONDITIONS}")
    arm = None if args.ceiling_only else args.arm
    paths = output_paths(args.out_dir, arm, args.split, args.max_windows)
    models = {
        c: output_paths(args.out_dir, arm, args.split, args.max_windows, c) for c in conditions
    }
    if args.ceiling_only:
        targets = [paths[k] for k in REFERENCES]
        if all(t.exists() for t in targets) and not args.overwrite:
            ap.error(f"{', '.join(map(str, targets))} exist; pass --overwrite to replace")
    else:
        existing = [str(p["model"]) for p in models.values() if p["model"].exists()]
        if existing and not args.overwrite:
            ap.error(f"{', '.join(existing)} exist; pass --overwrite to replace")
    required = ("gtf",) if args.ceiling_only else ("genome", "gtf", "base_checkpoint")
    for flag in required:
        if not getattr(args, flag):
            ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")

    windows = pd.read_parquet(args.windows_dir / SPLIT_FILES[args.split])
    if args.max_windows is not None:
        windows = windows.iloc[: args.max_windows]
    windows = windows.reset_index(drop=True)
    tracks_dir = args.tracks_dir or tracks_dir_of(None if args.ceiling_only else args.checkpoint)
    reuse = {k: paths[k].exists() and not args.overwrite for k in REFERENCES}
    for kind in REFERENCES:
        why = reuse[kind] and ceiling_mismatch(paths[kind], len(windows), tracks_dir)
        if why:
            ap.error(f"{why}; pass --overwrite to recompute it")
    genes = pd.read_parquet(args.expression).reset_index(drop=True)
    meta = {"split": args.split, "windows": len(windows), "tracks_dir": str(tracks_dir)}
    cds, cds_genes = load_mane_cds(args.gtf)

    for kind in REFERENCES:
        if reuse[kind]:
            print(f"keeping existing {paths[kind]}", flush=True)
            continue
        t0 = time.time()
        table = reference_metrics(kind, tracks_dir, CONDITIONS, genes, windows, cds, cds_genes)
        write_tsv(table.assign(source=kind, **meta), paths[kind])
        print(f"{kind} {time.time() - t0:.0f}s -> {paths[kind]}", flush=True)
    if args.ceiling_only:
        return

    if rnacond:
        try:
            check_rna_scales(args.checkpoint, tracks_dir, conditions)
        except ValueError as exc:
            ap.error(str(exc))
    t0 = time.time()
    head_kind = args.head or head_kind_of(args.checkpoint)
    transcripts = load_transcripts(head_kind, args.gtf)
    predict_fn = load_predict_fn(
        args.checkpoint, args.arm, base_checkpoint=args.base_checkpoint, head_kind=head_kind
    )
    trained = set(parse_rnacond_arm(args.arm)[2]) if rnacond else set()
    for condition in conditions:
        t0 = time.time()
        table = evaluate_checkpoint(
            args.checkpoint,
            args.arm,
            windows,
            fasta=args.genome,
            tracks_dir=tracks_dir,
            genes=genes,
            cds=cds,
            cds_genes=cds_genes,
            predict_fn=predict_fn,
            prefetch=args.prefetch,
            transcripts=transcripts,
            condition=condition,
        )
        table = table.assign(source="model", arm=args.arm, checkpoint=str(args.checkpoint), **meta)
        if rnacond:
            table["heldout"] = condition not in trained
        path = models[condition]["model"]
        write_tsv(table, path)
        print(f"model {time.time() - t0:.0f}s -> {path}", flush=True)
        print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
