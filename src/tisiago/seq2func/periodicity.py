"""CHX 3-nt periodicity diagnostic for fine-tuned checkpoints (P6.3).

Usage::

    python -m tisiago.seq2func.periodicity --checkpoint data/p6/runs/condition_hela/best
    python -m tisiago.seq2func.periodicity --checkpoint data/p6/runs/condition_hela/best \
        --max-windows 10 --run-name smoke
    python -m tisiago.seq2func.periodicity --reference replicate --arm condition:HeLa

Ribosome footprints are codon-quantized, but out-of-frame reads are real (uORFs, overlapping
ORFs, alternative frames), so the primary metrics are frame-agnostic and local: period-3
spectral power, block phase-composition KL (the training loss, :func:`block_phase_kl`) and
per-block dominant-phase agreement and switch rates. Annotation-anchored frame-0 fractions are
kept as secondary ``annot_*`` columns (MANE CDS only).

Regions are MANE_Select CDS and 5'UTRs (spliced, transcript-oriented, canonical P-sites) of
genes expressed in the channel's condition, each assigned to the window that contains it with
the nearest centre (``evaluate.assign_spans``); regions in no single window are skipped. 5'UTRs
need ``--min-utr-counts`` observed counts. CDS rows come per segment: ``all``, ``single``
(single-exon CDS), ``first`` (first CDS exon) and ``rest`` (later exons). Phase blocks use each
segment's own grid. Writes ``regions.tsv`` (one row per channel x region x segment) and
``summary.md`` to ``data/p6/periodicity/<run_name>/``. Runs on CPU by default.
Dominant-phase metrics use blocks with at least ``--min-block-counts`` observed counts.
``--reference replicate`` writes the same rows for raw replicate 2 scored against replicate 1
(``*_replicate`` files, no model): the ceiling the model rows are read against.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.evaluate import (
    EXCLUDED,
    OUTSIDE,
    SPLIT_FILES,
    _group,
    assign_spans,
    head_kind_of,
    load_mane_cds,
    load_predict_fn,
    load_transcripts,
    run_args,
    tracks_dir_of,
    write_tsv,
)
from tisiago.seq2func.overfit import HEAD_CHOICES
from tisiago.seq2func.targets import channel_layout
from tisiago.seq2func.train import parse_arm

DEFAULT_BLOCK = 10
DEFAULT_MIN_BLOCK_COUNTS = 10.0
DEFAULT_MIN_UTR_COUNTS = 10.0
REFERENCES = ("replicate",)
HF_MAX_PERIOD = 10
_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')


def region_positions(strand: str, intervals: Sequence[tuple[int, int]]):
    """Genomic positions in transcript order and the exon rank of each (0 = 5'-most)."""
    exons = sorted(intervals, reverse=strand == "-")
    positions, ranks = [], []
    for rank, (start, end) in enumerate(exons):
        span = range(start, end) if strand == "+" else range(end - 1, start - 1, -1)
        positions.extend(span)
        ranks.extend([rank] * (end - start))
    return np.asarray(positions, dtype=np.int64), np.asarray(ranks, dtype=np.int64)


def phase_counts(values: np.ndarray, block: int = 1) -> np.ndarray:
    """``(n_blocks, 3)`` per-phase sums over consecutive blocks of ``block`` codons."""
    values = np.asarray(values, dtype=np.float64)
    n = (len(values) // (3 * block)) * 3 * block
    return values[:n].reshape(-1, block, 3).sum(axis=1)


def block_phase_kl(pred: np.ndarray, obs: np.ndarray, block: int) -> tuple[float, float]:
    """``(sum n * KL(obs phase || pred phase), sum n)`` over blocks with observed counts."""
    p, o = phase_counts(pred, block), phase_counts(obs, block)
    n = o.sum(axis=1)
    keep = n > 0
    if not keep.any():
        return 0.0, 0.0
    qo = o[keep] / n[keep, None]
    qp = p[keep] / np.maximum(p[keep].sum(axis=1, keepdims=True), 1e-7)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(qo > 0, qo * (np.log(qo) - np.log(np.maximum(qp, 1e-7))), 0.0)
    return float((n[keep] * terms.sum(axis=1)).sum()), float(n[keep].sum())


def period3_power(values: np.ndarray) -> float:
    """Fraction of non-DC spectral power at period 3 (length trimmed to a multiple of 3)."""
    values = np.asarray(values, dtype=np.float64)
    n = (len(values) // 3) * 3
    if n < 6:
        return np.nan
    power = np.abs(np.fft.rfft(values[:n])) ** 2
    total = power[1:].sum()
    return float(power[n // 3] / total) if total > 0 else 0.0


def high_freq_power(values: np.ndarray, max_period: int = HF_MAX_PERIOD) -> float:
    """Spectral power at periods < ``max_period`` nt of the unit-sum profile (NaN if empty)."""
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if total <= 0 or len(values) < max_period:
        return np.nan
    power = np.abs(np.fft.rfft(values / total)) ** 2
    k = np.arange(len(power))
    return float(power[k * max_period > len(values)].sum())


def dominant_phases(values: np.ndarray, block: int, min_counts: float = 0.0) -> np.ndarray:
    """Dominant phase per block; -1 where the block total is below ``min_counts`` or tied."""
    counts = phase_counts(values, block)
    top = counts.max(axis=1)
    unique = (counts == top[:, None]).sum(axis=1) == 1
    return np.where(
        unique & (counts.sum(axis=1) >= min_counts) & (top > 0), counts.argmax(axis=1), -1
    )


def block_ranks(ranks: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Exon rank at the first and last position of each block of ``block`` codons."""
    size = 3 * block
    n = (len(ranks) // size) * size
    blocks = np.asarray(ranks[:n]).reshape(-1, size)
    return blocks[:, 0], blocks[:, -1]


def switch_counts(
    dominant: np.ndarray, first_rank: np.ndarray, last_rank: np.ndarray
) -> tuple[int, int, int, int]:
    """Dominant-phase switches between consecutive informative blocks (``dominant >= 0``).

    Returns ``(switches, pairs)`` for pairs inside one exon, then for pairs that span a splice
    junction (the first block's start and the second block's end lie in different exons).
    """
    informative = np.flatnonzero(dominant >= 0)
    a, b = informative[:-1], informative[1:]
    changed = dominant[a] != dominant[b]
    across = first_rank[a] != last_rank[b]
    return (
        int((changed & ~across).sum()),
        int((~across).sum()),
        int((changed & across).sum()),
        int(across.sum()),
    )


def segment_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    ranks: np.ndarray,
    block: int = DEFAULT_BLOCK,
    cds_offset: int | None = None,
    min_block_counts: float = DEFAULT_MIN_BLOCK_COUNTS,
) -> dict:
    """Frame-agnostic metrics of one transcript-ordered segment; ``annot_*`` when ``cds_offset``.

    Dominant-phase agreement and switch rates use only blocks whose observed total is at least
    ``min_block_counts`` (a shot-noise floor; ``dom_obs_blocks`` counts them). Predicted switch
    rates use the blocks where the observed dominant phase is defined, so both rates count the
    same block pairs. ``cds_offset`` is the segment's first position within the
    CDS (annotated frame = index % 3).
    """
    pred, obs = np.asarray(pred, np.float64), np.asarray(obs, np.float64)
    kl_num, kl_den = block_phase_kl(pred, obs, block)
    dom_obs = dominant_phases(obs, block, min_block_counts)
    dom_pred = dominant_phases(pred, block)
    both = (dom_obs >= 0) & (dom_pred >= 0)
    first, last = block_ranks(ranks, block)
    sw_obs = switch_counts(dom_obs, first, last)
    sw_pred = switch_counts(np.where(dom_obs >= 0, dom_pred, -1), first, last)
    row = {
        "length": len(obs),
        "obs_total": obs.sum(),
        "pred_total": pred.sum(),
        "p3_obs": period3_power(obs),
        "p3_pred": period3_power(pred),
        "hf_obs": high_freq_power(obs),
        "hf_pred": high_freq_power(pred),
        "block_kl_num": kl_num,
        "block_kl_den": kl_den,
        "dom_agree": int((dom_obs[both] == dom_pred[both]).sum()),
        "dom_blocks": int(both.sum()),
        "dom_obs_blocks": int((dom_obs >= 0).sum()),
        "min_block_counts": min_block_counts,
        "switch_obs_within": sw_obs[0],
        "pairs_obs_within": sw_obs[1],
        "switch_obs_junction": sw_obs[2],
        "pairs_obs_junction": sw_obs[3],
        "switch_pred_within": sw_pred[0],
        "pairs_pred_within": sw_pred[1],
        "switch_pred_junction": sw_pred[2],
        "pairs_pred_junction": sw_pred[3],
    }
    for name, track in (("obs", obs), ("pred", pred)):
        for f in range(3):
            row[f"annot_f{f}_{name}"] = (
                track[(np.arange(len(track)) + cds_offset) % 3 == f].sum()
                if cds_offset is not None
                else np.nan
            )
    return row


def load_mane_utr5(gtf: str | Path) -> tuple[list[tuple[str, str, list]], list[str]]:
    """MANE_Select 5'UTR ``(chrom, strand, intervals)`` (UTR pieces upstream of the CDS)."""
    utr: dict[str, list] = {}
    bounds: dict[str, list[int]] = {}
    genes: dict[str, str] = {}
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#") or 'tag "MANE_Select"' not in line:
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("UTR", "CDS"):
                continue
            tid = _TRANSCRIPT_ID.search(f[8]).group(1)
            start, end = int(f[3]) - 1, int(f[4])
            genes[tid] = _GENE_ID.search(f[8]).group(1)
            if f[2] == "CDS":
                b = bounds.setdefault(tid, [start, end])
                b[0], b[1] = min(b[0], start), max(b[1], end)
            else:
                utr.setdefault(tid, [f[0], f[6], []])[2].append((start, end))
    out, out_genes = [], []
    for tid, (chrom, strand, pieces) in utr.items():
        if tid not in bounds:
            continue
        lo, hi = bounds[tid]
        five = [(s, e) for s, e in pieces if (e <= lo if strand == "+" else s >= hi)]
        if five:
            out.append((chrom, strand, sorted(five)))
            out_genes.append(genes[tid])
    return out, out_genes


def region_table(regions, kind: str, gene_ids: Sequence[str]) -> pd.DataFrame:
    """Regions as a frame with ``chrom``/``start``/``end`` spans for window assignment."""
    return pd.DataFrame(
        {
            "region": kind,
            "chrom": [r[0] for r in regions],
            "strand": [r[1] for r in regions],
            "intervals": [list(r[2]) for r in regions],
            "gene_id": list(gene_ids),
            "start": [min(s for s, _ in r[2]) for r in regions],
            "end": [max(e for _, e in r[2]) for r in regions],
        }
    )


class PeriodicityAccumulator:
    """Per CHX channel x region x segment metrics over evaluation windows."""

    def __init__(
        self,
        layout: list[tuple[str, str, str]],
        genes: pd.DataFrame,
        windows: pd.DataFrame,
        regions: pd.DataFrame,
        *,
        block: int = DEFAULT_BLOCK,
        min_utr_counts: float = DEFAULT_MIN_UTR_COUNTS,
        min_block_counts: float = DEFAULT_MIN_BLOCK_COUNTS,
    ):
        """``regions``: :func:`region_table` rows (``region`` in ``cds``/``utr5``).

        ``not_in_one_window`` counts regions that overlap a window but lie inside none;
        ``outside_windows`` counts regions that overlap no window.
        """
        self.layout, self.block, self.min_utr_counts = layout, block, min_utr_counts
        self.min_block_counts = min_block_counts
        self.windows = windows.reset_index(drop=True)
        self.regions = regions.reset_index(drop=True)
        where = assign_spans(self.regions, self.windows)
        self.not_in_one_window = int((where == EXCLUDED).sum())
        self.outside_windows = int((where == OUTSIDE).sum())
        self._by_window = _group(where)
        expressed = genes[genes.expressed.astype(bool)]
        self._expressed = expressed.groupby("condition").gene_id.agg(set).to_dict()
        self._rna = (
            genes.set_index(["condition", "gene_id"]).rna_count.to_dict()
            if "rna_count" in genes
            else {}
        )
        self.rows: list[dict] = []

    def add(self, index: int, pred: np.ndarray, obs: np.ndarray) -> None:
        """Add window ``index``: ``(S, C)`` prediction and observed targets."""
        rows = self._by_window.get(index)
        if rows is None:
            return
        start = int(self.windows.start.iloc[index])
        for c, (condition, library, strand) in enumerate(self.layout):
            if library != "chx":
                continue
            ok = self._expressed.get(condition, set())
            for r in rows:
                region = self.regions.iloc[r]
                if region.strand != strand or region.gene_id not in ok:
                    continue
                positions, ranks = region_positions(strand, region.intervals)
                p, o = pred[positions - start, c], obs[positions - start, c]
                if region.region == "utr5" and o.sum() < self.min_utr_counts:
                    continue
                base = {
                    "condition": condition,
                    "strand": strand,
                    "channel": c,
                    "region": region.region,
                    "gene_id": region.gene_id,
                    "chrom": region.chrom,
                    "start": int(region.start),
                    "end": int(region.end),
                    "n_exons": int(ranks.max()) + 1,
                    "rna_count": self._rna.get((condition, region.gene_id), np.nan),
                }
                for segment, sl in self._segments(region.region, ranks):
                    offset = sl.start if region.region == "cds" else None
                    metrics = segment_metrics(
                        p[sl], o[sl], ranks[sl], self.block, offset, self.min_block_counts
                    )
                    self.rows.append({**base, "segment": segment, **metrics})

    @staticmethod
    def _segments(kind: str, ranks: np.ndarray) -> list[tuple[str, slice]]:
        whole = slice(0, len(ranks))
        if kind != "cds":
            return [("all", whole)]
        if ranks.max() == 0:
            return [("all", whole), ("single", whole)]
        first = int((ranks == 0).sum())
        return [("all", whole), ("first", slice(0, first)), ("rest", slice(first, len(ranks)))]

    def table(self) -> pd.DataFrame:
        """One row per channel x region x segment, with expression tertiles per condition."""
        table = pd.DataFrame(self.rows)
        if table.empty:
            return table
        table["block"] = self.block
        table["expr_tertile"] = expression_tertiles(table)
        return table


def expression_tertiles(table: pd.DataFrame) -> pd.Series:
    """Tertile (1-3) of ``rna_count`` among each condition's distinct genes in ``table``."""
    out = pd.Series(np.nan, index=table.index)
    for condition, rows in table.groupby("condition"):
        genes = rows.drop_duplicates("gene_id").set_index("gene_id").rna_count.dropna()
        if len(genes) < 3:
            continue
        ranks = genes.rank(method="first")
        tertile = pd.qcut(ranks, 3, labels=[1, 2, 3]).astype(int)
        out[rows.index] = rows.gene_id.map(tertile).to_numpy()
    return out


def _pooled(rows: pd.DataFrame) -> dict:
    """Pooled / median summaries of a group of segment rows."""

    def ratio(num, den):
        d = rows[den].sum()
        return rows[num].sum() / d if d else np.nan

    hf = (rows.hf_pred / rows.hf_obs).replace([np.inf, -np.inf], np.nan)
    annot_obs = rows[[f"annot_f{f}_obs" for f in range(3)]].sum()
    annot_pred = rows[[f"annot_f{f}_pred" for f in range(3)]].sum()
    return {
        "n": len(rows),
        "obs_counts": rows.obs_total.sum(),
        "p3_obs_median": rows.p3_obs.median(),
        "p3_pred_median": rows.p3_pred.median(),
        "hf_ratio_median": hf.median(),
        "block_kl": ratio("block_kl_num", "block_kl_den"),
        "dom_agree": ratio("dom_agree", "dom_blocks"),
        "dom_blocks": rows.dom_blocks.sum(),
        "switch_obs_within": ratio("switch_obs_within", "pairs_obs_within"),
        "switch_pred_within": ratio("switch_pred_within", "pairs_pred_within"),
        "switch_obs_junction": ratio("switch_obs_junction", "pairs_obs_junction"),
        "switch_pred_junction": ratio("switch_pred_junction", "pairs_pred_junction"),
        "annot_frame0_obs": annot_obs.iloc[0] / annot_obs.sum() if annot_obs.sum() else np.nan,
        "annot_frame0_pred": annot_pred.iloc[0] / annot_pred.sum() if annot_pred.sum() else np.nan,
    }


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Summary rows by condition x region x segment, and by expression tertile (``all``)."""
    out = []
    for (condition, region, segment), rows in table.groupby(
        ["condition", "region", "segment"], sort=False
    ):
        out.append({"condition": condition, "region": region, "stratum": segment, **_pooled(rows)})
    full = table[table.segment == "all"]
    for (condition, region, tertile), rows in full.groupby(
        ["condition", "region", "expr_tertile"], sort=True
    ):
        out.append(
            {
                "condition": condition,
                "region": region,
                "stratum": f"expr_tertile_{int(tertile)}",
                **_pooled(rows),
            }
        )
    return pd.DataFrame(out)


def markdown_table(table: pd.DataFrame) -> str:
    """Pipe table of ``table`` with floats at 3 decimals (no tabulate dependency)."""
    if table.empty:
        return "(no regions)"

    def cell(v):
        return f"{v:.3f}" if isinstance(v, float | np.floating) else str(v)

    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(False)]
    return "\n".join(lines)


def summary_markdown(summary: pd.DataFrame, meta: dict) -> str:
    """Short markdown report of :func:`summarize` output."""
    lines = [
        f"# CHX periodicity — {meta.get('run_name')}",
        "",
        *(f"- {k}: {v}" for k, v in meta.items()),
        "",
        "Primary (frame-agnostic): `p3_*_median` = median fraction of non-DC power at period 3;",
        "`block_kl` = counts-weighted KL of per-block phase composition (the training loss);",
        "`dom_agree` = predicted dominant phase equals observed per block (chance 1/3) over",
        "`dom_blocks` blocks with >= `min_block_counts` observed counts;",
        "`switch_*` = rate the dominant phase changes between consecutive such blocks,",
        "within exons vs across a splice junction; `hf_ratio_median` = predicted / observed",
        f"power at periods < {HF_MAX_PERIOD} nt (unit-sum profiles; observed includes shot noise).",
        "Secondary (annotation-anchored, CDS only): `annot_frame0_*`.",
        "Read model rows against the replicate reference (`--reference replicate`: raw rep2",
        "scored as the prediction of rep1), which shows what the data allow.",
        "",
        markdown_table(summary),
        "",
    ]
    return "\n".join(lines)


def _accumulator(arm, windows, genes, regions, block, min_utr_counts, min_block_counts):
    _, conditions = parse_arm(arm)
    genes = genes[genes.condition.isin(conditions)].reset_index(drop=True)
    acc = PeriodicityAccumulator(
        channel_layout(conditions),
        genes,
        windows.reset_index(drop=True),
        regions,
        block=block,
        min_utr_counts=min_utr_counts,
        min_block_counts=min_block_counts,
    )
    return acc, conditions, genes


def _counts(acc: PeriodicityAccumulator) -> dict[str, int]:
    return {"not_in_one_window": acc.not_in_one_window, "outside_windows": acc.outside_windows}


def run_periodicity(
    predict_fn: Callable[[dict], np.ndarray],
    arm: str,
    windows: pd.DataFrame,
    *,
    fasta: str | Path,
    tracks_dir: str | Path,
    genes: pd.DataFrame,
    regions: pd.DataFrame,
    block: int = DEFAULT_BLOCK,
    min_utr_counts: float = DEFAULT_MIN_UTR_COUNTS,
    min_block_counts: float = DEFAULT_MIN_BLOCK_COUNTS,
    prefetch: int = 4,
    transcripts=None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Region rows for the arm's predictions vs pooled targets, and unassigned-region counts.

    ``transcripts`` (``evaluate.load_transcripts``) adds the ``seqconv_tx`` input channels.
    """
    from tisiago.seq2func.data import iterate

    windows = windows.reset_index(drop=True)
    acc, conditions, genes = _accumulator(
        arm, windows, genes, regions, block, min_utr_counts, min_block_counts
    )
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
            pred = np.asarray(predict_fn(batch), dtype=np.float64)[0]
            acc.add(i, pred, np.asarray(batch["targets"][0], dtype=np.float64))
    finally:
        batches.close()
    return acc.table(), _counts(acc)


def run_reference(
    arm: str,
    windows: pd.DataFrame,
    *,
    tracks_dir: str | Path,
    genes: pd.DataFrame,
    regions: pd.DataFrame,
    block: int = DEFAULT_BLOCK,
    min_utr_counts: float = DEFAULT_MIN_UTR_COUNTS,
    min_block_counts: float = DEFAULT_MIN_BLOCK_COUNTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replicate reference: raw replicate 2 scored as the prediction of replicate 1.

    Same regions, filters and metrics as :func:`run_periodicity`, so model rows can be read
    against what two independent samples of the same library agree on (no model, CPU only).
    """
    from tisiago.seq2func.data import TrackReader

    windows = windows.reset_index(drop=True)
    acc, conditions, _ = _accumulator(
        arm, windows, genes, regions, block, min_utr_counts, min_block_counts
    )
    rep1, rep2 = (TrackReader(tracks_dir, conditions, rep=r) for r in (1, 2))
    try:
        for i, row in enumerate(windows.itertuples(index=False)):
            if i not in acc._by_window:
                continue
            start, end = int(row.start), int(row.end)
            obs = np.asarray(rep1.read(row.chrom, start, end), dtype=np.float64)
            pred = np.asarray(rep2.read(row.chrom, start, end), dtype=np.float64)
            acc.add(i, pred, obs)
    finally:
        rep1.close()
        rep2.close()
    return acc.table(), _counts(acc)


def output_paths(
    out_dir: Path, run_name: str, split: str, max_windows: int | None, reference: str | None
) -> tuple[Path, Path]:
    """``(regions TSV, summary markdown)``; ``_max<N>`` and ``_<reference>`` suffixes."""
    suffix = f"{split}_max{max_windows}" if max_windows else split
    if reference:
        suffix = f"{suffix}_{reference}"
    out = Path(out_dir) / run_name
    return out / f"regions_{suffix}.tsv", out / f"summary_{suffix}.md"


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None, help="run's best/ or last/")
    ap.add_argument(
        "--reference",
        choices=REFERENCES,
        default=None,
        help="score raw rep2 against rep1 instead of a checkpoint (needs --arm)",
    )
    ap.add_argument("--arm", default=None, help="default: the run's run.json arm")
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument(
        "--run-name",
        default=None,
        help="default: the run directory's name, or reference_<arm> with --reference",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/p6/periodicity"))
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/p6/expression/expression.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=None, help="default: run.json")
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows (smoke)")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="codons per phase block")
    ap.add_argument(
        "--min-block-counts",
        type=float,
        default=DEFAULT_MIN_BLOCK_COUNTS,
        help="observed counts a block needs for dominant-phase agreement / switch metrics",
    )
    ap.add_argument("--min-utr-counts", type=float, default=DEFAULT_MIN_UTR_COUNTS)
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"), help="MANE_Select GTF")
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--head", choices=HEAD_CHOICES, default=None, help="default: run.json")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    """CLI: periodicity diagnostic of one checkpoint (or the replicate reference) on a split."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    if (args.checkpoint is None) == (args.reference is None):
        ap.error("pass exactly one of --checkpoint or --reference")
    arm = args.arm or (run_args(args.checkpoint).get("arm") if args.checkpoint else None)
    if not arm:
        ap.error("--arm is required (no run.json next to the checkpoint)")
    try:
        parse_arm(arm)
    except ValueError as exc:
        ap.error(str(exc))
    if args.block < 1:
        ap.error("--block must be >= 1")
    required = ("gtf",) if args.reference else ("genome", "gtf", "base_checkpoint")
    for flag in required:
        if not getattr(args, flag):
            ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")
    if args.run_name:
        run_name = args.run_name
    elif args.reference:
        run_name = f"reference_{arm.replace(':', '_').lower()}"
    else:
        run_name = Path(args.checkpoint).resolve().parent.name
    tsv, md = output_paths(args.out_dir, run_name, args.split, args.max_windows, args.reference)
    if tsv.exists() and not args.overwrite:
        ap.error(f"{tsv} exists; pass --overwrite to replace")

    windows = pd.read_parquet(args.windows_dir / SPLIT_FILES[args.split])
    if args.max_windows is not None:
        windows = windows.iloc[: args.max_windows]
    tracks_dir = args.tracks_dir or tracks_dir_of(args.checkpoint)
    genes = pd.read_parquet(args.expression).reset_index(drop=True)
    cds, cds_genes = load_mane_cds(args.gtf)
    utr, utr_genes = load_mane_utr5(args.gtf)
    regions = pd.concat(
        [region_table(cds, "cds", cds_genes), region_table(utr, "utr5", utr_genes)],
        ignore_index=True,
    )
    regions = regions[regions.chrom.isin(set(windows.chrom))].reset_index(drop=True)
    settings = dict(
        genes=genes,
        regions=regions,
        block=args.block,
        min_utr_counts=args.min_utr_counts,
        min_block_counts=args.min_block_counts,
    )
    t0 = time.time()
    if args.reference:
        table, counts = run_reference(arm, windows, tracks_dir=tracks_dir, **settings)
        source = f"{args.reference}: rep2 raw vs rep1"
    else:
        if args.device == "cpu":
            os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no CUDA probing on a CPU node
        import jax

        device = jax.devices(args.device)[0]
        head_kind = args.head or head_kind_of(args.checkpoint)
        predict_fn = load_predict_fn(
            args.checkpoint,
            arm,
            base_checkpoint=args.base_checkpoint,
            head_kind=head_kind,
            device=device,
        )
        print(f"model load {time.time() - t0:.0f}s on {device}", flush=True)
        table, counts = run_periodicity(
            predict_fn,
            arm,
            windows,
            fasta=args.genome,
            tracks_dir=tracks_dir,
            prefetch=args.prefetch,
            transcripts=load_transcripts(head_kind, args.gtf),
            **settings,
        )
        source = str(args.checkpoint)
    meta = {
        "run_name": run_name,
        "source": source,
        "arm": arm,
        "split": args.split,
        "windows": len(windows),
        "block_codons": args.block,
        "min_block_counts": args.min_block_counts,
        "min_utr_counts": args.min_utr_counts,
        "regions_not_in_one_window": counts["not_in_one_window"],
        "regions_outside_windows": counts["outside_windows"],
        "tracks_dir": str(tracks_dir),
    }
    write_tsv(table.assign(source=source, split=args.split), tsv)
    summary = summarize(table) if not table.empty else pd.DataFrame()
    md.write_text(summary_markdown(summary, meta))
    print(f"periodicity {time.time() - t0:.0f}s -> {tsv}, {md}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
