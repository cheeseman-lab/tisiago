"""Concordance scorecard: predicted vs real Ribo-seq tracks in plain correlation units (P6 exp. 2).

Usage::

    python -m tisiago.seq2func.scorecard --split val --heldout HeLa \
        --source rnacond:HeLa=data/p6/runs/ar_rnacond_hela/predicted_{split}_float \
        --source base:HeLa=data/p6/runs/ar_hela_base/predicted_{split}_float
    python -m tisiago.seq2func.scorecard --split test --i-am-final ...

Each ``--source NAME:CONDITION=DIR`` (or ``NAME=DIR`` for every condition) is a
``predict_tracks --mode float`` output directory holding ``<C>/{tis,chx}_pooled.*.bw`` (Ribo-TISH
minus-strand convention, written over the split's windows, no ``--match-depth`` or
``--mask-unexpressed``); ``{split}`` is substituted. A source is scored on the conditions it has.

Units: one MANE_Select transcript (with a CDS) per gene expressed in the condition (expression
table ``expressed``), whose genomic span lies inside the union of the split's windows. For each
condition x library (``tis``, ``chx``) every unit gives spliced, 5'->3' 1-nt vectors of the
prediction and of replicates 1 and 2 (canonical P-sites). Each metric is Pearson r of the
prediction with replicate 2 (``value``) and of replicate 1 with replicate 2 (``yardstick``, the
same units), and ``percent = 100 * value / yardstick`` (NaN, "n/a" in the markdown, when the
yardstick is below ``PERCENT_MIN_YARDSTICK`` = 0.1 or not finite):

- ``gene``: log1p CDS totals across units; ``n`` = units. ``gene_mrna``: the same over the whole
  mRNA (TIS signal includes 5'UTR starts).
- ``bin30`` / ``codon`` / ``nt``: per-unit r along the mRNA (30-nt bins from the 5' end; 3-nt bins
  in the annotated CDS frame; single nt; trailing partial bins dropped), median over units with
  at least ``--min-reads`` replicate-2 reads on the mRNA; ``n`` = those units. A constant vector
  has r = 0.
- ``frame`` (CHX only): dominant-frame agreement with replicate 2 over CDS codons
  (:func:`tisiago.seq2func.periodicity.dominant_phases`). Per source, value and yardstick use the
  same codons: replicate 2 has at least ``--min-frame-reads`` reads and a unique dominant frame,
  and replicate 1 and the prediction each have a unique dominant frame (ties and empty codons are
  undefined); ``n`` = those codons.

Built-in comparator ``rna_scaled``: the condition's pooled RNA 5'-end track (no minus shift) in
counts per million of its genome-wide pooled RNA total (:func:`rna_cpm`), times one scalar per
library: total pooled Ribo-seq / total RNA CPM over the training windows (``train.parquet``),
pooled over the fit conditions. The fit conditions are ``--fit-conditions`` (default: all but
``--heldout``) minus the evaluated condition, so a condition's own Ribo-seq never enters its
baseline; with none left the comparator is skipped for that condition. The scalar only moves the
``gene`` metric (log1p), the within-transcript correlations are scale-free.

A source missing a condition's prediction directory is skipped for that condition with a warning
(listed under ``skipped`` in ``scorecard.json``).

Writes ``scorecard.tsv`` (long: condition, heldout, library, source, scale, value, yardstick,
percent, n),
``scorecard.md`` (one table per scale), ``scorecard.json`` (inputs, fit conditions, scalars) and
``plots/<C>_<rank>_<gene>.png`` for the ``--example-genes`` units with the highest RNA count in
the split (ties by gene_id): rows replicate 1, replicate 2, each source, ``rna_scaled`` (RNA CPM
where no scalar was fit); TIS and CHX
columns; transcript coordinates with the CDS shaded. Default output ``data/p6/scorecard/<split>``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from tisiago.seq2func.data import rna_total
from tisiago.seq2func.evaluate import SPLIT_FILES, write_tsv
from tisiago.seq2func.periodicity import dominant_phases, region_positions
from tisiago.seq2func.predict_tracks import pooled_prefix, region_of, strand_shift
from tisiago.seq2func.samples import CONDITIONS
from tisiago.seq2func.targets import STRAND_FILES
from tisiago.seq2func.tracks import track_prefix

LIBRARIES = ("tis", "chx")
SCALES = ("gene", "gene_mrna", "bin30", "codon", "nt")
WITHIN_SCALES = ("bin30", "codon", "nt")
FRAME = "frame"
BIN = 30
MIN_READS = 50
MIN_FRAME_READS = 5
EXAMPLE_GENES = 4
RNA_SCALED = "rna_scaled"
PERCENT_MIN_YARDSTICK = 0.1
RESERVED = ("rep1", "rep2", "rna", RNA_SCALED)
COLUMNS = (
    "condition",
    "heldout",
    "library",
    "source",
    "scale",
    "value",
    "yardstick",
    "percent",
    "n",
)
_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')


@dataclass(frozen=True)
class Unit:
    """One MANE transcript; ``exons`` / ``cds`` are 0-based half-open, ascending genomic order."""

    transcript_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]
    cds: tuple[tuple[int, int], ...]

    @cached_property
    def positions(self) -> np.ndarray:
        """Genomic position of each mRNA base, 5'->3'."""
        return region_positions(self.strand, self.exons)[0]

    @property
    def length(self) -> int:
        """Mature mRNA length."""
        return len(self.positions)

    @property
    def start(self) -> int:
        """First exonic base."""
        return self.exons[0][0]

    @property
    def end(self) -> int:
        """End of the last exon (exclusive)."""
        return self.exons[-1][1]

    @cached_property
    def cds_range(self) -> tuple[int, int]:
        """``[start, end)`` of the CDS in mRNA coordinates."""
        first = self.cds[0][0] if self.strand == "+" else self.cds[-1][1] - 1
        start = int(np.flatnonzero(self.positions == first)[0])
        return start, start + sum(e - s for s, e in self.cds)


def load_units(gtf: str | Path) -> list[Unit]:
    """MANE_Select transcripts with a CDS, one per gene (the first seen)."""
    records: dict[str, list] = {}
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#") or 'tag "MANE_Select"' not in line:
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("exon", "CDS"):
                continue
            tid = _TRANSCRIPT_ID.search(f[8]).group(1)
            gene = _GENE_ID.search(f[8]).group(1)
            name = _GENE_NAME.search(f[8])
            record = records.setdefault(
                tid, [f[0], f[6], gene, name.group(1) if name else gene, [], []]
            )
            record[4 if f[2] == "exon" else 5].append((int(f[3]) - 1, int(f[4])))
    out, seen = [], set()
    for tid, (chrom, strand, gene, name, exons, cds) in records.items():
        if gene in seen or not exons or not cds:
            continue
        seen.add(gene)
        out.append(Unit(tid, gene, name, chrom, strand, tuple(sorted(exons)), tuple(sorted(cds))))
    return out


def select_units(
    units: Sequence[Unit], genes: pd.DataFrame, condition: str, windows: pd.DataFrame
) -> list[Unit]:
    """Units of genes expressed in ``condition`` whose span lies inside the windows' union."""
    rows = genes[(genes.condition == condition) & genes.expressed.astype(bool)]
    expressed = set(rows.gene_id)
    region = region_of(windows)
    out = []
    for unit in units:
        spans = region.get(unit.chrom)
        if unit.gene_id not in expressed or not spans:
            continue
        starts = np.array([s for s, _ in spans])
        i = int(np.searchsorted(starts, unit.start, side="right")) - 1
        if i >= 0 and unit.end <= spans[i][1]:
            out.append(unit)
    return sorted(out, key=lambda u: (u.chrom, u.start, u.transcript_id))


class BigWigs:
    """Open-once 1-bp bigWig reader."""

    def __init__(self):
        """No handles yet."""
        self._handles: dict[str, object] = {}

    def _handle(self, path: str):
        if path not in self._handles:
            bw = pyBigWig.open(path)
            if bw is None:
                raise FileNotFoundError(path)
            self._handles[path] = bw
        return self._handles[path]

    def dense(self, path: str | Path, chrom: str, lo: int, hi: int) -> np.ndarray:
        """Values over file positions ``[lo, hi)``; 0 where absent or off-chromosome."""
        bw = self._handle(str(path))
        out = np.zeros(max(hi - lo, 0), dtype=np.float64)
        size = bw.chroms().get(chrom) or 0
        a, b = max(lo, 0), min(hi, size)
        if b > a:
            out[a - lo : b - lo] = np.nan_to_num(bw.values(chrom, a, b, numpy=True), nan=0.0)
        return out

    def region_sum(self, path: str | Path, region: Mapping[str, Sequence], shift: int = 0) -> float:
        """Sum over file positions ``p`` with ``p - shift`` in ``region``."""
        bw = self._handle(str(path))
        sizes = bw.chroms()
        total = 0.0
        for chrom, spans in region.items():
            size = sizes.get(chrom) or 0
            for lo, hi in spans:
                a, b = max(lo + shift, 0), min(hi + shift, size)
                if b > a:
                    total += bw.stats(chrom, a, b, type="sum", exact=True)[0] or 0.0
        return total

    def close(self) -> None:
        """Close every handle."""
        for bw in self._handles.values():
            bw.close()
        self._handles.clear()


def transcript_vector(reader: BigWigs, path: str | Path, unit: Unit, shift: int) -> np.ndarray:
    """Spliced 5'->3' values of ``unit``; file position ``p + shift`` holds canonical ``p``."""
    lo = unit.start
    dense = reader.dense(path, unit.chrom, lo + shift, unit.end + shift)
    return dense[unit.positions - lo]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; 0 when either vector is constant, NaN with fewer than 2 points."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bin_sums(values: np.ndarray, size: int, offset: int = 0) -> np.ndarray:
    """Sums of consecutive ``size``-nt bins starting at ``offset``; the partial tail is dropped."""
    values = np.asarray(values, dtype=np.float64)[offset:]
    n = (len(values) // size) * size
    return values[:n].reshape(-1, size).sum(axis=1)


def _within(unit: Unit, values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "bin30":
        return bin_sums(values, BIN)
    if scale == "codon":
        return bin_sums(values, 3, unit.cds_range[0] % 3)
    return values


def percent_of(value: float, yardstick: float) -> float:
    """``100 * value / yardstick``; NaN below :data:`PERCENT_MIN_YARDSTICK` or if not finite."""
    if not np.isfinite(value) or not np.isfinite(yardstick):
        return float("nan")
    if yardstick < PERCENT_MIN_YARDSTICK:
        return float("nan")
    return 100.0 * value / yardstick


def library_rows(
    units: Sequence[Unit],
    vectors: Mapping[str, Sequence[np.ndarray]],
    library: str,
    min_reads: float = MIN_READS,
    min_frame_reads: float = MIN_FRAME_READS,
) -> list[dict]:
    """Scorecard rows (no ``condition``) of each source in ``vectors`` for one library.

    ``vectors`` maps ``rep1``, ``rep2`` and every source name to per-unit 1-nt vectors aligned
    with ``units``.
    """
    rep1, rep2 = vectors["rep1"], vectors["rep2"]
    sources = [k for k in vectors if k not in ("rep1", "rep2")]
    rows: list[dict] = []

    def add(source, scale, value, yardstick, n):
        rows.append(
            {
                "library": library,
                "source": source,
                "scale": scale,
                "value": value,
                "yardstick": yardstick,
                "percent": percent_of(value, yardstick),
                "n": int(n),
            }
        )

    def totals(values, cds_only):
        out = []
        for unit, v in zip(units, values, strict=True):
            lo, hi = unit.cds_range if cds_only else (0, len(v))
            out.append(float(v[lo:hi].sum()))
        return np.log1p(out)

    for scale, cds_only in (("gene", True), ("gene_mrna", False)):
        obs = totals(rep2, cds_only)
        yard = pearson(totals(rep1, cds_only), obs)
        for source in sources:
            add(source, scale, pearson(totals(vectors[source], cds_only), obs), yard, len(units))

    eligible = [i for i, v in enumerate(rep2) if v.sum() >= min_reads]
    for scale in WITHIN_SCALES:
        pairs = []
        for i in eligible:
            unit = units[i]
            target = _within(unit, rep2[i], scale)
            r = [pearson(_within(unit, rep1[i], scale), target)]
            r += [pearson(_within(unit, vectors[s][i], scale), target) for s in sources]
            if np.isfinite(r[0]):
                pairs.append(r)
        table = np.array(pairs).reshape(len(pairs), 1 + len(sources))
        median = np.median(table, axis=0) if len(pairs) else np.full(1 + len(sources), np.nan)
        for k, source in enumerate(sources):
            add(source, scale, float(median[k + 1]), float(median[0]), len(pairs))

    if library == "chx":
        # per source: [model matches, replicate-1 matches, codons], over the same codons
        counts = np.zeros((len(sources), 3))
        for i, unit in enumerate(units):
            lo, hi = unit.cds_range
            target = dominant_phases(rep2[i][lo:hi], 1, min_frame_reads)
            yard = dominant_phases(rep1[i][lo:hi], 1, 0.0)
            base = (target >= 0) & (yard >= 0)
            if not base.any():
                continue
            for k, source in enumerate(sources):
                phases = dominant_phases(vectors[source][i][lo:hi], 1, 0.0)
                keep = base & (phases >= 0)
                counts[k] += [
                    (phases[keep] == target[keep]).sum(),
                    (yard[keep] == target[keep]).sum(),
                    keep.sum(),
                ]
        for k, source in enumerate(sources):
            model, replicate, n = counts[k]
            value = model / n if n else float("nan")
            add(source, FRAME, float(value), float(replicate / n) if n else float("nan"), n)
    return rows


def default_fit_conditions(heldout: Sequence[str]) -> tuple[str, ...]:
    """Every condition not held out."""
    return tuple(c for c in CONDITIONS if c not in set(heldout))


def rna_cpm(tracks_dir: str | Path, condition: str) -> float:
    """``1e6 / genome-wide pooled RNA reads`` of ``condition``."""
    total = rna_total(tracks_dir, condition)
    if not total > 0:
        raise ValueError(f"{condition}: no pooled RNA reads in {tracks_dir}")
    return 1e6 / total


def fit_conditions_for(condition: str, fit_conditions: Sequence[str]) -> tuple[str, ...]:
    """``fit_conditions`` without the evaluated ``condition``; raises when none is left."""
    out = tuple(c for c in fit_conditions if c != condition)
    if not out:
        raise ValueError(f"{condition}: no fit conditions left for {RNA_SCALED}")
    return out


def fit_rna_scalars(
    tracks_dir: str | Path,
    fit_conditions: Sequence[str],
    train_windows: pd.DataFrame,
    reader: BigWigs | None = None,
) -> dict[str, float]:
    """Per library: pooled Ribo-seq total / pooled RNA CPM total over the training windows."""
    own = reader is None
    reader = reader or BigWigs()
    region = region_of(train_windows)
    try:
        rna = 0.0
        for condition in fit_conditions:
            prefix = track_prefix(tracks_dir, condition, "rna", "pooled")
            raw = sum(reader.region_sum(f"{prefix}.{n}.bw", region) for n in STRAND_FILES.values())
            rna += raw * rna_cpm(tracks_dir, condition)
        if not rna > 0:
            raise ValueError(f"no RNA counts of {tuple(fit_conditions)} in the training windows")
        out = {}
        for library in LIBRARIES:
            ribo = 0.0
            for condition in fit_conditions:
                prefix = track_prefix(tracks_dir, condition, library, "pooled")
                for strand, name in STRAND_FILES.items():
                    ribo += reader.region_sum(f"{prefix}.{name}.bw", region, strand_shift(strand))
            out[library] = ribo / rna
        return out
    finally:
        if own:
            reader.close()


def parse_sources(items: Sequence[str], conditions: Sequence[str]) -> dict[str, dict[str, str]]:
    """``NAME:CONDITION=DIR`` / ``NAME=DIR`` (every condition) -> ``{name: {condition: dir}}``."""
    out: dict[str, dict[str, str]] = {}
    for item in items:
        key, sep, path = item.partition("=")
        name, colon, condition = key.partition(":")
        if not sep or not path or not name:
            raise ValueError(f"--source takes NAME[:CONDITION]=DIR, not {item!r}")
        if name in RESERVED:
            raise ValueError(f"source name {name!r} is reserved ({', '.join(RESERVED)})")
        if colon and condition not in CONDITIONS:
            raise ValueError(f"{item!r}: condition must be one of {CONDITIONS}")
        targets = [condition] if colon else list(conditions)
        mapping = out.setdefault(name, {})
        for c in targets:
            if c in mapping:
                raise ValueError(f"source {name!r} given twice for {c}")
            if c in conditions:
                mapping[c] = path
    return out


def example_genes(
    units: Sequence[Unit], genes: pd.DataFrame, condition: str, k: int = EXAMPLE_GENES
) -> list[Unit]:
    """The ``k`` units with the highest RNA count in ``condition`` (ties: gene_id)."""
    counts = genes[genes.condition == condition].set_index("gene_id").rna_count
    ranked = sorted(units, key=lambda u: (-float(counts.get(u.gene_id, 0)), u.gene_id))
    return ranked[:k]


def plot_unit(
    path: str | Path,
    unit: Unit,
    panels: Mapping[str, Mapping[str, np.ndarray]],
    title: str,
) -> None:
    """One PNG: rows = the tracks of ``panels[library]`` in order; columns = libraries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(next(iter(panels.values())))
    nrows = len(names)
    fig, axes = plt.subplots(
        nrows, len(panels), sharex=True, figsize=(6 * len(panels), 1.1 * nrows + 0.8), squeeze=False
    )
    x = np.arange(unit.length)
    lo, hi = unit.cds_range
    for col, (library, tracks) in enumerate(panels.items()):
        for row, name in enumerate(names):
            ax = axes[row, col]
            values = tracks[name]
            ax.vlines(x, 0, values, lw=0.6, color="C0" if row < 2 else "C1")
            ax.axvspan(lo, hi, color="0.85", lw=0, zorder=0)
            ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(library.upper(), fontsize=9)
        axes[-1, col].set_xlabel("transcript position (nt, 5'->3'; CDS shaded)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value: float, percent: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3f} ({percent:.0f}%)" if np.isfinite(percent) else f"{value:.3f} (n/a)"


def render_markdown(table: pd.DataFrame, meta: Mapping) -> str:
    """One table per scale: rows condition x library, columns the sources as ``r (percent%)``."""
    sources = list(dict.fromkeys(table.source))
    lines = [
        f"# P6 concordance scorecard ({meta['split']})",
        "",
        "Each cell: prediction vs replicate 2 (percent of replicate 1 vs replicate 2). Pearson r,",
        "except `frame`: dominant-frame agreement over CHX CDS codons. Units: MANE transcripts of",
        "expressed genes inside the split's windows. `gene`: log1p CDS totals; `gene_mrna`: whole",
        f"mRNA. Percent is n/a when the yardstick is below {PERCENT_MIN_YARDSTICK}.",
        f"Within-transcript scales: median over transcripts with >= {meta['min_reads']}",
        "replicate-2 reads. Held-out conditions are marked `*`.",
    ]
    for scale in (*SCALES, FRAME):
        part = table[table.scale == scale]
        if part.empty:
            continue
        lines += [
            "",
            f"## {scale}",
            "",
            "| condition | library | n | rep1 vs rep2 | " + " | ".join(sources) + " |",
            "|---|---|---|---|" + "---|" * len(sources),
        ]
        for (condition, library), group in part.groupby(["condition", "library"], sort=False):
            by = {r.source: r for r in group.itertuples()}
            first = group.iloc[0]
            label = f"{condition}*" if condition in meta.get("heldout", ()) else condition
            cells = [_fmt(by[s].value, by[s].percent) if s in by else "-" for s in sources]
            lines.append(
                f"| {label} | {library} | {int(first.n)} | {first.yardstick:.3f} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def run_scorecard(
    *,
    conditions: Sequence[str],
    sources: Mapping[str, Mapping[str, str]],
    split: str,
    windows_dir: str | Path,
    expression: str | Path,
    tracks_dir: str | Path,
    gtf: str | Path,
    out_dir: str | Path,
    heldout: Sequence[str] = (),
    fit_conditions: Sequence[str] | None = None,
    min_reads: float = MIN_READS,
    min_frame_reads: float = MIN_FRAME_READS,
    example_genes_n: int = EXAMPLE_GENES,
) -> pd.DataFrame:
    """Score every source against the replicates; write the outputs; return the long table."""
    from tisiago.seq2func.track_tis_eval import predicted_dirs

    fit = tuple(fit_conditions) if fit_conditions else default_fit_conditions(heldout)
    leaked = sorted(set(fit) & set(heldout))
    if leaked:
        raise ValueError(f"fit conditions {leaked} are held out")
    resolved: dict[str, dict[str, Path]] = {}
    skipped = []
    for name, templates in sources.items():
        resolved[name] = {}
        for condition, template in templates.items():
            try:
                found = predicted_dirs({condition: template}, split, [condition], windows_dir)
            except FileNotFoundError as exc:
                warnings.warn(f"{name} {condition}: skipped, no prediction ({exc})", stacklevel=2)
                skipped.append({"source": name, "condition": condition, "reason": "missing"})
                continue
            resolved[name].update(found)
    windows = pd.read_parquet(Path(windows_dir) / SPLIT_FILES[split])
    train_windows = None
    genes = pd.read_parquet(expression)
    units_all = load_units(gtf)
    out_dir = Path(out_dir)
    reader = BigWigs()
    rows: list[dict] = []
    scalar_cache: dict[tuple[str, ...], dict[str, float]] = {}
    meta: dict = {
        "split": split,
        "windows_dir": str(windows_dir),
        "expression": str(expression),
        "tracks_dir": str(tracks_dir),
        "conditions": list(conditions),
        "heldout": list(heldout),
        "fit_conditions": list(fit),
        "min_reads": min_reads,
        "min_frame_reads": min_frame_reads,
        "sources": {n: {c: str(p) for c, p in d.items()} for n, d in resolved.items()},
        "skipped": skipped,
        "units": {},
        "rna_scaled": {},
        "plots": {},
    }
    try:
        for condition in conditions:
            units = select_units(units_all, genes, condition, windows)
            meta["units"][condition] = len(units)
            print(f"{condition}: {len(units)} units", flush=True)
            if not units:
                meta["rna_scaled"][condition] = None
                continue

            def vectors_of(prefix: Path, shift_of) -> list[np.ndarray]:
                return [
                    transcript_vector(
                        reader,
                        f"{prefix}.{STRAND_FILES[u.strand]}.bw",
                        u,
                        shift_of(u.strand),
                    )
                    for u in units  # noqa: B023
                ]

            rna_prefix = track_prefix(tracks_dir, condition, "rna", "pooled")
            cpm = rna_cpm(tracks_dir, condition)
            rna = [v * cpm for v in vectors_of(rna_prefix, lambda s: 0)]
            try:
                fit_c = fit_conditions_for(condition, fit)
            except ValueError:
                fit_c = None
            scalars = None
            if fit_c is not None:
                if fit_c not in scalar_cache:
                    if train_windows is None:
                        train_windows = pd.read_parquet(Path(windows_dir) / "train.parquet")
                    scalar_cache[fit_c] = fit_rna_scalars(tracks_dir, fit_c, train_windows, reader)
                scalars = scalar_cache[fit_c]
                meta["rna_scaled"][condition] = {
                    "fit_conditions": list(fit_c),
                    "scalars": scalars,
                    "rna_cpm_scale": cpm,
                }
            else:
                meta["rna_scaled"][condition] = None

            panels: dict[str, dict[str, list[np.ndarray]]] = {}
            for library in LIBRARIES:
                vectors = {
                    f"rep{rep}": vectors_of(
                        track_prefix(tracks_dir, condition, library, rep), strand_shift
                    )
                    for rep in (1, 2)
                }
                for name, dirs in resolved.items():
                    if condition in dirs:
                        prefix = pooled_prefix(dirs[condition], condition, library)
                        vectors[name] = vectors_of(prefix, strand_shift)
                if scalars is not None:
                    vectors[RNA_SCALED] = [v * scalars[library] for v in rna]
                for row in library_rows(units, vectors, library, min_reads, min_frame_reads):
                    rows.append({"condition": condition, "heldout": condition in heldout, **row})
                panels[library] = vectors

            index = {u.transcript_id: i for i, u in enumerate(units)}
            meta["plots"][condition] = []
            for rank, unit in enumerate(example_genes(units, genes, condition, example_genes_n)):
                i = index[unit.transcript_id]
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", unit.gene_name)
                path = out_dir / "plots" / f"{condition}_{rank + 1}_{name}.png"
                plot_unit(
                    path,
                    unit,
                    {
                        lib: {
                            **{k: v[i] for k, v in vecs.items() if k != RNA_SCALED},
                            **(
                                {f"{RNA_SCALED}": vecs[RNA_SCALED][i]}
                                if RNA_SCALED in vecs
                                else {"RNA (CPM)": rna[i]}
                            ),
                        }
                        for lib, vecs in panels.items()
                    },
                    f"{condition} {unit.gene_name} ({unit.transcript_id}, {unit.strand} strand, "
                    f"{split})",
                )
                meta["plots"][condition].append(str(path.relative_to(out_dir)))
    finally:
        reader.close()

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    write_tsv(table, out_dir / "scorecard.tsv")
    (out_dir / "scorecard.md").write_text(render_markdown(table, meta))
    tmp = out_dir / f".scorecard.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, out_dir / "scorecard.json")
    return table


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(v for v in (value or "").split(",") if v)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME[:C]=DIR",
        help="predict_tracks --mode float output dir; '{split}' is substituted (repeatable)",
    )
    ap.add_argument("--condition", default=None, help="comma list (default: all six)")
    ap.add_argument("--heldout", default=None, help="comma list of held-out conditions")
    ap.add_argument(
        "--fit-conditions", default=None, help="rna_scaled fit conditions (default: not held out)"
    )
    ap.add_argument("--split", choices=tuple(SPLIT_FILES), default="val")
    ap.add_argument("--i-am-final", action="store_true", help="required for --split test")
    ap.add_argument("--windows-dir", type=Path, default=Path("data/p6/windows_new"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/p6/expression/expression.new.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/p6/tracks"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"), help="MANE_Select GTF")
    ap.add_argument("--out-dir", type=Path, help="default: data/p6/scorecard/<split>")
    ap.add_argument("--min-reads", type=float, default=MIN_READS)
    ap.add_argument("--min-frame-reads", type=float, default=MIN_FRAME_READS)
    ap.add_argument("--example-genes", type=int, default=EXAMPLE_GENES)
    return ap


def main() -> None:
    """CLI: score the sources on a split."""
    ap = build_parser()
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_final:
        ap.error("--split test reads the held-out chromosomes; pass --i-am-final to confirm")
    if not args.gtf:
        ap.error("--gtf is required (or set TISIAGO_GTF in .env)")
    conditions = _csv(args.condition) or CONDITIONS
    heldout = _csv(args.heldout)
    fit = _csv(args.fit_conditions) or None
    for flag, values in (("condition", conditions), ("heldout", heldout), ("fit", fit or ())):
        unknown = sorted(set(values) - set(CONDITIONS))
        if unknown:
            ap.error(f"--{flag}: unknown conditions {unknown}; choose from {CONDITIONS}")
    try:
        sources = parse_sources(args.source, conditions)
        run_scorecard(
            conditions=conditions,
            sources=sources,
            split=args.split,
            windows_dir=args.windows_dir,
            expression=args.expression,
            tracks_dir=args.tracks_dir,
            gtf=args.gtf,
            out_dir=args.out_dir or Path("data/p6/scorecard") / args.split,
            heldout=heldout,
            fit_conditions=fit,
            min_reads=args.min_reads,
            min_frame_reads=args.min_frame_reads,
            example_genes_n=args.example_genes,
        )
    except ValueError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
