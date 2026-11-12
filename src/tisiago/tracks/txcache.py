"""Transcript-space embedding cache for the fast loop and the ``spliced`` stage.

Usage::

    python -m tisiago.tracks.txcache index --name genomic_fc2d_k562      # CPU, seconds
    python -m tisiago.tracks.txcache tracks --name genomic_fc2d_k562     # CPU
    python -m tisiago.tracks.txcache embed --name genomic_fc2d_k562 \
        --checkpoint data/tracks/runs/tracks.genomic.free.pos.ho-K562.s0/best   # GPU
    python -m tisiago.tracks.txcache build ...                           # all three
    python -m tisiago.tracks.txcache build --test --i-am-final --name <x>_test \
        --pca-from data/tracks/txcache/<x> --checkpoint data/tracks/runs/<x>/best   # test cache

Units: one MANE_Select transcript with a CDS per gene (:func:`scorecard.load_units`) whose gene
is expressed in at least one condition and whose genomic span lies inside the union of the
``train`` or ``val`` windows (``test`` is never cached; chromosomes of ``test.parquet`` are
refused). Every vector is spliced and 5'->3' (cap to poly-A), as :mod:`scorecard`.

A TEST cache (``--test --i-am-final``, final scoring only) holds the units of the ``test``
windows alone, by the same rules, in shards ``test_<chrom>``, with what the val shards have
(``ckpt.npy``, ``seq.npy``; ``sites.parquet`` on test). Its ``pca.npz`` is a copy of the train
cache's (``--pca-from``; recorded in ``cache.json["pca_source"]``), never refit on test.

Layout of ``<cache>/`` (default ``data/tracks/txcache/<name>``)::

    index.parquet      one row per transcript: transcript_id, gene_id, gene_name, chrom, strand,
                       split, start, end, length, cds_start, cds_end (mRNA coordinates; metrics
                       only, never an input), exons, cds, junctions (mRNA index of each exon
                       start after the first), expressed_<C>, shard, offset
    sites.parquet      real pooled Ribo-TISH starts (val only, Jimmy's filter; evaluation only):
                       condition, transcript_id, tx_pos, codon, tistype
    pca.npz            mean (D,), components (K, D), explained_variance (K,), total_variance
    cache.json         provenance, sizes, build seconds per stage
    shards/<split>_<chrom>/
        emb.npy        (P, K) float16: PCA of the checkpoint's 1-bp embeddings (head input)
        rna.npy        (C, P, 2) float32: pooled RNA 5' ends (sense, antisense), no minus shift,
                       scaled to a 5e7-read library (:func:`data.rna_depth_scale`)
        tis.npy, chx.npy  (C, P, 3) float32: replicate 1, replicate 2, pooled; canonical P-sites
                       (minus strand un-shifted), sense strand
        seq.npy        (P,) int8 mRNA base codes A C G T = 0..3, N = 4 (test caches; other
                       caches via :mod:`tisiago.tracks.cache_seq`)
        ckpt.npy       (C, P, 2) float32, val / test only: the checkpoint's own TIS, CHX prediction
                       (sense strand; the full-model reference for the fast-loop scorer)

Rows of a shard are its transcripts concatenated in index order (``offset``, ``length``);
conditions follow ``cache.json["conditions"]``. Every ``.npy`` opens with ``mmap_mode="r"``.

Embeddings are condition-independent (the backbone sees DNA only): each window of the split is
run once and every position takes the window whose centre is nearest (ties to the earlier
window; :func:`predict_tracks.owned_spans`), as the scorecard's predicted tracks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.tracks.predict_tracks import owned_spans, region_of, strand_shift
from tisiago.tracks.scorecard import BigWigs, Unit, load_units
from tisiago.tracks.targets import STRAND_FILES
from tisiago.tracks.track_io import track_prefix

CACHE_VERSION = 1
PCA_DIM = 256
CACHE_SPLITS = ("train", "val")
TEST_SPLIT = "test"
SCORED_SPLITS = ("val", TEST_SPLIT)
REPS = ("rep1", "rep2", "pooled")
TRACK_LIBRARIES = ("tis", "chx")
INDEX = "index.parquet"
SITES = "sites.parquet"
PCA_FILE = "pca.npz"
CONFIG = "cache.json"
SHARDS = "shards"
DEFAULT_ROOT = Path("data/tracks/txcache")
OTHER_STRAND = {"+": "-", "-": "+"}


def junctions_of(unit: Unit) -> list[int]:
    """Transcript index of the first base of every exon after the 5'-most one."""
    lengths = [e - s for s, e in sorted(unit.exons, reverse=unit.strand == "-")]
    return np.cumsum(lengths)[:-1].astype(int).tolist()


def _inside(unit: Unit, region: Mapping[str, Sequence[tuple[int, int]]]) -> bool:
    spans = region.get(unit.chrom)
    if not spans:
        return False
    starts = np.array([s for s, _ in spans])
    i = int(np.searchsorted(starts, unit.start, side="right")) - 1
    return i >= 0 and unit.end <= spans[i][1]


def select_transcripts(
    units: Sequence[Unit],
    genes: pd.DataFrame,
    windows: Mapping[str, pd.DataFrame],
    conditions: Sequence[str],
    forbidden_chroms: Sequence[str] = (),
    test: bool = False,
) -> pd.DataFrame:
    """Index rows of the cached transcripts (see the module docstring), shards and offsets.

    ``windows`` maps a split (``train`` / ``val``) to its windows. A unit is kept when its gene
    is expressed in at least one of ``conditions`` and its span lies inside the union of one
    split's windows. Raises if a split is not cacheable or a window lies on ``forbidden_chroms``.
    ``test=True`` (a final test cache) takes ``windows == {"test": ...}`` only, same unit rules.
    """
    allowed = (TEST_SPLIT,) if test else CACHE_SPLITS
    bad = sorted(set(windows) - set(allowed))
    if bad:
        raise ValueError(f"splits {bad} are never cached (allowed: {allowed})")
    if test and set(windows) != {TEST_SPLIT}:
        raise ValueError("a test cache takes the test windows only")
    for split, frame in windows.items():
        if split == TEST_SPLIT:
            continue
        leaked = sorted(set(frame.chrom) & set(forbidden_chroms))
        if leaked:
            raise ValueError(f"{split} windows on held-out test chromosomes {leaked}")
    rows = genes[genes.condition.isin(conditions) & genes.expressed.astype(bool)]
    expressed = {c: set(rows.gene_id[rows.condition == c]) for c in conditions}
    anywhere = set().union(*expressed.values()) if expressed else set()
    regions = {split: region_of(frame) for split, frame in windows.items()}
    records = []
    for unit in units:
        if unit.gene_id not in anywhere:
            continue
        split = next((s for s in allowed if s in regions and _inside(unit, regions[s])), None)
        if split is None:
            continue
        cds_start, cds_end = unit.cds_range
        records.append(
            {
                "transcript_id": unit.transcript_id,
                "gene_id": unit.gene_id,
                "gene_name": unit.gene_name,
                "chrom": unit.chrom,
                "strand": unit.strand,
                "split": split,
                "start": unit.start,
                "end": unit.end,
                "length": unit.length,
                "cds_start": cds_start,
                "cds_end": cds_end,
                "exons": [list(e) for e in unit.exons],
                "cds": [list(e) for e in unit.cds],
                "junctions": junctions_of(unit),
                **{f"expressed_{c}": unit.gene_id in expressed[c] for c in conditions},
            }
        )
    index = pd.DataFrame(records)
    if index.empty:
        return index
    index = index.sort_values(["split", "chrom", "start", "transcript_id"]).reset_index(drop=True)
    index["shard"] = index.split + "_" + index.chrom
    index["offset"] = (index.groupby("shard").length.cumsum() - index.length).astype(np.int64)
    return index


def unit_of(row) -> Unit:
    """The :class:`scorecard.Unit` of an index row."""
    return Unit(
        row.transcript_id,
        row.gene_id,
        row.gene_name,
        row.chrom,
        row.strand,
        tuple((int(s), int(e)) for s, e in row.exons),
        tuple((int(s), int(e)) for s, e in row.cds),
    )


def shard_sizes(index: pd.DataFrame) -> dict[str, int]:
    """Rows (mRNA positions) per shard, in index order."""
    sizes = index.groupby("shard", sort=False).length.sum()
    return {str(k): int(v) for k, v in sizes.items()}


def spliced_values(reader: BigWigs, path: str | Path, unit: Unit, shift: int) -> np.ndarray:
    """:func:`scorecard.transcript_vector` read exon by exon (same values, less I/O)."""
    parts = [reader.dense(path, unit.chrom, s + shift, e + shift) for s, e in unit.exons]
    values = np.concatenate(parts) if parts else np.zeros(0)
    return values if unit.strand == "+" else values[::-1]


def transcript_tracks(
    reader: BigWigs, tracks_dir: str | Path, condition: str, unit: Unit, rna_scale: float
) -> dict[str, np.ndarray]:
    """``rna`` (L, 2) sense/antisense, ``tis`` / ``chx`` (L, 3) rep1, rep2, pooled (sense).

    RNA is read without a minus shift and times ``rna_scale``; TIS/CHX at canonical P-sites
    (the Ribo-TISH minus file read one base 3', as :func:`scorecard.transcript_vector`).
    """
    out = {}
    columns = []
    for strand in (unit.strand, OTHER_STRAND[unit.strand]):
        prefix = track_prefix(tracks_dir, condition, "rna", "pooled")
        columns.append(spliced_values(reader, f"{prefix}.{STRAND_FILES[strand]}.bw", unit, 0))
    out["rna"] = np.stack(columns, axis=1) * rna_scale
    name = STRAND_FILES[unit.strand]
    for library in TRACK_LIBRARIES:
        columns = []
        for rep in (1, 2, "pooled"):
            prefix = track_prefix(tracks_dir, condition, library, rep)
            columns.append(
                spliced_values(reader, f"{prefix}.{name}.bw", unit, strand_shift(unit.strand))
            )
        out[library] = np.stack(columns, axis=1)
    return out


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def shard_dir(cache: str | Path, shard: str) -> Path:
    """Directory of one shard."""
    return Path(cache) / SHARDS / shard


def build_shard_tracks(
    index: pd.DataFrame,
    shard: str,
    cache: str | Path,
    tracks_dir: str | Path,
    conditions: Sequence[str],
    rna_scales: Mapping[str, float],
) -> None:
    """Write ``rna.npy``, ``tis.npy``, ``chx.npy`` of ``shard``."""
    rows = index[index.shard == shard]
    size = int(rows.length.sum())
    arrays = {
        "rna": np.zeros((len(conditions), size, 2), np.float32),
        "tis": np.zeros((len(conditions), size, 3), np.float32),
        "chx": np.zeros((len(conditions), size, 3), np.float32),
    }
    reader = BigWigs()
    try:
        for row in rows.itertuples():
            unit = unit_of(row)
            lo, hi = int(row.offset), int(row.offset) + int(row.length)
            for k, condition in enumerate(conditions):
                scale = rna_scales[condition]
                tracks = transcript_tracks(reader, tracks_dir, condition, unit, scale)
                for key, array in arrays.items():
                    array[k, lo:hi] = tracks[key]
    finally:
        reader.close()
    out = shard_dir(cache, shard)
    out.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        _save_npy(out / f"{key}.npy", array)


def shard_positions(index: pd.DataFrame, shard: str) -> tuple[np.ndarray, np.ndarray]:
    """Genomic position of every row of ``shard`` and the strand of its transcript (+1 / -1)."""
    rows = index[index.shard == shard]
    positions, strands = [], []
    for row in rows.itertuples():
        unit = unit_of(row)
        positions.append(unit.positions)
        strands.append(np.full(unit.length, 1 if unit.strand == "+" else -1, np.int8))
    if not positions:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    return np.concatenate(positions), np.concatenate(strands)


def window_rows(
    windows: pd.DataFrame, chrom: str, positions: np.ndarray
) -> list[tuple[int, int, np.ndarray]]:
    """``(start, end, rows)`` per window of ``chrom`` owning at least one of ``positions``.

    A position belongs to the window whose centre is nearest (:func:`owned_spans` over every
    window of ``windows``, ties to the earlier window); ``rows`` index ``positions``. Raises if
    a position is owned by no window.
    """
    frame = windows.reset_index(drop=True)
    spans = owned_spans(frame)
    order = np.argsort(positions, kind="stable")
    ordered = positions[order]
    out, covered = [], 0
    for i, row in enumerate(frame.itertuples()):
        if row.chrom != chrom:
            continue
        lo, hi = spans[i]
        a, b = np.searchsorted(ordered, [lo, hi])
        if b > a:
            out.append((int(row.start), int(row.end), order[a:b]))
            covered += int(b - a)
    if covered != len(positions):
        raise ValueError(f"{chrom}: {len(positions) - covered} positions owned by no window")
    out.sort(key=lambda t: t[0])
    return out


def fill_rows(
    positions: np.ndarray,
    plan: Sequence[tuple[int, int, np.ndarray]],
    compute: Callable[[int, int], np.ndarray],
    width: tuple[int, ...],
    dtype=np.float32,
    progress: Callable[[int], None] | None = None,
) -> np.ndarray:
    """``(len(positions), *width)``: row ``r`` takes ``compute(start, end)[positions[r] - start]``.

    ``plan`` comes from :func:`window_rows`; ``compute(start, end)`` returns the per-base values
    of one window ``(end - start, *width)``.
    """
    out = np.zeros((len(positions), *width), dtype)
    for k, (start, end, rows) in enumerate(plan):
        values = np.asarray(compute(start, end))
        if values.shape[0] != end - start:
            raise ValueError(f"window [{start}, {end}) gave {values.shape[0]} positions")
        out[rows] = values[positions[rows] - start]
        if progress is not None:
            progress(k)
    return out


def select_strand(pred: np.ndarray, strands: np.ndarray) -> np.ndarray:
    """``(..., P, 4)`` TIS+, TIS-, CHX+, CHX- -> ``(..., P, 2)`` sense TIS, CHX per row strand."""
    plus = pred[..., [0, 2]]
    minus = pred[..., [1, 3]]
    return np.where((strands > 0)[:, None], plus, minus)


def fit_pca(samples: np.ndarray, dim: int = PCA_DIM) -> dict[str, np.ndarray]:
    """PCA of ``(N, D)`` samples: mean, top-``dim`` components (rows), explained variances."""
    x = np.asarray(samples, dtype=np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    cov = centred.T @ centred / max(len(x) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:dim]
    components = vectors[:, order].T
    # deterministic sign: the largest-magnitude loading of each component is positive
    signs = np.sign(components[np.arange(len(order)), np.abs(components).argmax(axis=1)])
    components = components * np.where(signs == 0, 1.0, signs)[:, None]
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance": values[order].astype(np.float64),
        "total_variance": np.float64(values.sum()),
    }


def pca_project(x, pca: Mapping[str, np.ndarray]):
    """``(..., D)`` -> ``(..., K)`` PCA coordinates (numpy or jax arrays)."""
    return (x - pca["mean"]) @ pca["components"].T


def pca_reconstruct(z, pca: Mapping[str, np.ndarray]):
    """Inverse of :func:`pca_project` on the retained components."""
    return z @ pca["components"] + pca["mean"]


def ribotish_sites(calls: pd.DataFrame, condition: str, index: pd.DataFrame) -> pd.DataFrame:
    """Filtered pooled Ribo-TISH starts mapped to mRNA coordinates of ``index``'s transcripts.

    ``calls`` is a ``*_pooled.calls.tsv`` table (already filtered). The start anchor is
    ``lo`` (+) / ``hi - 1`` (-) of ``GenomePos`` (canonical, as ``tis_peak_share.py``); each
    unique (chrom, strand, anchor) is kept once, on the first index transcript (index order)
    with that exonic base on its strand.
    """
    pattern = r"^(?P<chrom>[^:]+):(?P<lo>\d+)-(?P<hi>\d+):(?P<strand>[+-])$"
    pos = calls.GenomePos.str.extract(pattern)
    anchor = np.where(pos.strand == "+", pos.lo.astype(int), pos.hi.astype(int) - 1)
    sites = pd.DataFrame(
        {
            "chrom": pos.chrom.values,
            "strand": pos.strand.values,
            "anchor": anchor,
            "codon": calls.StartCodon.values,
            "tistype": calls.TisType.values,
        }
    ).drop_duplicates(["chrom", "strand", "anchor"])
    lookup: dict[tuple[str, str, int], tuple[str, int]] = {}
    for row in index.itertuples():
        for i, position in enumerate(unit_of(row).positions.tolist()):
            lookup.setdefault((row.chrom, row.strand, position), (row.transcript_id, i))
    out = []
    for site in sites.itertuples():
        hit = lookup.get((site.chrom, site.strand, int(site.anchor)))
        if hit is not None:
            out.append((condition, hit[0], hit[1], site.codon, site.tistype))
    return pd.DataFrame(out, columns=["condition", "transcript_id", "tx_pos", "codon", "tistype"])


SEQ_CODE = np.full(256, 4, np.int8)
for _i, _b in enumerate("ACGT"):
    SEQ_CODE[ord(_b)] = SEQ_CODE[ord(_b.lower())] = _i
SEQ_COMPLEMENT = np.array([3, 2, 1, 0, 4], np.int8)


def transcript_codes(fasta, unit: Unit) -> np.ndarray:
    """Base codes of the mRNA (A C G T = 0..3, N = 4), 5'->3' sense, as ``build_seq.py``."""
    parts = [str(fasta[unit.chrom][int(s) : int(e)]) for s, e in sorted(unit.exons)]
    codes = SEQ_CODE[np.frombuffer("".join(parts).encode(), np.uint8)]
    return codes if unit.strand == "+" else SEQ_COMPLEMENT[codes[::-1]]


def shard_codes(fasta, index: pd.DataFrame, shard: str) -> np.ndarray:
    """``seq.npy`` of ``shard``: :func:`transcript_codes` at each row's ``offset``."""
    rows = index[index.shard == shard]
    out = np.full(int(rows.length.sum()), 4, np.int8)
    for row in rows.itertuples():
        codes = transcript_codes(fasta, unit_of(row))
        if len(codes) != int(row.length):
            raise ValueError(f"{row.transcript_id}: {len(codes)} bases for length {row.length}")
        out[int(row.offset) : int(row.offset) + int(row.length)] = codes
    return out


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_pca(cache: str | Path, source: str | Path) -> dict:
    """Copy ``<source>/pca.npz`` into ``cache`` (never refit) and record it in ``cache.json``.

    An existing ``pca.npz`` in ``cache`` must be byte-identical to the source's. Returns the
    ``pca_source`` record (path, sha256, the source's ``cache.json["pca"]``).
    """
    import shutil

    cache, source = Path(cache), Path(source)
    if cache.resolve() == source.resolve():
        raise ValueError("--pca-from must be the train cache, not the cache being built")
    src = source / PCA_FILE
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist (embed the train cache first)")
    digest = _sha256(src)
    dest = cache / PCA_FILE
    if dest.exists():
        if _sha256(dest) != digest:
            raise ValueError(f"{dest} differs from {src}; a test cache never refits PCA")
    else:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.npz")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    record = {
        "path": str(src),
        "sha256": digest,
        "cache": str(source),
        "pca": _load_config(source).get("pca"),
    }
    _update_config(cache, pca_source=record)
    return record


class TxCache:
    """Read-only, memory-mapped view of a cache directory."""

    def __init__(self, path: str | Path):
        """Load the index and config; shard arrays are opened lazily."""
        self.path = Path(path)
        self.config = json.loads((self.path / CONFIG).read_text())
        self.index = pd.read_parquet(self.path / INDEX)
        self.conditions = tuple(self.config["conditions"])
        sites = self.path / SITES
        self.sites = pd.read_parquet(sites) if sites.exists() else None
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    def array(self, shard: str, key: str) -> np.ndarray:
        """``shards/<shard>/<key>.npy``, memory-mapped."""
        if (shard, key) not in self._arrays:
            path = shard_dir(self.path, shard) / f"{key}.npy"
            self._arrays[(shard, key)] = np.load(path, mmap_mode="r")
        return self._arrays[(shard, key)]

    def preload(self, split: str, keys: Sequence[str], threads: int = 8) -> None:
        """Read ``keys`` of every shard of ``split`` into memory (replacing the memory maps).

        Files are read concurrently by ``threads`` threads (network filesystems scale with
        parallel sequential reads); arrays already in memory are kept.
        """
        from concurrent.futures import ThreadPoolExecutor

        todo = [
            (shard, key)
            for shard in self.rows(split).shard.unique()
            for key in keys
            if isinstance(self._arrays.get((shard, key)), np.memmap)
            or (shard, key) not in self._arrays
        ]

        def read(item):
            shard, key = item
            return item, np.load(shard_dir(self.path, shard) / f"{key}.npy")

        with ThreadPoolExecutor(max(1, threads)) as pool:
            for item, array in pool.map(read, todo):
                self._arrays[item] = array

    def has(self, shard: str, key: str) -> bool:
        """Whether ``shards/<shard>/<key>.npy`` exists."""
        return (shard_dir(self.path, shard) / f"{key}.npy").exists()

    def condition_index(self, condition: str) -> int:
        """Position of ``condition`` in the per-condition arrays."""
        return self.conditions.index(condition)

    def rows(self, split: str) -> pd.DataFrame:
        """Index rows of ``split``."""
        return self.index[self.index.split == split]


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _load_config(cache: Path) -> dict:
    path = cache / CONFIG
    return json.loads(path.read_text()) if path.exists() else {}


def _update_config(cache: Path, **fields) -> dict:
    config = _load_config(cache)
    for key, value in fields.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    _write_json(cache / CONFIG, config)
    return config


def stage_index(args) -> pd.DataFrame:
    """Select transcripts, write ``index.parquet`` and ``sites.parquet``."""
    from tisiago.calling.ribotish import filter_predictions
    from tisiago.tracks.data import rna_depth_scales
    from tisiago.tracks.samples import CONDITIONS

    t0 = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    conditions = tuple(CONDITIONS)
    test = bool(getattr(args, "test", False))
    windows = load_windows(args.windows_dir, test)
    test_path = Path(args.windows_dir) / "test.parquet"
    forbidden = sorted(set(pd.read_parquet(test_path).chrom)) if test_path.exists() else []
    genes = pd.read_parquet(args.expression)
    index = select_transcripts(load_units(args.gtf), genes, windows, conditions, forbidden, test)
    index.to_parquet(cache / INDEX, index=False)
    sites = []
    if args.calls_run:
        val = index[index.split.isin(SCORED_SPLITS)]
        for condition in conditions:
            path = Path(args.calls_run) / condition / "pooled" / f"{condition}_pooled.calls.tsv"
            calls = filter_predictions(pd.read_csv(path, sep="\t"))
            calls = calls[calls.GenomePos.str.split(":").str[0].isin(set(val.chrom))]
            sites.append(ribotish_sites(calls, condition, val))
    sites_frame = (
        pd.concat(sites, ignore_index=True)
        if sites
        else pd.DataFrame(columns=["condition", "transcript_id", "tx_pos", "codon", "tistype"])
    )
    sites_frame.to_parquet(cache / SITES, index=False)
    counts = index.groupby("split").agg(transcripts=("length", "size"), positions=("length", "sum"))
    _update_config(
        cache,
        version=CACHE_VERSION,
        conditions=list(conditions),
        windows_dir=str(args.windows_dir),
        expression=str(args.expression),
        tracks_dir=str(args.tracks_dir),
        calls_run=str(args.calls_run) if args.calls_run else None,
        forbidden_chroms=forbidden,
        **({"split": TEST_SPLIT, "test_chroms": forbidden} if test else {}),
        rna_reference_depth=5e7,
        rna_depth_scales=rna_depth_scales(args.tracks_dir, conditions),
        splits={s: {k: int(v) for k, v in r.items()} for s, r in counts.iterrows()},
        shards=shard_sizes(index),
        sites=int(len(sites_frame)),
        git_commit=_git_commit(),
        seconds={"index": time.time() - t0},
    )
    print(f"index: {len(index)} transcripts, {len(sites_frame)} sites", flush=True)
    return index


def load_windows(windows_dir: str | Path, test: bool = False) -> dict[str, pd.DataFrame]:
    """Windows of the cached splits: ``train`` and ``val``, or ``test`` alone for a test cache."""
    splits = (TEST_SPLIT,) if test else CACHE_SPLITS
    return {s: pd.read_parquet(Path(windows_dir) / f"{s}.parquet") for s in splits}


def _tracks_job(job) -> str:
    index, shard, cache, tracks_dir, conditions, scales = job
    build_shard_tracks(index, shard, cache, tracks_dir, conditions, scales)
    return shard


def stage_tracks(args) -> None:
    """Write every shard's RNA / TIS / CHX arrays (``--workers`` processes)."""
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    cache = Path(args.cache)
    config = _load_config(cache)
    index = pd.read_parquet(cache / INDEX)
    conditions = config["conditions"]
    scales = config["rna_depth_scales"]
    todo = [
        s
        for s in shard_sizes(index)
        if args.overwrite or not (shard_dir(cache, s) / "chx.npy").exists()
    ]
    jobs = [(index[index.shard == s], s, cache, args.tracks_dir, conditions, scales) for s in todo]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            for shard in pool.map(_tracks_job, jobs):
                print(f"tracks {shard} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for job in jobs:
            print(f"tracks {_tracks_job(job)} ({time.time() - t0:.0f}s)", flush=True)
    _update_config(cache, seconds={"tracks": time.time() - t0})


def _window_onehot(fasta, chrom: str, start: int, end: int) -> np.ndarray:
    from tisiago.tracks.data import fetch_sequence, one_hot

    return one_hot(fetch_sequence(fasta, chrom, start, end))


class _Embeddings1bp:
    """Stand-in for AlphaGenome ``Embeddings`` exposing only the 1-bp embeddings (head input)."""

    def __init__(self, x):
        self._x = x

    def get_sequence_embeddings(self, resolution: int):
        if resolution != 1:
            raise ValueError("only the 1-bp embeddings are cached")
        return self._x


def load_backbone(checkpoint: Path, base_checkpoint: str, arm: str, head_kind: str):
    """``(embed(onehot (S, 4)) -> (S, D), head(x (S, D), rna (K, S, 2)) -> (K, S, 4))``, jitted.

    The checkpoint's backbone and head (``load_finetuned``); ``head`` is the checkpoint's own
    rnacond head applied to cached-style embeddings and each RNA input.
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphagenome_ft.custom_heads import create_registered_head
    from alphagenome_research.model import dna_model

    from tisiago.tracks.model import ORGANISM, _alphagenome_backbone, load_finetuned
    from tisiago.tracks.train_genomic import parse_rnacond_arm

    head, _, _ = parse_rnacond_arm(arm)
    model = load_finetuned(str(checkpoint), str(base_checkpoint), head, 4, head_kind)
    params, state = model._params, model._state
    backbone = _alphagenome_backbone()
    org = dna_model.convert_to_organism_index(ORGANISM)
    num_organisms = len(dna_model.Organism)

    @hk.transform_with_state
    def embed_forward(onehot):
        organism = jnp.full((onehot.shape[0],), org, jnp.int32)
        return backbone(onehot, organism).get_sequence_embeddings(resolution=1)

    @hk.transform_with_state
    def head_forward(x, sequence):
        organism = jnp.full((sequence.shape[0],), org, jnp.int32)
        with hk.name_scope("head"):
            module = create_registered_head(head, metadata=None, num_organisms=num_organisms)
            return module(_Embeddings1bp(x), organism, sequence=sequence)

    def split(tree, keep):
        return {k: v for k, v in tree.items() if keep(k)}

    is_head = lambda k: k.startswith("head/")  # noqa: E731
    not_head = lambda k: not is_head(k)  # noqa: E731
    b_params, b_state = split(params, not_head), split(state, not_head)
    h_params, h_state = split(params, is_head), split(state, is_head)

    @jax.jit
    def embed(onehot):
        out, _ = embed_forward.apply(b_params, b_state, None, onehot[None])
        return out[0]

    @jax.jit
    def head_apply(x, onehot, rna):
        k = rna.shape[0]
        xs = jnp.broadcast_to(x[None], (k, *x.shape))
        seq = jnp.concatenate(
            [jnp.broadcast_to(onehot[None], (k, *onehot.shape)), rna.astype(onehot.dtype)], -1
        )
        out, _ = head_forward.apply(h_params, h_state, None, xs, seq)
        return out

    return embed, head_apply


def stage_embed(args) -> None:
    """PCA fit (if absent) and every shard's ``emb.npy`` (+ ``ckpt.npy`` for val), on the GPU.

    A test cache (``args.test``) copies the train cache's PCA (:func:`reuse_pca`, never a fit)
    and also writes ``seq.npy`` per shard.
    """
    import jax
    import jax.numpy as jnp
    from pyfaidx import Fasta

    from tisiago.tracks.data import RnaCondReaders
    from tisiago.tracks.evaluate import head_kind_of, run_args

    t0 = time.time()
    cache = Path(args.cache)
    config = _load_config(cache)
    index = pd.read_parquet(cache / INDEX)
    conditions = config["conditions"]
    checkpoint = Path(args.checkpoint)
    arm = args.arm or run_args(checkpoint).get("arm")
    head_kind = args.head or head_kind_of(checkpoint)
    embed, head_apply = load_backbone(checkpoint, args.base_checkpoint, arm, head_kind)
    print(f"model load {time.time() - t0:.1f}s on {jax.devices()}", flush=True)
    test = bool(getattr(args, "test", False))
    if test != (config.get("split") == TEST_SPLIT):
        raise ValueError(f"{cache}: --test must match how the index was built")
    windows = load_windows(args.windows_dir, test)
    fasta = Fasta(str(args.genome), as_raw=True)
    rng = np.random.default_rng(args.seed)
    pca_path = cache / PCA_FILE
    shards = list(shard_sizes(index))
    if args.shards:
        shards = [s for s in shards if s in set(args.shards.split(","))]

    def plan_of(shard):
        split, chrom = shard.split("_", 1)
        positions, strands = shard_positions(index, shard)
        return positions, strands, window_rows(windows[split], chrom, positions)

    if test:
        reuse_pca(cache, args.pca_from)
    elif not pca_path.exists():
        t1 = time.time()
        plans = [(s, *plan_of(s)) for s in shard_sizes(index) if s.startswith("train_")]
        candidates = [(s, p, w) for s, p, _, plan in plans for w in plan]
        picks = np.linspace(0, len(candidates) - 1, min(args.pca_windows, len(candidates)))
        samples = []
        for k in np.unique(picks.astype(int)):
            shard, positions, (start, end, rows) = candidates[k]
            chrom = shard.split("_", 1)[1]
            take = rng.choice(rows, size=min(args.pca_per_window, len(rows)), replace=False)
            x = embed(jnp.asarray(_window_onehot(fasta, chrom, start, end)))
            samples.append(np.asarray(x[jnp.asarray(positions[take] - start)], np.float32))
        pca = fit_pca(np.concatenate(samples), PCA_DIM)
        np.savez(pca_path.with_suffix(".tmp.npz"), **pca)
        os.replace(pca_path.with_suffix(".tmp.npz"), pca_path)
        ratio = float(pca["explained_variance"].sum() / pca["total_variance"])
        _update_config(
            cache,
            pca={
                "dim": PCA_DIM,
                "windows": len(samples),
                "samples": int(sum(len(s) for s in samples)),
                "explained_variance_ratio": ratio,
                "embedding_dim": int(pca["mean"].shape[0]),
                "seconds": time.time() - t1,
            },
        )
        print(f"pca: {sum(len(s) for s in samples)} samples, {ratio:.4f} variance", flush=True)
    pca = dict(np.load(pca_path))
    mean, comps = jnp.asarray(pca["mean"]), jnp.asarray(pca["components"])
    project = jax.jit(lambda x: ((x.astype(jnp.float32) - mean) @ comps.T).astype(jnp.float16))
    readers = RnaCondReaders(args.tracks_dir)
    scales = config["rna_depth_scales"]
    try:
        for shard in shards:
            out = shard_dir(cache, shard)
            if (out / "emb.npy").exists() and not args.overwrite:
                continue
            t1 = time.time()
            split, chrom = shard.split("_", 1)
            positions, strands, plan = plan_of(shard)
            val = split in SCORED_SPLITS
            emb = np.zeros((len(positions), PCA_DIM), np.float16)
            ckpt = np.zeros((len(conditions), len(positions), 4), np.float32) if val else None
            for start, end, rows in plan:
                onehot = jnp.asarray(_window_onehot(fasta, chrom, start, end))
                x = embed(onehot)
                emb[rows] = np.asarray(project(x))[positions[rows] - start]
                if val:
                    rna = np.stack(
                        [
                            readers.get(c)[1].read(chrom, start, end) * np.float32(scales[c])
                            for c in conditions
                        ]
                    )
                    pred = np.asarray(head_apply(x, onehot, jnp.asarray(rna)), np.float32)
                    ckpt[:, rows] = pred[:, positions[rows] - start]
            out.mkdir(parents=True, exist_ok=True)
            _save_npy(out / "emb.npy", emb)
            if val:
                _save_npy(out / "ckpt.npy", select_strand(ckpt, strands))
            if test:
                _save_npy(out / "seq.npy", shard_codes(fasta, index, shard))
            print(
                f"embed {shard}: {len(plan)} windows, {len(positions)} rows, "
                f"{time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)",
                flush=True,
            )
    finally:
        readers.close()
        fasta.close()
    _update_config(
        cache,
        checkpoint=str(checkpoint),
        arm=arm,
        head=head_kind,
        seconds={f"embed{'_' + args.shards if args.shards else ''}": time.time() - t0},
    )


def cache_size(cache: str | Path) -> int:
    """Bytes under ``cache``."""
    return sum(p.stat().st_size for p in Path(cache).rglob("*") if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=("index", "tracks", "embed", "build", "size"))
    ap.add_argument("--name", default=None, help="cache name, e.g. genomic_fc2d_k562")
    ap.add_argument("--cache", type=Path, help="default: data/tracks/txcache/<name>")
    ap.add_argument(
        "--checkpoint", type=Path, default=None, help="stage-1 run best/ (embed, build)"
    )
    ap.add_argument("--arm", default=None, help="default: the run's run.json")
    ap.add_argument("--head", default=None, help="default: the run's run.json")
    ap.add_argument("--windows-dir", type=Path, default=Path("data/tracks/windows"))
    ap.add_argument(
        "--expression", type=Path, default=Path("data/tracks/expression/expression.parquet")
    )
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/tracks/tracks"))
    ap.add_argument("--gtf", default=os.environ.get("TISIAGO_GTF"))
    ap.add_argument("--genome", default=os.environ.get("TISIAGO_GENOME"))
    ap.add_argument("--base-checkpoint", default=os.environ.get("ALPHAGENOME_WEIGHTS_PATH"))
    ap.add_argument("--calls-run", default=os.environ.get("TISIAGO_RIBOTISH_RUN"))
    ap.add_argument("--shards", default=None, help="comma list of shards (embed)")
    ap.add_argument("--pca-windows", type=int, default=150)
    ap.add_argument("--pca-per-window", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1, help="processes for the tracks stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--test",
        action="store_true",
        help="build a final TEST cache (test windows only; needs --i-am-final and --pca-from)",
    )
    ap.add_argument("--i-am-final", action="store_true", help="required for --test")
    ap.add_argument(
        "--pca-from", type=Path, default=None, help="train cache whose pca.npz is reused (--test)"
    )
    return ap


def check_args(ap: argparse.ArgumentParser, args) -> None:
    """CLI guards: a cache path; a test cache needs ``--i-am-final``, ``--pca-from``, a new name."""
    if args.cache is None:
        ap.error("give --name or --cache")
    if args.test:
        if not args.i_am_final:
            ap.error("--test reads the held-out test chromosomes; pass --i-am-final to confirm")
        if args.stage in ("embed", "build") and args.pca_from is None:
            ap.error("--test needs --pca-from <train cache> (PCA is never refit on test)")
        if (
            args.pca_from is not None
            and Path(args.pca_from).resolve() == Path(args.cache).resolve()
        ):
            ap.error("--pca-from must be the train cache, not the test cache")
    elif args.pca_from is not None:
        ap.error("--pca-from is only for --test")
    if (Path(args.cache) / INDEX).exists():
        is_test = _load_config(Path(args.cache)).get("split") == TEST_SPLIT
        if is_test and not args.test:
            ap.error(f"{args.cache} is a test cache; pass --test --i-am-final")
        if args.test and not is_test:
            ap.error(f"{args.cache} is a train/val cache; give a new --name for the test cache")


def main() -> None:
    """CLI: build (a stage of) the transcript cache."""
    ap = build_parser()
    args = ap.parse_args()
    if args.cache is None and args.name is not None:
        args.cache = DEFAULT_ROOT / args.name
    check_args(ap, args)
    if args.stage in ("index", "build") and not args.gtf:
        ap.error("--gtf is required (or set TISIAGO_GTF in .env)")
    if args.stage in ("embed", "build"):
        if args.checkpoint is None:
            ap.error("--checkpoint is required for embed / build")
        for flag in ("genome", "base_checkpoint"):
            if not getattr(args, flag):
                ap.error(f"--{flag.replace('_', '-')} is required (or set it in .env)")
    if args.stage == "index" or (args.stage == "build" and not (args.cache / INDEX).exists()):
        stage_index(args)
    if args.stage in ("tracks", "build"):
        stage_tracks(args)
    if args.stage in ("embed", "build"):
        stage_embed(args)
    size = cache_size(args.cache)
    _update_config(Path(args.cache), size_bytes=size)
    print(f"cache {args.cache}: {size / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
