"""Augmented, prefetching batch iterator over genomic windows with Ribo-seq track targets."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.augment import reverse_complement, shift
from tisiago.seq2func.masking import expression_mask
from tisiago.seq2func.psite import RIBOTISH_MINUS_SHIFT
from tisiago.seq2func.samples import LIBRARIES
from tisiago.seq2func.targets import (
    RNACOND_LIBRARIES,
    STRAND_FILES,
    channel_layout,
    rnacond_layout,
    strand_pairs,
)
from tisiago.seq2func.tracks import track_prefix

_LOOKUP = np.zeros((256, 4), dtype=np.float32)
for _i, _base in enumerate("ACGT"):
    _LOOKUP[ord(_base), _i] = 1.0
    _LOOKUP[ord(_base.lower()), _i] = 1.0

_DONE = object()
RNA_REFERENCE_DEPTH = 5e7


class _WorkerError:
    def __init__(self, exc: BaseException):
        self.exc = exc


def one_hot(seq: str) -> np.ndarray:
    """One-hot ``(len(seq), 4)`` float32 over ACGT; N and other characters are all zeros."""
    return _LOOKUP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def fetch_sequence(fasta, chrom: str, start: int, end: int) -> str:
    """Uppercase sequence for ``[start, end)``, N-padded past the chromosome ends."""
    size = len(fasta[chrom])
    lo = min(max(start, 0), end)
    hi = max(min(end, size), lo)
    body = str(fasta[chrom][lo:hi]).upper() if hi > lo else ""
    return "N" * (lo - start) + body + "N" * (end - hi)


class TrackReader:
    """Reads :func:`tisiago.seq2func.targets.load_targets` windows from cached bigWig handles."""

    def __init__(
        self,
        tracks_dir: str | Path,
        conditions: Sequence[str],
        rep: int | str = "pooled",
        libraries: Sequence[str] = LIBRARIES,
    ):
        """Resolve the per-channel paths; handles are opened lazily on first read."""
        self.channels = []
        for condition, library, strand in channel_layout(conditions, libraries):
            prefix = track_prefix(tracks_dir, condition, library, rep)
            shift_ = RIBOTISH_MINUS_SHIFT if strand == "-" and library != "rna" else 0
            self.channels.append((f"{prefix}.{STRAND_FILES[strand]}.bw", shift_))
        self._handles: dict[str, object] = {}

    def _handle(self, path: str):
        if path not in self._handles:
            import pyBigWig

            bw = pyBigWig.open(path)
            if bw is None:
                raise FileNotFoundError(path)
            self._handles[path] = bw
        return self._handles[path]

    def read(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Targets ``(end - start, C)``, identical to ``load_targets``."""
        columns = []
        for path, offset in self.channels:
            lo_req, hi_req = start + offset, end + offset
            out = np.zeros(end - start, dtype=np.float32)
            bw = self._handle(path)
            size = bw.chroms().get(chrom, 0)
            lo, hi = max(lo_req, 0), min(hi_req, size)
            if hi > lo:
                values = np.asarray(bw.values(chrom, lo, hi), dtype=np.float32)
                out[lo - lo_req : hi - lo_req] = np.nan_to_num(values)
            columns.append(out)
        return np.stack(columns, axis=1)

    def close(self) -> None:
        """Close all open handles."""
        for bw in self._handles.values():
            bw.close()
        self._handles.clear()


def rna_total(tracks_dir: str | Path, condition: str) -> float:
    """Genome-wide pooled RNA reads of ``condition``: the pooled bigWigs' ``sumData``, both strands.

    The header sum covers every chromosome, so it does not depend on the windows or the split
    (it equals the summed ``rna_rep{1,2}.json`` totals of :func:`tisiago.seq2func.tracks.pool`).
    """
    import pyBigWig

    total = 0.0
    prefix = track_prefix(tracks_dir, condition, "rna", "pooled")
    for name in STRAND_FILES.values():
        bw = pyBigWig.open(f"{prefix}.{name}.bw")
        if bw is None:
            raise FileNotFoundError(f"{prefix}.{name}.bw")
        try:
            total += float(bw.header()["sumData"])
        finally:
            bw.close()
    return total


def rna_depth_scale(tracks_dir: str | Path, condition: str) -> float:
    """``RNA_REFERENCE_DEPTH / rna_total``: scales ``condition``'s RNA to a 5e7-read library.

    Read units are kept (one read of a 50M-read library stays 1), so ``log1p`` still compresses.
    """
    total = rna_total(tracks_dir, condition)
    if not total > 0:
        raise ValueError(f"{condition}: no pooled RNA reads in {tracks_dir}")
    return RNA_REFERENCE_DEPTH / total


def rna_depth_scales(tracks_dir: str | Path, conditions: Sequence[str]) -> dict[str, float]:
    """:func:`rna_depth_scale` of each condition."""
    return {c: rna_depth_scale(tracks_dir, c) for c in conditions}


class RnaCondReaders:
    """Per-condition readers of ``rnacond`` examples: TIS/CHX targets, RNA input, depth scale."""

    def __init__(self, tracks_dir: str | Path):
        """Readers are created on first use of a condition."""
        self.tracks_dir = tracks_dir
        self._readers: dict[str, tuple[TrackReader, TrackReader, float]] = {}

    def get(self, condition: str) -> tuple[TrackReader, TrackReader, float]:
        """``(targets reader, RNA reader, RNA depth scale)`` of ``condition``."""
        if condition not in self._readers:
            self._readers[condition] = (
                TrackReader(self.tracks_dir, [condition], libraries=RNACOND_LIBRARIES),
                TrackReader(self.tracks_dir, [condition], libraries=("rna",)),
                rna_depth_scale(self.tracks_dir, condition),
            )
        return self._readers[condition]

    def close(self) -> None:
        """Close every reader."""
        for targets, rna, _ in self._readers.values():
            targets.close()
            rna.close()
        self._readers.clear()


def _window_sequence(fasta, chrom: str, start: int, end: int, transcripts=None) -> np.ndarray:
    """One-hot (plus transcript features) of ``[start, end)`` from a path or an open handle."""

    def read(handle):
        sequence = one_hot(fetch_sequence(handle, chrom, start, end))
        if transcripts is None:
            return sequence
        return np.concatenate([sequence, transcripts.window(handle, chrom, start, end)], axis=1)

    if isinstance(fasta, (str, Path)):
        from pyfaidx import Fasta

        handle = Fasta(str(fasta), as_raw=True)
        try:
            return read(handle)
        finally:
            handle.close()
    return read(fasta)


def make_example(
    fasta: str | Path | object,
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    reader: TrackReader | None = None,
    transcripts=None,
) -> dict[str, np.ndarray]:
    """One window: ``sequences (S, 4)``, ``targets (S, C)``, ``mask (S, C)``.

    Pass ``reader`` to reuse open bigWig handles across calls. With ``transcripts`` (a
    :class:`tisiago.seq2func.tx_features.TranscriptFeatures`) ``sequences`` is ``(S, 4 + 20)``:
    one-hot, then the transcript-coordinate open-frame features.
    """
    sequences = _window_sequence(fasta, chrom, start, end, transcripts)
    layout = channel_layout(conditions)
    if reader is None:
        reader = TrackReader(tracks_dir, conditions)
        try:
            targets = reader.read(chrom, start, end)
        finally:
            reader.close()
    else:
        targets = reader.read(chrom, start, end)
    return {
        "sequences": sequences,
        "targets": targets,
        "mask": expression_mask(layout, genes, chrom, start, end),
    }


def make_rnacond_example(
    fasta: str | Path | object,
    tracks_dir: str | Path,
    condition: str,
    genes: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    readers: RnaCondReaders | None = None,
) -> dict[str, np.ndarray]:
    """One (window, condition) pair: ``sequences (S, 6)``, ``targets (S, 4)``, ``mask (S, 4)``.

    ``sequences`` is the one-hot, then ``condition``'s pooled RNA 5'-end counts on the plus and
    minus strand (read like the ``rna`` target channels: no minus-strand shift), scaled to a
    library of :data:`RNA_REFERENCE_DEPTH` reads by its genome-wide pooled RNA total
    (:func:`rna_depth_scale`). ``targets``
    are its pooled TIS+, TIS-, CHX+, CHX- (:func:`tisiago.seq2func.targets.rnacond_layout`,
    canonical P-sites) and ``mask`` their expression mask. Pass ``readers`` to reuse handles.
    """
    sequence = _window_sequence(fasta, chrom, start, end)
    own = readers is None
    readers = RnaCondReaders(tracks_dir) if own else readers
    try:
        target_reader, rna_reader, scale = readers.get(condition)
        targets = target_reader.read(chrom, start, end)
        rna = rna_reader.read(chrom, start, end) * np.float32(scale)
    finally:
        if own:
            readers.close()
    return {
        "sequences": np.concatenate([sequence, rna], axis=1),
        "targets": targets,
        "mask": expression_mask(rnacond_layout(condition), genes, chrom, start, end),
    }


def augment(example: dict, rng, pairs, max_shift: int = 3) -> dict[str, np.ndarray]:
    """Reverse-complement with p=0.5, then shift by an offset in ``[-max_shift, max_shift]``.

    Extra ``sequences`` channels (stranded feature blocks) follow :func:`reverse_complement`.
    """
    onehot, targets, mask = example["sequences"], example["targets"], example["mask"]
    if rng.random() < 0.5:
        onehot, targets, mask = reverse_complement(onehot, targets, mask, pairs)
    offset = int(rng.integers(-max_shift, max_shift + 1))
    onehot, targets, mask = shift(onehot, targets, mask, offset)
    return {"sequences": onehot, "targets": targets, "mask": mask}


def _put(out: queue.Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            out.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _produce(out, stop, worker, workers, windows, cfg, seed, augment_examples, loop):
    from pyfaidx import Fasta

    reader = handle = None
    try:
        fasta, tracks_dir, conditions, genes, transcripts, rna_conditioned = cfg
        handle = Fasta(str(fasta), as_raw=True)
        if rna_conditioned:
            reader = RnaCondReaders(tracks_dir)
            pairs = strand_pairs(rnacond_layout(conditions[0]))
            items = [(w, c) for c in conditions for w in range(len(windows))]
        else:
            reader = TrackReader(tracks_dir, conditions)
            pairs = strand_pairs(channel_layout(conditions))
            items = [(w, None) for w in range(len(windows))]
        rng = np.random.default_rng([seed, worker])
        epoch = 0
        while not stop.is_set():
            order = np.arange(len(items))
            if loop:
                order = np.random.default_rng(seed + epoch).permutation(order)
            for i in order[worker::workers]:
                if stop.is_set():
                    return
                w, condition = items[i]
                row = windows.iloc[w]
                window = (row.chrom, int(row.start), int(row.end))
                if rna_conditioned:
                    example = make_rnacond_example(
                        handle, tracks_dir, condition, genes, *window, readers=reader
                    )
                else:
                    example = make_example(
                        handle,
                        tracks_dir,
                        conditions,
                        genes,
                        *window,
                        reader=reader,
                        transcripts=transcripts,
                    )
                if augment_examples:
                    example = augment(example, rng, pairs)
                if not _put(out, {k: v[None] for k, v in example.items()}, stop):
                    return
            if not loop:
                break
            epoch += 1
        _put(out, _DONE, stop)
    except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
        _put(out, _WorkerError(exc), stop)
    finally:
        if reader is not None:
            reader.close()
        if handle is not None:
            handle.close()


def iterate(
    windows: pd.DataFrame,
    *,
    fasta: str | Path,
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    seed: int,
    augment_examples: bool,
    prefetch: int = 4,
    loop: bool = True,
    workers: int = 4,
    transcripts=None,
    rna_conditioned: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batch-1 dicts (leading axis 1) prefetched by daemon threads into a bounded queue.

    ``loop=True`` reshuffles each epoch with ``default_rng(seed + epoch)``, splits it across
    ``workers`` threads (each with its own handle cache and rng stream) and never ends; order
    is not preserved. ``loop=False`` uses one worker and yields each window once, in order.
    Worker errors are re-raised here; closing the generator stops and joins the workers.
    ``transcripts`` (shared by the workers) adds the ``seqconv_tx`` channels: see
    :func:`make_example`.

    ``rna_conditioned`` yields :func:`make_rnacond_example` (window, condition) pairs over every
    window and every condition in ``conditions`` instead: shuffled jointly per epoch with
    ``loop=True``; condition-major (each condition's windows in order) with ``loop=False``.
    """
    if rna_conditioned and transcripts is not None:
        raise ValueError("rna_conditioned examples take no transcripts")
    workers = max(1, workers) if loop else 1
    out: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    cfg = (fasta, tracks_dir, conditions, genes, transcripts, rna_conditioned)
    threads = [
        threading.Thread(
            target=_produce,
            args=(out, stop, k, workers, windows, cfg, seed, augment_examples, loop),
            daemon=True,
        )
        for k in range(workers)
    ]
    for thread in threads:
        thread.start()
    try:
        while True:
            item = out.get()
            if item is _DONE:
                return
            if isinstance(item, _WorkerError):
                raise item.exc
            yield item
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
