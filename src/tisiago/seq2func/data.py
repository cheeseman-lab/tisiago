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
from tisiago.seq2func.targets import channel_layout, load_targets, strand_pairs

_LOOKUP = np.zeros((256, 4), dtype=np.float32)
for _i, _base in enumerate("ACGT"):
    _LOOKUP[ord(_base), _i] = 1.0
    _LOOKUP[ord(_base.lower()), _i] = 1.0

_DONE = object()


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


def make_example(
    fasta: str | Path | object,
    tracks_dir: str | Path,
    conditions: Sequence[str],
    genes: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    """One window: ``sequences (S, 4)``, ``targets (S, C)``, ``mask (S, C)``."""
    if isinstance(fasta, (str, Path)):
        from pyfaidx import Fasta

        handle = Fasta(str(fasta), as_raw=True)
        try:
            seq = fetch_sequence(handle, chrom, start, end)
        finally:
            handle.close()
    else:
        seq = fetch_sequence(fasta, chrom, start, end)
    layout = channel_layout(conditions)
    return {
        "sequences": one_hot(seq),
        "targets": load_targets(tracks_dir, conditions, chrom, start, end),
        "mask": expression_mask(layout, genes, chrom, start, end),
    }


def augment(example: dict, rng, pairs, max_shift: int = 3) -> dict[str, np.ndarray]:
    """Reverse-complement with p=0.5, then shift by an offset in ``[-max_shift, max_shift]``."""
    onehot, targets, mask = example["sequences"], example["targets"], example["mask"]
    if rng.random() < 0.5:
        onehot, targets, mask = reverse_complement(onehot, targets, mask, pairs)
    onehot, targets, mask = shift(
        onehot, targets, mask, int(rng.integers(-max_shift, max_shift + 1))
    )
    return {"sequences": onehot, "targets": targets, "mask": mask}


def _produce(out, windows, fasta, tracks_dir, conditions, genes, seed, augment_examples, loop):
    from pyfaidx import Fasta

    try:
        handle = Fasta(str(fasta), as_raw=True)
        pairs = strand_pairs(channel_layout(conditions))
        rng = np.random.default_rng(seed)
        epoch = 0
        while True:
            order = np.arange(len(windows))
            if loop:
                order = np.random.default_rng(seed + epoch).permutation(order)
            for i in order:
                row = windows.iloc[i]
                example = make_example(
                    handle, tracks_dir, conditions, genes, row.chrom, int(row.start), int(row.end)
                )
                if augment_examples:
                    example = augment(example, rng, pairs)
                out.put({k: v[None] for k, v in example.items()})
            if not loop:
                break
            epoch += 1
        out.put(_DONE)
    except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
        out.put(_WorkerError(exc))


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
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batch-1 dicts (leading axis 1); a daemon thread prefetches into a bounded queue.

    ``loop=True`` reshuffles each epoch with ``default_rng(seed + epoch)`` and never ends;
    ``loop=False`` yields each window once, in order. Worker errors are re-raised here.
    """
    out: queue.Queue = queue.Queue(maxsize=prefetch)
    thread = threading.Thread(
        target=_produce,
        args=(out, windows, fasta, tracks_dir, conditions, genes, seed, augment_examples, loop),
        daemon=True,
    )
    thread.start()
    while True:
        item = out.get()
        if item is _DONE:
            return
        if isinstance(item, _WorkerError):
            raise item.exc
        yield item
