"""Open-frame features in transcript coordinates for the ``seqconv_tx`` head (CPU, data loader).

One transcript per gene: the ``MANE_Select``-tagged one (genes without one are skipped). Only the
exon structure is used, never CDS / start / stop annotation, so frames come from sequence. Each
transcript overlapping a window is spliced in full (exons outside the window included), oriented
5'->3', and scored with the ``orf_features`` definition (:func:`strand_features_np`: per relative
frame k, scaled distance to the upstream in-frame stop, the same for ATG, and ``open``). Values
are mapped back to the genomic positions of its exonic bases.

Window layout ``(S, 20)``: per genomic strand (plus, then minus) the 9 features followed by an
``exonic`` indicator. Non-exonic positions are 0 in all 10 channels of that strand. Where two
transcripts of one strand overlap, the one with the longer mRNA wins (equal lengths: the smaller
transcript id).
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tisiago.seq2func.orf_features import DEFAULT_CAP, strand_features_np

STRAND_CHANNELS = 10
NUM_TX_CHANNELS = 2 * STRAND_CHANNELS
TX_CHANNEL_NAMES = tuple(
    name
    for strand in ("plus", "minus")
    for name in (
        *(
            f"{strand}_tx_{kind}_k{k}"
            for kind in ("stop_dist", "atg_dist", "open")
            for k in range(3)
        ),
        f"{strand}_exonic",
    )
)
DEFAULT_CACHE_BYTES = 512 * 2**20
_GENE_ID = re.compile(rb'gene_id "([^"]+)"')
_TRANSCRIPT_ID = re.compile(rb'transcript_id "([^"]+)"')


@dataclass(frozen=True)
class Transcript:
    """One spliced transcript; ``exons`` are 0-based half-open, in ascending genomic order."""

    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]

    @property
    def length(self) -> int:
        """Mature mRNA length."""
        return sum(end - start for start, end in self.exons)

    @property
    def start(self) -> int:
        """First exonic base."""
        return self.exons[0][0]

    @property
    def end(self) -> int:
        """End of the last exon (exclusive)."""
        return self.exons[-1][1]


def load_mane_transcripts(gtf: str | Path) -> list[Transcript]:
    """MANE_Select transcripts (exon lines only), one per gene (the first seen)."""
    exons: dict[bytes, list] = {}
    with open(gtf, "rb") as handle:
        for line in handle:
            if b"MANE_Select" not in line or line.startswith(b"#"):
                continue
            f = line.split(b"\t", 8)
            if f[2] != b"exon" or b'tag "MANE_Select"' not in f[8]:
                continue
            tid = _TRANSCRIPT_ID.search(f[8]).group(1)
            record = exons.setdefault(tid, [f[0], f[6], _GENE_ID.search(f[8]).group(1), []])
            record[3].append((int(f[3]) - 1, int(f[4])))
    out, seen = [], set()
    for tid, (chrom, strand, gene, spans) in exons.items():
        if gene in seen:
            continue
        seen.add(gene)
        out.append(
            Transcript(
                tid.decode(), gene.decode(), chrom.decode(), strand.decode(), tuple(sorted(spans))
            )
        )
    return out


def _mrna_index(tx: Transcript):
    """Per exon ``(genomic start, genomic end, mRNA offset)`` with 5'->3' offsets."""
    order = tx.exons if tx.strand == "+" else tx.exons[::-1]
    offset, out = 0, []
    for start, end in order:
        out.append((start, end, offset))
        offset += end - start
    return out


class TranscriptFeatures:
    """Window features ``(S, 20)`` from a fixed set of transcripts; thread-safe, LRU-cached.

    The per-chromosome interval index is built once; the per-transcript ``(L, 9)`` mRNA features
    are cached up to ``cache_bytes`` (windows overlap, so neighbours reuse them).
    """

    def __init__(
        self,
        transcripts,
        cap: int = DEFAULT_CAP,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ):
        """Index ``transcripts`` by chromosome."""
        self.cap = cap
        self.cache_bytes = cache_bytes
        self._index: dict[str, tuple[np.ndarray, np.ndarray, list[Transcript]]] = {}
        by_chrom: dict[str, list[Transcript]] = {}
        for tx in transcripts:
            by_chrom.setdefault(tx.chrom, []).append(tx)
        for chrom, txs in by_chrom.items():
            txs.sort(key=lambda t: (t.start, t.transcript_id))
            starts = np.array([t.start for t in txs], dtype=np.int64)
            ends = np.array([t.end for t in txs], dtype=np.int64)
            self._index[chrom] = (starts, ends, txs)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.Lock()

    @classmethod
    def from_gtf(cls, gtf: str | Path, cap: int = DEFAULT_CAP, **kwargs) -> TranscriptFeatures:
        """Build from the MANE_Select transcripts of ``gtf``."""
        return cls(load_mane_transcripts(gtf), cap=cap, **kwargs)

    def overlapping(self, chrom: str, start: int, end: int) -> list[Transcript]:
        """Transcripts with at least one exon base in ``[start, end)``."""
        if chrom not in self._index:
            return []
        starts, ends, txs = self._index[chrom]
        hits = np.flatnonzero((starts < end) & (ends > start))
        return [txs[i] for i in hits if any(s < end and e > start for s, e in txs[i].exons)]

    def mrna_features(self, fasta, tx: Transcript) -> np.ndarray:
        """``(L, 9)`` float32 features of the spliced, 5'->3' mRNA (cached)."""
        with self._lock:
            cached = self._cache.get(tx.transcript_id)
            if cached is not None:
                self._cache.move_to_end(tx.transcript_id)
                return cached
        from tisiago.seq2func.data import fetch_sequence, one_hot

        seq = "".join(fetch_sequence(fasta, tx.chrom, s, e) for s, e in tx.exons)
        onehot = one_hot(seq)
        if tx.strand == "-":
            onehot = onehot[::-1, ::-1]
        features = strand_features_np(onehot, self.cap)
        features.setflags(write=False)
        with self._lock:
            if tx.transcript_id not in self._cache:
                self._cache[tx.transcript_id] = features
                self._cached_bytes += features.nbytes
                while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
                    _, old = self._cache.popitem(last=False)
                    self._cached_bytes -= old.nbytes
        return features

    def window(self, fasta, chrom: str, start: int, end: int) -> np.ndarray:
        """``(end - start, 20)`` float32 transcript-coordinate features of a genomic window."""
        out = np.zeros((end - start, NUM_TX_CHANNELS), np.float32)
        overlapping = self.overlapping(chrom, start, end)
        for block, strand in enumerate("+-"):
            txs = [t for t in overlapping if t.strand == strand]
            # Painted in order, so the last writer wins: longest mRNA, then smallest id.
            txs = sorted(txs, key=lambda t: t.transcript_id, reverse=True)
            txs.sort(key=lambda t: t.length)
            view = out[:, block * STRAND_CHANNELS : (block + 1) * STRAND_CHANNELS]
            for tx in txs:
                features = self.mrna_features(fasta, tx)
                for s, e, offset in _mrna_index(tx):
                    lo, hi = max(s, start), min(e, end)
                    if hi <= lo:
                        continue
                    if strand == "+":
                        values = features[offset + lo - s : offset + hi - s]
                    else:
                        values = features[offset + e - hi : offset + e - lo][::-1]
                    view[lo - start : hi - start, :9] = values
                    view[lo - start : hi - start, 9] = 1.0
        return out
