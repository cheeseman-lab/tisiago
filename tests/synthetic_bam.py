"""Synthetic BAMs whose Ribo-TISH P-sites reproduce a stranded 1-bp track.

Each count at position ``x`` becomes a 2-bp read starting at ``x - 1``. With the para table
``{2: 1}``, Ribo-TISH places the P-site at ``x`` on both strands, and a 2-bp read always passes
its splice-compatibility check (``--compatiblemis`` 2), so no transcript model is needed.
"""

from __future__ import annotations

from pathlib import Path

import pyBigWig
import pysam

from tisiago.tracks.track_io import read_bigwig_counts

READ_LENGTH = 2
READ_OFFSET = 1


def write_para(path: str | Path) -> None:
    """Write the Ribo-TISH offset table that matches the synthetic reads."""
    Path(path).write_text(f"offdict = {{{READ_LENGTH}: {READ_OFFSET}}}\n")


def tracks_to_bam(
    plus_bw: str | Path, minus_bw: str | Path, out_bam: str | Path, scale: float = 1.0
) -> int:
    """Write a coordinate-sorted, indexed BAM for a stranded track; return the read count."""
    bw = pyBigWig.open(str(plus_bw))
    chrom_sizes = list(bw.chroms().items())
    bw.close()
    plus = read_bigwig_counts(plus_bw)
    minus = read_bigwig_counts(minus_bw)
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in chrom_sizes]}
    n_reads = 0
    with pysam.AlignmentFile(str(out_bam), "wb", header=header) as bam:
        for reference_id, (chrom, _) in enumerate(chrom_sizes):
            events = []
            for is_reverse, counter in ((False, plus.get(chrom, {})), (True, minus.get(chrom, {}))):
                for position, value in counter.items():
                    count = int(round(value * scale))
                    if position >= READ_OFFSET and count > 0:
                        events.append((position - READ_OFFSET, is_reverse, count))
            for start, is_reverse, count in sorted(events):
                for _ in range(count):
                    read = pysam.AlignedSegment(bam.header)
                    read.query_name = f"s{n_reads}"
                    read.query_sequence = "N" * READ_LENGTH
                    read.reference_id = reference_id
                    read.reference_start = start
                    read.cigartuples = [(0, READ_LENGTH)]
                    read.mapping_quality = 255
                    read.flag = 16 if is_reverse else 0
                    read.set_tag("NH", 1)
                    read.set_tag("MD", str(READ_LENGTH))
                    bam.write(read)
                    n_reads += 1
    pysam.index(str(out_bam))
    return n_reads
