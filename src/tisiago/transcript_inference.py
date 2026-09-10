"""Pure construction of exon-spliced, transcript-oriented Evo2 requests."""

from __future__ import annotations

from dataclasses import dataclass

from tisiago.sequence import spliced_transcript_sequence

TRANSCRIPT_LENGTH_TAG = "TXP"
TRANSCRIPT_OFFSETS = (0, 3, 6, 9)


@dataclass(frozen=True)
class TranscriptRequest:
    """One complete-transcript Evo2 input and its requested candidate sites."""

    transcript_id: str
    sequence: str
    positions: list[int]
    requests: list[tuple[int, int, int]]


def build_transcript_request(
    rows,
    model: dict,
    fasta,
    *,
    offsets: tuple[int, ...] = TRANSCRIPT_OFFSETS,
    bucket_size: int = 0,
) -> TranscriptRequest:
    """Build a candidate-independent spliced-transcript position-gather plan.

    The model input is always the complete mature transcript in biological 5′→3′
    orientation, followed by ``max(offsets)`` N bases. Thus a candidate receives
    identical model input whether it came from the curated or dense manifest.
    Optional length bucketing is experimental: the current Evo2/Vortex embedding
    implementation is not exactly invariant to suffix padding.
    """
    if bucket_size < 0:
        raise ValueError("bucket_size must be non-negative")
    if not offsets or min(offsets) < 0:
        raise ValueError("transcript offsets must be a non-empty, non-negative tuple")

    records = list(rows.itertuples(index=False))
    if not records:
        raise ValueError("cannot build a transcript request without candidate rows")
    transcript_ids = {str(row.transcript_id) for row in records}
    if len(transcript_ids) != 1:
        raise ValueError("all request rows must belong to one transcript")
    transcript_id = next(iter(transcript_ids))
    if any(row.chrom != model["chrom"] or row.strand != model["strand"] for row in records):
        raise ValueError(f"{transcript_id}: manifest and GTF chromosome/strand disagree")

    transcript_sequence = spliced_transcript_sequence(fasta, model)
    requested: list[tuple[int, int, int]] = []
    seen_rows: set[int] = set()
    for row in records:
        row_idx = int(row.row_idx)
        if row_idx in seen_rows:
            raise ValueError(f"{transcript_id}: duplicate row_idx {row_idx}")
        seen_rows.add(row_idx)
        anchor = int(row.mrna_index)
        if anchor < 0 or anchor + 3 > len(transcript_sequence):
            raise ValueError(f"{transcript_id}: candidate {row_idx} is outside the transcript")
        observed = transcript_sequence[anchor : anchor + 3]
        expected = str(row.codon).upper().replace("U", "T")
        if observed != expected:
            raise ValueError(
                f"{transcript_id}: row {row_idx} codon mismatch at mRNA {anchor}: "
                f"manifest={expected}, sequence={observed}"
            )
        requested.extend((row_idx, offset, anchor + offset) for offset in offsets)

    positions = sorted({position for _, _, position in requested})
    position_index = {position: index for index, position in enumerate(positions)}
    requests = [
        (row_idx, offset, position_index[position])
        for row_idx, offset, position in requested
    ]

    sequence = transcript_sequence + ("N" * max(offsets))
    if positions[-1] >= len(sequence):
        raise AssertionError("fixed transcript tail does not cover requested offsets")
    if bucket_size:
        padded_length = ((len(sequence) + bucket_size - 1) // bucket_size) * bucket_size
        sequence += "N" * (padded_length - len(sequence))

    return TranscriptRequest(
        transcript_id=transcript_id,
        sequence=sequence,
        positions=positions,
        requests=requests,
    )


def expected_transcript_keys(layers, offsets=TRANSCRIPT_OFFSETS) -> set[str]:
    """Return archive keys for the namespaced complete-transcript representation."""
    return {
        f"evo2::{TRANSCRIPT_LENGTH_TAG}::{layer}::off{offset}"
        for layer in layers
        for offset in offsets
    }
