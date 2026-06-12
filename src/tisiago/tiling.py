"""Per-candidate window tiling for TIS embedding extraction.

Pure, CPU-testable geometry. Each candidate codon must be embedded with adequate
genomic context, but the head only ever sees the sliced per-candidate vector — so
the candidate's absolute position inside the forward-pass window is irrelevant.
That freedom lets nearby candidates in a transcript share one forward pass.

Strategy — **grid snap**: snap each candidate's A position to a coarse grid so
candidates within ``step`` bp collapse onto the same tile, while the grid spacing
guarantees the candidate lands comfortably off the window edges:

* ``centered`` (AlphaGenome, bidirectional): tile centred on the grid point;
  candidate offset ∈ [W/4, 3W/4].
* ``left_heavy`` (Evo-2, causal): candidate placed ~60-70% through the tile so it
  carries generous real upstream context (the half the causal receptive field sees).

Coordinates are plus-strand genomic; minus-strand tiles are reverse-complemented at
fetch time, so the within-tile offset is computed in the fed (5'→3') orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Model tile specs. window/step in bp; offsets applied on top of each candidate
# (Evo-2 grabs downstream Kozak/frame positions its causality hides at the codon).
MODEL_SPECS: dict[str, dict] = {
    "ag16k": {"backend": "alphagenome_jax", "window": 16384, "placement": "centered",
              "step": 8192, "offsets": [0], "length_tag": "L16k"},
    "ag131k": {"backend": "alphagenome_jax", "window": 131072, "placement": "centered",
               "step": 65536, "offsets": [0], "length_tag": "L131k"},
    "evo2_8k": {"backend": "evo2", "window": 8192, "placement": "left_heavy",
                "step": 2048, "up_anchor": 5120, "offsets": [0, 3, 6, 9], "length_tag": "W8k"},
}


@dataclass
class TileMember:
    """One candidate sliced from a tile."""

    row_idx: int
    offset: int  # within-tile, fed-orientation index of the candidate's A
    codon: str


@dataclass
class Tile:
    """A single forward-pass window covering one or more candidates."""

    chrom: str
    start: int  # plus-strand genomic start (may be < 0 → N-pad at fetch)
    strand: str
    window: int
    members: list[TileMember]

    @property
    def end(self) -> int:
        return self.start + self.window


def a_plus_of(gstart: int, strand: str) -> int:
    """Plus-strand genomic position of the codon's A from the standardized gstart."""
    return gstart if strand == "+" else gstart - 1


def candidate_tile(a_plus: int, strand: str, spec: dict) -> tuple[int, int]:
    """Return (tile_start, within_tile_offset) for a candidate's A position.

    The within-tile offset is in fed (5'→3') orientation: for ``+`` it is the
    distance from ``tile_start``; for ``-`` it is measured from the tile's 3' end
    because the sequence is reverse-complemented before the forward pass.
    """
    w, step = spec["window"], spec["step"]
    snap = round(a_plus / step) * step
    if spec["placement"] == "centered":
        tile_start = snap - w // 2 if strand == "+" else snap - w // 2
    else:  # left_heavy: candidate sits ~up_anchor into the fed sequence
        up = spec["up_anchor"]
        tile_start = (snap - up) if strand == "+" else (snap - (w - up))
    offset = (a_plus - tile_start) if strand == "+" else (tile_start + w - 1 - a_plus)
    return tile_start, offset


def group_into_tiles(candidates, spec: dict) -> list[Tile]:
    """Group candidate rows into deduplicated tiles.

    Args:
        candidates: iterable of objects/rows with ``row_idx``, ``chrom``,
            ``gstart``, ``strand``, ``codon`` attributes (e.g. namedtuples from
            ``DataFrame.itertuples``).
        spec: a MODEL_SPECS entry.

    Returns:
        List of Tile, deduplicated by (chrom, tile_start, strand).
    """
    tiles: dict[tuple[str, int, str], Tile] = {}
    w = spec["window"]
    for c in candidates:
        a = a_plus_of(int(c.gstart), c.strand)
        tile_start, offset = candidate_tile(a, c.strand, spec)
        if offset < 0 or offset >= w:  # safety: should not happen by construction
            continue
        key = (c.chrom, tile_start, c.strand)
        tile = tiles.get(key)
        if tile is None:
            tile = Tile(chrom=c.chrom, start=tile_start, strand=c.strand, window=w, members=[])
            tiles[key] = tile
        tile.members.append(TileMember(row_idx=int(c.row_idx), offset=offset, codon=c.codon))
    return list(tiles.values())
