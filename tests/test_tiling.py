from types import SimpleNamespace

from tisiago.tiling import (
    MODEL_SPECS,
    Tile,
    TileMember,
    candidate_tile,
    group_into_tiles,
    tile_position_requests,
)


def test_all_model_specs_keep_requested_offsets_inside_window():
    for spec in MODEL_SPECS.values():
        for strand in ("+", "-"):
            for a_plus in range(-20_000, 20_001, 127):
                _, offset = candidate_tile(a_plus, strand, spec)
                assert offset >= 0
                assert offset + max(spec["offsets"]) < spec["window"]


def test_evo2_s4k_ablation_uses_about_half_as_many_tiles():
    candidates = [
        SimpleNamespace(
            row_idx=i,
            chrom="chr1",
            gstart=i * 3,
            strand="+",
            codon="ATG",
        )
        for i in range(20_000)
    ]
    baseline = group_into_tiles(candidates, MODEL_SPECS["evo2_8k"])
    fast = group_into_tiles(candidates, MODEL_SPECS["evo2_8k_s4k"])
    ratio = len(baseline) / len(fast)
    assert 1.8 < ratio < 2.2


def test_dense_offset_requests_reuse_overlapping_hidden_states():
    tile = Tile(
        chrom="chr1",
        start=0,
        strand="+",
        window=100,
        members=[TileMember(row_idx=i, offset=20 + i, codon="ATG") for i in range(20)],
    )
    positions, requests = tile_position_requests(tile, [0, 3, 6, 9])
    assert len(requests) == 80
    assert len(positions) == 29  # positions 20..48, not 80 repeated gathers
    for row, offset, source in requests:
        assert positions[source] == 20 + row + offset
