"""The aTIS Ribo-seq sample sheet and Ribo-TISH P-site offset files."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

CONDITIONS = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")
CELL_LINE_GROUPS = {
    "HeLa": ("HeLa",),
    "K562": ("K562",),
    "U2OS": ("U2OS",),
    "RPE1": ("RPE1_Async", "RPE1_Que", "RPE1_Sen"),
}
LIBRARIES = ("tis", "chx", "rna")
REPLICATES = (1, 2)
_RPE1 = {"Async": "RPE1_Async", "Quiescent": "RPE1_Que", "Senescent": "RPE1_Sen"}
_LIBRARY = {("TIS", "RPF"): "tis", ("CHX", "RPF"): "chx", ("TIS", "Input"): "rna"}


def condition_name(cell_line: str, condition: str) -> str:
    """Return the condition label used by Ribo-TISH outputs and the tisiago manifest."""
    if cell_line == "RPE1":
        return _RPE1[condition]
    return cell_line


def parse_para(path: str | Path) -> tuple[dict[int, int], dict[int, int] | None]:
    """Parse a Ribo-TISH ``.para.py`` file into (offsets, m0 offsets or None)."""
    _, _, literal = Path(path).read_text().partition("=")
    offdict = ast.literal_eval(literal.strip())
    m0 = offdict.pop("m0", None)
    offsets = {int(length): int(offset) for length, offset in offdict.items()}
    if m0 is not None:
        m0 = {int(length): int(offset) for length, offset in m0.items()}
    return offsets, m0


def load_samples(atis_root: str | Path, bam_overrides: str | Path | None = None) -> pd.DataFrame:
    """Load the 6 conditions x 3 libraries x 2 replicates used by P6.

    ``bam_overrides`` is an optional TSV (``sample_id``, ``bam``) that replaces the BAM of the
    listed samples, e.g. rebuilt alignments. P-site offset files stay those of the sheet.
    """
    root = Path(atis_root)
    sheet = pd.read_csv(root / "ribosome_profiling" / "manifest.csv", dtype=str).fillna("")
    rows = []
    for record in sheet[sheet.bam_path != ""].itertuples():
        library = _LIBRARY.get((record.treatment, record.type))
        if library is None:
            continue
        bam = root / record.bam_path
        rows.append(
            {
                "condition": condition_name(record.cell_line, record.condition),
                "library": library,
                "rep": int(float(record.rep)),
                "sample_id": record.sample_id,
                "bam": bam,
                "para": None if library == "rna" else bam.with_name(bam.name + ".para.py"),
            }
        )
    samples = pd.DataFrame(rows)
    if bam_overrides is not None:
        table = pd.read_csv(bam_overrides, sep="\t", dtype=str)
        unknown = sorted(set(table.sample_id) - set(samples.sample_id))
        if unknown:
            raise ValueError(f"unknown sample_id in {bam_overrides}: {unknown}")
        new = dict(zip(table.sample_id, table.bam))
        samples["bam"] = [Path(new.get(s, b)) for s, b in zip(samples.sample_id, samples.bam)]
    expected = len(CONDITIONS) * len(LIBRARIES) * len(REPLICATES)
    if len(samples) != expected or samples.duplicated(["condition", "library", "rep"]).any():
        raise ValueError(f"expected {expected} unique condition/library/rep samples")
    return samples.sort_values(["condition", "library", "rep"]).reset_index(drop=True)
