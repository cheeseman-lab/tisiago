"""Per-condition expressed-gene table from TIS-input RNA counts (manifest-equivalent rule)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tisiago.seq2func.samples import load_samples

_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_LANE_SUFFIX = re.compile(r"_L\d+$")


def read_gene_spans(gtf: str | Path) -> pd.DataFrame:
    """GTF ``gene`` lines as 0-based half-open spans."""
    rows = []
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            rows.append(
                {
                    "gene_id": _GENE_ID.search(f[8]).group(1),
                    "chrom": f[0],
                    "start": int(f[3]) - 1,
                    "end": int(f[4]),
                    "strand": f[6],
                }
            )
    return pd.DataFrame(rows, columns=["gene_id", "chrom", "start", "end", "strand"])


def read_counts(path: str | Path) -> pd.Series:
    """htseq-count gene counts, summary lines (``__*``) dropped."""
    table = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "count"])
    table = table[~table.gene_id.str.startswith("__")]
    return table.set_index("gene_id")["count"].astype(int)


def counts_path(root: str | Path, sample_id: str, counts_dir: str | Path | None = None) -> Path:
    """Resolve counts file for a sample, handling lane suffix mismatches.

    First tries exact {sample_id}_htseqcount.txt; if not found, searches for
    {library}_L*_htseqcount.txt where library = sample_id with lane suffix removed.

    Args:
        root: directory containing counts/ subdirectory
        sample_id: sample identifier (may include _L[digit]+ lane suffix)
        counts_dir: counts directory; default ``root/counts``

    Returns:
        Path to the htseq-count file

    Raises:
        FileNotFoundError: if no matching file found
        ValueError: if multiple lane variants match
    """
    root = Path(root)
    counts_dir = Path(counts_dir) if counts_dir is not None else root / "counts"
    exact_path = counts_dir / f"{sample_id}_htseqcount.txt"

    # Try exact match first
    if exact_path.exists():
        return exact_path

    # Try lane fallback: strip lane suffix and search for any _L* variant
    library = _LANE_SUFFIX.sub("", sample_id)
    pattern = f"{library}_L*_htseqcount.txt"
    matches = list(counts_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"no counts file found for {sample_id} (tried {exact_path} and {pattern})"
        )
    if len(matches) > 1:
        raise ValueError(f"multiple lane variants found for {library}: {[m.name for m in matches]}")

    return matches[0]


def expression_table(
    atis_root: str | Path,
    gtf: str | Path,
    min_count: int = 2,
    counts_dir: str | Path | None = None,
) -> pd.DataFrame:
    """One row per condition x gene: summed TIS-input counts and ``expressed`` flag.

    ``counts_dir`` replaces ``atis_root/counts`` (e.g. counts rebuilt from rebuilt BAMs).
    """
    root = Path(atis_root)
    spans = read_gene_spans(gtf)
    samples = load_samples(root)
    frames = []
    for condition, rows in samples[samples.library == "rna"].groupby("condition"):
        total = pd.Series(0, index=spans.gene_id, dtype=int)
        for sample_id in rows.sample_id:
            path = counts_path(root, sample_id, counts_dir)
            counts = read_counts(path)
            total = total.add(counts.reindex(spans.gene_id, fill_value=0), fill_value=0)
        frame = spans.copy()
        frame.insert(0, "condition", condition)
        frame["rna_count"] = total.to_numpy().astype(int)
        frame["expressed"] = frame.rna_count >= min_count
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
