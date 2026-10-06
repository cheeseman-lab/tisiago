"""Per-condition expressed-gene table from TIS-input RNA counts (manifest-equivalent rule)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tisiago.seq2func.samples import load_samples

_GENE_ID = re.compile(r'gene_id "([^"]+)"')


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
    path = Path(path)
    if not path.exists():
        # If exact file doesn't exist, return empty Series
        return pd.Series(dtype=int)
    table = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "count"])
    table = table[~table.gene_id.str.startswith("__")]
    return table.set_index("gene_id")["count"].astype(int)


def expression_table(atis_root: str | Path, gtf: str | Path, min_count: int = 2) -> pd.DataFrame:
    """One row per condition x gene: summed TIS-input counts and ``expressed`` flag."""
    root = Path(atis_root)
    spans = read_gene_spans(gtf)
    samples = load_samples(root)
    frames = []
    for condition, rows in samples[samples.library == "rna"].groupby("condition"):
        total = pd.Series(0, index=spans.gene_id, dtype=int)
        for sample_id in rows.sample_id:
            counts = read_counts(root / "counts" / f"{sample_id}_htseqcount.txt")
            total = total.add(counts.reindex(spans.gene_id, fill_value=0), fill_value=0)
        frame = spans.copy()
        frame.insert(0, "condition", condition)
        frame["rna_count"] = total.to_numpy().astype(int)
        frame["expressed"] = frame.rna_count >= min_count
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
