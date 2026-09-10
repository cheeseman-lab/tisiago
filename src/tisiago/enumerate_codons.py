"""GTF-driven dense codon enumeration -> a scan manifest for the global caller.

Walks each transcript's exons (GENCODE v49 GTF) into spliced mRNA coordinates,
emits one row per reading position (every ``mrna_index``), reads its genome 3-mer,
classifies the codon, and labels it against the curated manifest's called TIS. The
output parquet is schema-compatible with ``extract.py`` (columns ``row_idx,
transcript_id, chrom, gstart, strand, codon``), so the existing GPU extraction +
``store.py`` assemble a dense scan store unchanged.

Coordinate conventions: GTF is 1-based inclusive; manifest ``gstart`` is 0-based.
The Task-4 correctness test pins the mapping by requiring enumerated ``(gstart,
codon)`` to equal the curated manifest for every existing candidate.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import numpy as np

from tisiago.sequence import spliced_transcript_sequence

if TYPE_CHECKING:
    import pandas as pd

# The 9 single-substitution neighbours of ATG (matches the curated negatives).
NEAR_COGNATES = {"CTG", "GTG", "TTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"}


def classify_codon(codon: str) -> str:
    """Classify a 3-mer as ``AUG`` / ``near_cognate`` / ``non_cognate``.

    Args:
        codon: a 3-letter codon (case-insensitive); non-ACGT bases -> non_cognate.

    Returns:
        ``"AUG"``, ``"near_cognate"``, or ``"non_cognate"``.
    """
    c = codon.upper()
    if c == "ATG":
        return "AUG"
    if c in NEAR_COGNATES:
        return "near_cognate"
    return "non_cognate"


def spliced_positions(exons, strand: str) -> np.ndarray:
    """Plus-strand genomic coordinate of each base in mRNA 5'->3' order.

    Args:
        exons: list of (start, end) 0-based half-open genomic intervals (any order).
        strand: ``"+"`` or ``"-"``.

    Returns:
        int array of length = total exon length; element ``i`` is the 0-based
        genomic position of mRNA base ``i`` (reading 5'->3').
    """
    intervals = sorted(exons)  # ascending genomic
    coords = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in intervals])
    return coords if strand == "+" else coords[::-1].copy()


_TX_RE = re.compile(r'transcript_id "([^"]+)"')


def parse_gtf_exons(gtf_path: str, keep: set[str] | None = None) -> dict:
    """Parse exon features from a GTF into per-transcript models.

    Args:
        gtf_path: path to a GENCODE/Ensembl GTF.
        keep: if given, only transcripts whose id is in this set are returned.

    Returns:
        dict ``transcript_id -> {"chrom", "strand", "exons"}`` where ``exons`` is a
        genomic-ascending list of 0-based half-open ``(start, end)`` intervals.
    """
    models: dict[str, dict] = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            mt = _TX_RE.search(f[8])
            if mt is None:
                continue
            tx = mt.group(1)
            if keep is not None and tx not in keep:
                continue
            start = int(f[3]) - 1  # 1-based inclusive -> 0-based
            end = int(f[4])  # inclusive end -> half-open
            rec = models.get(tx)
            if rec is None:
                models[tx] = {"chrom": f[0], "strand": f[6], "exons": [(start, end)]}
            else:
                rec["exons"].append((start, end))
    for rec in models.values():
        rec["exons"].sort()
    return models


def enumerate_transcript(tx: str, model: dict, fa) -> "pd.DataFrame":
    """Enumerate every codon of one transcript into a DataFrame.

    Args:
        tx: transcript id.
        model: ``{"chrom", "strand", "exons"}`` from ``parse_gtf_exons``.
        fa: an open ``pyfaidx.Fasta`` over the genome.

    Returns:
        DataFrame with ``transcript_id, chrom, strand, mrna_index, gstart, codon,
        codon_class`` — one row per reading position ``0..L-3`` (codons with N skipped).
    """
    import pandas as pd

    chrom, strand, exons = model["chrom"], model["strand"], model["exons"]
    coords = spliced_positions(exons, strand)
    mrna = spliced_transcript_sequence(fa, model)
    # gstart convention (confirmed against the curated manifest):
    #   + : gstart = coords[i]      ;  - : gstart = coords[i] + 1
    # (so tiling.a_plus_of(gstart, '-') = gstart - 1 = coords[i], the plus-strand A.)
    g_off = 0 if strand == "+" else 1
    recs = []
    L = len(mrna)
    for i in range(L - 2):
        codon = mrna[i : i + 3]
        if "N" in codon:
            continue
        recs.append((tx, chrom, strand, i, int(coords[i]) + g_off, codon, classify_codon(codon)))
    cols = ["transcript_id", "chrom", "strand", "mrna_index", "gstart", "codon", "codon_class"]
    return pd.DataFrame(recs, columns=cols)


def main() -> None:
    """Enumerate every codon of held-out transcripts into a dense scan manifest."""
    import argparse
    from pathlib import Path

    import pandas as pd
    from pyfaidx import Fasta

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/store/manifest.parquet")
    ap.add_argument(
        "--gtf",
        default=os.environ.get("TISIAGO_GTF"),
        help="Transcript annotation GTF (default: TISIAGO_GTF)",
    )
    ap.add_argument(
        "--genome",
        default=os.environ.get("TISIAGO_GENOME"),
        help="Indexed reference FASTA (default: TISIAGO_GENOME)",
    )
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="data/scan_manifest.parquet")
    args = ap.parse_args()
    if not args.gtf or not args.genome:
        ap.error("--gtf and --genome are required (or set TISIAGO_GTF/TISIAGO_GENOME)")

    man = pd.read_parquet(args.manifest)
    sub = man[man.split.isin(args.splits)]
    tx_split = sub.drop_duplicates("transcript_id").set_index("transcript_id").split.to_dict()
    keep = set(tx_split)
    print(f"enumerating {len(keep)} transcripts from splits {args.splits}", flush=True)

    models = parse_gtf_exons(args.gtf, keep=keep)
    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    pos_keys = set(
        zip(
            man.loc[man.label_tis == 1, "transcript_id"],
            man.loc[man.label_tis == 1, "mrna_index"],
        )
    )

    frames = []
    for n, tx in enumerate(keep):
        if tx not in models:
            continue
        df = enumerate_transcript(tx, models[tx], fa)
        df["split"] = tx_split[tx]
        df["label_tis"] = [int((tx, i) in pos_keys) for i in df.mrna_index]
        frames.append(df)
        if n % 200 == 0:
            print(f"  {n}/{len(keep)}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "row_idx", np.arange(len(out), dtype=np.int64))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    cls = out.codon_class.value_counts().to_dict()
    print(f"wrote {args.out}: {len(out)} positions, {out.transcript_id.nunique()} transcripts")
    print(f"  positives={int(out.label_tis.sum())}  codon_class={cls}")


if __name__ == "__main__":
    main()
