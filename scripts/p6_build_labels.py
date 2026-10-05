"""Rebuild per-condition TIS labels from round-trip Ribo-TISH output with swissisoform."""

import argparse
from pathlib import Path

import pandas as pd
from swissisoform.io.rnaseq import sum_replicate_counts
from swissisoform.pipeline import run_sample

SWISS = Path("/lab/barcheese01/mdiberna/swissisoform-v2")
CONDITIONS = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")


def main() -> None:
    """Run swissisoform ``run_sample`` on each condition's round-trip predictions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrip-dir", default="data/p6/roundtrip")
    ap.add_argument("--atis-root", default="/lab/barcheese01/aTIS_data")
    ap.add_argument("--out-dir", default="data/p6/labels")
    ap.add_argument("--min-gene-counts", type=float, default=10.0)
    ap.add_argument(
        "--genome",
        default=str(SWISS / "data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa"),
    )
    ap.add_argument(
        "--protein", default=str(SWISS / "data/reference/gencode.v49.pc_translations.fa")
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(SWISS / "data/reference/ribotish_replicate_manifest.csv")
    gtf = Path(args.atis_root) / "reference/gencode.v49.primary_assembly.annotation.gtf"
    for condition in CONDITIONS:
        counts = [SWISS / p for p in reps[reps["sample"] == condition].rnaseq_count_file]
        predict = Path(args.roundtrip_dir) / condition / f"{condition}_TIS_predict_all.txt"
        final, _ = run_sample(
            predict,
            counts,
            gtf,
            sample=condition,
            genome_fasta=args.genome,
            protein_fasta=args.protein,
        )
        final.to_parquet(out_dir / f"{condition}_labels.parquet", index=False)
        expression = sum_replicate_counts(counts).rename_axis("Gid").reset_index()
        expression["expressed"] = expression.GeneRNASeqCounts >= args.min_gene_counts
        expression_dir = out_dir.parent / "expression"
        expression_dir.mkdir(parents=True, exist_ok=True)
        expression.to_parquet(expression_dir / f"{condition}.parquet", index=False)
        print(
            condition,
            len(final),
            int((~final.Imputed).sum()),
            int(expression.expressed.sum()),
            flush=True,
        )


if __name__ == "__main__":
    main()
