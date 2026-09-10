"""Extract frozen Evo2 features from complete exon-spliced transcripts.

This is an accuracy-gated alternative to genomic ``W8k`` extraction. Each input
is the complete mature transcript in biological 5′→3′ orientation with a fixed N
tail for downstream offsets. All outputs use the distinct ``evo2/TXP`` namespace;
a downstream head must be retrained for this representation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from tisiago.enumerate_codons import parse_gtf_exons
from tisiago.extraction_io import PreallocatedShard, shard_is_complete
from tisiago.linear_head import LinearHeadArtifact
from tisiago.manifest import transcript_shard_mask
from tisiago.transcript_inference import (
    TRANSCRIPT_OFFSETS,
    build_transcript_request,
    expected_transcript_keys,
)
from tisiago.transcript_projection import (
    project_transcript_result,
    txp_feature_keys,
    validate_txp_artifacts,
)


def main() -> None:
    """Extract one transcript-sharded part with a frozen Evo2 backend."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True, help="gruyerenome Evo2 YAML")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Equal-length transcript inputs per backend call (default: config.batch_size)",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=0,
        help=(
            "Experimental right-padding multiple; 0 (default) avoids suffix-dependent "
            "embedding changes"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--head-artifact",
        action="append",
        default=[],
        help=(
            "Portable LinearHeadArtifact to project during extraction; repeat for an "
            "ensemble. Emits additive TXP logits instead of full embeddings."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate transcript reconstruction and report workload without loading Evo2",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress NPZ output (default: enabled)",
    )
    args = parser.parse_args()

    import pandas as pd
    from pyfaidx import Fasta

    with open(args.config) as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    if raw_config.get("model") != "evo2":
        raise ValueError("transcript-oriented extraction currently requires config.model=evo2")
    layers = list(raw_config.get("evo2_layers") or ["blocks.28.mlp.l3"])
    artifacts = {}
    for artifact_path in args.head_artifact:
        name = Path(artifact_path).stem
        if name in artifacts:
            raise ValueError(f"duplicate head artifact name: {name}")
        artifacts[name] = LinearHeadArtifact.load(artifact_path)
    if artifacts:
        validate_txp_artifacts(artifacts, txp_feature_keys(layers))
    if args.dry_run:
        config = None
        batch_size = 1
    else:
        from gruyerenome import Config, load_backend

        config = Config.from_yaml(args.config)
        batch_size = args.batch_size or config.batch_size
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
    expected_keys = (
        {f"partial_logit::{name}" for name in artifacts}
        if artifacts
        else expected_transcript_keys(layers)
    )

    started = time.perf_counter()
    manifest = pd.read_parquet(args.manifest)
    required = {
        "row_idx",
        "transcript_id",
        "mrna_index",
        "chrom",
        "strand",
        "codon",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    mask = transcript_shard_mask(
        manifest.transcript_id, args.shard_id, args.n_shards
    )
    shard = manifest[mask]
    row_idx = np.sort(shard.row_idx.to_numpy(dtype=np.int64))
    if len(np.unique(row_idx)) != len(row_idx):
        raise ValueError("manifest row_idx values must be unique")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_kind = "logits" if artifacts else "shard"
    out_path = out_dir / f"evo2_txp_{output_kind}{args.shard_id:05d}.npz"
    resume_keys = expected_keys if len(row_idx) else set()
    if not args.dry_run and args.resume and shard_is_complete(
        out_path, row_idx, resume_keys
    ):
        print(f"[shard {args.shard_id}] complete output exists; skipping {out_path}")
        return
    if len(shard) == 0:
        if args.dry_run:
            print(f"[shard {args.shard_id}] dry run passed; shard is empty", flush=True)
        else:
            PreallocatedShard(row_idx).save(out_path, compressed=args.compress)
        return

    transcript_ids = set(shard.transcript_id.astype(str))
    models = parse_gtf_exons(args.gtf, keep=transcript_ids)
    missing_models = transcript_ids - models.keys()
    if missing_models:
        examples = sorted(missing_models)[:5]
        raise ValueError(
            f"GTF is missing {len(missing_models)} manifest transcripts; examples={examples}"
        )
    fasta = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    requests = [
        build_transcript_request(
            rows,
            models[str(transcript_id)],
            fasta,
            offsets=TRANSCRIPT_OFFSETS,
            bucket_size=args.bucket_size,
        )
        for transcript_id, rows in shard.groupby("transcript_id", sort=False)
    ]
    requests.sort(key=lambda item: (len(item.sequence), item.transcript_id))
    input_bases = sum(len(item.sequence) for item in requests)
    unique_positions = sum(len(item.positions) for item in requests)
    print(
        f"[shard {args.shard_id}/{args.n_shards}] {len(row_idx):,} candidates / "
        f"{len(requests):,} complete transcripts / {input_bases:,} input bases / "
        f"{unique_positions:,} unique hidden rows / "
        f"max input={max(len(item.sequence) for item in requests):,}",
        flush=True,
    )
    if args.dry_run:
        print(f"[shard {args.shard_id}] dry run passed; Evo2 was not loaded", flush=True)
        return

    if config is None:  # narrowed by the dry-run return above
        raise AssertionError("real extraction requires a parsed gruyerenome config")
    backend = load_backend(config)
    output = PreallocatedShard(
        row_idx, dtype=np.float32 if artifacts else np.float16
    )
    inference_seconds = collect_seconds = 0.0
    completed = 0
    group_start = 0
    while group_start < len(requests):
        length = len(requests[group_start].sequence)
        group_end = group_start + 1
        while group_end < len(requests) and len(requests[group_end].sequence) == length:
            group_end += 1
        for start in range(group_start, group_end, batch_size):
            batch = requests[start : min(start + batch_size, group_end)]
            stage_started = time.perf_counter()
            results = backend.embed_positions(
                [item.sequence for item in batch],
                [item.positions for item in batch],
            )
            inference_seconds += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            for item, result in zip(batch, results, strict=True):
                if artifacts:
                    projected_rows, partials = project_transcript_result(
                        item, result, artifacts
                    )
                    for name, values in partials.items():
                        output.add(
                            f"partial_logit::{name}", projected_rows, values[:, None]
                        )
                else:
                    member_rows = np.fromiter(
                        (row for row, _, _ in item.requests), dtype=np.int64
                    )
                    member_offsets = np.fromiter(
                        (offset for _, offset, _ in item.requests), dtype=np.int64
                    )
                    source_rows = np.fromiter(
                        (source for _, _, source in item.requests), dtype=np.int64
                    )
                    for layer, values in result.items():
                        for offset in TRANSCRIPT_OFFSETS:
                            selected = member_offsets == offset
                            key = f"evo2::TXP::{layer}::off{offset}"
                            output.add(
                                key,
                                member_rows[selected],
                                values[source_rows[selected]],
                            )
            collect_seconds += time.perf_counter() - stage_started
            completed += len(batch)
            if completed == len(requests) or completed % 100 == 0:
                print(
                    f"[shard {args.shard_id}] transcripts {completed:,}/{len(requests):,}",
                    flush=True,
                )
        group_start = group_end

    write_started = time.perf_counter()
    output.save(
        out_path,
        expected_keys=expected_keys,
        compressed=args.compress,
    )
    write_seconds = time.perf_counter() - write_started
    elapsed = time.perf_counter() - started
    print(f"[shard {args.shard_id}] wrote {out_path}", flush=True)
    print(
        f"[shard {args.shard_id}] timing: total={elapsed:.1f}s "
        f"inference={inference_seconds:.1f}s collect={collect_seconds:.1f}s "
        f"write={write_seconds:.1f}s throughput="
        f"{len(requests) / max(inference_seconds, 1e-9):.2f} transcripts/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
