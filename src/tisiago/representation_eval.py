"""Multi-seed, leakage-free comparison of W8k and transcript Evo2 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tisiago.caller import (
    evaluate_at_threshold,
    fit_calibrated_head,
    recall_at_fp_budget,
)
from tisiago.linear_head import LinearHeadArtifact
from tisiago.manifest import split_grouped_indices, unique_site_indices

AG16 = "alphagenome_jax/L16k/decoder_1bp/off0.npy"
AG131 = "alphagenome_jax/L131k/decoder_1bp/off0.npy"
KOZAK = "onehot/kozakW20.npy"


def _evo_keys(length_tag: str) -> list[str]:
    """Return the four offset keys for one Evo2 representation."""
    return [
        f"evo2/{length_tag}/blocks.28.mlp.l3/off{offset}.npy"
        for offset in (0, 3, 6, 9)
    ]


REPRESENTATIONS = {
    "w8k": _evo_keys("W8k"),
    "txp": _evo_keys("TXP"),
    "ag_w8k": [AG16, AG131, *_evo_keys("W8k"), KOZAK],
    "ag_txp": [AG16, AG131, *_evo_keys("TXP"), KOZAK],
}


def sample_training_rows(
    rows: np.ndarray, labels: np.ndarray, *, negative_cap: int, seed: int
) -> np.ndarray:
    """Keep every positive and draw a reproducible capped negative sample."""
    rows = np.asarray(rows, dtype=np.int64)
    labels = np.asarray(labels).astype(bool)
    if rows.ndim != 1 or labels.ndim != 1:
        raise ValueError("rows and labels must be one-dimensional")
    if len(rows) and (rows.min() < 0 or rows.max() >= len(labels)):
        raise IndexError("training rows fall outside labels")
    if negative_cap <= 0:
        raise ValueError("negative_cap must be positive")
    positive = rows[labels[rows]]
    negative = rows[~labels[rows]]
    if not len(positive) or not len(negative):
        raise ValueError("training rows must contain both classes")
    if len(negative) > negative_cap:
        negative = np.random.default_rng(seed).choice(
            negative, negative_cap, replace=False
        )
    return np.sort(np.concatenate([positive, negative]))


def load_feature_rows(
    embedding_root: Path, feature_keys: list[str], rows: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Load only selected aligned rows into one preallocated float32 matrix."""
    arrays = []
    missing = []
    for key in feature_keys:
        path = embedding_root / key
        if not path.is_file():
            missing.append(key)
        else:
            arrays.append(np.load(path, mmap_mode="r"))
    if missing:
        raise FileNotFoundError("missing feature arrays: " + ", ".join(missing))
    widths = [int(array.shape[1]) for array in arrays]
    if any(array.shape[0] <= int(rows[-1]) for array in arrays if len(rows)):
        raise ValueError("feature arrays do not cover the selected rows")
    output = np.empty((len(rows), sum(widths)), dtype=np.float32)
    start = 0
    for key, array, width in zip(feature_keys, arrays, widths, strict=True):
        print(f"  loading {key} rows={len(rows):,} width={width:,}", flush=True)
        output[:, start : start + width] = array[rows]
        start += width
    return output, widths


def near_neighbor_pairs(
    manifest: pd.DataFrame, rows: np.ndarray, *, distance: int = 64
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return positive/near-decoy pairs using mature-transcript distance."""
    selected = manifest.iloc[rows].reset_index(drop=True)
    if "mrna_index" not in selected:
        raise ValueError("near-neighbor evaluation requires mrna_index")
    labels = selected.label_tis.to_numpy().astype(bool)
    positions = selected.mrna_index.to_numpy()
    pair_groups = []
    positive_parts = []
    negative_parts = []
    for indices in selected.groupby("transcript_id", sort=False).indices.values():
        indices = np.asarray(indices, dtype=np.int64)
        positive = indices[labels[indices]]
        negative = indices[~labels[indices]]
        group_positive = []
        group_negative = []
        for positive_index in positive:
            near = negative[
                np.abs(positions[negative] - positions[positive_index]) <= distance
            ]
            if len(near):
                group_positive.append(
                    np.full(len(near), positive_index, dtype=np.int64)
                )
                group_negative.append(near)
        if group_positive:
            positive_array = np.concatenate(group_positive)
            negative_array = np.concatenate(group_negative)
        else:
            positive_array = np.empty(0, dtype=np.int64)
            negative_array = np.empty(0, dtype=np.int64)
        pair_groups.append((positive_array, negative_array))
        positive_parts.append(positive_array)
        negative_parts.append(negative_array)
    return (
        np.concatenate(positive_parts),
        np.concatenate(negative_parts),
        pair_groups,
    )


def _metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    transcripts: np.ndarray,
    threshold: float,
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
) -> dict[str, float]:
    """Compute ranking, fixed-threshold, and near-neighbor metrics."""
    fixed = evaluate_at_threshold(probability, labels, transcripts, threshold)
    has_both_classes = len(np.unique(labels)) == 2
    return {
        "auprc": (
            float(average_precision_score(labels, probability))
            if np.any(labels)
            else float("nan")
        ),
        "auroc": (
            float(roc_auc_score(labels, probability))
            if has_both_classes
            else float("nan")
        ),
        "recall": fixed["recall"],
        "precision": float(
            fixed["true_positives"]
            / max(1, fixed["true_positives"] + fixed["false_positives"])
        ),
        "fp_per_transcript": fixed["fp_per_transcript"],
        "winrate64": float(
            np.mean(probability[positive_pairs] > probability[negative_pairs])
        )
        if len(positive_pairs)
        else float("nan"),
    }


def _bootstrap(
    predictions: dict[str, np.ndarray],
    thresholds: dict[str, float],
    labels: np.ndarray,
    transcripts: np.ndarray,
    pair_groups: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_bootstrap: int,
    seed: int = 20260909,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transcript-bootstrap arm estimates and paired TXP-minus-W8k differences."""
    groups = [
        np.asarray(indices, dtype=np.int64)
        for indices in pd.Series(np.arange(len(labels)))
        .groupby(transcripts, sort=False)
        .apply(list)
    ]
    if len(groups) != len(pair_groups):
        raise AssertionError("transcript row groups and near-neighbor groups are misaligned")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    metric_names = (
        "auprc",
        "auroc",
        "recall",
        "precision",
        "fp_per_transcript",
        "winrate64",
    )
    metrics = {
        arm: {name: [] for name in metric_names} for arm in predictions
    }
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in sampled])
        sampled_labels = labels[rows]
        for arm, probability in predictions.items():
            sampled_probability = probability[rows]
            metrics[arm]["auprc"].append(
                float(average_precision_score(sampled_labels, sampled_probability))
                if np.any(sampled_labels)
                else float("nan")
            )
            metrics[arm]["auroc"].append(
                float(roc_auc_score(sampled_labels, sampled_probability))
                if len(np.unique(sampled_labels)) == 2
                else float("nan")
            )
            admitted = sampled_probability >= thresholds[arm]
            metrics[arm]["recall"].append(
                float(
                    np.sum(admitted & sampled_labels)
                    / max(1, np.sum(sampled_labels))
                )
            )
            metrics[arm]["precision"].append(
                float(
                    np.sum(admitted & sampled_labels)
                    / max(1, np.sum(admitted))
                )
            )
            metrics[arm]["fp_per_transcript"].append(
                float(np.sum(admitted & ~sampled_labels) / len(sampled))
            )
            wins = total = 0
            for index in sampled:
                positive, negative = pair_groups[index]
                wins += int(np.sum(probability[positive] > probability[negative]))
                total += len(positive)
            metrics[arm]["winrate64"].append(wins / total if total else np.nan)

    rows = []
    for arm, arm_metrics in metrics.items():
        for name, values in arm_metrics.items():
            values = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "arm": arm,
                    "metric": name,
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                }
            )
    paired_rows = []
    for reference, candidate in (("w8k", "txp"), ("ag_w8k", "ag_txp")):
        if reference not in metrics or candidate not in metrics:
            continue
        for name in metrics[reference]:
            difference = np.asarray(metrics[candidate][name]) - np.asarray(
                metrics[reference][name]
            )
            paired_rows.append(
                {
                    "comparison": f"{candidate}-{reference}",
                    "metric": name,
                    "ci_low": float(np.nanquantile(difference, 0.025)),
                    "ci_high": float(np.nanquantile(difference, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def main() -> None:
    """Train each representation across seeds and write deployable head artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/store")
    parser.add_argument(
        "--arms", nargs="+", choices=sorted(REPRESENTATIONS), default=list(REPRESENTATIONS)
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--train-negatives",
        type=int,
        default=60_000,
        help="Negative rows per seed; all positive training rows are always retained.",
    )
    parser.add_argument("--C", type=float, default=0.00075)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument(
        "--evaluation-scope",
        choices=("curated", "dense"),
        default="curated",
        help="Recorded interpretation of this run; does not alter row selection.",
    )
    parser.add_argument(
        "--test-codon-classes",
        nargs="+",
        default=None,
        help=(
            "Optional test-set codon classes, e.g. AUG near_cognate. Training and "
            "validation retain their manifest-defined candidate populations."
        ),
    )
    parser.add_argument("--out-dir", default="runs/txp_representation")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds:
        raise ValueError("at least one seed is required")
    store = Path(args.store)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(store / "manifest.parquet")
    labels = manifest.label_tis.to_numpy()
    unique = unique_site_indices(manifest)
    train_all = unique[manifest.split.to_numpy()[unique] == "train"]
    validation = unique[manifest.split.to_numpy()[unique] == "val"]
    test = unique[manifest.split.to_numpy()[unique] == "test"]
    if args.test_codon_classes is not None:
        requested = set(args.test_codon_classes)
        available = set(manifest.codon_class.astype(str))
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown test codon classes: {sorted(unknown)}")
        test = test[
            manifest.codon_class.astype(str).isin(requested).to_numpy()[test]
        ]
    if not len(train_all) or not len(validation) or not len(test):
        raise ValueError("train, validation, and selected test rows must all be non-empty")
    calibration, operating = split_grouped_indices(
        validation,
        manifest.transcript_id.to_numpy()[validation],
        fraction=0.5,
        seed=0,
    )
    train_rows = {
        seed: sample_training_rows(
            train_all, labels, negative_cap=args.train_negatives, seed=seed
        )
        for seed in seeds
    }
    needed = np.unique(
        np.concatenate([calibration, operating, test, *train_rows.values()])
    )
    lookup = np.full(len(manifest), -1, dtype=np.int64)
    lookup[needed] = np.arange(len(needed))
    positive_pairs, negative_pairs, pair_groups = near_neighbor_pairs(manifest, test)
    test_labels = labels[test]
    test_transcripts = manifest.transcript_id.to_numpy()[test]
    test_strands = manifest.strand.astype(str).to_numpy()[test]
    test_codon_classes = manifest.codon_class.astype(str).to_numpy()[test]
    np.savez(
        output_dir / "row_selections.npz",
        unique=unique,
        calibration=calibration,
        operating=operating,
        test=test,
        **{f"train_seed{seed}": rows for seed, rows in train_rows.items()},
    )
    run_metadata = {
        "schema_version": 1,
        "store": str(store.resolve()),
        "arms": list(args.arms),
        "feature_keys": {arm: REPRESENTATIONS[arm] for arm in args.arms},
        "seeds": seeds,
        "train_negative_cap": args.train_negatives,
        "regularization_C": args.C,
        "fp_per_transcript_budget": args.budget,
        "bootstrap_replicates": args.bootstrap,
        "test_codon_classes": args.test_codon_classes,
        "split_counts": {
            "unique": int(len(unique)),
            "calibration": int(len(calibration)),
            "operating": int(len(operating)),
            "test": int(len(test)),
            "train_by_seed": {
                str(seed): int(len(rows)) for seed, rows in train_rows.items()
            },
        },
        "evaluation_scope": args.evaluation_scope,
    }

    metric_rows = []
    ensemble_predictions = {}
    ensemble_thresholds = {}
    for arm in args.arms:
        keys = REPRESENTATIONS[arm]
        print(f"\n=== arm={arm} keys={len(keys)} ===", flush=True)
        features, widths = load_feature_rows(store / "embeddings", keys, needed)
        operating_predictions = []
        test_predictions = []
        for seed in seeds:
            head = fit_calibrated_head(
                features[lookup[train_rows[seed]]],
                labels[train_rows[seed]],
                features[lookup[calibration]],
                labels[calibration],
                C=args.C,
                max_iter=1_000,
            )
            p_operating = head["predict"](features[lookup[operating]])
            p_test = head["predict"](features[lookup[test]])
            selected = recall_at_fp_budget(
                p_operating,
                labels[operating],
                manifest.transcript_id.to_numpy()[operating],
                budget=args.budget,
            )
            metrics = _metrics(
                p_test,
                test_labels,
                test_transcripts,
                selected["threshold"],
                positive_pairs,
                negative_pairs,
            )
            metric_rows.append(
                {"arm": arm, "seed": seed, "threshold": selected["threshold"], **metrics}
            )
            artifact = LinearHeadArtifact.from_fitted_head(
                head,
                keys,
                widths,
                operating_threshold=selected["threshold"],
            )
            artifact.save(output_dir / f"{arm}_seed{seed}.npz")
            operating_predictions.append(p_operating)
            test_predictions.append(p_test)
            print(f"  seed={seed} {metrics}", flush=True)

        ensemble_operating = np.mean(operating_predictions, axis=0)
        ensemble_test = np.mean(test_predictions, axis=0)
        selected = recall_at_fp_budget(
            ensemble_operating,
            labels[operating],
            manifest.transcript_id.to_numpy()[operating],
            budget=args.budget,
        )
        metrics = _metrics(
            ensemble_test,
            test_labels,
            test_transcripts,
            selected["threshold"],
            positive_pairs,
            negative_pairs,
        )
        metric_rows.append(
            {"arm": arm, "seed": "ensemble", "threshold": selected["threshold"], **metrics}
        )
        ensemble_predictions[arm] = ensemble_test
        ensemble_thresholds[arm] = selected["threshold"]
        run_metadata.setdefault("ensemble_thresholds", {})[arm] = selected["threshold"]
        print(f"  ensemble {metrics}", flush=True)
        del features

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "metrics.tsv", sep="\t", index=False)
    stratum_masks = {"all": np.ones(len(test), dtype=bool)}
    stratum_masks.update(
        {f"strand:{strand}": test_strands == strand for strand in sorted(set(test_strands))}
    )
    stratum_masks.update(
        {
            f"codon_class:{codon_class}": test_codon_classes == codon_class
            for codon_class in sorted(set(test_codon_classes))
        }
    )
    stratum_rows = []
    no_pairs = np.empty(0, dtype=np.int64)
    for arm, probability in ensemble_predictions.items():
        for stratum, mask in stratum_masks.items():
            pair_mask = mask[positive_pairs] & mask[negative_pairs]
            metrics = _metrics(
                probability[mask],
                test_labels[mask],
                test_transcripts[mask],
                ensemble_thresholds[arm],
                no_pairs,
                no_pairs,
            )
            if pair_mask.any():
                metrics["winrate64"] = float(
                    np.mean(
                        probability[positive_pairs[pair_mask]]
                        > probability[negative_pairs[pair_mask]]
                    )
                )
            stratum_rows.append({"arm": arm, "stratum": stratum, **metrics})
    pd.DataFrame(stratum_rows).to_csv(
        output_dir / "stratified_metrics.tsv", sep="\t", index=False
    )
    intervals, paired = _bootstrap(
        ensemble_predictions,
        ensemble_thresholds,
        test_labels,
        test_transcripts,
        pair_groups,
        n_bootstrap=args.bootstrap,
    )
    intervals.to_csv(output_dir / "bootstrap_ci.tsv", sep="\t", index=False)
    paired.to_csv(output_dir / "paired_bootstrap_ci.tsv", sep="\t", index=False)
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"\n{metrics_frame.to_string(index=False)}")
    print(f"\nbootstrap intervals\n{intervals.to_string(index=False)}")
    if len(paired):
        print(f"\npaired TXP differences\n{paired.to_string(index=False)}")
    print(f"\nwrote results and head artifacts to {output_dir}")


if __name__ == "__main__":
    main()
