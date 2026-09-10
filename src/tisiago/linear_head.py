"""Portable, closure-free artifacts for calibrated linear inference heads."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LinearHeadArtifact:
    """A standardized logistic head folded into raw-feature coefficients."""

    feature_keys: tuple[str, ...]
    feature_dims: tuple[int, ...]
    coefficient: np.ndarray
    intercept: float
    isotonic_x: np.ndarray
    isotonic_y: np.ndarray
    operating_threshold: float

    def __post_init__(self) -> None:
        """Validate dimensions and calibration knots."""
        if not self.feature_keys or len(self.feature_keys) != len(self.feature_dims):
            raise ValueError("feature keys and dimensions must be non-empty and aligned")
        if len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("feature keys must be unique")
        if any(width <= 0 for width in self.feature_dims):
            raise ValueError("feature dimensions must be positive")
        coefficient = np.asarray(self.coefficient, dtype=np.float32)
        isotonic_x = np.asarray(self.isotonic_x, dtype=np.float64)
        isotonic_y = np.asarray(self.isotonic_y, dtype=np.float64)
        if coefficient.shape != (sum(self.feature_dims),):
            raise ValueError("coefficient width does not match feature dimensions")
        if isotonic_x.ndim != 1 or isotonic_y.shape != isotonic_x.shape:
            raise ValueError("isotonic knots must be aligned one-dimensional arrays")
        if len(isotonic_x) < 2 or np.any(isotonic_x[1:] <= isotonic_x[:-1]):
            raise ValueError("isotonic x knots must be strictly increasing")
        if (
            np.any((isotonic_x < 0) | (isotonic_x > 1))
            or np.any((isotonic_y < 0) | (isotonic_y > 1))
            or np.any(isotonic_y[1:] < isotonic_y[:-1])
        ):
            raise ValueError("isotonic knots must define a monotone probability mapping")
        threshold = float(self.operating_threshold)
        if np.isnan(threshold) or threshold < 0 or (np.isfinite(threshold) and threshold > 1):
            raise ValueError("operating threshold must be in [0, 1] or positive infinity")
        if not (
            np.isfinite(coefficient).all()
            and np.isfinite(self.intercept)
            and np.isfinite(isotonic_x).all()
            and np.isfinite(isotonic_y).all()
        ):
            raise ValueError("linear-head parameters must be finite")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "isotonic_x", isotonic_x)
        object.__setattr__(self, "isotonic_y", isotonic_y)

    @classmethod
    def from_fitted_head(
        cls,
        head: Mapping,
        feature_keys: list[str] | tuple[str, ...],
        feature_dims: list[int] | tuple[int, ...],
        *,
        operating_threshold: float,
    ) -> LinearHeadArtifact:
        """Create an artifact from :func:`caller.fit_calibrated_head` output."""
        calibrator = head["calibrator"]
        return cls(
            feature_keys=tuple(feature_keys),
            feature_dims=tuple(int(width) for width in feature_dims),
            coefficient=np.asarray(head["linear_coef"], dtype=np.float32),
            intercept=float(head["linear_intercept"]),
            isotonic_x=np.asarray(calibrator.X_thresholds_, dtype=np.float64),
            isotonic_y=np.asarray(calibrator.y_thresholds_, dtype=np.float64),
            operating_threshold=float(operating_threshold),
        )

    def feature_slices(self) -> dict[str, slice]:
        """Return each named feature's coefficient slice."""
        result = {}
        start = 0
        for key, width in zip(self.feature_keys, self.feature_dims, strict=True):
            result[key] = slice(start, start + width)
            start += width
        return result

    def logit_from_blocks(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        """Accumulate logits one feature block at a time without concatenation."""
        if set(blocks) != set(self.feature_keys):
            missing = sorted(set(self.feature_keys) - set(blocks))
            extra = sorted(set(blocks) - set(self.feature_keys))
            raise ValueError(f"feature blocks differ: missing={missing}, extra={extra}")
        return self.partial_logit_from_blocks(blocks, include_intercept=True)

    def partial_logit_from_blocks(
        self,
        blocks: Mapping[str, np.ndarray],
        *,
        include_intercept: bool = False,
    ) -> np.ndarray:
        """Project any known feature subset for coefficient-fused extraction.

        The default excludes the intercept so independent model extractors can emit
        additive partial logits. Add the intercept exactly once after all model
        contributions have been summed.
        """
        if not blocks:
            raise ValueError("at least one feature block is required")
        unknown = sorted(set(blocks) - set(self.feature_keys))
        if unknown:
            raise ValueError(f"unknown feature blocks: {unknown}")
        feature_slices = self.feature_slices()
        logits = None
        for key, width in zip(self.feature_keys, self.feature_dims, strict=True):
            if key not in blocks:
                continue
            values = np.asarray(blocks[key], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(f"{key}: expected (*, {width}), got {values.shape}")
            if logits is None:
                initial = self.intercept if include_intercept else 0.0
                logits = np.full(values.shape[0], initial, dtype=np.float32)
            elif values.shape[0] != len(logits):
                raise ValueError("feature blocks have different row counts")
            logits += values @ self.coefficient[feature_slices[key]]
        if logits is None:  # guarded by non-empty features, retained for type narrowing
            raise AssertionError("linear head has no selected features")
        return logits

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply the logistic link and stored isotonic calibration."""
        logits = np.asarray(logits, dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("logits must be finite")
        probability = np.empty_like(logits)
        positive = logits >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_x = np.exp(logits[~positive])
        probability[~positive] = exp_x / (1.0 + exp_x)
        return np.interp(
            probability,
            self.isotonic_x,
            self.isotonic_y,
            left=self.isotonic_y[0],
            right=self.isotonic_y[-1],
        )

    def predict_blocks(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return calibrated probabilities for aligned feature blocks."""
        return self.calibrate_logits(self.logit_from_blocks(blocks))

    def save(self, path: str | Path) -> None:
        """Atomically save the artifact without pickle-dependent Python objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        )
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                np.savez(
                    tmp,
                    format_version=np.asarray(1, dtype=np.int64),
                    feature_keys=np.asarray(self.feature_keys, dtype=np.str_),
                    feature_dims=np.asarray(self.feature_dims, dtype=np.int64),
                    coefficient=self.coefficient,
                    intercept=np.asarray(self.intercept, dtype=np.float64),
                    isotonic_x=self.isotonic_x,
                    isotonic_y=self.isotonic_y,
                    operating_threshold=np.asarray(
                        self.operating_threshold, dtype=np.float64
                    ),
                )
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> LinearHeadArtifact:
        """Load and validate an artifact saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 1:
                raise ValueError("unsupported linear-head artifact version")
            return cls(
                feature_keys=tuple(str(key) for key in data["feature_keys"]),
                feature_dims=tuple(int(width) for width in data["feature_dims"]),
                coefficient=data["coefficient"],
                intercept=float(data["intercept"]),
                isotonic_x=data["isotonic_x"],
                isotonic_y=data["isotonic_y"],
                operating_threshold=float(data["operating_threshold"]),
            )
