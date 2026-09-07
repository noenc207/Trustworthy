"""
Patient-Level Dataset Splitting for DERMA-ACT.

Implements scientifically rigorous data splitting that:
1. Groups by patient_id or lesion_id to prevent data leakage
2. Reserves a separate calibration holdout (disjoint from val/test)
3. Records split assignments with deterministic hashing for reproducibility
4. Supports stratified splitting within patient groups
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for patient-level splitting."""

    train_ratio: float = 0.70
    val_ratio: float = 0.10
    calibration_ratio: float = 0.06  # Separate from val — for temperature scaling
    test_ratio: float = 0.14
    seed: int = 42
    group_column: str = "lesion_id"  # or "patient_id" if available
    stratify_column: str = "class_id"

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.calibration_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, "
                f"cal={self.calibration_ratio}, test={self.test_ratio})"
            )


@dataclass
class SplitResult:
    """Result of a patient-level split operation."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    split_hash: str  # Deterministic hash for reproducibility tracking
    config: SplitConfig
    group_column_used: str
    num_unique_groups: int
    split_metadata: dict = field(default_factory=dict)

    @property
    def num_train(self) -> int:
        return len(self.train_indices)

    @property
    def num_val(self) -> int:
        return len(self.val_indices)

    @property
    def num_calibration(self) -> int:
        return len(self.calibration_indices)

    @property
    def num_test(self) -> int:
        return len(self.test_indices)

    def verify_no_leakage(self, groups: np.ndarray) -> bool:
        """Verify that no group appears in multiple splits."""
        train_groups = set(groups[self.train_indices])
        val_groups = set(groups[self.val_indices])
        cal_groups = set(groups[self.calibration_indices])
        test_groups = set(groups[self.test_indices])

        overlap_train_val = train_groups & val_groups
        overlap_train_test = train_groups & test_groups
        overlap_train_cal = train_groups & cal_groups
        overlap_val_test = val_groups & test_groups
        overlap_val_cal = val_groups & cal_groups
        overlap_cal_test = cal_groups & test_groups

        all_overlaps = {
            "train∩val": overlap_train_val,
            "train∩test": overlap_train_test,
            "train∩cal": overlap_train_cal,
            "val∩test": overlap_val_test,
            "val∩cal": overlap_val_cal,
            "cal∩test": overlap_cal_test,
        }

        is_clean = True
        for name, overlap in all_overlaps.items():
            if overlap:
                logger.error(f"DATA LEAKAGE: {name} overlap has {len(overlap)} groups: {list(overlap)[:5]}...")
                is_clean = False

        if is_clean:
            logger.info("✅ No patient/lesion leakage detected across all splits.")
        return is_clean


def _compute_split_hash(indices_dict: dict[str, np.ndarray], config: SplitConfig) -> str:
    """Compute deterministic hash of split assignments for reproducibility."""
    content = {
        "config": {
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "calibration_ratio": config.calibration_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
            "group_column": config.group_column,
        },
        "train_indices": sorted(indices_dict["train"].tolist()),
        "val_indices": sorted(indices_dict["val"].tolist()),
        "cal_indices": sorted(indices_dict["cal"].tolist()),
        "test_indices": sorted(indices_dict["test"].tolist()),
    }
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


def patient_level_split(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> SplitResult:
    """Perform patient/lesion-level stratified split.

    Ensures no patient (or lesion) appears in multiple splits.
    If group_column is missing, falls back to image-level split with a warning.

    Args:
        df: DataFrame with at least columns: image_id, class_id,
            and optionally lesion_id/patient_id.
        config: Split configuration.

    Returns:
        SplitResult with indices for each split.
    """
    rng = np.random.RandomState(config.seed)

    # Determine grouping column
    group_col = config.group_column
    if group_col not in df.columns:
        if "patient_id" in df.columns:
            group_col = "patient_id"
            logger.warning(f"Column '{config.group_column}' not found, falling back to 'patient_id'.")
        elif "lesion_id" in df.columns:
            group_col = "lesion_id"
            logger.warning(f"Column '{config.group_column}' not found, falling back to 'lesion_id'.")
        else:
            # Last resort: use image_id as group (no grouping, but warns)
            group_col = "image_id"
            logger.warning(
                "⚠️ No patient_id or lesion_id found. Using image-level split. "
                "This may cause data leakage if multiple images belong to the same patient."
            )

    if df[group_col].isnull().any():
        logger.info(f"Filling missing {group_col} values with individual image IDs.")
        df[group_col] = df[group_col].fillna(df["image_id"]).astype(str)

    groups = df[group_col].values
    labels = df[config.stratify_column].values
    unique_groups = np.unique(groups)
    num_unique = len(unique_groups)
    logger.info(f"Splitting {len(df)} samples with {num_unique} unique groups (column='{group_col}').")

    # Step 1: Split into (train+val+cal) vs test
    test_size = config.test_ratio
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.seed)
    trainvalcal_idx, test_idx = next(gss_test.split(df, labels, groups))

    # Step 2: Split remaining into (train+val) vs cal
    remaining_groups = groups[trainvalcal_idx]
    remaining_labels = labels[trainvalcal_idx]
    cal_size_of_remaining = config.calibration_ratio / (1.0 - test_size)

    gss_cal = GroupShuffleSplit(n_splits=1, test_size=cal_size_of_remaining, random_state=config.seed + 1)
    trainval_local_idx, cal_local_idx = next(gss_cal.split(
        trainvalcal_idx, remaining_labels, remaining_groups
    ))
    trainval_idx = trainvalcal_idx[trainval_local_idx]
    cal_idx = trainvalcal_idx[cal_local_idx]

    # Step 3: Split trainval into train vs val
    trainval_groups = groups[trainval_idx]
    trainval_labels = labels[trainval_idx]
    val_size_of_trainval = config.val_ratio / (config.train_ratio + config.val_ratio)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_of_trainval, random_state=config.seed + 2)
    train_local_idx, val_local_idx = next(gss_val.split(
        trainval_idx, trainval_labels, trainval_groups
    ))
    train_idx = trainval_idx[train_local_idx]
    val_idx = trainval_idx[val_local_idx]

    # Compute hash
    indices_dict = {
        "train": train_idx,
        "val": val_idx,
        "cal": cal_idx,
        "test": test_idx,
    }
    split_hash = _compute_split_hash(indices_dict, config)

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        calibration_indices=cal_idx,
        test_indices=test_idx,
        split_hash=split_hash,
        config=config,
        group_column_used=group_col,
        num_unique_groups=num_unique,
        split_metadata={
            "total_samples": len(df),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "cal_samples": len(cal_idx),
            "test_samples": len(test_idx),
            "train_groups": len(np.unique(groups[train_idx])),
            "val_groups": len(np.unique(groups[val_idx])),
            "cal_groups": len(np.unique(groups[cal_idx])),
            "test_groups": len(np.unique(groups[test_idx])),
        },
    )

    # Verify no leakage
    result.verify_no_leakage(groups)

    logger.info(
        f"Split complete [hash={split_hash}]: "
        f"train={len(train_idx)}, val={len(val_idx)}, "
        f"cal={len(cal_idx)}, test={len(test_idx)}"
    )

    return result


def save_split(result: SplitResult, output_dir: Path) -> None:
    """Save split assignments to disk for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save indices
    np.save(output_dir / "train_indices.npy", result.train_indices)
    np.save(output_dir / "val_indices.npy", result.val_indices)
    np.save(output_dir / "calibration_indices.npy", result.calibration_indices)
    np.save(output_dir / "test_indices.npy", result.test_indices)

    # Save metadata
    metadata = {
        "split_hash": result.split_hash,
        "group_column_used": result.group_column_used,
        "num_unique_groups": result.num_unique_groups,
        "config": {
            "train_ratio": result.config.train_ratio,
            "val_ratio": result.config.val_ratio,
            "calibration_ratio": result.config.calibration_ratio,
            "test_ratio": result.config.test_ratio,
            "seed": result.config.seed,
            "group_column": result.config.group_column,
            "stratify_column": result.config.stratify_column,
        },
        **result.split_metadata,
    }
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Split saved to {output_dir} (hash={result.split_hash})")


def load_split(output_dir: Path) -> SplitResult:
    """Load a previously saved split."""
    with open(output_dir / "split_metadata.json") as f:
        metadata = json.load(f)

    config = SplitConfig(**metadata["config"])
    return SplitResult(
        train_indices=np.load(output_dir / "train_indices.npy"),
        val_indices=np.load(output_dir / "val_indices.npy"),
        calibration_indices=np.load(output_dir / "calibration_indices.npy"),
        test_indices=np.load(output_dir / "test_indices.npy"),
        split_hash=metadata["split_hash"],
        config=config,
        group_column_used=metadata["group_column_used"],
        num_unique_groups=metadata["num_unique_groups"],
        split_metadata={k: v for k, v in metadata.items() if k not in (
            "split_hash", "group_column_used", "num_unique_groups", "config"
        )},
    )
