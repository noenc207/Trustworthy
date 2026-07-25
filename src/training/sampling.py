"""
Advanced sampling and splitting strategies.
Supports class balancing, patient-aware splits, and cross-validation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler


class WeightedClassSampler(WeightedRandomSampler):
    """
    Weighted random sampling based on inverse class frequency.
    """
    def __init__(self, labels: list[int] | np.ndarray) -> None:
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        super().__init__(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has roughly equal class representation.
    """
    def __init__(self, labels: list[int] | np.ndarray, batch_size: int) -> None:
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        # Calculate how many samples from each class per batch
        self.samples_per_class = max(1, self.batch_size // len(self.classes))
        self.num_batches = len(self.labels) // self.batch_size

    def __iter__(self) -> Iterator[int]:
        batch_indices = []
        # Create copies of indices to draw from
        indices = {c: self.class_indices[c].copy() for c in self.classes}
        for c in self.classes:
            np.random.shuffle(indices[c])
            
        for _ in range(self.num_batches):
            batch = []
            for c in self.classes:
                # If we run out of indices for a class, re-shuffle
                if len(indices[c]) < self.samples_per_class:
                    indices[c] = self.class_indices[c].copy()
                    np.random.shuffle(indices[c])
                
                # Take samples_per_class elements
                batch.extend(indices[c][:self.samples_per_class])
                indices[c] = indices[c][self.samples_per_class:]
                
            # If batch is slightly smaller than batch_size due to integer division, fill it randomly
            while len(batch) < self.batch_size:
                c = np.random.choice(self.classes)
                if len(indices[c]) < 1:
                    indices[c] = self.class_indices[c].copy()
                    np.random.shuffle(indices[c])
                batch.append(indices[c][0])
                indices[c] = indices[c][1:]
                
            np.random.shuffle(batch)
            batch_indices.extend(batch)
            
        return iter(batch_indices)

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class PatientAwareSampler(Sampler):
    """
    Random sampler that groups by patient, but standard splits are usually better
    handled beforehand in data preparation. Included for specific dataloader dynamic logic.
    """
    def __init__(self, patient_ids: list[str] | np.ndarray) -> None:
        self.patient_ids = np.array(patient_ids)
        self.unique_patients = np.unique(self.patient_ids)
        self.indices_len = len(self.patient_ids)

    def __iter__(self) -> Iterator[int]:
        np.random.shuffle(self.unique_patients)
        indices = []
        for pid in self.unique_patients:
            patient_indices = np.where(self.patient_ids == pid)[0]
            indices.extend(patient_indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.indices_len


def create_stratified_split(
    df: pd.DataFrame, 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float, 
    seed: int, 
    patient_column: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create splits. Supports patient-level grouping if patient_column provided.
    Returns indices for train, val, test.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    if patient_column and patient_column in df.columns:
        # Step 1: Split Train vs (Val + Test) grouped by patient
        gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, temp_idx = next(gss.split(df, groups=df[patient_column]))
        
        # Step 2: Split Val vs Test grouped by patient
        temp_df = df.iloc[temp_idx].reset_index(drop=True)
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_relative_ratio, random_state=seed)
        val_temp_idx, test_temp_idx = next(gss2.split(temp_df, groups=temp_df[patient_column]))
        
        val_idx = temp_idx[val_temp_idx]
        test_idx = temp_idx[test_temp_idx]
        return train_idx, val_idx, test_idx
    else:
        # Standard Stratified split
        y = df['class_id'].values
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, temp_idx = next(sss.split(np.zeros(len(y)), y))
        
        y_temp = y[temp_idx]
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_relative_ratio, random_state=seed)
        val_temp_idx, test_temp_idx = next(sss2.split(np.zeros(len(y_temp)), y_temp))
        
        val_idx = temp_idx[val_temp_idx]
        test_idx = temp_idx[test_temp_idx]
        return train_idx, val_idx, test_idx


def create_kfold_splits(
    df: pd.DataFrame, 
    n_splits: int, 
    seed: int, 
    patient_column: str | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create K-Fold splits, respecting patient grouping if provided.
    Returns list of (train_idx, val_idx).
    """
    if patient_column and patient_column in df.columns:
        gkf = GroupKFold(n_splits=n_splits)
        # GroupKFold doesn't take random_state or shuffle, it is deterministic based on groups
        splits = list(gkf.split(df, groups=df[patient_column]))
        return splits
    else:
        y = df['class_id'].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))
        return splits
