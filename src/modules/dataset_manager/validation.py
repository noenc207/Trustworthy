"""
Dataset Validation Utilities.
Provides analytical functions to ensure dataset integrity.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from loguru import logger


def detect_duplicates(df: pd.DataFrame) -> list[str]:
    """Find duplicated image IDs in the dataset."""
    duplicates = df[df.duplicated(subset=['image_id'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate records.")
    return duplicates['image_id'].unique().tolist()


def detect_corrupted_images(df: pd.DataFrame, num_workers: int = 4) -> list[str]:
    """Find unreadable or corrupted images using multi-threading."""
    def check_image(path: str) -> str | None:
        if not Path(path).exists():
            return path
        img = cv2.imread(path)
        if img is None:
            return path
        return None

    paths = df['path'].tolist()
    corrupted = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(check_image, paths)
        corrupted = [p for p in results if p is not None]

    if corrupted:
        logger.error(f"Detected {len(corrupted)} corrupted or missing images.")
    return corrupted


def analyze_class_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Calculate the frequency of each class in the dataset."""
    dist = df['class_name'].value_counts().to_dict()
    return dist


def detect_patient_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, patient_column: str) -> bool:
    """
    Check if any patient appears in more than one split.
    Returns True if leakage is detected.
    """
    if patient_column not in train_df.columns:
        logger.info("No patient column provided, skipping leakage check.")
        return False
        
    train_patients = set(train_df[patient_column].unique())
    val_patients = set(val_df[patient_column].unique())
    test_patients = set(test_df[patient_column].unique())
    
    leak_train_val = train_patients.intersection(val_patients)
    leak_train_test = train_patients.intersection(test_patients)
    leak_val_test = val_patients.intersection(test_patients)
    
    has_leak = False
    if leak_train_val:
        logger.error(f"Leakage Train/Val! {len(leak_train_val)} patients leak.")
        has_leak = True
    if leak_train_test:
        logger.error(f"Leakage Train/Test! {len(leak_train_test)} patients leak.")
        has_leak = True
    if leak_val_test:
        logger.error(f"Leakage Val/Test! {len(leak_val_test)} patients leak.")
        has_leak = True
        
    return has_leak


def generate_dataset_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate and save a comprehensive JSON report of dataset health."""
    report = {
        "total_images": len(df),
        "class_distribution": analyze_class_distribution(df),
        "duplicates": len(detect_duplicates(df)),
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Dataset report saved to {output_path}")
