"""
Validate dataset splits for data leakage.
"""
import pandas as pd
from pathlib import Path
import sys
from loguru import logger

def validate_splits(dataset_name: str, splits_dir: Path, cleaned_csv: Path):
    if not cleaned_csv.exists():
        logger.error(f"Cleaned CSV not found at {cleaned_csv}")
        return

    df = pd.read_csv(cleaned_csv)
    if 'lesion_id' not in df.columns and 'patient_id' not in df.columns:
        logger.warning(f"Dataset {dataset_name} has no lesion_id or patient_id. Cannot validate group leakage.")
        return

    group_col = 'lesion_id' if 'lesion_id' in df.columns else 'patient_id'
    id_to_group = dict(zip(df['image_id'].astype(str), df[group_col].astype(str)))

    splits = {}
    for split_name in ['train', 'val', 'cal', 'test']:
        idx_file = splits_dir / f"{split_name}_indices.csv"
        if idx_file.exists():
            ids = pd.read_csv(idx_file, header=None)[0].astype(str).tolist()
            groups = {id_to_group.get(i, "UNKNOWN") for i in ids if id_to_group.get(i, "UNKNOWN") != "UNKNOWN"}
            splits[split_name] = groups

    if not splits:
        logger.error("No split indices found!")
        return

    pairs = [('train', 'val'), ('train', 'cal'), ('train', 'test'), 
             ('val', 'cal'), ('val', 'test'), ('cal', 'test')]
    
    leakage_found = False
    for s1, s2 in pairs:
        if s1 in splits and s2 in splits:
            overlap = splits[s1].intersection(splits[s2])
            if overlap:
                logger.error(f"LEAKAGE DETECTED: {s1} and {s2} share {len(overlap)} groups! Examples: {list(overlap)[:5]}")
                leakage_found = True
            else:
                logger.info(f"OK: {s1} - {s2} == empty")

    if leakage_found:
        raise RuntimeError("Data leakage detected across splits. Pipeline halted.")
    else:
        logger.info(f"Dataset {dataset_name} passes leakage validation.")

if __name__ == "__main__":
    base_dir = Path("d:/Trustworthy/data/datasets")
    if not base_dir.exists():
        base_dir = Path("d:/Trustworthy/data")
    
    datasets = ["ham10000", "isic2019", "test_dataset", "test_dataset_v2"]
    for ds in datasets:
        ds_dir = base_dir / ds
        if ds_dir.exists():
            splits_dir = ds_dir / "splits"
            cleaned_csv = ds_dir / "labels/cleaned.csv"
            if splits_dir.exists():
                validate_splits(ds, splits_dir, cleaned_csv)
