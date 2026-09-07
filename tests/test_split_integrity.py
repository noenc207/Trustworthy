import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.modules.dataset_manager.pytorch_dataset import SkinLesionDataset
from src.core.constants import LesionClass, NORMALIZE_MEAN, NORMALIZE_STD


SPLITS_DIR = Path("data/isic2019/splits")
LABELS_PATH = Path("data/isic2019/labels/cleaned.csv")


@pytest.fixture
def cleaned_df():
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    pytest.skip("cleaned.csv not available")


@pytest.fixture
def split_ids():
    result = {}
    for name, filename in [("train", "train_indices.csv"), ("val", "val_indices.csv"), 
                            ("cal", "cal_indices.csv"), ("test", "test_indices.csv")]:
        path = SPLITS_DIR / filename
        if path.exists():
            result[name] = set(pd.read_csv(path, header=None)[0].tolist())
    if not result:
        pytest.skip("Split files not available")
    return result


def test_no_full_dataset_fallback():
    """Missing split manifest with split_name must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="CRITICAL DATA INTEGRITY ERROR"):
        SkinLesionDataset(
            cleaned_csv_path=LABELS_PATH if LABELS_PATH.exists() else "dummy.csv",
            indices_csv_path=None,
            split_name="test",
        )


def test_missing_manifest_file_raises():
    """Non-existent manifest path must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="Split manifest file not found"):
        SkinLesionDataset(
            cleaned_csv_path=LABELS_PATH if LABELS_PATH.exists() else "dummy.csv",
            indices_csv_path="nonexistent_file.csv",
            split_name="test",
        )


def test_split_disjointness(split_ids):
    """No image should appear in multiple splits."""
    splits = list(split_ids.items())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            name_a, ids_a = splits[i]
            name_b, ids_b = splits[j]
            overlap = ids_a & ids_b
            assert len(overlap) == 0, f"{name_a}-{name_b} overlap: {len(overlap)} images"


def test_group_disjointness(split_ids, cleaned_df):
    """No lesion group should appear in multiple splits."""
    if "lesion_id" not in cleaned_df.columns:
        pytest.skip("No lesion_id column")
    
    id_to_group = {}
    for _, row in cleaned_df.iterrows():
        lid = row.get("lesion_id", "")
        if pd.isna(lid) or lid == "":
            lid = row["image_id"]
        id_to_group[row["image_id"]] = str(lid)
    
    split_groups = {}
    for name, ids in split_ids.items():
        split_groups[name] = set(id_to_group.get(img_id, img_id) for img_id in ids)
    
    groups_list = list(split_groups.items())
    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            name_a, groups_a = groups_list[i]
            name_b, groups_b = groups_list[j]
            overlap = groups_a & groups_b
            assert len(overlap) == 0, f"{name_a}-{name_b} group overlap: {len(overlap)}"


def test_prediction_count_matches_manifest(split_ids):
    """If prediction artifacts exist, their count must match the manifest."""
    pred_file = Path("research/baseline_v2/results/predictions_baseline_v2_test.npz")
    if not pred_file.exists():
        pytest.skip("No predictions file")
    if "test" not in split_ids:
        pytest.skip("No test split")
    
    data = np.load(pred_file)
    n_preds = len(data["image_id"])
    n_manifest = len(split_ids["test"])
    assert n_preds == n_manifest, (
        f"Prediction count ({n_preds}) != test manifest count ({n_manifest}). "
        f"This indicates the predictions were generated from wrong split!"
    )


def test_class_order_consistency():
    """Class order in LesionClass enum must be consistent."""
    expected = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
    actual = [c.value for c in LesionClass]
    assert actual == expected, f"Class order mismatch: {actual} != {expected}"


def test_normalization_consistency():
    """Normalization values must be ImageNet standard."""
    assert NORMALIZE_MEAN == (0.485, 0.456, 0.406), f"Mean mismatch: {NORMALIZE_MEAN}"
    assert NORMALIZE_STD == (0.229, 0.224, 0.225), f"Std mismatch: {NORMALIZE_STD}"


def test_no_checkpoint_strict_false():
    """Evaluate script must use strict=True for checkpoint loading."""
    eval_script = Path("scripts/evaluate_baseline_v2.py")
    if not eval_script.exists():
        pytest.skip("Evaluate script not found")
    content = eval_script.read_text()
    assert "strict=False" not in content, "Found strict=False in evaluate script!"
    assert "strict=True" in content, "strict=True not found in evaluate script!"


def test_splits_sum_to_total(split_ids, cleaned_df):
    """All split sizes must sum to total dataset size."""
    total_in_splits = sum(len(ids) for ids in split_ids.values())
    assert total_in_splits == len(cleaned_df), (
        f"Split sum ({total_in_splits}) != total ({len(cleaned_df)})"
    )
