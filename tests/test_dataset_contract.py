import pytest
import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from src.modules.dataset_manager.pytorch_dataset import SkinLesionDataset
from src.training.augmentation import get_train_transforms, get_val_transforms

DATA_DIR = Path(r"d:\Trustworthy\data\test_dataset_v2")

def test_dataset_directory_exists():
    assert DATA_DIR.exists()
    assert (DATA_DIR / "raw").exists()
    assert (DATA_DIR / "labels").exists()
    assert (DATA_DIR / "splits").exists()

def test_metadata_csv_exists():
    assert (DATA_DIR / "raw" / "HAM10000_metadata.csv").exists() or (DATA_DIR / "labels" / "cleaned.csv").exists()

def test_referenced_images_exist():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    for _, row in df.iterrows():
        assert Path(row['path']).exists(), f"Image {row['path']} does not exist"

def test_labels_valid():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    assert df['class_id'].min() >= 0
    assert df['class_id'].max() <= 6

def test_class_mapping_valid():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    unique_classes = df['class_name'].unique()
    valid_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for cls in unique_classes:
        assert cls in valid_classes

def test_image_ids_unique():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    assert df['image_id'].is_unique

def test_group_ids_exist():
    raw_df = pd.read_csv(DATA_DIR / "raw" / "HAM10000_metadata.csv")
    has_group = 'lesion_id' in raw_df.columns or 'patient_id' in raw_df.columns
    assert has_group, "lesion_id or patient_id must exist in raw metadata"

def test_transforms_accept_real_image():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    img_path = df.iloc[0]['path']
    img = np.array(Image.open(img_path).convert("RGB"))
    
    train_transform = get_train_transforms(224)
    val_transform = get_val_transforms(224)
    
    train_out = train_transform(image=img)
    assert 'image' in train_out
    assert isinstance(train_out['image'], torch.Tensor)
    
    val_out = val_transform(image=img)
    assert 'image' in val_out
    assert isinstance(val_out['image'], torch.Tensor)

def test_dataset_returns_expected_shape():
    dataset = SkinLesionDataset(
        cleaned_csv_path=DATA_DIR / "labels" / "cleaned.csv",
        transform=get_val_transforms(224),
        image_size=224
    )
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)

def test_dataset_returns_real_image_ids():
    df = pd.read_csv(DATA_DIR / "labels" / "cleaned.csv")
    assert not df.iloc[0]['image_id'].startswith("img_0_0")
