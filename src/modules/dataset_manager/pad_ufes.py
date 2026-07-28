"""
Dataset Manager for PAD-UFES-20.
Clinical image dataset for skin lesions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.core.constants import LesionClass, MIN_QUALITY_SCORE
from src.modules.dataset_manager.base import BaseDatasetManager
from src.modules.dataset_manager.config import DatasetConfig

logger = logging.getLogger(__name__)

# PAD-UFES-20 to ISIC mapping
# PAD-UFES classes: BCC, MEL, NEV (Nevus), ACK (Actinic Keratosis), SEK (Seborrheic Keratosis), SCC (Squamous Cell Carcinoma)
PAD_UFES_TO_ISIC = {
    "BCC": LesionClass.BCC,
    "MEL": LesionClass.MEL,
    "NEV": LesionClass.NV,
    "ACK": LesionClass.AKIEC,
    "SEK": LesionClass.BKL,
    "SCC": LesionClass.AKIEC,
}

class PADUFES20Manager(BaseDatasetManager):
    """Dataset manager for PAD-UFES-20 clinical dataset."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.raw_dir = Path(config.base_path) / "raw"
        self.metadata_path = self.raw_dir / "metadata.csv"
        # The images are typically inside an 'images' subfolder or in the root depending on extraction
        self.images_dir = self.raw_dir

    def process_and_clean(self) -> pd.DataFrame:
        """Parse PAD-UFES-20 metadata and map to standardized ISIC classes."""
        logger.info(f"Processing PAD-UFES-20 metadata from {self.metadata_path}")
        self.setup_directories()
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"PAD-UFES-20 metadata not found at {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Standardize columns
        # Assuming typical PAD-UFES structure: 'img_id', 'diagnostic', 'patient_id'
        cleaned_data = []

        for _, row in df.iterrows():
            img_id = str(row.get("img_id", ""))
            diag = str(row.get("diagnostic", ""))
            patient_id = str(row.get("patient_id", ""))

            # Handle different image naming schemes in the unzipped folder
            img_path = self.images_dir / f"{img_id}"
            if not img_path.exists():
                # Sometimes images are inside an 'images/' folder
                img_path = self.images_dir / "images" / f"{img_id}"
            
            # Map diagnostic
            target_class = PAD_UFES_TO_ISIC.get(diag.upper())
            if not target_class:
                continue

            cleaned_data.append({
                "image_id": img_id,
                "path": str(img_path.absolute()),
                "class_name": target_class.value,
                "class_id": list(LesionClass).index(target_class),
                "patient_id": patient_id,
                "quality_score": 1.0,  # Clinical images are generally assumed 1.0 here
                "dataset_source": "PAD-UFES-20"
            })

        df_clean = pd.DataFrame(cleaned_data)
        
        # Save unified labels
        cleaned_path = self.labels_dir / "cleaned.csv"
        df_clean.to_csv(cleaned_path, index=False)
        logger.info(f"Saved {len(df_clean)} unified records to {cleaned_path}")
        
        return df_clean


