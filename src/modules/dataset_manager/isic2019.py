"""
ISIC 2019 Dataset Manager.
Handles specific parsing logic for the ISIC 2019 metadata format.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.core.constants import LesionClass
from src.core.exceptions import DatasetNotFoundError
from src.modules.dataset_manager.base import BaseDatasetManager


class ISIC2019Manager(BaseDatasetManager):
    """
    Parser for the ISIC 2019 dataset.
    Expects `ISIC_2019_Training_GroundTruth.csv` in the `raw/` directory.
    """

    def process_and_clean(self) -> pd.DataFrame:
        """
        Parses the raw ISIC 2019 CSV and maps classes to the internal LesionClass enum.
        Handles one-hot encoded ground truth.
        """
        logger.info(f"Processing dataset: {self.config.name}")
        self.setup_directories()

        metadata_file = self.raw_dir / "ISIC_2019_Training_GroundTruth.csv"
        if not metadata_file.exists():
            raise DatasetNotFoundError(
                message=f"Missing metadata file for {self.config.name}",
                detail=str(metadata_file)
            )

        # Load raw data
        df = pd.read_csv(metadata_file)

        # Ensure required image column exists
        if "image" not in df.columns:
            raise ValueError("Metadata missing required 'image' column")

        metadata_extra_file = self.raw_dir / "ISIC_2019_Training_Metadata.csv"
        if metadata_extra_file.exists():
            meta_df = pd.read_csv(metadata_extra_file)
            if "image" in meta_df.columns and "lesion_id" in meta_df.columns:
                df = pd.merge(df, meta_df[["image", "lesion_id"]], on="image", how="left")

        cleaned_records = []
        for _, row in df.iterrows():
            img_id = row["image"]

            # Map one-hot encoded row to a single class
            lesion_class = None
            if row.get("MEL", 0.0) == 1.0:
                lesion_class = LesionClass.MEL
            elif row.get("NV", 0.0) == 1.0:
                lesion_class = LesionClass.NV
            elif row.get("BCC", 0.0) == 1.0:
                lesion_class = LesionClass.BCC
            elif row.get("AK", 0.0) == 1.0:
                lesion_class = LesionClass.AKIEC
            elif row.get("BKL", 0.0) == 1.0:
                lesion_class = LesionClass.BKL
            elif row.get("DF", 0.0) == 1.0:
                lesion_class = LesionClass.DF
            elif row.get("VASC", 0.0) == 1.0:
                lesion_class = LesionClass.VASC
            elif row.get("SCC", 0.0) == 1.0:
                lesion_class = LesionClass.AKIEC
            else:
                logger.warning(f"Skipping row with unknown/unmapped class for image {img_id}")
                continue

            img_path = self.raw_dir / f"{img_id}.jpg"
            
            record = {
                "image_id": img_id,
                "class_id": list(LesionClass).index(lesion_class),
                "class_name": lesion_class.value,
                "path": str(img_path.absolute()),
            }
            if "lesion_id" in row and pd.notna(row["lesion_id"]):
                record["lesion_id"] = row["lesion_id"]
                
            cleaned_records.append(record)

        cleaned_df = pd.DataFrame(cleaned_records)

        # Save unified labels
        cleaned_path = self.labels_dir / "cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved {len(cleaned_df)} unified records to {cleaned_path}")

        return cleaned_df
