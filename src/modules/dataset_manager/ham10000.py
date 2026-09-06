"""
HAM10000 Dataset Manager.
Handles specific parsing logic for the HAM10000 metadata format.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.core.constants import LesionClass
from src.core.exceptions import DatasetNotFoundError
from src.modules.dataset_manager.base import BaseDatasetManager


class HAM10000Manager(BaseDatasetManager):
    """
    Parser for the HAM10000 dataset.
    Expects `HAM10000_metadata.csv` in the `raw/` directory.
    """

    def process_and_clean(self) -> pd.DataFrame:
        """
        Parses the raw HAM10000 CSV and maps classes to the internal LesionClass enum.
        """
        logger.info(f"Processing dataset: {self.config.name}")
        self.setup_directories()

        metadata_file = self.raw_dir / "HAM10000_metadata.csv"
        if not metadata_file.exists():
            raise DatasetNotFoundError(
                message=f"Missing metadata file for {self.config.name}",
                detail=str(metadata_file)
            )

        # Load raw data
        df = pd.read_csv(metadata_file)

        # Ensure required columns exist
        required_cols = {"image_id", "dx"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Metadata missing required columns: {required_cols - set(df.columns)}")

        # Clean and standardise
        cleaned_records = []
        for _, row in df.iterrows():
            img_id = row["image_id"]
            dx = row["dx"].lower().strip()

            # Map raw string to LesionClass enum
            try:
                class_enum = LesionClass(dx)
            except ValueError:
                logger.warning(f"Skipping row with unmapped class '{dx}' for image {img_id}")
                continue

            # In a real dataset, images might be scattered in subfolders (part_1, part_2).
            # For this unified parser, we assume they are extracted directly into raw/
            # or we construct the path by searching, but sticking to standard structure:
            img_path = self.raw_dir / f"{img_id}.jpg"

            record = {
                "image_id": img_id,
                "class_id": list(LesionClass).index(class_enum),
                "class_name": class_enum.value,
                "path": str(img_path.absolute()),
            }
            if "lesid" in row and pd.notna(row["lesid"]):
                record["lesion_id"] = row["lesid"]
            elif "lesion_id" in row and pd.notna(row["lesion_id"]):
                record["lesion_id"] = row["lesion_id"]
                
            cleaned_records.append(record)

        cleaned_df = pd.DataFrame(cleaned_records)

        # Save unified labels
        cleaned_path = self.labels_dir / "cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved {len(cleaned_df)} unified records to {cleaned_path}")

        return cleaned_df
