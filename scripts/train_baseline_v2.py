"""
Train Baseline V2 (Real Implementation)
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
from src.training.train_pipeline import train

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Enforce Baseline V2 specifics according to directive
    cfg.trainer.strict_mode = True
    
    if cfg.dataset.name == "synthetic":
        raise ValueError("Synthetic mock dataset is forbidden for real training/evaluation.")
    
    if cfg.dataset.base_path == "data/test_dataset_v2":
        from loguru import logger
        logger.warning("USING TEST / DEVELOPMENT DATA ONLY. THESE ARE NOT RESEARCH RESULTS.")
        
    train(cfg)

if __name__ == "__main__":
    main()
