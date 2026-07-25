import os
import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.training.train_pipeline import SkinLesionDataModule, SkinLesionLightningModule

@pytest.fixture
def dummy_cfg():
    return OmegaConf.create({
        "seed": 42,
        "model": {
            "architecture": "simple_cnn"
        },
        "dataset": {
            "name": "synthetic",
            "batch_size": 4,
            "num_workers": 0
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "fast_dev_run": True,
            "label_smoothing": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4
        }
    })

def test_data_module(dummy_cfg):
    dm = SkinLesionDataModule(dummy_cfg)
    dm.setup(stage="fit")
    
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.shape == (4, 3, 224, 224)
    assert y.shape == (4,)

def test_lightning_module_synthetic_run(dummy_cfg):
    pl.seed_everything(42, workers=True)
    dm = SkinLesionDataModule(dummy_cfg)
    
    import torch.nn as nn
    model_instance = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 7)
    )
    model = SkinLesionLightningModule(dummy_cfg, model=model_instance)
    
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    
    # If it completed without crashing, it succeeded.
    assert True

def test_nan_loss_handling(dummy_cfg):
    import torch.nn as nn
    model_instance = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 7)
    )
    model = SkinLesionLightningModule(dummy_cfg, model=model_instance)
    # Inject NaN into the input tensor to force NaN output
    nan_tensor = torch.full((4, 3, 224, 224), float('nan'))
    target = torch.randint(0, 7, (4,))
    
    loss = model.training_step((nan_tensor, target), batch_idx=0)
    # We implemented graceful NaN handling by returning None
    assert loss is None
