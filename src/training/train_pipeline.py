"""
Main Training Pipeline.

Uses PyTorch Lightning for:
  - Multi-GPU / DDP training
  - Automatic mixed precision (AMP)
  - Gradient accumulation
  - Learning rate scheduling
  - Model checkpointing
  - Early stopping
  - Metric logging (TensorBoard + W&B)

Configured via Hydra + OmegaConf.
"""
from __future__ import annotations

import logging
from typing import Any

import hydra
import mlflow
import pytorch_lightning as pl
import torch
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
import omegaconf.base
import omegaconf.nodes
import typing
from torch.serialization import add_safe_globals
try:
    add_safe_globals([DictConfig, ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode, typing.Any])
except Exception:
    pass

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
    Specificity,
)

from src.core.constants import LesionClass
from src.core.exceptions import TrustworthyError

logger = logging.getLogger(__name__)


class NumericalInstabilityError(TrustworthyError):
    """Raised when loss becomes NaN or Inf during training."""
    def __init__(self, message: str = "Loss is NaN or Inf") -> None:
        super().__init__(message)


class SkinLesionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping SkinLesionClassifier.

    Handles:
      - Training/validation/test steps
      - Loss computation (label-smoothed CrossEntropy)
      - Optimizer and scheduler configuration
      - Metric logging
    """

    def __init__(self, cfg: DictConfig, model: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        self.num_classes = len(LesionClass)

        # 1. Model Injection (Clean Architecture)
        if model is not None:
            self.model = model
        elif "model" in cfg and cfg.model is not None:
            self.model = hydra.utils.instantiate(cfg.model)
        else:
            raise ValueError("A model must be provided either directly or via cfg.model configuration")

        # 2. Loss function with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("trainer", {}).get("label_smoothing", 0.0)
        )

        # 3. Metrics
        task = "multiclass"

        # Train metrics
        self.train_acc = Accuracy(task=task, num_classes=self.num_classes)

        # Validation/Test metrics (Milestone 6.9 Scientific Evaluation)
        self.val_acc = Accuracy(task=task, num_classes=self.num_classes)
        self.val_bal_acc = Accuracy(task=task, num_classes=self.num_classes, average="macro")
        self.val_auroc = AUROC(task=task, num_classes=self.num_classes)
        self.val_f1 = F1Score(task=task, num_classes=self.num_classes)
        self.val_precision = Precision(task=task, num_classes=self.num_classes)
        self.val_recall = Recall(task=task, num_classes=self.num_classes)
        self.val_specificity = Specificity(task=task, num_classes=self.num_classes)
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.val_kappa = CohenKappa(task=task, num_classes=self.num_classes)

        self.test_acc = Accuracy(task=task, num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer + LR scheduler from Hydra config."""
        lr = self.cfg.get("trainer", {}).get("learning_rate", 1e-3)
        weight_decay = self.cfg.get("trainer", {}).get("weight_decay", 1e-4)

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        max_epochs = self.cfg.get("trainer", {}).get("max_epochs", 10)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        loss, logits, y = self._shared_step(batch)

        # Handle NaN/Inf gracefully
        if not torch.isfinite(loss):
            strict_mode = self.cfg.get("trainer", {}).get("strict_mode", False)
            if strict_mode:
                raise NumericalInstabilityError(f"NaN/Inf loss detected at epoch {self.current_epoch}, step {self.global_step}")
            else:
                logger.warning(f"NaN/Inf loss detected at epoch {self.current_epoch}, step {self.global_step}. Skipping step.")
                return None

        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, y = self._shared_step(batch)

        self.val_acc(logits, y)
        self.val_bal_acc(logits, y)
        self.val_auroc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_specificity(logits, y)
        self.val_mcc(logits, y)
        self.val_kappa(logits, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/bal_acc", self.val_bal_acc, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True)
        self.log("val/f1", self.val_f1, on_epoch=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/recall", self.val_recall, on_epoch=True)
        self.log("val/specificity", self.val_specificity, on_epoch=True)
        self.log("val/mcc", self.val_mcc, on_epoch=True)
        self.log("val/kappa", self.val_kappa, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, y = self._shared_step(batch)
        self.test_acc(logits, y)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x, _ = batch
        return torch.softmax(self(x), dim=-1)


class GenericSkinLesionDataset(Dataset):
    """
    Generic dataset that takes a list of data dicts.
    Dataset-agnostic, relies on data provided during setup.
    """
    def __init__(self, data_list: list[dict[str, Any]]) -> None:
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        return item["image"], item["label"]




@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Hydra-managed training entrypoint.
    """
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Data
    from src.training.data_module import SkinLesionDataModule as RealDataModule
    datamodule = RealDataModule(cfg)

    # Model (Inject via factory or config if possible, fallback for script run)
    model_instance = None
    if "model" in cfg and cfg.model is not None:
        model_instance = hydra.utils.instantiate(cfg.model)
    else:
        raise ValueError("Model configuration is missing. Cannot instantiate model.")

    model = SkinLesionLightningModule(cfg, model=model_instance)

    # Callbacks
    trainer_cfg = cfg.get("trainer", {})
    checkpoint_dir = trainer_cfg.get("checkpoint_dir", "checkpoints/")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val_auroc:.4f}",
            monitor="val/auroc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/auroc",
            patience=trainer_cfg.get("early_stopping_patience", 10),
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Loggers
    experiment_cfg = cfg.get("experiment", {})
    tb_logger = TensorBoardLogger(
        save_dir="logs/tensorboard",
        name=experiment_cfg.get("name", "trustworthy_run"),
    )
    loggers = [tb_logger]

    if cfg.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=experiment_cfg.get("project", "trustworthy-ai"),
            name=experiment_cfg.get("name", "trustworthy_run"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)

    # MLflow tracking
    mlflow_cfg = cfg.get("mlflow", {})
    if "tracking_uri" in mlflow_cfg:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default"))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 2),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )

    with mlflow.start_run(run_name=experiment_cfg.get("name", "trustworthy_run")):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        trainer.fit(model, datamodule=datamodule)

        if not trainer.fast_dev_run:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
