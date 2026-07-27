"""Checkpoint manager for training engine."""

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingCheckpoint:
    epoch: int
    global_step: int
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    scaler_state_dict: dict | None
    ema_state_dict: dict | None
    best_metric: float
    config: dict
    seed: int

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, save_top_k: int = 3, monitor: str = 'val/auroc', mode: str = 'max'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode

        self.best_k_models: dict[str, float] = {}

    def save(self, checkpoint: TrainingCheckpoint, metric_value: float) -> Path:
        filepath = self.checkpoint_dir / f"checkpoint_epoch_{checkpoint.epoch}_step_{checkpoint.global_step}.pt"

        torch.save({
            'epoch': checkpoint.epoch,
            'global_step': checkpoint.global_step,
            'model_state_dict': checkpoint.model_state_dict,
            'optimizer_state_dict': checkpoint.optimizer_state_dict,
            'scheduler_state_dict': checkpoint.scheduler_state_dict,
            'scaler_state_dict': checkpoint.scaler_state_dict,
            'ema_state_dict': checkpoint.ema_state_dict,
            'best_metric': checkpoint.best_metric,
            'config': checkpoint.config,
            'seed': checkpoint.seed
        }, filepath)

        self.best_k_models[str(filepath)] = metric_value
        self.cleanup_old()
        return filepath

    def load_best(self) -> TrainingCheckpoint:
        if not self.best_k_models:
            raise FileNotFoundError("No checkpoints found.")

        if self.mode == 'max':
            best_path = max(self.best_k_models.items(), key=lambda x: x[1])[0]
        else:
            best_path = min(self.best_k_models.items(), key=lambda x: x[1])[0]

        return self._load_path(Path(best_path))

    def load_latest(self) -> TrainingCheckpoint:
        checkpoints = glob.glob(str(self.checkpoint_dir / "checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in directory.")
        latest_path = max(checkpoints, key=os.path.getctime)
        return self._load_path(Path(latest_path))

    def _load_path(self, path: Path) -> TrainingCheckpoint:
        ckpt_data = torch.load(path, map_location='cpu')
        return TrainingCheckpoint(
            epoch=ckpt_data['epoch'],
            global_step=ckpt_data['global_step'],
            model_state_dict=ckpt_data['model_state_dict'],
            optimizer_state_dict=ckpt_data['optimizer_state_dict'],
            scheduler_state_dict=ckpt_data['scheduler_state_dict'],
            scaler_state_dict=ckpt_data['scaler_state_dict'],
            ema_state_dict=ckpt_data['ema_state_dict'],
            best_metric=ckpt_data['best_metric'],
            config=ckpt_data['config'],
            seed=ckpt_data['seed']
        )

    def can_resume(self) -> bool:
        return len(glob.glob(str(self.checkpoint_dir / "checkpoint_*.pt"))) > 0

    def cleanup_old(self) -> None:
        if len(self.best_k_models) > self.save_top_k:
            if self.mode == 'max':
                worst_path = min(self.best_k_models.items(), key=lambda x: x[1])[0]
            else:
                worst_path = max(self.best_k_models.items(), key=lambda x: x[1])[0]

            del self.best_k_models[worst_path]
            try:
                os.remove(worst_path)
            except OSError:
                pass
