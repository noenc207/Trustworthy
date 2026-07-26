"""
Script Training lại Model với bộ lọc chống học vẹt (Anti-Shortcut Learning).

Chạy trên máy ảo:
  python3 retrain_anti_shortcut.py

Tính năng mới:
  - RandomResizedCrop zoom mạnh (0.5-0.85) để cắt bỏ logo/watermark
  - CoarseDropout che mọi chữ/mũi tên còn sót lại
  - Strong color augmentation chống color-based shortcuts
  - Label Smoothing 0.1 chống overfitting
  - WeightedRandomSampler chống class imbalance (thiên vị lớp NV)
"""
import os
import sys
import torch
import omegaconf

# Fix PyTorch 2.6+ serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

import pytorch_lightning as pl
from hydra import initialize, compose
from hydra.utils import instantiate

from src.training.data_module import SkinLesionDataModule
from src.training.train_pipeline import SkinLesionLightningModule

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

torch.set_float32_matmul_precision('medium')

print("=" * 60)
print("🔧 ANTI-SHORTCUT RETRAINING")
print("  Mục tiêu: Loại bỏ hiệu ứng Clever Hans (Học vẹt)")
print("  Kỹ thuật: Aggressive Crop + CoarseDropout + Color Jitter")
print("=" * 60)

# 1. Nạp cấu hình
print("\n📋 1. Đang nạp cấu hình...")
with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="train", overrides=[
        "model.backbone=efficientnet_b4",
        "dataset.name=isic2019",
        "dataset.base_path=/home/ezycloudx-admin/Trustworthy/ISIC_2019",
        "dataset.use_weighted_sampler=true",
        "trainer.max_epochs=40",
        "trainer.learning_rate=3e-4",
        "trainer.weight_decay=1e-4",
        "trainer.label_smoothing=0.1",
        "trainer.precision=16-mixed",
        "trainer.early_stopping_patience=8",
    ])

# 2. Khởi tạo DataModule
print("📂 2. Đang chuẩn bị dữ liệu với bộ lọc Anti-Shortcut...")
datamodule = SkinLesionDataModule(cfg)

# 3. Khởi tạo Model mới (KHÔNG load checkpoint cũ - train từ đầu)
print("🧠 3. Đang khởi tạo Model EfficientNet-B4 mới...")
model_instance = instantiate(cfg.model)
module = SkinLesionLightningModule(cfg, model=model_instance)

# 4. Callbacks
print("⚙️  4. Đang thiết lập Callbacks...")
checkpoint_dir = "checkpoints_v2"
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="antishortcut-{epoch:02d}-{val_auroc:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
    ),
    EarlyStopping(
        monitor="val/auroc",
        patience=8,
        mode="max",
        verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

# 5. Trainer
print("🚀 5. Bắt đầu Training...")
print("   - Epochs tối đa: 40")
print("   - Early Stopping patience: 8")
print("   - Precision: Mixed 16-bit (FP16)")
print("   - Label Smoothing: 0.1")
print("   - Anti-Shortcut: RandomResizedCrop(0.5-0.85) + CoarseDropout(p=0.7)")
print("-" * 60)

trainer = pl.Trainer(
    max_epochs=40,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,
    callbacks=callbacks,
    deterministic=False,  # Tắt deterministic để tăng tốc
    enable_progress_bar=True,
)

# 6. Bắt đầu huấn luyện
trainer.fit(module, datamodule=datamodule)

# 7. Test trên tập test
print("\n🏆 6. Đang chấm điểm trên tập Test...")
trainer.test(module, datamodule=datamodule, ckpt_path="best")

print("\n" + "=" * 60)
print("✅ HOÀN TẤT TRAINING ANTI-SHORTCUT!")
print(f"   Checkpoint tốt nhất nằm trong: {checkpoint_dir}/")
print("=" * 60)
