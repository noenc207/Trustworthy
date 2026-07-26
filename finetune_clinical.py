"""
Phase 2: Cross-Domain Transfer Learning (Fine-Tuning on Clinical Images).

This script:
1. Loads the best checkpoint from Phase 1 (ISIC 2019 - Dermoscopic)
2. Loads the PAD-UFES-20 dataset (Clinical/Smartphone images)
3. Freezes the backbone (optional) or uses a very small learning rate
4. Fine-tunes the model for 15 epochs to adapt to clinical image domain
"""
import os
import sys
import glob
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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

torch.set_float32_matmul_precision('medium')

print("=" * 60)
print("🩺 PHASE 2: CROSS-DOMAIN FINE-TUNING (PAD-UFES-20)")
print("  Mục tiêu: Thích nghi mô hình với ảnh chụp lâm sàng (Smartphone)")
print("=" * 60)

# 1. Tìm checkpoint tốt nhất từ Phase 1
ckpt_files = glob.glob('checkpoints_v2/*.ckpt')
if not ckpt_files:
    print("❌ LỖI: Không tìm thấy Checkpoint từ Giai đoạn 1 (checkpoints_v2/). Vui lòng chạy Phase 1 trước!")
    sys.exit(1)
best_ckpt_path = max(ckpt_files, key=os.path.getmtime)
print(f"📦 Đã tìm thấy Checkpoint nền tảng: {best_ckpt_path}")

# 2. Nạp cấu hình cho PAD-UFES-20
print("\n📋 Đang nạp cấu hình cho bộ dữ liệu Clinical...")
with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="train", overrides=[
        "model.backbone=efficientnet_b4",
        "dataset.name=pad_ufes20",
        "dataset.base_path=/home/ezycloudx-admin/Trustworthy/PAD_UFES_20",
        "dataset.use_weighted_sampler=true",
        "trainer.max_epochs=15",
        "trainer.learning_rate=1e-5", # Học cực chậm để không quên kiến thức ISIC
        "trainer.weight_decay=1e-4",
        "trainer.label_smoothing=0.1",
        "trainer.precision=16-mixed",
        "trainer.early_stopping_patience=5",
    ])

# 3. Khởi tạo DataModule
print("📂 Đang chuẩn bị dữ liệu PAD-UFES-20...")
datamodule = SkinLesionDataModule(cfg)

# 4. Load Model từ Checkpoint Phase 1
print("🧠 Đang load bộ nhớ từ Giai đoạn 1...")
model_instance = instantiate(cfg.model)
module = SkinLesionLightningModule.load_from_checkpoint(
    best_ckpt_path,
    cfg=cfg,
    model=model_instance,
    strict=False
)

# 5. Callbacks
checkpoint_dir = "checkpoints_finetuned"
os.makedirs(checkpoint_dir, exist_ok=True)
callbacks = [
    ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="clinical-{epoch:02d}-{val_auroc:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=2,
        save_last=True,
    ),
    EarlyStopping(
        monitor="val/auroc",
        patience=5,
        mode="max",
        verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

# 6. Trainer
print("🚀 Bắt đầu Fine-Tuning...")
print("   - Epochs tối đa: 15")
print("   - Learning Rate: 1e-5 (Rất nhỏ)")
print("-" * 60)

trainer = pl.Trainer(
    max_epochs=15,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    gradient_clip_val=1.0,
    callbacks=callbacks,
    deterministic=False,
    enable_progress_bar=True,
)

# 7. Huấn luyện Giai đoạn 2
trainer.fit(module, datamodule=datamodule)

# 8. Test trên tập test lâm sàng
print("\n🏆 Đang chấm điểm trên tập Test Lâm sàng...")
trainer.test(module, datamodule=datamodule, ckpt_path="best")

print("\n" + "=" * 60)
print("✅ HOÀN TẤT GIAI ĐOẠN 2: CROSS-DOMAIN TRANSFER LEARNING!")
print(f"   Model cuối cùng sẵn sàng tại: {checkpoint_dir}/")
print("=" * 60)
