#!/bin/bash
# =============================================================
# TRUSTWORTHY AI - SETUP & TRAIN (Chạy trên máy ảo mới thuê)
# Chỉ cần 1 lệnh: bash setup_and_train.sh
# =============================================================

set -e  # Dừng ngay nếu có lỗi

echo "========================================================"
echo "  TRUSTWORTHY MEDICAL AI - AUTO SETUP & TRAIN"
echo "========================================================"

# =============================================
# BƯỚC 1: Cài đặt môi trường
# =============================================
echo ""
echo "📦 BƯỚC 1: Cài đặt các thư viện cần thiết..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q pytorch-lightning hydra-core omegaconf timm \
    albumentations opencv-python-headless \
    torchmetrics loguru pandas scikit-learn \
    pytorch-grad-cam mlflow

echo "✅ Cài đặt xong!"

# =============================================
# BƯỚC 2: Tải dữ liệu ISIC 2019
# =============================================
echo ""
echo "📂 BƯỚC 2: Chuẩn bị dữ liệu ISIC 2019..."

# Kiểm tra xem dữ liệu đã có chưa
if [ -d "ISIC_2019/raw" ] && [ "$(ls -A ISIC_2019/raw 2>/dev/null)" ]; then
    echo "✅ Dữ liệu ISIC 2019 đã có sẵn, bỏ qua bước tải!"
else
    echo "   Đang tải dữ liệu từ ISIC Archive..."
    mkdir -p ISIC_2019/raw
    
    # Tải ảnh ISIC 2019 (phần 1)
    wget -q --show-progress -O ISIC_2019/raw/ISIC_2019_Training_Input.zip \
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
    
    # Tải nhãn
    wget -q --show-progress -O ISIC_2019/raw/ISIC_2019_Training_GroundTruth.csv \
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
    
    wget -q --show-progress -O ISIC_2019/raw/ISIC_2019_Training_Metadata.csv \
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
    
    echo "   Đang giải nén..."
    unzip -q ISIC_2019/raw/ISIC_2019_Training_Input.zip -d ISIC_2019/raw/
    echo "✅ Tải và giải nén xong!"
fi

# =============================================
# BƯỚC 3: Bắt đầu Training Anti-Shortcut
# =============================================
echo ""
echo "🚀 BƯỚC 3: Bắt đầu Training với Anti-Shortcut Learning..."
echo "   Backbone: EfficientNet-B4"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "   Epochs: 40 (Early Stopping patience=8)"
echo "   Kỹ thuật: Aggressive Crop + CoarseDropout + Label Smoothing"
echo "--------------------------------------------------------"

python3 retrain_anti_shortcut.py

# =============================================
# BƯỚC 4: Upload checkpoint lên mây
# =============================================
echo ""
echo "☁️  BƯỚC 4: Đang upload checkpoint tốt nhất lên mây để tải về..."

BEST_CKPT=$(ls -t checkpoints_v2/*.ckpt 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t checkpoints/*.ckpt 2>/dev/null | head -1)
fi

if [ -n "$BEST_CKPT" ]; then
    echo "   File checkpoint: $BEST_CKPT"
    UPLOAD_URL=$(curl -s -F "reqtype=fileupload" \
        -F "fileToUpload=@$BEST_CKPT" \
        -F "time=72h" \
        "https://litterbox.catbox.moe/resources/internals/api.php")
    echo ""
    echo "========================================================"
    echo "  ✅ TRAINING HOÀN TẤT!"
    echo ""
    echo "  📥 LINK TẢI CHECKPOINT VỀ MÁY TÍNH:"
    echo "  $UPLOAD_URL"
    echo ""
    echo "  👆 Copy link trên, dán vào Chrome để tải về!"
    echo "  Sau đó bỏ vào D:\Trustworthy\checkpoints_v2\"
    echo "========================================================"
else
    echo "⚠️  Không tìm thấy checkpoint. Vui lòng kiểm tra thư mục checkpoints_v2/"
fi
