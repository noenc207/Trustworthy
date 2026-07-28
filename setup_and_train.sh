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
sudo apt-get update -yqq
sudo apt-get install -yqq python3-pip unzip

# Dùng pip3 thay vì pip để chắc chắn ăn vào Python 3
pip3 install --break-system-packages -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install --break-system-packages -q pytorch-lightning hydra-core omegaconf timm \
    albumentations opencv-python-headless \
    torchmetrics loguru pandas scikit-learn \
    matplotlib seaborn "grad-cam" pydantic-settings mlflow

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
# BƯỚC 4: Tải dữ liệu Lâm sàng (PAD-UFES-20)
# =============================================
echo ""
echo "📂 BƯỚC 4: Chuẩn bị dữ liệu PAD-UFES-20 (Ảnh lâm sàng)..."

if [ -d "PAD_UFES_20/raw" ] && [ "$(ls -A PAD_UFES_20/raw 2>/dev/null)" ]; then
    echo "✅ Dữ liệu PAD-UFES-20 đã có sẵn!"
else
    echo "   Đang tải dữ liệu từ Mendeley Data (AWS)..."
    mkdir -p PAD_UFES_20/raw
    wget -q --show-progress -O PAD_UFES_20/raw/pad_ufes_20.zip \
        "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
    
    echo "   Đang giải nén..."
    unzip -q -j PAD_UFES_20/raw/pad_ufes_20.zip -d PAD_UFES_20/raw/
    echo "✅ Tải và giải nén xong!"
fi

# =============================================
# BƯỚC 5: Giai đoạn 2 - Fine-Tuning
# =============================================
echo ""
echo "🚀 BƯỚC 5: Bắt đầu GIAI ĐOẠN 2 (Fine-Tuning trên ảnh lâm sàng)..."
echo "   Sử dụng Checkpoint tốt nhất từ GĐ 1 để dạy bổ túc 15 Epochs."
echo "--------------------------------------------------------"

python3 finetune_clinical.py

# =============================================
# BƯỚC 6: Upload checkpoint cực phẩm lên mây
# =============================================
echo ""
echo "☁️  BƯỚC 6: Đang upload checkpoint TỐT NHẤT (Đã Fine-tune) lên mây..."

BEST_CKPT=$(ls -t checkpoints_finetuned/*.ckpt 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t checkpoints_v2/*.ckpt 2>/dev/null | head -1)
fi

if [ -n "$BEST_CKPT" ]; then
    echo "   File checkpoint: $BEST_CKPT"
    UPLOAD_URL=$(curl -s -F "reqtype=fileupload" \
        -F "fileToUpload=@$BEST_CKPT" \
        -F "time=72h" \
        "https://litterbox.catbox.moe/resources/internals/api.php")
    echo ""
    echo "========================================================"
    echo "  ✅ CHÚC MỪNG! TRAINING CẢ 2 GIAI ĐOẠN ĐÃ HOÀN TẤT!"
    echo ""
    echo "  📥 LINK TẢI BỘ NÃO 'TRÙM CUỐI' VỀ MÁY TÍNH:"
    echo "  $UPLOAD_URL"
    echo ""
    echo "  👆 Copy link trên, dán vào Chrome để tải về!"
    echo "  Sau đó bỏ vào D:\Trustworthy\checkpoints_finetuned\"
    echo "========================================================"
else
    echo "⚠️  Không tìm thấy checkpoint!"
fi
