# =====================================================================
# HƯỚNG DẪN TRAINING HOÀN CHỈNH
# Trustworthy Skin Cancer AI
# =====================================================================
# Tài liệu này hướng dẫn chi tiết cách:
#   1. Bắt đầu training
#   2. Dừng giữa chừng mà KHÔNG mất dữ liệu
#   3. Tiếp tục training từ chỗ đã dừng
#   4. Đẩy kết quả ngược lại lên GitHub
# =====================================================================


# =====================================================================
# PHẦN 1: CHUẨN BỊ MÔI TRƯỜNG (Chỉ làm 1 lần duy nhất)
# =====================================================================

# Bước 1: Clone code từ GitHub về máy ảo
git clone https://github.com/noenc207/Trustworthy.git
cd Trustworthy

# Bước 2: Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Bước 3: Cài đặt PyTorch + CUDA (cho RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Bước 4: Cài đặt các thư viện còn lại
pip install -r requirements.txt

# Bước 5: Tải dataset (ví dụ ISIC 2019 qua Kaggle)
pip install kaggle
mkdir -p data/isic2019/raw
kaggle datasets download -d andrewmvd/isic-2019 -p data/isic2019/raw/
unzip data/isic2019/raw/isic-2019.zip -d data/isic2019/raw/

# Bước 6: Kiểm tra GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"


# =====================================================================
# PHẦN 2: BẮT ĐẦU TRAINING
# =====================================================================

# --- Cách 1: Chạy với config mặc định ---
python -m src.training.train_pipeline

# --- Cách 2: Chạy với tùy chỉnh (KHUYẾN NGHỊ) ---
python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  dataset.batch_size=64 \
  trainer.max_epochs=100 \
  trainer.precision=16-mixed \
  trainer.learning_rate=1e-3 \
  loss.name=focal \
  seed=42

# --- Cách 3: Chạy trong nền (QUAN TRỌNG cho máy ảo) ---
# Nếu bạn đóng terminal thì training vẫn tiếp tục chạy!
nohup python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  trainer.max_epochs=100 \
  trainer.precision=16-mixed \
  > training.log 2>&1 &

# Xem log training đang chạy:
tail -f training.log

# --- Cách 4: Chạy bằng tmux (TỐT NHẤT cho máy ảo) ---
# tmux giúp bạn thoát SSH mà training vẫn chạy
tmux new -s training
python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  trainer.max_epochs=100
# Nhấn Ctrl+B rồi nhấn D để thoát tmux (training vẫn chạy)
# Quay lại xem training: tmux attach -t training


# =====================================================================
# PHẦN 3: DỪNG GIỮA CHỪNG VÀ TIẾP TỤC
# =====================================================================

# -----------------------------------------------------------
# CƠ CHẾ TỰ ĐỘNG LƯU (Không cần làm gì thêm!)
# -----------------------------------------------------------
# Hệ thống của chúng ta ĐÃ TÍCH HỢP SẴN các cơ chế sau:
#
# 1. ModelCheckpoint: Tự động lưu model tốt nhất và model cuối cùng
#    vào thư mục checkpoints/
#    - checkpoints/last.ckpt         (model cuối cùng)
#    - checkpoints/epoch=XX-val_auroc=0.XXXX.ckpt  (top 3 model tốt nhất)
#
# 2. EarlyStopping: Tự dừng nếu val/auroc không cải thiện sau 10 epoch
#
# 3. Mọi thứ được lưu: model weights, optimizer state, scheduler state,
#    epoch hiện tại, random seed → Tiếp tục training CHÍNH XÁC từ chỗ dừng

# -----------------------------------------------------------
# CÁCH DỪNG AN TOÀN
# -----------------------------------------------------------

# Cách 1: Nhấn Ctrl+C (Dừng nhẹ nhàng)
# PyTorch Lightning sẽ tự động lưu checkpoint trước khi thoát

# Cách 2: Đợi hết epoch hiện tại rồi tự dừng
# (Ctrl+C 1 lần = dừng sau epoch hiện tại)
# (Ctrl+C 2 lần = dừng ngay lập tức, vẫn lưu checkpoint)

# -----------------------------------------------------------
# CÁCH TIẾP TỤC TRAINING TỪ CHỖ ĐÃ DỪNG (RESUME)
# -----------------------------------------------------------

# PyTorch Lightning hỗ trợ resume tự động từ checkpoint:
python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  trainer.max_epochs=100 \
  +ckpt_path=checkpoints/last.ckpt

# Giải thích:
# +ckpt_path=checkpoints/last.ckpt  →  Load lại toàn bộ trạng thái:
#   - Model weights (trọng số)
#   - Optimizer state (momentum, adaptive learning rates)
#   - Scheduler state (learning rate hiện tại)
#   - Epoch đã train tới
#   - Random seed
# → Training tiếp tục CHÍNH XÁC như chưa hề bị dừng!


# =====================================================================
# PHẦN 4: SAU KHI TRAINING XONG — ĐẨY KẾT QUẢ LÊN GITHUB
# =====================================================================

# -----------------------------------------------------------
# Bước 1: Kiểm tra kết quả training
# -----------------------------------------------------------

# Xem kết quả test
cat training.log | grep "test/"

# Xem danh sách checkpoint đã lưu
ls -la checkpoints/

# Xem log TensorBoard (nếu muốn xem biểu đồ trực quan)
tensorboard --logdir logs/tensorboard --port 6006
# Mở trình duyệt: http://localhost:6006

# -----------------------------------------------------------
# Bước 2: Chuẩn bị file kết quả để push
# -----------------------------------------------------------

# Tạo thư mục chứa kết quả
mkdir -p results

# Copy checkpoint tốt nhất (thường < 100MB, push được lên GitHub)
cp checkpoints/last.ckpt results/
# Nếu file > 100MB, dùng Git LFS (xem bên dưới)

# Copy log training
cp training.log results/

# -----------------------------------------------------------
# Bước 3: Cấu hình Git trên máy ảo
# -----------------------------------------------------------

# Thiết lập thông tin Git (chỉ làm 1 lần)
git config --global user.name "noenc207"
git config --global user.email "email_cua_ban@gmail.com"

# -----------------------------------------------------------
# Bước 4: Commit và Push kết quả
# -----------------------------------------------------------

# Thêm tất cả file kết quả
git add results/
git add training.log
git add configs/

# KHÔNG push data/ và checkpoints/ nặng (đã có .gitignore chặn)

# Commit
git commit -m "Training complete: EfficientNet-B4, ISIC2019, 100 epochs"

# Push lên GitHub
git push origin main

# -----------------------------------------------------------
# NẾU CHECKPOINT > 100MB: Dùng Git LFS
# -----------------------------------------------------------

# Cài Git LFS (Large File Storage)
sudo apt install git-lfs
git lfs install

# Track file checkpoint lớn
git lfs track "*.ckpt"
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes

# Sau đó commit và push bình thường
git add results/best_model.ckpt
git commit -m "Add best model checkpoint"
git push origin main

# -----------------------------------------------------------
# CÁCH KHÁC: Upload checkpoint lên Google Drive / HuggingFace
# -----------------------------------------------------------

# Nếu không muốn dùng Git LFS, upload lên Google Drive:
pip install gdown
# Upload thủ công qua giao diện web Google Drive
# Sau đó share link trong README

# Hoặc upload lên HuggingFace Hub (miễn phí, không giới hạn):
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints/last.ckpt',
    path_in_repo='checkpoints/last.ckpt',
    repo_id='noenc207/trustworthy-skin-ai',
    repo_type='model',
)
print('Upload complete!')
"


# =====================================================================
# PHẦN 5: MẸO HAY (TIPS & TRICKS)
# =====================================================================

# -----------------------------------------------------------
# Tip 1: Xem GPU đang làm gì
# -----------------------------------------------------------
watch -n 1 nvidia-smi

# -----------------------------------------------------------
# Tip 2: Training nhiều backbone cùng lúc (nếu có nhiều GPU)
# -----------------------------------------------------------

# GPU 0: EfficientNet
CUDA_VISIBLE_DEVICES=0 python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 trainer.max_epochs=100 &

# GPU 1: ViT (nếu có 2 GPU)
CUDA_VISIBLE_DEVICES=1 python -m src.training.train_pipeline \
  model.backbone=vit_base_patch16_224 trainer.max_epochs=100 &

# -----------------------------------------------------------
# Tip 3: Chạy nhanh test thử trước khi training dài
# -----------------------------------------------------------

# Fast dev run: chạy 1 batch train + 1 batch val để kiểm tra lỗi
python -m src.training.train_pipeline \
  trainer.max_epochs=1 \
  dataset.name=synthetic \
  +trainer.fast_dev_run=true

# -----------------------------------------------------------
# Tip 4: Tự động tắt máy ảo sau khi training xong (tiết kiệm tiền!)
# -----------------------------------------------------------

# Chạy training rồi tự tắt máy:
python -m src.training.train_pipeline trainer.max_epochs=100 && sudo shutdown -h now

# -----------------------------------------------------------
# Tip 5: Theo dõi training từ điện thoại
# -----------------------------------------------------------

# Dùng Weights & Biases (miễn phí):
pip install wandb
wandb login
python -m src.training.train_pipeline use_wandb=true
# Mở app wandb.ai trên điện thoại để theo dõi loss/accuracy real-time!
