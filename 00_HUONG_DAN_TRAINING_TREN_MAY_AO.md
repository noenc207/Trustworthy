# 🚀 HƯỚNG DẪN TRAINING TRÊN MÁY ẢO (TỪ A ĐẾN Z) 🚀

Tài liệu này là "kim chỉ nam" giúp bạn làm chủ toàn bộ quá trình huấn luyện mô hình trên máy ảo (như RunPod, Vast.ai, AWS, v.v.). Bạn có thể yên tâm treo máy, ngắt kết nối, hay tắt máy đột ngột mà **không bao giờ sợ mất dữ liệu**.

---

## 🟢 1. CÁCH BẮT ĐẦU TRAINING ĐÚNG CHUẨN

Khi thuê máy ảo, nếu bạn chạy lệnh bình thường rồi vô tình đóng trình duyệt/mất mạng, quá trình training sẽ bị chết. Vì vậy, **BẮT BUỘC** phải dùng `tmux`.

```bash
# BƯỚC 1: Mở môi trường tmux (giúp giữ session sống mãi mãi)
tmux new -s training

# BƯỚC 2: Chạy lệnh training (Ví dụ với EfficientNet-B4)
python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  trainer.max_epochs=100 \
  trainer.precision=16-mixed

# BƯỚC 3: Thoát màn hình tmux một cách an toàn
# Nhấn tổ hợp phím: Ctrl + B  rồi nhả ra, bấm tiếp phím D
# Lúc này bạn có thể tắt máy tính cá nhân đi ngủ. Training vẫn đang chạy trên máy ảo!

# BƯỚC 4: Sáng hôm sau thức dậy, xem lại tiến độ training
tmux attach -t training
```

---

## 🔴 2. CÁCH DỪNG TRAINING GIỮA CHỪNG (KHÔNG MẤT DỮ LIỆU)

Hệ thống đã được lập trình sẵn cơ chế **Auto-Save (Lưu tự động)** qua PyTorch Lightning.
Sau mỗi Epoch, hệ thống sẽ tự động lưu lại toàn bộ trạng thái vào thư mục `checkpoints/`.

*   `checkpoints/last.ckpt` (Trạng thái mới nhất)
*   `checkpoints/epoch=...val_auroc=...ckpt` (Top 3 Model xịn nhất)

**Cách dừng:**
Bạn chỉ cần nhấn **`Ctrl + C`** ngay tại màn hình đang chạy. Hệ thống sẽ bắt tín hiệu, hoàn tất việc lưu Checkpoint và dừng lại nhẹ nhàng. Dữ liệu của bạn được bảo toàn 100%.

---

## 🔄 3. CÁCH TIẾP TỤC TRAINING (RESUME) TỪ CHỖ ĐÃ DỪNG

Nếu hôm qua bạn train đến Epoch 45 rồi tắt, hôm nay bật lên muốn chạy tiếp từ 45 đến 100, bạn chỉ cần thêm tham số `+ckpt_path` vào lệnh ban đầu:

```bash
python -m src.training.train_pipeline \
  model.backbone=efficientnet_b4 \
  dataset.name=isic2019 \
  dataset.base_path=data/isic2019 \
  trainer.max_epochs=100 \
  +ckpt_path=checkpoints/last.ckpt
```

Hệ thống sẽ "nhớ" toàn bộ: Trọng số (weights), Optimizer (AdamW), Learning Rate hiện tại (Scheduler), và con số Epoch 45. Nó sẽ chạy tiếp Epoch 46 như chưa hề có cuộc chia ly!

---

## ⬆️ 4. SAU KHI TRAINING XONG: ĐẨY KẾT QUẢ VỀ GITHUB

Khi hoàn tất, bạn không cần push cục Data khổng lồ về. Bạn chỉ cần push File kết quả (Model) và Log.

```bash
# 1. Cấu hình danh tính Git (Nếu máy ảo mới thuê)
git config --global user.name "Tên_Của_Bạn"
git config --global user.email "Email_Của_Bạn@gmail.com"

# 2. Tạo một thư mục để gom gọn kết quả
mkdir -p results
cp checkpoints/last.ckpt results/
cp -r logs/tensorboard results/

# 3. Add và Commit lên GitHub
git add results/
git commit -m "🚀 Hoàn thành training EfficientNet-B4 ISIC2019 (100 Epochs)"
git push origin main
```

### ⚠️ LƯU Ý QUAN TRỌNG VỀ DUNG LƯỢNG (Nếu file .ckpt lớn hơn 100MB)
GitHub **cấm** up file > 100MB. Nếu file `last.ckpt` của bạn nặng hơn 100MB, hãy dùng HuggingFace (miễn phí, không giới hạn dung lượng) để lưu Model thay vì GitHub:

```bash
# Cài đặt thư viện
pip install huggingface_hub

# Đăng nhập HuggingFace (Lấy token từ trang web huggingface.co)
huggingface-cli login

# Chạy lệnh Python nhỏ này để đẩy Model lên HuggingFace
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='checkpoints/last.ckpt', 
    path_in_repo='checkpoints/best_model.ckpt', 
    repo_id='TÊN_TÀI_KHOẢN_HUGGINGFACE_CỦA_BẠN/trustworthy-skin-ai', 
    repo_type='model'
)
print('🚀 Đã đẩy Model lên HuggingFace thành công!')
"
```

---

## 💡 5. "TUYỆT CHIÊU" TIẾT KIỆM TIỀN THUÊ MÁY ẢO

Nếu bạn train qua đêm, khi train xong mà máy ảo vẫn bật thì bạn sẽ vẫn bị trừ tiền theo giờ. Hãy dùng câu lệnh "Train xong tự tắt máy":

```bash
# Máy ảo sẽ tự động Shut Down ngay khi Epoch 100 kết thúc
python -m src.training.train_pipeline trainer.max_epochs=100 && sudo shutdown -h now
```
