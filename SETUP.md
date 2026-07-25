# Hướng dẫn Thiết lập và Tải Dự án (Setup & Download Guide)

Tài liệu này hướng dẫn cách đưa dự án Trustworthy Medical AI Platform từ máy cá nhân lên một máy ảo (VM) có GPU (như RunPod, AWS, Vast.ai) và thiết lập môi trường để bắt đầu huấn luyện.

---

## Bước 1: Clone kho lưu trữ (Repository) từ GitHub
Sau khi bạn đã đẩy code từ máy cá nhân lên GitHub, hãy đăng nhập vào máy ảo và chạy lệnh sau để tải toàn bộ source code về:

```bash
# Clone dự án về máy ảo
git clone https://github.com/TEN_TAI_KHOAN_CUA_BAN/Trustworthy.git

# Di chuyển vào thư mục dự án
cd Trustworthy
```

---

## Bước 2: Thiết lập Môi trường (Environment Setup)

Máy ảo thường chạy hệ điều hành Ubuntu/Linux. Hãy đảm bảo bạn sử dụng một môi trường ảo (Virtual Environment) hoặc Conda để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo (tuỳ chọn nhưng khuyến nghị)
python3 -m venv venv
source venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip

# Cài đặt PyTorch với hỗ trợ CUDA (Phù hợp cho RTX 5090 / GPU đời mới)
# Lưu ý: Cài phiên bản PyTorch tương thích với CUDA 11.8 hoặc 12.1 trở lên
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện lõi của dự án
pip install -r requirements.txt
```
*(Lưu ý: Nếu dự án chưa có file `requirements.txt`, hãy chạy lệnh `pip install numpy scipy scikit-learn matplotlib fastapi uvicorn pydantic loguru prometheus-fastapi-instrumentator captum pytorch-grad-cam opencv-python`)*

---

## Bước 3: Thiết lập xác thực Kaggle (Để tải Dataset)

Vì bộ dữ liệu ISIC 2019 rất lớn, cách tốt nhất là tải qua Kaggle API để đảm bảo tốc độ cao trên máy ảo.

1. Truy cập [Kaggle](https://www.kaggle.com/), vào **Account Settings** -> Chọn **Create New API Token**. File `kaggle.json` sẽ được tải về máy bạn.
2. Mở nội dung file `kaggle.json` (nó chứa `username` và `key`).
3. Trên máy ảo, thiết lập Kaggle:

```bash
# Cài đặt Kaggle CLI
pip install kaggle

# Tạo thư mục ẩn .kaggle
mkdir -p ~/.kaggle

# Tạo file kaggle.json trên máy ảo bằng nano (hoặc vi)
nano ~/.kaggle/kaggle.json
# (Paste nội dung của kaggle.json vào đây, sau đó nhấn Ctrl+O -> Enter -> Ctrl+X để lưu)

# Cấp quyền bảo mật cho file
chmod 600 ~/.kaggle/kaggle.json
```

---

## Bước 4: Tải và Giải nén Dataset

Tạo thư mục chứa dữ liệu và bắt đầu quá trình tải. Do ổ cứng trên VM thường được mount ở `/workspace` hoặc thư mục gốc, hãy linh hoạt điều chỉnh đường dẫn (nhưng code mặc định sẽ đọc từ thư mục `data/` nằm trong thư mục gốc của dự án).

```bash
# Tạo cấu trúc thư mục dữ liệu
mkdir -p data/raw
cd data/raw

# 1. Tải ISIC 2019 qua Kaggle API (~15GB)
kaggle datasets download -d andrewmvd/isic-2019
unzip -q isic-2019.zip -d ISIC_2019
rm isic-2019.zip

# 2. Tải PAD-UFES-20 (Dữ liệu ảnh chụp điện thoại lâm sàng)
# (Sử dụng wget từ link chia sẻ hoặc script download cụ thể)
wget -O pad_ufes_20.zip "LINK_TAI_TRUC_TIEP"
unzip -q pad_ufes_20.zip -d PAD_UFES_20
rm pad_ufes_20.zip

cd ../..
```

---

## Bước 5: Kiểm tra Sức khỏe Hệ thống (Health Check)

Sau khi code và dữ liệu đã sẵn sàng, hãy chạy thử script xác minh tự động của dự án để đảm bảo môi trường CUDA và các thuật toán hoạt động hoàn hảo trước khi cắm máy chạy Training nhiều ngày.

```bash
# Đảm bảo đang ở thư mục gốc của dự án (Trustworthy/)
python verify_batch7.py
```

Nếu console in ra dòng chữ **"Verification SUCCESS: All Batch 7 checks passed."**, chúc mừng bạn! Môi trường máy ảo đã sẵn sàng 100% cho quá trình huấn luyện hệ thống Trustworthy Skin Cancer AI.
