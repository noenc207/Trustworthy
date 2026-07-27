# 📚 DANH SÁCH & LINK TẢI DATASET Y TẾ TỐT NHẤT 📚

Vì bạn đã quyết định thuê máy ảo 24h (Rất sáng suốt!), thời gian tải dữ liệu bằng băng thông siêu tốc của máy ảo sẽ chỉ mất vài phút. Dưới đây là danh sách các bộ dữ liệu Skin Cancer chuẩn Quốc tế và link tải trực tiếp để bạn kéo về máy ảo.

---

## 1. ISIC 2019 (Bộ dữ liệu Chuẩn nhất & Toàn diện nhất)
Đây là bộ dữ liệu khổng lồ (25,331 ảnh) chứa 8 nhãn bệnh lý về da, bao gồm cả các hình ảnh từ HAM10000 và BCN_20000. Đây là lựa chọn số 1 để training.

*   **Dung lượng:** ~15 GB
*   **Trang chủ Kaggle:** [andrewmvd/isic-2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019)
*   **Lệnh tải siêu tốc trên máy ảo (Cần cài `kaggle`):**
    ```bash
    # Đảm bảo đã copy file kaggle.json vào ~/.kaggle/
    mkdir -p data/isic2019/raw
    kaggle datasets download -d andrewmvd/isic-2019 -p data/isic2019/raw/
    
    # Giải nén
    unzip -q data/isic2019/raw/isic-2019.zip -d data/isic2019/raw/
    ```

---

## 2. HAM10000 (Bộ dữ liệu Phổ biến nhất)
Nếu bạn không muốn tải bộ ISIC 2019 quá nặng, bạn có thể tải riêng bộ HAM10000 (10,015 ảnh với 7 nhãn bệnh lý). Rất nhiều bài báo nghiên cứu State-of-the-Art đều dùng bộ này.

*   **Dung lượng:** ~2.7 GB
*   **Trang chủ Dataverse / Kaggle:** [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
*   **Lệnh tải siêu tốc trên máy ảo:**
    ```bash
    mkdir -p data/ham10000/raw
    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000/raw/
    
    # Giải nén
    unzip -q data/ham10000/raw/skin-cancer-mnist-ham10000.zip -d data/ham10000/raw/
    ```

---

## 3. PAD-UFES-20 (Dữ liệu Out-of-Distribution lâm sàng)
Bộ dữ liệu độc đáo gồm 2,298 ảnh được chụp bằng **điện thoại thông minh** (không phải kính hiển vi). Cực kỳ tuyệt vời để đánh giá tính thực tế của Model (Framework của chúng ta hỗ trợ tính năng Out-of-Distribution - OOD trên bộ này).

*   **Dung lượng:** ~1.4 GB
*   **Trang chủ Mendeley Data:** [Link tải trực tiếp](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
*   **Lệnh tải trực tiếp bằng wget (Không cần tài khoản):**
    ```bash
    mkdir -p data/pad_ufes_20/raw
    # Tải file zip trực tiếp từ máy chủ
    wget -O data/pad_ufes_20/raw/pad_ufes_20.zip "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
    
    # Giải nén
    unzip -q data/pad_ufes_20/raw/pad_ufes_20.zip -d data/pad_ufes_20/raw/
    ```

---

## 4. BCN_20000 (Khó & Đa dạng cao)
Bộ dữ liệu từ Bệnh viện Clinic Barcelona. Rất nhiều ảnh kích thước lớn, độ phân giải siêu cao và nhiều ca bệnh khó. Khuyên dùng nếu muốn model cực kỳ mạnh.

*   **Dung lượng:** ~23 GB
*   **Trang chủ ISIC Challenge:** [Link tải ISIC](https://challenge2019.isic-archive.com/data.html)
*   *(Ghi chú: Bộ này đã được gộp chung trong ISIC 2019 ở mục số 1, nên nếu tải mục 1 thì không cần tải mục này).*

---

## 5. Fitzpatrick 17k (Dữ liệu Đỉnh cao về Tính công bằng - Fairness AI)
Đây là bộ dữ liệu "Premium" để đánh giá độ tin cậy của AI (Trustworthy AI). Nó chứa 17,000 ảnh nhưng được phân loại cực kỳ chi tiết theo **Màu da (Skin Tone)** từ sáng nhất đến tối nhất (Fitzpatrick scale). Dùng bộ này sẽ giúp model của bạn không bị "phân biệt chủng tộc" (ví dụ: chỉ nhận diện giỏi trên da trắng mà đoán sai trên da đen). Rất ăn điểm nếu đưa vào báo cáo!

*   **Trang chủ:** [Fitzpatrick 17k GitHub](https://github.com/mattgroh/fitzpatrick17k)
*   **Lệnh tải (Kaggle):**
    ```bash
    mkdir -p data/fitzpatrick17k/raw
    kaggle datasets download -d mattgroh/fitzpatrick17k -p data/fitzpatrick17k/raw/
    unzip -q data/fitzpatrick17k/raw/fitzpatrick17k.zip -d data/fitzpatrick17k/raw/
    ```

---

## 6. Derm7pt (Dữ liệu Giải thích Y khoa Cao cấp - Explainability)
Bộ dữ liệu gồm 1,011 ca bệnh, nhưng mỗi ca không chỉ có 1 ảnh. Nó có cả ảnh thường (clinical), ảnh nội soi (dermoscopy) VÀ **chẩn đoán theo 7 tiêu chí y khoa (7-point checklist)** của bác sĩ. Nếu bạn muốn Model giải thích được "TẠI SAO lại dự đoán là ung thư", bộ này là ĐỈNH NHẤT.

*   **Dung lượng:** ~3 GB
*   **Trang chủ:** [Derm7pt Homepage](http://derm.cs.sfu.ca/)
*   **Lệnh tải trực tiếp:**
    ```bash
    mkdir -p data/derm7pt/raw
    wget -O data/derm7pt/raw/release_v0.zip "http://derm.cs.sfu.ca/downloads/release_v0.zip"
    unzip -q data/derm7pt/raw/release_v0.zip -d data/derm7pt/raw/
    ```

---

## 🚀 ĐỀ XUẤT WORKFLOW DÀNH CHO BẠN TRONG 24H:

1. **Bước 1 (Base Model):** Dành 15 phút tải bộ **ISIC 2019** (Số 1) và train `efficientnet_b4` để lấy kết quả chuẩn.
2. **Bước 2 (Nâng cao Đỉnh chóp):** Tải bộ **Fitzpatrick 17k** (Số 5) để Test xem Model của bạn có công bằng với mọi loại màu da không, hoặc bộ **Derm7pt** (Số 6) để bật tính năng giải thích y khoa (Explainability).
3. **Bước 3:** Bật lệnh train, dùng máy ảo RTX 5090 cày cuốc, bạn đi ngủ.
4. **Bước 4:** Sáng dậy, vào máy ảo tải Checkpoint về và bắt đầu viết báo cáo luận văn điểm A+!
