"""
Trustworthy Medical AI - Web Demo with Explainability (XAI).

Features:
  - Auto-preprocessing: Removes watermarks/logos by center-cropping
  - Grad-CAM Heatmap: Shows where the AI is looking
  - Explainability Report: Focus Score, Background Leakage, Trustworthiness
  - Clinical Description: Vietnamese medical explanation for each diagnosis
"""
import gradio as gr
import torch
import cv2
import numpy as np

# === Fix PyTorch 2.6+ serialization ===
try:
    import omegaconf
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
except ImportError:
    pass

from src.core.constants import LesionClass, LESION_CLASS_NAMES, LESION_URGENCY
from src.modules.classification.classifier import SkinLesionClassifier
from pytorch_grad_cam.utils.image import show_cam_on_image
from xai_explainer import explain_all

# =============================================
# 1. LOAD MODEL (ENSEMBLE: EfficientNet-B4 + ResNet-50 + DenseNet-121)
# =============================================
print("🔄 Đang tải Hội đồng Y khoa (3 Chuyên gia)...")
import os

model1 = SkinLesionClassifier(backbone='efficientnet_b4', num_classes=7, pretrained=False)
model2 = SkinLesionClassifier(backbone='resnet50', num_classes=7, pretrained=False)
model3 = SkinLesionClassifier(backbone='densenet121', num_classes=7, pretrained=False)

def load_ckpt(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model1 = load_ckpt(model1, "checkpoints/best_model.ckpt")
model2 = load_ckpt(model2, "checkpoints/resnet50.ckpt")
model3 = load_ckpt(model3, "checkpoints/densenet121.ckpt")
print("✅ Hội đồng Y khoa 3 thành viên đã sẵn sàng!")

# === Setup Grad-CAM (Using Model 1 for visualization) ===
try:
    target_layer = model1.backbone.conv_head
except AttributeError:
    target_layer = list(model1.backbone.children())[-2]
cam = GradCAM(model=model1, target_layers=[target_layer])

# =============================================
# 2. CLINICAL DESCRIPTIONS (Vietnamese)
# =============================================
CLINICAL_INFO = {
    "mel": {
        "name": "Ung thư hắc tố (Melanoma)",
        "urgency": "🔴 KHẨN CẤP - Cần gặp bác sĩ da liễu NGAY LẬP TỨC",
        "desc": "Melanoma là dạng ung thư da nguy hiểm nhất. Đặc điểm ABCDE: Asymmetry (Bất đối xứng), Border (Viền không đều), Color (Nhiều màu sắc), Diameter (>6mm), Evolving (Thay đổi theo thời gian)."
    },
    "nv": {
        "name": "Nốt ruồi lành tính (Melanocytic Nevi)",
        "urgency": "🟢 AN TOÀN - Theo dõi định kỳ",
        "desc": "Nốt ruồi thông thường, lành tính. Cần theo dõi nếu có thay đổi về kích thước, hình dạng hoặc màu sắc."
    },
    "bcc": {
        "name": "Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)",
        "urgency": "🟡 CẦN KHÁM - Đặt lịch gặp bác sĩ da liễu trong 2 tuần",
        "desc": "Loại ung thư da phổ biến nhất (chiếm 80%). Thường xuất hiện ở vùng da tiếp xúc nắng. Di căn rất hiếm nhưng có thể phá hủy mô xung quanh nếu không điều trị."
    },
    "akiec": {
        "name": "Dày sừng quang hóa / Bệnh Bowen (Actinic Keratosis)",
        "urgency": "🟡 CẦN KHÁM - Tổn thương tiền ung thư",
        "desc": "Tổn thương tiền ung thư do tích lũy tia UV. Nếu không điều trị có thể tiến triển thành ung thư biểu mô tế bào vảy (SCC)."
    },
    "bkl": {
        "name": "Dày sừng lành tính (Benign Keratosis)",
        "urgency": "🟢 AN TOÀN - Không cần can thiệp y tế",
        "desc": "Bao gồm dày sừng tiết bã (seborrheic keratosis), dày sừng giống lichen và dày sừng mặt trời. Hoàn toàn lành tính."
    },
    "df": {
        "name": "U xơ bì (Dermatofibroma)",
        "urgency": "🟢 AN TOÀN - Theo dõi định kỳ",
        "desc": "Khối u lành tính phổ biến ở chi dưới. Cứng, nhỏ, màu nâu. Không cần điều trị trừ khi gây khó chịu."
    },
    "vasc": {
        "name": "Tổn thương mạch máu (Vascular Lesions)",
        "urgency": "🔵 THEO DÕI - Tham vấn bác sĩ nếu thay đổi",
        "desc": "Bao gồm u mạch máu (hemangioma), giãn mạch, u hạt sinh mủ. Phần lớn lành tính nhưng cần theo dõi."
    },
}


# =============================================
# 3. PREPROCESSING - Anti-Watermark
# =============================================
def preprocess_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Tiền xử lý ảnh: Cắt viền 15% mỗi bên để loại bỏ logo/watermark,
    sau đó resize về kích thước chuẩn."""
    h, w = img.shape[:2]

    # Cắt bỏ 15% viền mỗi bên (logo thường nằm ở góc/rìa)
    crop_pct = 0.15
    top = int(h * crop_pct)
    bottom = int(h * (1 - crop_pct))
    left = int(w * crop_pct)
    right = int(w * (1 - crop_pct))

    img_cropped = img[top:bottom, left:right]
    img_resized = cv2.resize(img_cropped, (target_size, target_size))
    return img_resized


# =============================================
# 4. PREDICTION FUNCTION
# =============================================
def predict(img: np.ndarray):
    if img is None:
        return None, None, "⚠️ Vui lòng tải ảnh lên trước."

    # --- Tiền xử lý: Cắt logo/watermark ---
    img_clean = preprocess_image(img, target_size=224)
    img_float = np.float32(img_clean) / 255.0

    # --- Chuyển thành Tensor ---
    tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    # --- Dự đoán (Hội chẩn 3 Chuyên gia) ---
    with torch.no_grad():
        logits1 = model1(tensor)
        probs1 = torch.softmax(logits1, dim=-1).squeeze().numpy()
        
        logits2 = model2(tensor)
        probs2 = torch.softmax(logits2, dim=-1).squeeze().numpy()
        
        logits3 = model3(tensor)
        probs3 = torch.softmax(logits3, dim=-1).squeeze().numpy()
        
        # Hội chẩn: Lấy trung bình cộng dự đoán của cả 3 chuyên gia
        probs = (probs1 + probs2 + probs3) / 3.0

    classes = [c.value for c in LesionClass]
    result_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

    # --- Grad-CAM Heatmap ---
    heatmap = cam(input_tensor=tensor, targets=None)[0, :]
    heatmap_img = show_cam_on_image(img_float, heatmap, use_rgb=True)
    heatmap_img = np.uint8(255 * heatmap_img)

    # --- Phân tích XAI ---
    top_class = classes[np.argmax(probs)]
    top_prob = float(np.max(probs))
    clinical = CLINICAL_INFO.get(top_class, {})

    focus_score = float(np.mean(heatmap[heatmap > 0.5]) * 100) if np.any(heatmap > 0.5) else 0.0
    bg_leakage = float(np.mean(heatmap[heatmap < 0.15]) * 100)

    # Entropy (đo độ "không chắc chắn" của AI)
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    max_entropy = float(np.log(len(classes)))
    uncertainty_pct = (entropy / max_entropy) * 100

    report = f"## 🏥 KẾT QUẢ CHẨN ĐOÁN\n\n"
    report += f"### {clinical.get('name', top_class)}\n"
    report += f"**Độ tin cậy:** `{top_prob*100:.1f}%`\n\n"
    report += f"**Mức độ khẩn cấp:** {clinical.get('urgency', 'N/A')}\n\n"
    report += f"**Mô tả lâm sàng:** {clinical.get('desc', 'N/A')}\n\n"
    report += f"---\n\n"
    report += f"## 📋 BÁO CÁO GIẢI THÍCH Y KHOA (XAI REPORT)\n\n"
    report += f"| Chỉ số | Giá trị | Đánh giá |\n"
    report += f"|--------|---------|----------|\n"

    focus_eval = "✅ TỐT" if focus_score > 55 else "⚠️ TRUNG BÌNH" if focus_score > 30 else "❌ YẾU"
    report += f"| Độ tập trung (Focus) | `{focus_score:.1f}%` | {focus_eval} |\n"

    bg_eval = "✅ RẤT THẤP" if bg_leakage < 15 else "⚠️ CÓ NHIỄU" if bg_leakage < 30 else "❌ NHIỄU NẶNG"
    report += f"| Nhiễu hậu cảnh (Leakage) | `{bg_leakage:.1f}%` | {bg_eval} |\n"

    unc_eval = "✅ RẤT TỰ TIN" if uncertainty_pct < 30 else "⚠️ CÓ DO DỰ" if uncertainty_pct < 60 else "❌ KHÔNG CHẮC CHẮN"
    report += f"| Độ không chắc chắn (Uncertainty) | `{uncertainty_pct:.1f}%` | {unc_eval} |\n"

    report += f"\n**Đánh giá tổng thể:** "
    if focus_score > 55 and bg_leakage < 15 and uncertainty_pct < 30:
        report += "**KẾT QUẢ ĐÁNG TIN CẬY ✅** — AI tập trung đúng vùng tổn thương."
    elif focus_score > 30 and bg_leakage < 30:
        report += "**CẦN THAM KHẢO BÁC SĨ ⚠️** — AI có dấu hiệu không chắc chắn."
    else:
        report += "**KHÔNG NÊN TIN TƯỞNG ❌** — AI bị nhiễu hoặc không tập trung đúng vùng."

    report += "\n\n> ⚠️ **Lưu ý:** Kết quả AI chỉ mang tính tham khảo, KHÔNG thay thế chẩn đoán của bác sĩ chuyên khoa."

    return result_dict, heatmap_img, report


def predict_xai(img: np.ndarray):
    """Chay tat ca 4 phuong phap XAI, tra ve 4 anh overlay de so sanh."""
    if img is None:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return blank, blank, blank, blank, blank

    img_clean = preprocess_image(img, target_size=224)
    img_float = np.float32(img_clean) / 255.0

    tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    try:
        target_layer = model1.backbone.conv_head
    except AttributeError:
        target_layer = list(model1.backbone.children())[-2]

    print("Bat dau XAI Panel...")
    xai = explain_all(model1, tensor, img_float, target_layer)

    orig = (img_float * 255).astype(np.uint8)
    return (
        orig,
        xai["GradCAM"],
        xai["GradCAM++"],
        xai["LIME"],
        xai["IntGrad"],
        xai["Occlusion"],
    )


# =============================================
# 5. GRADIO WEB INTERFACE
# =============================================
with gr.Blocks(title="Trustworthy Medical AI - Phan loai Ung Thu Da") as demo:
    gr.Markdown("""
    # 🏥 Tro ly Y khoa AI Dang tin cay (Trustworthy Medical AI)
    He thong chan doan ung thu da tu dong voi kha nang **Giai thich Y khoa (Explainable AI)**,
    giup bac si hieu ro **tai sao** AI dua ra quyet dinh.

    > 🔬 **Tinh nang chong hoc vet (Anti-Shortcut):** Anh se duoc tu dong cat vien 15%
    > de loai bo logo benh vien, watermark, va mui ten nhan tao truoc khi phan tich.
    """)

    with gr.Tabs():
        # ── Tab 1: Chẩn đoán chính ──────────────────────────────────
        with gr.Tab("🔍 Chan Doan Chinh"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="📷 Tai anh Not ruoi / Da benh len day")
                    btn = gr.Button("🔍 Bat dau Phan tich", variant="primary", size="lg")
                    gr.Markdown("""
                    **Huong dan su dung:**
                    1. Tai anh chup vung da can phan tich
                    2. Bam nut "Bat dau Phan tich"
                    3. Doc ket qua chan doan va bao cao XAI
                    """)
                with gr.Column(scale=2):
                    out_label   = gr.Label(num_top_classes=4, label="📊 Du doan cua AI")
                    out_heatmap = gr.Image(label="🔥 Anh Grad-CAM Heatmap")
                    out_report  = gr.Markdown(label="📋 Bao cao Giai thich (XAI)")
            btn.click(predict, inputs=input_img,
                      outputs=[out_label, out_heatmap, out_report])

        # ── Tab 2: So sánh XAI Panel ─────────────────────────────────
        with gr.Tab("🔬 So Sanh XAI (4 Phuong Phap)"):
            gr.Markdown("""
            ### Panel So sanh 4 Phuong phap Giai thich AI
            Moi phuong phap nhin anh theo mot cach khac nhau.
            Vung sang = vung AI cho la quan trong nhat de dua ra quyet dinh.

            | Phuong phap | Mo ta |
            |-------------|-------|
            | **Grad-CAM** | Gradient qua lop cuoi cung |
            | **Grad-CAM++** | Nang cap, chinh xac hon khi nhieu vung benh |
            | **LIME** | Che tung cum superpixel, xem thay doi gi |
            | **Integrated Gradients** | Tich luy gradient (Google Research) |
            | **Occlusion** | Trat o vuong den, do do giam xac suat |
            """)
            with gr.Row():
                xai_img   = gr.Image(label="📷 Anh goc (da xu ly)")
                btn_xai   = gr.Button("🔬 Chay XAI Panel", variant="secondary", size="lg")
            with gr.Row():
                out_gc    = gr.Image(label="1️⃣ Grad-CAM")
                out_gcpp  = gr.Image(label="2️⃣ Grad-CAM++")
                out_lime  = gr.Image(label="3️⃣ LIME")
            with gr.Row():
                out_ig    = gr.Image(label="4️⃣ Integrated Gradients")
                out_occ   = gr.Image(label="5️⃣ Occlusion Sensitivity")
                gr.Markdown("""**Ghi chu:** LIME co the mat 20-30 giay.
                Cac phuong phap khac rat nhanh (~2-5 giay).
                """)
            btn_xai.click(
                predict_xai,
                inputs=input_img,
                outputs=[xai_img, out_gc, out_gcpp, out_lime, out_ig, out_occ]
            )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
