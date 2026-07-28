"""
XAI Explainer Module — Trustworthy Medical AI
==============================================
Cung cap 5 phuong phap giai thich AI:
  1. Grad-CAM    (da co, giu lai)
  2. Grad-CAM++  (nang cap Grad-CAM)
  3. LIME        (superpixel, model-agnostic)
  4. Integrated Gradients (Google Research)
  5. Occlusion Sensitivity Map

Yeu cau them:
  pip install grad-cam lime captum
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ==============================================================
# 1. GRAD-CAM (cu) va GRAD-CAM++ (moi)
# ==============================================================
def run_gradcam(model, tensor, target_layer, use_plusplus=False):
    """
    Chay Grad-CAM hoac Grad-CAM++ va tra ve heatmap (H, W) float [0,1].
    use_plusplus=True -> Grad-CAM++ (chinh xac hon khi co nhieu vung benh)
    """
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    with CAMClass(model=model, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=tensor, targets=None)[0, :]
    return heatmap  # shape (H, W), float32 [0,1]


# ==============================================================
# 2. LIME
# ==============================================================
def run_lime(model, img_float_rgb, device="cpu", num_samples=500):
    """
    Chay LIME: chia anh thanh superpixel, che tung cum xem thay doi gi.
    img_float_rgb: np.ndarray float32 [0,1] shape (H, W, 3)
    Tra ve mask (H, W) float [0,1] — vung sang = quan trong.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        raise ImportError("Cai them LIME: pip install lime scikit-image")

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def predict_fn(images):
        """LIME goi ham nay voi mang anh uint8 (N, H, W, 3)."""
        batch = []
        for img in images:
            x = img.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1)).float()
            batch.append(x)
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    img_uint8 = (img_float_rgb * 255).astype(np.uint8)

    explanation = explainer.explain_instance(
        img_uint8,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    top_label = explanation.top_labels[0]
    # Lay mask cua class co xac suat cao nhat, chi giu phan duong (positive)
    _, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )
    # Chuyen mask nhi phan (0/1) -> float [0,1]
    heatmap = mask.astype(np.float32)
    # Lam mo nhe de dep hon
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap  # (H, W) float [0,1]


# ==============================================================
# 3. INTEGRATED GRADIENTS (Google Research / Captum)
# ==============================================================
def run_integrated_gradients(model, tensor, n_steps=50):
    """
    Tich luy gradient tu baseline (anh den) den anh that.
    Tra ve attribution map (H, W) float [0,1].
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("Cai them Captum: pip install captum")

    model.eval()
    ig = IntegratedGradients(model)

    # Lay class co xac suat cao nhat
    with torch.no_grad():
        logits = model(tensor)
        target_class = int(logits.argmax(dim=-1).item())

    tensor.requires_grad_(True)
    attrs = ig.attribute(
        tensor,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=1,
    )  # shape (1, C, H, W)

    # Tong hop theo kenh mau, lay tri tuyet doi
    attr_map = attrs.squeeze(0).detach().cpu().numpy()   # (C, H, W)
    attr_map = np.abs(attr_map).sum(axis=0)              # (H, W)

    if attr_map.max() > 0:
        attr_map = attr_map / attr_map.max()
    return attr_map.astype(np.float32)  # (H, W) [0,1]


# ==============================================================
# 4. OCCLUSION SENSITIVITY MAP
# ==============================================================
def run_occlusion(model, tensor, patch_size=32, stride=16):
    """
    Trat mot o vuong den len anh, xem xac suat thay doi bao nhieu.
    Vung bi giam nhieu nhat = vung quan trong nhat.
    Tra ve heatmap (H, W) float [0,1].
    """
    model.eval()
    _, _, H, W = tensor.shape

    with torch.no_grad():
        base_prob = torch.softmax(model(tensor), dim=-1)
        target_class = int(base_prob.argmax(dim=-1).item())
        base_score = base_prob[0, target_class].item()

    sensitivity = np.zeros((H, W), dtype=np.float32)
    counts      = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0  # o den

            with torch.no_grad():
                prob = torch.softmax(model(occluded), dim=-1)
                score = prob[0, target_class].item()

            drop = base_score - score  # drop lon = vung quan trong
            sensitivity[y:y+patch_size, x:x+patch_size] += drop
            counts[y:y+patch_size, x:x+patch_size]      += 1

    counts[counts == 0] = 1
    heatmap = sensitivity / counts
    heatmap = np.clip(heatmap, 0, None)  # chi lay drop duong
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.astype(np.float32)  # (H, W) [0,1]


# ==============================================================
# 5. TIEN ICH: Chuyen heatmap thanh anh overlay dep
# ==============================================================
def overlay_heatmap(img_float_rgb, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Chong heatmap (H,W) len anh goc (H,W,3) float [0,1].
    Tra ve anh RGB uint8.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image
    result = show_cam_on_image(img_float_rgb, heatmap, use_rgb=True, colormap=colormap)
    return np.uint8(result)


# ==============================================================
# 6. CHAY TAT CA 4 PHUONG PHAP — Ham tien ich chinh
# ==============================================================
def explain_all(model, tensor, img_float_rgb, target_layer, device="cpu"):
    """
    Chay ca 4 phuong phap XAI va tra ve dict chua 4 anh overlay.

    Returns:
        {
          "GradCAM":   np.ndarray uint8 (H, W, 3),
          "GradCAM++": np.ndarray uint8 (H, W, 3),
          "LIME":      np.ndarray uint8 (H, W, 3),
          "IntGrad":   np.ndarray uint8 (H, W, 3),
          "Occlusion": np.ndarray uint8 (H, W, 3),
        }
    """
    results = {}

    print("  [1/5] Grad-CAM...")
    h = run_gradcam(model, tensor, target_layer, use_plusplus=False)
    results["GradCAM"] = overlay_heatmap(img_float_rgb, h)

    print("  [2/5] Grad-CAM++...")
    h = run_gradcam(model, tensor, target_layer, use_plusplus=True)
    results["GradCAM++"] = overlay_heatmap(img_float_rgb, h)

    print("  [3/5] LIME (chay lau ~10-20s)...")
    try:
        h = run_lime(model, img_float_rgb, device=device, num_samples=300)
        results["LIME"] = overlay_heatmap(img_float_rgb, h, colormap=cv2.COLORMAP_COOL)
    except Exception as e:
        print(f"    LIME bi loi: {e}")
        results["LIME"] = (img_float_rgb * 255).astype(np.uint8)

    print("  [4/5] Integrated Gradients...")
    try:
        h = run_integrated_gradients(model, tensor.clone())
        h_resized = cv2.resize(h, (img_float_rgb.shape[1], img_float_rgb.shape[0]))
        results["IntGrad"] = overlay_heatmap(img_float_rgb, h_resized, colormap=cv2.COLORMAP_MAGENTA)
    except Exception as e:
        print(f"    IntGrad bi loi: {e}")
        results["IntGrad"] = (img_float_rgb * 255).astype(np.uint8)

    print("  [5/5] Occlusion Sensitivity...")
    h = run_occlusion(model, tensor, patch_size=28, stride=14)
    h_resized = cv2.resize(h, (img_float_rgb.shape[1], img_float_rgb.shape[0]))
    results["Occlusion"] = overlay_heatmap(img_float_rgb, h_resized, colormap=cv2.COLORMAP_HOT)

    print("  XAI hoan thanh!")
    return results
