from __future__ import annotations
import cv2
import numpy as np
from .action_space import ObservationAction

class ViewGenerator:
    """Generates specific views or transformations for a given image based on observation actions."""
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def generate(self, image: np.ndarray, action: ObservationAction) -> np.ndarray:
        """Generate a view from image for the given action. Returns (H, W, 3) uint8."""
        h, w = image.shape[:2]
        
        if action == ObservationAction.STOP or action == ObservationAction.ABSTAIN:
            raise ValueError("Cannot generate view for terminal action")
            
        elif action == ObservationAction.KEEP_FULL:
            return cv2.resize(image, (self.image_size, self.image_size))
            
        elif action == ObservationAction.ZOOM_CENTER:
            ch, cw = h // 2, w // 2
            h_crop, w_crop = int(h * 0.6), int(w * 0.6)
            y1, y2 = max(0, ch - h_crop // 2), min(h, ch + h_crop // 2)
            x1, x2 = max(0, cw - w_crop // 2), min(w, cw + w_crop // 2)
            cropped = image[y1:y2, x1:x2]
            return cv2.resize(cropped, (self.image_size, self.image_size))
            
        elif action == ObservationAction.ZOOM_BORDER:
            mask = np.ones((h, w), dtype=np.uint8)
            ch, cw = h // 2, w // 2
            h_mask, w_mask = int(h * 0.4), int(w * 0.4)
            y1, y2 = max(0, ch - h_mask // 2), min(h, ch + h_mask // 2)
            x1, x2 = max(0, cw - w_mask // 2), min(w, cw + w_mask // 2)
            mask[y1:y2, x1:x2] = 0
            masked = cv2.bitwise_and(image, image, mask=mask)
            return cv2.resize(masked, (self.image_size, self.image_size))
            
        elif action == ObservationAction.LEFT_REGION:
            left_half = image[:, :w // 2]
            return cv2.resize(left_half, (self.image_size, self.image_size))
            
        elif action == ObservationAction.RIGHT_REGION:
            right_half = image[:, w // 2:]
            return cv2.resize(right_half, (self.image_size, self.image_size))
            
        elif action == ObservationAction.TOP_REGION:
            top_half = image[:h // 2, :]
            return cv2.resize(top_half, (self.image_size, self.image_size))
            
        elif action == ObservationAction.BOTTOM_REGION:
            bottom_half = image[h // 2:, :]
            return cv2.resize(bottom_half, (self.image_size, self.image_size))
            
        elif action == ObservationAction.PIGMENT_REGION:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_suppressed = cv2.addWeighted(l, 0.5, np.zeros_like(l), 0.5, 0)
            lab_suppressed = cv2.merge((l_suppressed, a, b))
            pigment_enhanced = cv2.cvtColor(lab_suppressed, cv2.COLOR_LAB2RGB)
            return cv2.resize(pigment_enhanced, (self.image_size, self.image_size))
            
        elif action == ObservationAction.TEXTURE_REGION:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            responses = []
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                responses.append(filtered)
            magnitude = np.max(np.stack(responses, axis=0), axis=0)
            magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            texture_rgb = cv2.cvtColor(magnitude_norm, cv2.COLOR_GRAY2RGB)
            return cv2.resize(texture_rgb, (self.image_size, self.image_size))
            
        elif action == ObservationAction.COLOR_NORMALIZED:
            img_float = image.astype(np.float32)
            p = 6
            illuminant = np.power(np.mean(np.power(img_float, p), axis=(0, 1)), 1.0 / p)
            illuminant = illuminant / np.sqrt(np.sum(illuminant ** 2) + 1e-8)
            normalized = np.clip(img_float / (illuminant + 1e-8) * (1.0 / np.sqrt(3)), 0, 255).astype(np.uint8)
            return cv2.resize(normalized, (self.image_size, self.image_size))
            
        elif action == ObservationAction.ARTIFACT_SUPPRESSED:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            inpainted = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)
            return cv2.resize(inpainted, (self.image_size, self.image_size))
            
        return cv2.resize(image, (self.image_size, self.image_size))
