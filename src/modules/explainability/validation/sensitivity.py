import json
import os
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np


class MetricSensitivityEngine:
    """
    Evaluates explainability metrics under various input perturbations
    and calculates robustness statistics.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def perturb_image(self, image: np.ndarray, pert_type: str) -> np.ndarray:
        """Apply perturbation. Assumes image is uint8 [0, 255] or float [0, 1]."""
        if pert_type == "Brightness":
            return np.clip(image * 1.2, 0, 255 if image.dtype == np.uint8 else 1).astype(image.dtype)
        elif pert_type == "Contrast":
            mean = np.mean(image, axis=(0,1), keepdims=True)
            return np.clip((image - mean) * 1.2 + mean, 0, 255 if image.dtype == np.uint8 else 1).astype(image.dtype)
        elif pert_type == "Noise":
            noise = np.random.normal(0, 0.05 * (255 if image.dtype == np.uint8 else 1), image.shape)
            return np.clip(image + noise, 0, 255 if image.dtype == np.uint8 else 1).astype(image.dtype)
        elif pert_type == "Gaussian Blur":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif pert_type == "JPEG":
            img_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return decimg if image.dtype == np.uint8 else (decimg / 255.0).astype(image.dtype)
        elif pert_type == "Rotation":
            h, w = image.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
            return cv2.warpAffine(image, rot_mat, (w, h))
        elif pert_type == "Crop":
            h, w = image.shape[:2]
            crop_img = image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            return cv2.resize(crop_img, (w, h))
        elif pert_type == "Hair":
            res = image.copy()
            cv2.line(res, (0,0), (res.shape[1], res.shape[0]), (0,0,0) if image.dtype == np.uint8 else 0.0, 2)
            return res
        elif pert_type == "Bubble":
            res = image.copy()
            cv2.circle(res, (res.shape[1]//2, res.shape[0]//2), 20, (255,255,255) if image.dtype == np.uint8 else 1.0, 2)
            return res
        elif pert_type == "Flash":
            res = image.copy()
            cv2.circle(res, (res.shape[1]//2, res.shape[0]//2), 50, (255,255,255) if image.dtype == np.uint8 else 1.0, -1)
            return cv2.addWeighted(image, 0.7, res, 0.3, 0)
        return image

    def evaluate_perturbation(self, metric_func: Callable, original_images: np.ndarray, **kwargs) -> dict[str, Any]:
        baseline_scores = np.array(metric_func(original_images, **kwargs))
        perturbations = ["Brightness", "Contrast", "Noise", "Gaussian Blur", "JPEG", "Rotation", "Crop", "Hair", "Bubble", "Flash"]

        results = {}
        pert_sensitivities = {}

        for p in perturbations:
            p_images = np.array([self.perturb_image(img, p) for img in original_images])
            pert_scores = np.array(metric_func(p_images, **kwargs))

            diffs = np.abs(baseline_scores - pert_scores)
            sensitivity = float(np.mean(diffs))
            variance = float(np.var(diffs))
            cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else 0.0
            ci_95 = 1.96 * float(np.std(diffs) / np.sqrt(len(diffs))) if len(diffs) > 0 else 0.0

            pert_sensitivities[p] = sensitivity

            results[p] = {
                "Sensitivity Index": sensitivity,
                "Variance": variance,
                "CV": cv,
                "95% CI": ci_95
            }

        if pert_sensitivities:
            worst = max(pert_sensitivities, key=pert_sensitivities.get)
            best = min(pert_sensitivities, key=pert_sensitivities.get)
            ranking = sorted(pert_sensitivities, key=pert_sensitivities.get)
        else:
            worst, best, ranking = None, None, []

        summary = {
            "Perturbations": results,
            "Worst Perturbation": worst,
            "Best Perturbation": best,
            "Robustness Ranking": ranking
        }
        return summary

    def generate_report(self, summary: dict[str, Any], filename: str = "sensitivity_report.json"):
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        return filepath
