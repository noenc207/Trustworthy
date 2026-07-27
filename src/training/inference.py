"""
Inference engine for model prediction and test-time evaluation.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


class InferenceEngine:
    def __init__(self, model: nn.Module, device: str = 'auto', precision: str = 'fp32'):
        self.model = model

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.precision = precision
        if precision == 'fp16':
            self.model.half()

    def _prepare_tensor(self, image: np.ndarray, transforms=None) -> torch.Tensor:
        if transforms:
            # Assuming transforms expect PIL Image or handle numpy
            if isinstance(image, np.ndarray) and transforms.__class__.__name__ != 'A':
                # Quick hack check if using torchvision (requires PIL or tensor)
                # But we'll just pass to transforms and see
                try:
                    tensor = transforms(image)
                except:
                    tensor = transforms(Image.fromarray(image))
            else:
                tensor = transforms(image)
                if isinstance(tensor, dict) and 'image' in tensor: # albumentations
                    tensor = tensor['image']
        else:
            # Default fallback conversion
            if image.ndim == 3:
                tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:
                tensor = torch.from_numpy(image).float()

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if self.precision == 'fp16':
            tensor = tensor.half()

        return tensor.unsqueeze(0).to(self.device) # Add batch dim

    def predict_single(self, image: np.ndarray, transforms=None) -> dict:
        tensor = self._prepare_tensor(image, transforms)

        with torch.no_grad():
            outputs = self.model(tensor)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            conf, pred = torch.max(probs, dim=0)

        return {
            'class_id': pred.item(),
            'confidence': conf.item(),
            'probabilities': probs.cpu().numpy()
        }

    def predict_batch(self, images: list[np.ndarray], transforms=None, batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            tensors = [self._prepare_tensor(img, transforms).squeeze(0) for img in batch_images]
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)

            for j in range(len(batch_images)):
                results.append({
                    'class_id': preds[j].item(),
                    'confidence': confs[j].item(),
                    'probabilities': probs[j].cpu().numpy()
                })
        return results

    def predict_folder(self, folder_path: Path, transforms=None, batch_size: int = 32) -> pd.DataFrame:
        from PIL import Image

        valid_exts = {'.jpg', '.jpeg', '.png'}
        image_paths = [p for p in Path(folder_path).glob('*') if p.suffix.lower() in valid_exts]

        results = []
        images_batch = []
        paths_batch = []

        for path in image_paths:
            img = np.array(Image.open(path).convert('RGB'))
            images_batch.append(img)
            paths_batch.append(path.name)

            if len(images_batch) >= batch_size:
                batch_preds = self.predict_batch(images_batch, transforms, batch_size)
                for p, pred in zip(paths_batch, batch_preds):
                    pred['filename'] = p
                    results.append(pred)
                images_batch = []
                paths_batch = []

        if images_batch:
            batch_preds = self.predict_batch(images_batch, transforms, batch_size)
            for p, pred in zip(paths_batch, batch_preds):
                pred['filename'] = p
                results.append(pred)

        return pd.DataFrame(results)

    def predict_with_tta(self, image: np.ndarray, tta_transforms: list[Any]) -> dict:
        probs_list = []
        for transform in tta_transforms:
            pred = self.predict_single(image, transform)
            probs_list.append(pred['probabilities'])

        avg_probs = np.mean(probs_list, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])

        return {
            'class_id': pred_idx,
            'confidence': confidence,
            'probabilities': avg_probs
        }

    def predict_with_calibration(self, image: np.ndarray, calibrator=None) -> dict:
        pred = self.predict_single(image)
        if calibrator is not None:
            # Assume calibrator takes numpy array of probs and returns calibrated probs
            calibrated_probs = calibrator.calibrate(np.expand_dims(pred['probabilities'], 0))[0]
            pred_idx = int(np.argmax(calibrated_probs))
            confidence = float(calibrated_probs[pred_idx])

            return {
                'class_id': pred_idx,
                'confidence': confidence,
                'probabilities': calibrated_probs
            }
        return pred
