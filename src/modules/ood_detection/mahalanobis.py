from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf

from .base import BaseOODDetector

logger = logging.getLogger(__name__)


class MahalanobisDetector(BaseOODDetector):
    """
    Mahalanobis Distance OOD Detection.
    Measures the distance from test sample features to the closest class-conditional Gaussian.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_layer_name: str | None = None,
        threshold: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.feature_layer_name = feature_layer_name
        self.threshold = threshold

        self.class_means_t: torch.Tensor | None = None
        self.precision_t: torch.Tensor | None = None
        self.features_out: torch.Tensor | None = None
        self._hook_handle = None

        self._attach_hook()

    def _attach_hook(self) -> None:
        def hook_fn(module: nn.Module, input_data: Any, output: torch.Tensor) -> None:
            self.features_out = output

        layer_found = False

        if self.feature_layer_name:
            for name, module in self.model.named_modules():
                if name == self.feature_layer_name:
                    self._hook_handle = module.register_forward_hook(hook_fn)
                    layer_found = True
                    break

        # Fallback to the layer right before the final Linear layer
        if not layer_found:
            modules = list(self.model.modules())
            for module in reversed(modules):
                if isinstance(module, nn.Linear):
                    continue
                # We found the layer before linear
                self._hook_handle = module.register_forward_hook(hook_fn)
                layer_found = True
                break

        if not layer_found:
            logger.warning("Could not automatically find suitable feature layer for Mahalanobis.")

    def fit(self, dataloader: Any, device: str = "cpu") -> None:
        """
        Fit class means and precision matrix using training data.
        Needs to be run on In-Distribution data.
        """
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                self.model(x)
                if self.features_out is None:
                    continue
                # Flatten spatial dimensions if they exist
                f = self.features_out.view(x.size(0), -1).cpu().numpy()
                features_list.append(f)
                labels_list.append(y.cpu().numpy())

        if not features_list:
            raise RuntimeError("Failed to extract features during fit().")

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        num_classes = len(np.unique(labels))
        class_means = []
        centered_features = []

        for c in range(num_classes):
            class_f = features[labels == c]
            if len(class_f) == 0:
                class_means.append(np.zeros(features.shape[1]))
                continue
            mean = np.mean(class_f, axis=0)
            class_means.append(mean)
            centered_features.append(class_f - mean)

        class_means = np.array(class_means)
        centered_features = np.concatenate(centered_features, axis=0)

        # Use Ledoit-Wolf shrinkage to handle ill-conditioned covariance matrix
        cov_estimator = LedoitWolf()
        cov_estimator.fit(centered_features)
        precision_matrix = cov_estimator.precision_

        self.class_means_t = torch.tensor(class_means, dtype=torch.float32).to(device)
        self.precision_t = torch.tensor(precision_matrix, dtype=torch.float32).to(device)

    def save_statistics(self, filepath: str) -> None:
        """Serialize and cache fitted statistics to disk for fast inference."""
        if self.class_means_t is None or self.precision_t is None:
            raise RuntimeError("Cannot save statistics before fitting.")
        stats = {
            "class_means": self.class_means_t.cpu(),
            "precision": self.precision_t.cpu(),
            "feature_layer_name": self.feature_layer_name,
        }
        torch.save(stats, filepath)

    def load_statistics(self, filepath: str, device: str = "cpu") -> None:
        """Load cached statistics from disk."""
        stats = torch.load(filepath, map_location=device, weights_only=True)
        self.class_means_t = stats["class_means"].to(device)
        self.precision_t = stats["precision"].to(device)
        if stats.get("feature_layer_name") != self.feature_layer_name:
            logger.warning("Loaded statistics feature layer mismatches current detector configuration.")

    def get_method_name(self) -> str:
        return "Mahalanobis"

    def get_threshold(self) -> float:
        return self.threshold

    def compute_score(self, x: torch.Tensor) -> float:
        if self.class_means_t is None or self.precision_t is None:
            raise RuntimeError("MahalanobisDetector must be fitted before computing scores.")

        with torch.no_grad():
            self.model(x)
            if self.features_out is None:
                raise RuntimeError("Failed to extract features during inference.")
            f = self.features_out.view(x.size(0), -1)

        # Compute Mahalanobis distance for each class
        # score = min_c ( (f - u_c)^T Precision (f - u_c) )
        distances = []
        for c in range(self.class_means_t.shape[0]):
            mean = self.class_means_t[c].unsqueeze(0)
            diff = f - mean
            # Batched quadratic form: sum( (diff @ precision) * diff, dim=-1 )
            dist = torch.sum(torch.matmul(diff, self.precision_t) * diff, dim=-1)
            distances.append(dist.unsqueeze(1))

        distances = torch.cat(distances, dim=1)
        min_dist = torch.min(distances, dim=1)[0]

        return float(min_dist.mean().item())

    def __del__(self) -> None:
        if self._hook_handle:
            self._hook_handle.remove()
