from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance


@dataclass
class DriftResult:
    status: str
    kl_divergence: float
    js_divergence: float
    earth_mover_distance: float
    ssim: float
    cosine_similarity: float
    pearson_correlation: float

class DriftAnalyzer:
    def __init__(self, benchmark_engine=None):
        self.benchmark_engine = benchmark_engine

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        p = p / np.sum(p)
        q = q / np.sum(q)
        m = 0.5 * (p + q)
        return 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

    def _ssim_approx(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if img1.shape != img2.shape:
            return 0.0
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.cov(img1.flatten(), img2.flatten())[0][1] if len(img1) > 1 else 0
        c1 = 1e-4
        c2 = 9e-4
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        return float(ssim)

    def analyze_drift(self, source_data: np.ndarray, target_data: np.ndarray) -> DriftResult:
        if source_data.size == 0 or target_data.size == 0:
            return DriftResult("error", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        src_flat = source_data.flatten()
        tgt_flat = target_data.flatten()

        min_len = min(len(src_flat), len(tgt_flat))
        src_trunc = src_flat[:min_len]
        tgt_trunc = tgt_flat[:min_len]

        kl_div = self._kl_divergence(src_trunc, tgt_trunc)
        js_div = self._js_divergence(src_trunc, tgt_trunc)
        emd = wasserstein_distance(src_flat, tgt_flat)

        try:
            ssim_val = self._ssim_approx(src_trunc, tgt_trunc)
        except:
            ssim_val = 0.0

        try:
            cos_sim = 1.0 - cosine(src_trunc, tgt_trunc)
        except:
            cos_sim = 0.0

        try:
            if len(src_trunc) > 1:
                pearson_corr, _ = pearsonr(src_trunc, tgt_trunc)
            else:
                pearson_corr = 0.0
        except:
            pearson_corr = 0.0

        status = "normal"
        if emd > 0.5 or js_div > 0.2:
            status = "drift_detected"

        return DriftResult(
            status=status,
            kl_divergence=kl_div,
            js_divergence=js_div,
            earth_mover_distance=emd,
            ssim=ssim_val,
            cosine_similarity=cos_sim,
            pearson_correlation=float(pearson_corr)
        )
