from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mutual_info_score


def distance_correlation(x, y):
    x = np.asarray(x)[:, None]
    y = np.asarray(y)[:, None]
    if x.size <= 1:
        return 0.0
    a = squareform(pdist(x))
    b = squareform(pdist(y))
    a_mat = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    b_mat = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (a_mat * b_mat).sum() / float(x.size * x.size)
    dcov2_xx = (a_mat * a_mat).sum() / float(x.size * x.size)
    dcov2_yy = (b_mat * b_mat).sum() / float(x.size * x.size)
    dvar = np.sqrt(dcov2_xx * dcov2_yy)
    if dvar > 0:
        return np.sqrt(dcov2_xy / dvar)
    return 0.0

def calc_mutual_information(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

class MetricCorrelationAnalyzer:
    """
    Computes correlation metrics (Pearson, Spearman, Kendall, Distance Correlation, Mutual Information)
    and identifies relationship classes.
    """
    def __init__(self, redundancy_threshold: float = 0.95, weak_threshold: float = 0.3, high_threshold: float = 0.7):
        self.redundancy_threshold = redundancy_threshold
        self.weak_threshold = weak_threshold
        self.high_threshold = high_threshold

    def classify_relationship(self, p, s, k):
        max_corr = max(abs(p), abs(s), abs(k))

        if p < -self.high_threshold or s < -self.high_threshold or k < -self.high_threshold:
            return "Contradictory"
        if max_corr >= self.redundancy_threshold:
            return "Redundant"
        if max_corr >= self.high_threshold:
            return "Highly correlated"
        if max_corr <= self.weak_threshold:
            return "Independent" if max_corr < 0.1 else "Weakly related"
        return "Weakly related"

    def analyze(self, metrics_data: dict[str, np.ndarray]) -> dict[str, Any]:
        metrics = list(metrics_data.keys())
        n_metrics = len(metrics)

        pearson_mat = np.zeros((n_metrics, n_metrics))
        spearman_mat = np.zeros((n_metrics, n_metrics))
        kendall_mat = np.zeros((n_metrics, n_metrics))
        dist_corr_mat = np.zeros((n_metrics, n_metrics))
        mi_mat = np.zeros((n_metrics, n_metrics))

        relationships = []

        for i in range(n_metrics):
            for j in range(n_metrics):
                if i == j:
                    pearson_mat[i, j] = 1.0
                    spearman_mat[i, j] = 1.0
                    kendall_mat[i, j] = 1.0
                    dist_corr_mat[i, j] = 1.0
                    x = np.asarray(metrics_data[metrics[i]])
                    mi_mat[i, j] = calc_mutual_information(x, x)
                    continue

                x = np.asarray(metrics_data[metrics[i]])
                y = np.asarray(metrics_data[metrics[j]])

                if np.std(x) == 0 or np.std(y) == 0:
                    p, s, k, dc, mi = 0.0, 0.0, 0.0, 0.0, 0.0
                else:
                    p, _ = pearsonr(x, y)
                    s, _ = spearmanr(x, y)
                    k, _ = kendalltau(x, y)
                    dc = distance_correlation(x, y)
                    mi = calc_mutual_information(x, y)

                pearson_mat[i, j] = p
                spearman_mat[i, j] = s
                kendall_mat[i, j] = k
                dist_corr_mat[i, j] = dc
                mi_mat[i, j] = mi

                if i < j:
                    classification = self.classify_relationship(p, s, k)
                    relationships.append({
                        "metric1": metrics[i],
                        "metric2": metrics[j],
                        "pearson": float(p),
                        "spearman": float(s),
                        "kendall": float(k),
                        "distance_correlation": float(dc),
                        "mutual_information": float(mi),
                        "classification": classification
                    })

        return {
            "metrics": metrics,
            "matrices": {
                "pearson": pearson_mat.tolist(),
                "spearman": spearman_mat.tolist(),
                "kendall": kendall_mat.tolist(),
                "distance_correlation": dist_corr_mat.tolist(),
                "mutual_information": mi_mat.tolist()
            },
            "relationships": relationships
        }
