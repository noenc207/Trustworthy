"""Evaluation metrics for the training engine."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, brier_score_loss,
    top_k_accuracy_score
)
from typing import Callable, Tuple, Dict, Any

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {}

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    
    specificity = np.divide(tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    sensitivity = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    metrics['specificity'] = float(np.mean(specificity))
    metrics['sensitivity'] = float(np.mean(sensitivity))
    
    try:
        if num_classes == 2:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob[:, 1]))
            metrics['brier_score'] = float(brier_score_loss(y_true, y_prob[:, 1]))
        else:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob, average='macro'))
            metrics['brier_score'] = float(np.mean([brier_score_loss(y_true == c, y_prob[:, c]) for c in range(num_classes)]))
    except Exception:
        metrics['auroc'] = 0.0
        metrics['pr_auc'] = 0.0
        metrics['brier_score'] = 0.0

    metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['ece'] = compute_ece(y_true, y_prob)
    
    try:
        metrics['top_k_accuracy'] = float(top_k_accuracy_score(y_true, y_prob, k=min(3, num_classes), labels=np.arange(num_classes)))
    except Exception:
        metrics['top_k_accuracy'] = 0.0

    return metrics

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    np.random.seed(seed)
    n = len(y_true)
    scores = []
    
    if n == 0:
        return 0.0, 0.0, 0.0
        
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        except Exception:
            continue
            
    if not scores:
        return 0.0, 0.0, 0.0
        
    alpha = (1.0 - ci) / 2.0
    scores_np = np.array(scores)
    scores_np.sort()
    
    lower = float(np.percentile(scores_np, alpha * 100))
    upper = float(np.percentile(scores_np, (1.0 - alpha) * 100))
    mean = float(np.mean(scores_np))
    
    return mean, lower, upper

def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = np.max(y_prob, axis=1)
        y_pred = np.argmax(y_prob, axis=1) if len(y_prob.shape) > 1 else (y_prob > 0.5).astype(int)
        y_true = (y_true == y_pred).astype(int)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return prob_true, prob_pred, bin_total[nonzero]

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    prob_true, prob_pred, bin_total = compute_calibration_curve(y_true, y_prob, n_bins)
    if len(bin_total) == 0:
        return 0.0
    return float(np.sum(np.abs(prob_pred - prob_true) * bin_total) / np.sum(bin_total))
