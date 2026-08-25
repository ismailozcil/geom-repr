from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
) 
from geom_repr.methods.prototypes import PrototypeBank


@dataclass
class MultiLabelMetrics:
    tpr: float 
    fpr: float 
    iou: float 
    precision: float 
    recall: float
    f1: float


def macro_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]: 
    """Per-class AUROC & AP averaged over valid classes."""
    aucs, aps = [], [] 
    for c in range(y_true.shape[1]): 
        p = y_true[:, c].sum() 
        if 0 < p < y_true.shape[0]: 
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c])) 
            aps.append(average_precision_score(y_true[:, c], y_score[:, c]))
    return (
        float(np.mean(aucs)) if aucs else float("nan"), 
        float(np.mean(aps)) if aps else float("nan"), 
    )


def evaluate_multilabel(preds: np.ndarray, gt: np.ndarray) -> MultiLabelMetrics: 
    """Evaluates multi-label prediction matrix against binary ground truth."""
    n, c = gt.shape 
    tpr_list, fpr_list, iou_list = [], [], [] 
    for i in range(n): 
        g, p = gt[i], preds[i] 
        tp = int(np.sum((p == 1) & (g == 1))) 
        fp = int(np.sum((p == 1) & (g == 0))) 
        fn = int(np.sum((p == 0) & (g == 1)))
        n_pos, n_neg = g.sum(), c - g.sum() 
        if n_pos > 0: 
            tpr_list.append(tp / n_pos)
        if n_neg > 0: 
            fpr_list.append(fp / n_neg) 
        if tp + fp + fn > 0: 
            iou_list.append(tp / (tp + fp + fn)) 

    tp_all = int(np.sum((preds == 1) & (gt == 1)))
    fp_all = int(np.sum((preds == 1) & (gt == 0)))
    fn_all = int(np.sum((preds == 0) & (gt == 1)))

    return MultiLabelMetrics(
        tpr=100 * float(np.mean(tpr_list)) if tpr_list else 0.0, 
        fpr=100 * float(np.mean(fpr_list)) if fpr_list else 0.0, 
        iou=100 * float(np.mean(iou_list)) if iou_list else 0.0,
        precision=100 * (tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0),
        recall=100 * (tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0), 
        f1=100 * (2 * tp_all / (2 * tp_all + fp_all + fn_all) if (2 * tp_all + fp_all + fn_all) else 0.0), 
    )


def predict_spm(bank: PrototypeBank, query_features: torch.Tensor, n_classes: Optional[int] = None) -> np.ndarray:
    """Generates binary predictions using standard SPM thresholds."""
    n = query_features.shape[0]
    class_ids = sorted(bank.class_ids)
    width = n_classes if n_classes is not None else (max(class_ids) + 1 if class_ids else 0)
    preds = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        q = query_features[i]
        for c in class_ids:
            score = bank.spm_score(q, c)
            preds[i, c] = float(score > bank.spm_threshold[c])
    return preds


def predict_mcm(
    bank: PrototypeBank,
    query_features: torch.Tensor,
    k_neighbors: int,
    per_class_thresholds: Optional[Dict[int, float]] = None,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Generates binary predictions using MCM consistency scores and calibrated thresholds."""
    n = query_features.shape[0]
    class_ids = sorted(bank.class_ids)
    width = n_classes if n_classes is not None else (max(class_ids) + 1 if class_ids else 0)
    preds = np.zeros((n, width), dtype=np.float32)
    thrs = per_class_thresholds or bank.mcm_threshold
    for i in range(n):
        q = query_features[i]
        for c in class_ids:
            score = bank.mcm_score(q, c, k_neighbors)
            preds[i, c] = float(score > thrs[c])
    return preds
