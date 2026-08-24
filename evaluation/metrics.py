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
) #[cite: 2]
from geom_repr.methods.prototypes import PrototypeBank


@dataclass
class MultiLabelMetrics:
    tpr: float #[cite: 3]
    fpr: float #[cite: 3]
    iou: float #[cite: 3]
    precision: float #[cite: 3]
    recall: float #[cite: 3]
    f1: float #[cite: 3]


def macro_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]: #[cite: 2]
    """Per-class AUROC & AP averaged over valid classes[cite: 2]."""
    aucs, aps = [], [] #[cite: 2]
    for c in range(y_true.shape[1]): #[cite: 2]
        p = y_true[:, c].sum() #[cite: 2]
        if 0 < p < y_true.shape[0]: #[cite: 2]
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c])) #[cite: 2]
            aps.append(average_precision_score(y_true[:, c], y_score[:, c])) #[cite: 2]
    return (
        float(np.mean(aucs)) if aucs else float("nan"), #[cite: 2]
        float(np.mean(aps)) if aps else float("nan"), #[cite: 2]
    )


def evaluate_multilabel(preds: np.ndarray, gt: np.ndarray) -> MultiLabelMetrics: #[cite: 3]
    """Evaluates multi-label prediction matrix against binary ground truth[cite: 3]."""
    n, c = gt.shape #[cite: 3]
    tpr_list, fpr_list, iou_list = [], [], [] #[cite: 3]
    for i in range(n): #[cite: 3]
        g, p = gt[i], preds[i] #[cite: 3]
        tp = int(np.sum((p == 1) & (g == 1))) #[cite: 2, 3]
        fp = int(np.sum((p == 1) & (g == 0))) #[cite: 2, 3]
        fn = int(np.sum((p == 0) & (g == 1))) #[cite: 2, 3]
        n_pos, n_neg = g.sum(), c - g.sum() #[cite: 2, 3]
        if n_pos > 0: #[cite: 2, 3]
            tpr_list.append(tp / n_pos) #[cite: 2, 3]
        if n_neg > 0: #[cite: 2, 3]
            fpr_list.append(fp / n_neg) #[cite: 2, 3]
        if tp + fp + fn > 0: #[cite: 2, 3]
            iou_list.append(tp / (tp + fp + fn)) #[cite: 2, 3]

    tp_all = int(np.sum((preds == 1) & (gt == 1))) #[cite: 3]
    fp_all = int(np.sum((preds == 1) & (gt == 0))) #[cite: 3]
    fn_all = int(np.sum((preds == 0) & (gt == 1))) #[cite: 3]

    return MultiLabelMetrics(
        tpr=100 * float(np.mean(tpr_list)) if tpr_list else 0.0, #[cite: 2, 3]
        fpr=100 * float(np.mean(fpr_list)) if fpr_list else 0.0, #[cite: 2, 3]
        iou=100 * float(np.mean(iou_list)) if iou_list else 0.0, #[cite: 2, 3]
        precision=100 * (tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0), #[cite: 3]
        recall=100 * (tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0), #[cite: 3]
        f1=100 * (2 * tp_all / (2 * tp_all + fp_all + fn_all) if (2 * tp_all + fp_all + fn_all) else 0.0), #[cite: 3]
    )


def predict_spm(bank: PrototypeBank, query_features: torch.Tensor) -> np.ndarray: #[cite: 3]
    """Generates binary predictions using standard SPM thresholds[cite: 3]."""
    n = query_features.shape[0] #[cite: 3]
    class_ids = sorted(bank.class_ids) #[cite: 3]
    preds = np.zeros((n, len(class_ids)), dtype=np.float32) #[cite: 3]
    for i in range(n): #[cite: 3]
        q = query_features[i] #[cite: 3]
        for j, c in enumerate(class_ids): #[cite: 3]
            score = bank.spm_score(q, c) #[cite: 3]
            preds[i, j] = float(score > bank.spm_threshold[c]) #[cite: 3]
    return preds #[cite: 3]


def predict_mcm(
    bank: PrototypeBank,
    query_features: torch.Tensor,
    k_neighbors: int,
    per_class_thresholds: Optional[Dict[int, float]] = None, #[cite: 3]
) -> np.ndarray:
    """Generates binary predictions using MCM consistency scores and calibrated thresholds[cite: 3]."""
    n = query_features.shape[0] #[cite: 3]
    class_ids = sorted(bank.class_ids) #[cite: 3]
    preds = np.zeros((n, len(class_ids)), dtype=np.float32) #[cite: 3]
    thrs = per_class_thresholds or bank.mcm_threshold #[cite: 3]
    for i in range(n): #[cite: 3]
        q = query_features[i] #[cite: 3]
        for j, c in enumerate(class_ids): #[cite: 3]
            score = bank.mcm_score(q, c, k_neighbors) #[cite: 3]
            preds[i, j] = float(score > thrs[c]) #[cite: 3]
    return preds #[cite: 3]
