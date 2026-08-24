"""Geometric Representation Learning Package."""

from geom_repr.datasets.transforms import SquarePad, build_standard_transform
from geom_repr.extractors.backbones import get_backbone, extract_features
from geom_repr.methods.geometry import (
    zero_mean,
    nearest_columns,
    principal_angle_score,
    leverage_scores,
)
from geom_repr.methods.prototypes import PrototypeBank
from geom_repr.methods.csdm import apply_csdm
from geom_repr.evaluation.metrics import (
    evaluate_multilabel,
    macro_auc_ap,
    predict_spm,
    predict_mcm,
    MultiLabelMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "SquarePad",
    "build_standard_transform",
    "get_backbone",
    "extract_features",
    "zero_mean",
    "nearest_columns",
    "principal_angle_score",
    "leverage_scores",
    "PrototypeBank",
    "apply_csdm",
    "evaluate_multilabel",
    "macro_auc_ap",
    "predict_spm",
    "predict_mcm",
    "MultiLabelMetrics",
]
