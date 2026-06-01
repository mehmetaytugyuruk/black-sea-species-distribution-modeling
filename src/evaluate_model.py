"""Model evaluation and feature-importance exports."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .train_model import TrainedModelResult


def evaluate_trained_models(results: list[TrainedModelResult]) -> pd.DataFrame:
    """Create model metrics for every trained species/model pair."""

    rows: list[dict[str, object]] = []
    for result in results:
        roc_auc = _safe_roc_auc(result)
        rows.append(
            {
                "species": result.species,
                "model": result.model_name,
                "roc_auc": roc_auc,
                "precision": precision_score(result.y_test, result.y_pred, zero_division=0),
                "recall": recall_score(result.y_test, result.y_pred, zero_division=0),
                "f1_score": f1_score(result.y_test, result.y_pred, zero_division=0),
                "threshold": 0.5,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_total": result.n_total,
            }
        )
    return pd.DataFrame(rows)


def _safe_roc_auc(result: TrainedModelResult) -> float:
    try:
        return float(roc_auc_score(result.y_test, result.y_score))
    except ValueError:
        return math.nan


def extract_feature_importance(results: list[TrainedModelResult]) -> pd.DataFrame:
    """Export feature importance for models that expose feature_importances_."""

    rows: list[dict[str, object]] = []
    for result in results:
        importances = getattr(result.final_model, "feature_importances_", None)
        if importances is None:
            continue
        for feature, importance in zip(result.feature_columns, importances):
            rows.append(
                {
                    "species": result.species,
                    "model": result.model_name,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    return pd.DataFrame(rows)

