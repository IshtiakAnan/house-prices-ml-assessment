"""
Evaluation and Validation Module for Ames Housing Dataset.
Implements K-Fold CV metrics, Out-of-Fold generation, and residual diagnostic tools.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def evaluate_cv(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = 5, 
    random_state: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Compute K-Fold Cross-Validation Root Mean Squared Logarithmic Error (RMSLE).
    Returns (mean_rmse, std_rmse, fold_scores).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = np.sqrt(-cross_val_score(
        model, X, y, 
        scoring='neg_mean_squared_error', 
        cv=kf, 
        n_jobs=-1
    ))
    return float(scores.mean()), float(scores.std()), scores


def generate_oof_predictions(
    models: Dict[str, Any], 
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = 5, 
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Compute out-of-fold (OOF) cross-validation predictions for unbiased validation and blending.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = {name: np.zeros(len(y)) for name in models.keys()}

    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model.predict(X_va)

    return oof_preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression evaluation metrics on log-transformed target.
    """
    return {
        'RMSLE': float(root_mean_squared_error(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred))
    }
