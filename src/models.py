"""
Models Module for Ames Housing Advanced Regression.
Defines baseline, candidate estimators, and ensembling pipelines.
"""

from typing import Dict, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


def get_baseline_model(alpha: float = 15.0) -> Pipeline:
    """
    Establish a fast, robust linear baseline using Ridge Regression with RobustScaler.
    Handles multicollinear and high-dimensional one-hot features via L2 regularization.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('model', Ridge(alpha=alpha, random_state=42))
    ])


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns candidate models spanning linear regularized models, bagging, and boosting:
    1. Ridge Regression (L2 regularized linear model)
    2. Lasso Regression (L1 regularized sparse linear model)
    3. Random Forest Regressor (Non-linear bagging ensemble from course template)
    4. Gradient Boosting Regressor (Scikit-Learn sequential boosting)
    5. LightGBM Regressor (High-performance gradient boosted decision trees)
    """
    return {
        'Ridge': Pipeline([
            ('scaler', RobustScaler()),
            ('model', Ridge(alpha=15.0, random_state=42))
        ]),
        'Lasso': Pipeline([
            ('scaler', RobustScaler()),
            ('model', Lasso(alpha=0.0005, max_iter=5000, random_state=42))
        ]),
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    }
