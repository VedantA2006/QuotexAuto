"""
LightGBM model wrapper for binary candle direction prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LGBM_PARAMS, MODEL_DIR
from utils.logger import get_logger

logger = get_logger("binary_predictor")


class LGBMModel:
    """LightGBM classifier wrapper for candle direction prediction."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or LGBM_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: list[str] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMModel":
        """
        Train the LightGBM model with optional early stopping.
        """
        self.feature_names = X_train.columns.tolist()

        self.model = lgb.LGBMClassifier(**self.params)

        fit_params = {"feature_name": self.feature_names}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_metric"] = "logloss"
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importances_ = self.model.feature_importances_

        n_trees = self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ > 0 else self.params.get('n_estimators', 0)
        logger.info(
            f"LightGBM trained: {n_trees} iterations, "
            f"train samples={len(X_train):,}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for class 1 (UP)."""
        proba = self.model.predict_proba(X[self.feature_names])
        return proba[:, 1]

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        path = path or MODEL_DIR / "lgbm_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names},
            path,
        )
        logger.info(f"LightGBM model saved → {path}")
        return path

    def load(self, path: Optional[Path] = None) -> "LGBMModel":
        """Load model from disk."""
        path = path or MODEL_DIR / "lgbm_model.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importances_ = self.model.feature_importances_
        logger.info(f"LightGBM model loaded ← {path}")
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.feature_importances_ is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
