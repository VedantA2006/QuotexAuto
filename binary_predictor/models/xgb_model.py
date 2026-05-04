"""
XGBoost model wrapper for binary candle direction prediction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import XGB_PARAMS, XGB_EARLY_STOPPING, MODEL_DIR
from utils.logger import get_logger

logger = get_logger("binary_predictor")


class XGBModel:
    """XGBoost classifier wrapper for candle direction prediction."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS.copy()
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBModel":
        """
        Train the XGBoost model with optional early stopping.
        """
        self.feature_names = X_train.columns.tolist()

        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_neg > 0:
            self.params["scale_pos_weight"] = n_neg / n_pos

        # Extract early_stopping_rounds from params (not a constructor param in newer xgb)
        early_stopping = self.params.pop("early_stopping_rounds", XGB_EARLY_STOPPING)

        self.model = xgb.XGBClassifier(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        # Use callbacks for early stopping
        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=early_stopping,
                metric_name="logloss",
                save_best=True,
            )
        ]

        self.model.fit(
            X_train, y_train,
            callbacks=callbacks if X_val is not None else None,
            **fit_params,
        )

        self.feature_importances_ = self.model.feature_importances_

        # Restore early_stopping_rounds to params
        self.params["early_stopping_rounds"] = early_stopping

        n_trees = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.params.get('n_estimators', 0)
        logger.info(
            f"XGBoost trained: {n_trees} trees, "
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
        path = path or MODEL_DIR / "xgb_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names},
            path,
        )
        logger.info(f"XGBoost model saved → {path}")
        return path

    def load(self, path: Optional[Path] = None) -> "XGBModel":
        """Load model from disk."""
        path = path or MODEL_DIR / "xgb_model.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importances_ = self.model.feature_importances_
        logger.info(f"XGBoost model loaded ← {path}")
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
