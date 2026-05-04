"""
Weighted ensemble of XGBoost + LightGBM + CNN models.

Implements confidence filtering — the single most important accuracy lever.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ENSEMBLE_WEIGHTS,
    MIN_CONFIDENCE,
    MODEL_DIR,
    ADAPTIVE_CONFIDENCE,
    ADAPTIVE_WINDOW,
    ADAPTIVE_LOW_WR,
    ADAPTIVE_HIGH_WR,
    ADAPTIVE_HIGH_CONF,
    ADAPTIVE_LOW_CONF,
)
from models.xgb_model import XGBModel
from models.lgbm_model import LGBMModel
from models.cnn_model import CNNModel
from utils.logger import get_logger

logger = get_logger("binary_predictor")


class EnsembleModel:
    """
    Weighted ensemble of XGBoost + LightGBM + CNN.
    
    Only generates a signal when confidence exceeds MIN_CONFIDENCE.
    """

    def __init__(
        self,
        weights: list[float] = None,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()
        self.min_confidence = min_confidence
        self.xgb_model = XGBModel()
        self.lgbm_model = LGBMModel()
        self.cnn_model = CNNModel()
        self._models_loaded = False

        # Adaptive confidence tracking
        self._recent_outcomes: list[int] = []

    def set_models(
        self,
        xgb_model: XGBModel,
        lgbm_model: LGBMModel,
        cnn_model: CNNModel,
    ):
        """Set pre-trained model instances."""
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.cnn_model = cnn_model
        self._models_loaded = True

    def predict_proba(
        self,
        X: pd.DataFrame,
        X_full: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Compute weighted ensemble probability for class 1 (UP).

        Parameters
        ----------
        X : DataFrame of tabular features (for XGB/LGBM)
        X_full : DataFrame with OHLCV + features (for CNN, optional)

        Returns
        -------
        Array of probabilities [0, 1] for each sample.
        """
        w = np.array(self.weights)
        w = w / w.sum()

        # XGBoost prediction
        try:
            xgb_proba = self.xgb_model.predict_proba(X)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            xgb_proba = np.full(len(X), 0.5)

        # LightGBM prediction
        try:
            lgbm_proba = self.lgbm_model.predict_proba(X)
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            lgbm_proba = np.full(len(X), 0.5)

        # CNN prediction
        try:
            if X_full is not None and self.cnn_model._is_fitted:
                cnn_proba = self.cnn_model.predict_proba(X_full)
            else:
                cnn_proba = np.full(len(X), 0.5)
        except Exception as e:
            logger.warning(f"CNN prediction failed: {e}")
            cnn_proba = np.full(len(X), 0.5)

        # Weighted average
        ensemble_proba = (
            w[0] * xgb_proba
            + w[1] * lgbm_proba
            + w[2] * cnn_proba
        )

        return ensemble_proba

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        X_full: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """
        Generate predictions with confidence filtering.

        Returns a list of signal dicts, one per sample.
        Signals with confidence < min_confidence are marked as 'SKIP'.
        """
        proba = self.predict_proba(X, X_full)
        threshold = self._get_current_threshold()

        signals = []
        for i, p in enumerate(proba):
            confidence = max(p, 1 - p)  # distance from 0.5
            direction = "UP" if p > 0.5 else "DOWN"

            if confidence >= threshold:
                signals.append({
                    "index": i,
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    "probability": round(p, 4),
                    "action": "TRADE",
                })
            else:
                signals.append({
                    "index": i,
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    "probability": round(p, 4),
                    "action": "SKIP",
                })

        return signals

    def predict_single(
        self,
        X: pd.DataFrame,
        X_full: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Predict for the last row only (latest candle)."""
        signals = self.predict_with_confidence(X, X_full)
        return signals[-1] if signals else {"action": "SKIP", "confidence": 0}

    def _get_current_threshold(self) -> float:
        """
        Get the adaptive confidence threshold based on recent performance.
        """
        if not ADAPTIVE_CONFIDENCE or len(self._recent_outcomes) < ADAPTIVE_WINDOW:
            return self.min_confidence

        recent_wr = np.mean(self._recent_outcomes[-ADAPTIVE_WINDOW:])

        if recent_wr < ADAPTIVE_LOW_WR:
            threshold = ADAPTIVE_HIGH_CONF
            logger.info(
                f"Adaptive: WR={recent_wr:.1%} < {ADAPTIVE_LOW_WR:.0%}, "
                f"raising threshold to {threshold}"
            )
        elif recent_wr > ADAPTIVE_HIGH_WR:
            threshold = ADAPTIVE_LOW_CONF
        else:
            threshold = self.min_confidence

        return threshold

    def record_outcome(self, won: bool):
        """Record a trade outcome for adaptive threshold."""
        self._recent_outcomes.append(int(won))
        if len(self._recent_outcomes) > 200:
            self._recent_outcomes = self._recent_outcomes[-200:]

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_full_val: Optional[pd.DataFrame] = None,
    ) -> list[float]:
        """
        Optimize ensemble weights on validation data using grid search.
        """
        logger.info("Optimizing ensemble weights on validation set...")

        # Get individual predictions
        try:
            xgb_p = self.xgb_model.predict_proba(X_val)
        except:
            xgb_p = np.full(len(X_val), 0.5)
        try:
            lgbm_p = self.lgbm_model.predict_proba(X_val)
        except:
            lgbm_p = np.full(len(X_val), 0.5)
        try:
            if X_full_val is not None and self.cnn_model._is_fitted:
                cnn_p = self.cnn_model.predict_proba(X_full_val)
            else:
                cnn_p = np.full(len(X_val), 0.5)
        except:
            cnn_p = np.full(len(X_val), 0.5)

        y = y_val.values

        best_acc = 0
        best_weights = self.weights.copy()

        # Grid search over weight combinations
        for w1 in np.arange(0.1, 0.8, 0.05):
            for w2 in np.arange(0.1, 0.8, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue

                combo = w1 * xgb_p + w2 * lgbm_p + w3 * cnn_p
                preds = (combo > 0.5).astype(int)

                # Only consider high-confidence predictions
                confidence = np.maximum(combo, 1 - combo)
                mask = confidence >= self.min_confidence

                if mask.sum() < 50:
                    continue

                acc = (preds[mask] == y[mask]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_weights = [round(w1, 3), round(w2, 3), round(w3, 3)]

        self.weights = best_weights
        logger.info(
            f"Optimal weights: {best_weights}, "
            f"filtered accuracy: {best_acc:.4f}"
        )
        return best_weights

    def save(self, path: Optional[Path] = None) -> Path:
        """Save all ensemble components."""
        path = path or MODEL_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.xgb_model.save(path / "xgb_model.pkl")
        self.lgbm_model.save(path / "lgbm_model.pkl")
        self.cnn_model.save(path / "cnn_model")

        meta = {
            "weights": self.weights,
            "min_confidence": self.min_confidence,
        }
        joblib.dump(meta, path / "ensemble_meta.pkl")
        logger.info(f"Ensemble saved → {path}")
        return path

    def load(self, path: Optional[Path] = None) -> "EnsembleModel":
        """Load all ensemble components."""
        path = path or MODEL_DIR
        path = Path(path)

        self.xgb_model.load(path / "xgb_model.pkl")
        self.lgbm_model.load(path / "lgbm_model.pkl")

        try:
            self.cnn_model.load(path / "cnn_model")
        except Exception as e:
            logger.warning(f"Could not load CNN model: {e}")

        meta_path = path / "ensemble_meta.pkl"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.weights = meta.get("weights", ENSEMBLE_WEIGHTS)
            self.min_confidence = meta.get("min_confidence", MIN_CONFIDENCE)

        self._models_loaded = True
        logger.info(f"Ensemble loaded ← {path}")
        return self
