"""
Feature importance analysis and selection.

Uses gain-based importance + SHAP to identify top features
and optionally prune low-value ones.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger("binary_predictor")


def compute_feature_importance(
    model,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model.

    Parameters
    ----------
    model : fitted XGBoost or LightGBM model
    feature_names : list of feature column names
    importance_type : 'gain', 'weight', or 'cover'

    Returns
    -------
    DataFrame with columns [feature, importance] sorted descending.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            score = booster.get_score(importance_type=importance_type)
            importances = np.array(
                [score.get(f"f{i}", score.get(fn, 0))
                 for i, fn in enumerate(feature_names)]
            )
        else:
            importances = np.ones(len(feature_names))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        importances = np.ones(len(feature_names))

    total = importances.sum() + 1e-9
    importances = importances / total

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> pd.DataFrame:
    """
    Compute SHAP values for feature importance ranking.

    Parameters
    ----------
    model : fitted model (XGBoost or LightGBM)
    X : feature DataFrame
    max_samples : subsample size for SHAP computation

    Returns
    -------
    DataFrame with columns [feature, shap_importance] sorted descending.
    """
    try:
        import shap

        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum() + 1e-9
        mean_abs_shap = mean_abs_shap / total

        df = pd.DataFrame({
            "feature": X.columns.tolist(),
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

        logger.info(f"SHAP importance computed for {len(df)} features")
        return df

    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis")
        return pd.DataFrame(columns=["feature", "shap_importance"])
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return pd.DataFrame(columns=["feature", "shap_importance"])


def select_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    min_importance: float = 0.005,
    importance_col: str = "importance",
) -> list[str]:
    """
    Select top N features based on importance scores.

    Returns list of feature names to keep.
    """
    filtered = importance_df[
        importance_df[importance_col] >= min_importance
    ]
    selected = filtered.head(top_n)["feature"].tolist()
    logger.info(f"Selected {len(selected)} features (top {top_n}, min_imp={min_importance})")
    return selected


def combined_importance(
    gain_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    gain_weight: float = 0.5,
    shap_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Combine gain-based and SHAP importance into a single ranking.
    """
    if shap_df.empty:
        return gain_df.copy()

    merged = gain_df.merge(shap_df, on="feature", how="left")
    merged["shap_importance"] = merged["shap_importance"].fillna(0)
    merged["combined"] = (
        merged["importance"] * gain_weight
        + merged["shap_importance"] * shap_weight
    )
    merged = merged.sort_values("combined", ascending=False).reset_index(drop=True)
    return merged


def save_importance_report(
    importance_df: pd.DataFrame,
    filepath: str | Path,
) -> None:
    """Save feature importance DataFrame to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(filepath, index=False)
    logger.info(f"Feature importance saved → {filepath}")
