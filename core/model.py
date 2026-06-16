"""
Model training and prediction interface.

Decoupled from feature computation. Accepts a feature matrix and targets,
returns a trained model that can predict on new features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_PARAMS: dict = {
    'objective':        'huber',   # robust to PnL outliers
    'n_estimators':     500,
    'learning_rate':    0.05,
    'num_leaves':       63,
    'min_child_samples': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_lambda':       1.0,
    'n_jobs':           -1,
    'verbose':          -1,
}


def train_model(
    features:      pd.DataFrame,
    target:        pd.Series,
    sample_weight: pd.Series | None = None,
    model_params:  dict      | None = None,
) -> Any:
    """
    Train a LGBMRegressor on features → target.

    Parameters
    ----------
    features      : feature matrix (NaN/inf should be cleaned before calling)
    target        : regression target (e.g. pnl_{tau})
    sample_weight : per-sample weight (e.g. clipped notional w_i)
    model_params  : override any default LightGBM hyperparameters

    Returns
    -------
    Fitted model with a sklearn-compatible .predict(features) method.
    """
    from lightgbm import LGBMRegressor

    params = {**_DEFAULT_PARAMS, **(model_params or {})}
    model  = LGBMRegressor(**params)
    model.fit(
        features,
        target,
        sample_weight=sample_weight,
    )
    return model


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """
    Run model prediction.

    Returns float64 array aligned with features rows.
    Higher score = better expected trade (less likely to filter out).
    """
    return model.predict(features).astype(np.float64)
