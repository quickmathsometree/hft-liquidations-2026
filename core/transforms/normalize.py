"""
Normalization transforms: zscore, clamp, winsorize, NaN cleanup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.transforms.base import Transform


# ---------------------------------------------------------------------------
# Low-level helpers (operate on a single Series)
# ---------------------------------------------------------------------------

def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mean = series.rolling(window, min_periods=2).mean()
    std  = series.rolling(window, min_periods=2).std()
    return (series - mean) / std.replace(0, np.nan)


def _winsorize(series: pd.Series, quantile: float) -> pd.Series:
    lo = series.quantile(quantile)
    hi = series.quantile(1 - quantile)
    return series.clip(lo, hi)


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class Zscore(Transform):
    """
    Rolling z-score applied to specified columns (or all numeric columns).

        z = (x - rolling_mean(window)) / rolling_std(window)
    """

    def __init__(self, window: int = 1000, columns: list[str] | None = None):
        self.window  = window
        self.columns = columns

    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns or list(features.select_dtypes(include='number').columns)
        out  = features.copy()
        for c in cols:
            if c in out.columns:
                out[c] = _zscore(out[c], self.window)
        return out


class Clamp(Transform):
    """
    Hard-clip values to [lo, hi] for specified columns (or all numeric columns).
    """

    def __init__(self, lo: float = -5.0, hi: float = 5.0, columns: list[str] | None = None):
        self.lo      = lo
        self.hi      = hi
        self.columns = columns

    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns or list(features.select_dtypes(include='number').columns)
        out  = features.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].clip(self.lo, self.hi)
        return out


class Winsorize(Transform):
    """
    Clip values to [quantile, 1-quantile] for specified columns (or all numeric columns).
    """

    def __init__(self, quantile: float = 0.01, columns: list[str] | None = None):
        self.quantile = quantile
        self.columns  = columns

    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns or list(features.select_dtypes(include='number').columns)
        out  = features.copy()
        for c in cols:
            if c in out.columns:
                out[c] = _winsorize(out[c], self.quantile)
        return out


class FillNanInf(Transform):
    """Replace NaN and ±inf with fill_value. Should be the last transform."""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        return features.replace([np.inf, -np.inf], np.nan).fillna(self.fill_value)
