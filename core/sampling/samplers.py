"""
Sampling strategies: every trade, volume-triggered, time-triggered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numba-accelerated inner loop for VolumeThreshold
# ---------------------------------------------------------------------------

def _volume_threshold_impl(notional: np.ndarray, threshold: float) -> np.ndarray:
    mask   = np.zeros(len(notional), dtype=np.bool_)
    cumsum = 0.0
    for i in range(len(notional)):
        cumsum += notional[i]
        if cumsum >= threshold:
            mask[i] = True
            cumsum   = 0.0
    return mask


try:
    import numba
    _volume_threshold_loop = numba.njit(_volume_threshold_impl)
except ImportError:
    _volume_threshold_loop = _volume_threshold_impl


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Sampler(ABC):

    @abstractmethod
    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        """
        Return a boolean mask of length len(trades).
        True = this trade is a datapoint in the supervised dataset.
        """
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class EveryTrade(Sampler):
    """Use every trade as a datapoint. Default for the liquidation filter task."""

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        return np.ones(len(trades), dtype=bool)


class VolumeThreshold(Sampler):
    """
    Emit a datapoint every time cumulative traded notional exceeds a threshold.
    The emitted row is the trade that crossed the threshold.
    Accelerated with numba.njit when numba is available.
    """

    def __init__(self, notional_threshold: float = 100_000.0):
        self.notional_threshold = notional_threshold

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        notional = (trades['price'].values * trades['amount'].values).astype(np.float64)
        return _volume_threshold_loop(notional, float(self.notional_threshold))


class TimeInterval(Sampler):
    """Emit a datapoint every N seconds (the last trade in each interval)."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        ts       = trades['timestamp'].values
        interval = int(self.interval_s * 1_000_000)
        buckets  = ts // interval
        mask     = np.zeros(len(trades), dtype=bool)
        _, last_indices = np.unique(buckets[::-1], return_index=True)
        mask[len(trades) - 1 - last_indices] = True
        return mask
