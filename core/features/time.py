"""
Time features.

Time-of-day, day-of-week, and time-to-known-event encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock

_FUNDING_INTERVAL_S = 8 * 3600   # 00:00 / 08:00 / 16:00 UTC
_DAY_S              = 86_400


class TimeFeatures(FeatureBlock):
    """
    Calendar and schedule features derived purely from timestamps.

    Outputs:
        time_to_funding_s   seconds until next funding event (0, 28800]
        hour_utc            UTC hour of day  [0, 23]
        minute_utc          UTC minute of hour [0, 59]

    All outputs are raw integers / floats — suitable for LightGBM, which
    finds its own splits and doesn't need cyclic encoding.
    No searchsorted or prefix sums: every column is O(n) arithmetic.
    """

    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        t_us  = trades['timestamp'].values
        t_sec = t_us // 1_000_000

        t_in_day = t_sec % _DAY_S

        return pd.DataFrame({
            'time_to_funding_s': (_FUNDING_INTERVAL_S - t_in_day % _FUNDING_INTERVAL_S).astype(np.float64),
            'hour_utc':          (t_in_day // 3600).astype(np.float64),
            'minute_utc':        (t_in_day % 3600 // 60).astype(np.float64),
        }, index=trades.index)

    @property
    def feature_names(self) -> list[str]:
        return ['time_to_funding_s', 'hour_utc', 'minute_utc']
