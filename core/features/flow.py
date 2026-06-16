"""
Signed trade flow features: signed flow, total flow, taker imbalance, trade count.

Leading indicator of cascade onset: aggressive directional flow building
before liquidations fire. Computed from the Binance trades stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock

_NOTIONAL_CAP = 100_000.0


class FlowFeatures(FeatureBlock):
    """
    Trade pressure over one or more rolling windows, in one shared pass.

    Per window w, outputs four columns:
        signed_flow_{w}s       = Σ s_j * min(notional_j, 100k)  in [t-w, t)
        total_flow_{w}s        = Σ       min(notional_j, 100k)  in [t-w, t)
        taker_imbalance_{w}s   = signed_flow / total_flow  ∈ [-1, 1]
        trade_count_{w}s       = number of trades                in [t-w, t)

    Both prefix sums are built once and shared across all windows.
    trade_count is free — it is just (right - left) from the window boundaries.
    DirectionRelativize converts signed_flow → same_side_flow and
    taker_imbalance → same_side_taker_imbalance later.
    """

    def __init__(self, windows_s: tuple[float, ...] = (5.0,)):
        self.windows_s = tuple(windows_s)

    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ts       = trades['timestamp'].values
        s        = np.where(trades['side'].values == 'buy', 1.0, -1.0)
        notional = np.minimum(trades['price'].values * trades['amount'].values, _NOTIONAL_CAP)

        # Both prefix sums built once, shared across all windows.
        # right[i] = i for unique sorted ts → sums trades 0..i-1, excludes self.
        prefix_signed   = np.concatenate([[0.0], np.cumsum(s * notional)])
        prefix_unsigned = np.concatenate([[0.0], np.cumsum(notional)])
        right = np.searchsorted(ts, ts, side='left')

        out: dict[str, np.ndarray] = {}
        for w_s in self.windows_s:
            w_us = int(w_s * 1_000_000)
            left = np.searchsorted(ts, ts - w_us, side='left')

            signed_flow = prefix_signed[right]   - prefix_signed[left]
            total_flow  = prefix_unsigned[right]  - prefix_unsigned[left]

            imbalance = np.zeros(len(signed_flow))
            pos = total_flow > 0
            imbalance[pos] = signed_flow[pos] / total_flow[pos]

            out[f'signed_flow_{w_s}s']     = signed_flow
            out[f'total_flow_{w_s}s']      = total_flow
            out[f'taker_imbalance_{w_s}s'] = imbalance
            out[f'trade_count_{w_s}s']     = (right - left).astype(float)

        return pd.DataFrame(out, index=trades.index)

    @property
    def feature_names(self) -> list[str]:
        return [
            name
            for w in self.windows_s
            for name in (f'signed_flow_{w}s', f'total_flow_{w}s',
                          f'taker_imbalance_{w}s', f'trade_count_{w}s')
        ]
