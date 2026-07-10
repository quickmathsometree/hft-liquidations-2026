"""
Book / L1 features from Binance BBO stream.

All features use the last BBO update strictly before each trade (causal).
bid_amount / ask_amount columns are kept absolute here; DirectionRelativize
converts them to same_side_depth / opp_side_depth later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock


class BookFeatures(FeatureBlock):
    """
    All L1 book features in one pass over the BBO array.

    Outputs (always):
        spread_bps, imbalance, microprice_dev, bid_amount, ask_amount

    Outputs (per window in depth_delta_windows_s):
        bid_amount_delta_{w}s, ask_amount_delta_{w}s

    Single searchsorted for current-state features; one additional
    searchsorted per delta window for the past-state lookup.
    """

    def __init__(self, depth_delta_windows_s: tuple[float, ...] = ()):
        self.depth_delta_windows_s = tuple(depth_delta_windows_s)

    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        bbo      = kwargs['bbo']
        bbo_ts   = bbo['timestamp'].values
        query_ts = trades['timestamp'].values

        # One searchsorted for the current BBO state
        idx_now   = np.searchsorted(bbo_ts, query_ts, side='left') - 1
        valid_now = idx_now >= 0

        def _lookup(col: str) -> np.ndarray:
            out = np.full(len(query_ts), np.nan)
            out[valid_now] = bbo[col].values[idx_now[valid_now]]
            return out

        bid_p = _lookup('bid_price')
        ask_p = _lookup('ask_price')
        bid_a = _lookup('bid_amount')
        ask_a = _lookup('ask_amount')

        mid        = (bid_p + ask_p) / 2
        microprice = (bid_p * ask_a + ask_p * bid_a) / (bid_a + ask_a)

        out: dict[str, np.ndarray] = {
            'spread_bps':     (ask_p - bid_p) / mid * 10_000,
            'imbalance':      (bid_a - ask_a) / (bid_a + ask_a),
            'microprice_dev': (microprice - mid) / mid * 10_000,
            'bid_amount':     bid_a,
            'ask_amount':     ask_a,
        }

        # One searchsorted per delta window for the past-state lookup
        for w_s in self.depth_delta_windows_s:
            w_us      = int(w_s * 1_000_000)
            idx_past  = np.searchsorted(bbo_ts, query_ts - w_us, side='left') - 1
            valid_past = idx_past >= 0

            bid_past = np.full(len(query_ts), np.nan)
            ask_past = np.full(len(query_ts), np.nan)
            bid_past[valid_past] = bbo['bid_amount'].values[idx_past[valid_past]]
            ask_past[valid_past] = bbo['ask_amount'].values[idx_past[valid_past]]

            out[f'bid_amount_delta_{w_s}s'] = bid_a - bid_past
            out[f'ask_amount_delta_{w_s}s'] = ask_a - ask_past

        return pd.DataFrame(out, index=trades.index)

    @property
    def feature_names(self) -> list[str]:
        base  = ['spread_bps', 'imbalance', 'microprice_dev', 'bid_amount', 'ask_amount']
        delta = [
            name
            for w in self.depth_delta_windows_s
            for name in (f'bid_amount_delta_{w}s', f'ask_amount_delta_{w}s')
        ]
        return base + delta
