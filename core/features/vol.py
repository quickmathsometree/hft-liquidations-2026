"""
Volatility and regime-context features.

Conditioning layer: modulates how dangerous a given level of liq pressure is
given the current market regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock

# ---------------------------------------------------------------------------
# Helper: causal rolling rank percentile over the trailing n trades
# ---------------------------------------------------------------------------

def _rolling_rank_impl(values: np.ndarray, n: int) -> np.ndarray:
    """
    For each i, the fraction of non-NaN values in values[max(0, i-n+1) : i+1]
    that are <= values[i] (ties included, current value included in the
    window since it is already observed at trade time — not a future leak).

    NaN in/NaN out: if values[i] is NaN, the rank is NaN.
    O(n_rows * n) — fine for n in the hundreds to low thousands.
    """
    m   = len(values)
    out = np.empty(m)
    for i in range(m):
        cur = values[i]
        if np.isnan(cur):
            out[i] = np.nan
            continue
        lo = i - n + 1
        if lo < 0:
            lo = 0
        count  = 0
        length = 0
        for j in range(lo, i + 1):
            v = values[j]
            if np.isnan(v):
                continue
            length += 1
            if v <= cur:
                count += 1
        out[i] = count / length if length > 0 else np.nan
    return out


try:
    import numba
    _rolling_rank = numba.njit(_rolling_rank_impl, cache=True)
except ImportError:
    _rolling_rank = _rolling_rank_impl


class VolFeatures(FeatureBlock):
    """
    Volatility and regime context, in one shared pass.

    Per window w in windows_s, outputs:
        rolling_vol_{w}s       = std(log(price[j]/price[j-1]))  for trades in [t-w, t)

    Plus, using the trailing rank_window trades (count-based, not time-based):
        spread_bps_rank_{n}    = rolling rank percentile of spread_bps    (0-1)
        depth_rank_{n}         = rolling rank percentile of top-of-book size, i.e.
                                  bid_amount + ask_amount                 (0-1)
        vol_rank_{n}           = rolling rank percentile of rolling_vol at the
                                  shortest configured window (windows_s[0])  (0-1)

    Prefix sums of log-returns are built once and shared across all windows_s.
    spread_bps / depth are looked up from the BBO stream the same way
    BookFeatures does (last update strictly before each trade).
    """

    def __init__(
        self,
        windows_s:   tuple[float, ...] = (60.0, 300.0),
        rank_window: int               = 1000,
    ):
        self.windows_s   = tuple(windows_s)
        self.rank_window = rank_window

    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ts     = trades['timestamp'].values
        prices = trades['price'].values

        log_ret    = np.diff(np.log(prices), prepend=np.log(prices[0]))
        log_ret[0] = 0.0

        # Prefix sums built once, shared across all windows.
        # right[i] = i for unique sorted ts — covers log_ret[0..i-1], excludes self.
        cum_x  = np.concatenate([[0.0], np.cumsum(log_ret)])
        cum_x2 = np.concatenate([[0.0], np.cumsum(log_ret ** 2)])
        right  = np.searchsorted(ts, ts, side='left')

        out: dict[str, np.ndarray] = {}
        for w_s in self.windows_s:
            w_us = int(w_s * 1_000_000)
            left = np.searchsorted(ts, ts - w_us, side='left')

            n      = (right - left).astype(float)
            sum_x  = cum_x[right]  - cum_x[left]
            sum_x2 = cum_x2[right] - cum_x2[left]

            variance = np.zeros(len(n))
            mask = n > 1
            variance[mask] = sum_x2[mask] / n[mask] - (sum_x[mask] / n[mask]) ** 2

            out[f'rolling_vol_{w_s}s'] = np.sqrt(np.maximum(variance, 0.0))

        # --- BBO lookup for spread_bps / depth (last update strictly before each trade) ---
        bbo       = kwargs['bbo']
        bbo_ts    = bbo['timestamp'].values
        idx_now   = np.searchsorted(bbo_ts, ts, side='left') - 1
        valid_now = idx_now >= 0

        def _lookup(col: str) -> np.ndarray:
            vals = np.full(len(ts), np.nan)
            vals[valid_now] = bbo[col].values[idx_now[valid_now]]
            return vals

        bid_p, ask_p = _lookup('bid_price'), _lookup('ask_price')
        bid_a, ask_a = _lookup('bid_amount'), _lookup('ask_amount')

        mid        = (bid_p + ask_p) / 2
        spread_bps = (ask_p - bid_p) / mid * 10_000
        depth      = bid_a + ask_a

        n = self.rank_window
        out[f'spread_bps_rank_{n}'] = _rolling_rank(spread_bps, n)
        out[f'depth_rank_{n}']      = _rolling_rank(depth, n)
        out[f'vol_rank_{n}']        = _rolling_rank(out[f'rolling_vol_{self.windows_s[0]}s'], n)

        return pd.DataFrame(out, index=trades.index)

    @property
    def feature_names(self) -> list[str]:
        n = self.rank_window
        names = [f'rolling_vol_{w}s' for w in self.windows_s]
        names += [f'spread_bps_rank_{n}', f'depth_rank_{n}', f'vol_rank_{n}']
        return names

