"""
Liquidation pressure features.

Primary signal for cascade detection. Computes EWMA of liquidation notional
across venues (Binance, Bybit) and sides (buy, sell), plus intensity metrics
(event count, time since last liq).

Bybit timestamps must be pre-shifted (+200ms) before calling these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock


# ---------------------------------------------------------------------------
# Helper: EWMA recurrence — O(n + m), numerically stable
# ---------------------------------------------------------------------------

def _ewma_recurrence_impl(
    query_ts:   np.ndarray,
    event_ts:   np.ndarray,
    event_vals: np.ndarray,
    halflife_us: float,
) -> np.ndarray:
    """
    Recurrence-based EWMA over a sorted event stream at sorted query times.
    Strictly causal: events at exactly query_ts are excluded.

    O(n + m) single merge pass — no overflow risk since only small dt values
    appear in exp(-dt/tau).
    """
    tau = halflife_us / np.log(2)
    n_q = len(query_ts)
    n_e = len(event_ts)
    result = np.zeros(n_q)

    ptr = 0
    state = 0.0
    last_t = 0
    has_event = False

    for i in range(n_q):
        t_q = query_ts[i]
        while ptr < n_e and event_ts[ptr] < t_q:
            if not has_event:
                state = event_vals[ptr]
                has_event = True
            else:
                dt = event_ts[ptr] - last_t
                state = state * np.exp(-dt / tau) + event_vals[ptr]
            last_t = event_ts[ptr]
            ptr += 1
        if has_event:
            dt = t_q - last_t
            result[i] = state * np.exp(-dt / tau)

    return result


try:
    import numba
    _ewma_event_state_at = numba.njit(_ewma_recurrence_impl)
except ImportError:
    _ewma_event_state_at = _ewma_recurrence_impl


# ---------------------------------------------------------------------------
# FeatureBlock
# ---------------------------------------------------------------------------

class LiqFeatures(FeatureBlock):
    """
    Liquidation pressure features for one (venue, side) pair.

    Filters the liquidation stream by venue and side once, then computes:
      - EWMA of liquidation notional (one per halflife)
      - Rolling count of liquidation events (one per count window)
      - Time since the last liquidation event (microseconds)

    All computations share the same filtered event arrays.
    DirectionRelativize converts buy/sell columns → same/opp later.

    Output columns (in order):
        liq_ewma_{venue}_{side}_{hl}s       per halflife in halflives_s
        liq_count_{venue}_{side}_{w}s       per window in count_windows_s
        liq_time_since_{venue}_{side}        if include_time_since=True
    """

    def __init__(
        self,
        venue:              str,
        side:               str,
        halflives_s:        tuple[float, ...] = (),
        count_windows_s:    tuple[float, ...] = (),
        include_time_since: bool              = True,
    ):
        assert venue in ('binance', 'bybit'), f"Unknown venue: {venue}"
        assert side  in ('buy', 'sell'),      f"Unknown side: {side}"
        self.venue              = venue
        self.side               = side
        self.halflives_s        = tuple(halflives_s)
        self.count_windows_s    = tuple(count_windows_s)
        self.include_time_since = include_time_since

    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        liq = kwargs.get(f'liq_{self.venue}')

        # Filter by side once — all helpers share these arrays
        if liq is not None and len(liq) > 0:
            mask       = liq['side'].values == self.side
            event_ts_f = liq['timestamp'].values[mask].astype(np.float64)   # for EWMA
            event_ts_i = liq['timestamp'].values[mask].astype(np.int64)     # for count/time_since
            event_vals = (liq['price'].values[mask] * liq['amount'].values[mask]).astype(np.float64)
        else:
            event_ts_f = np.array([], dtype=np.float64)
            event_ts_i = np.array([], dtype=np.int64)
            event_vals = np.array([], dtype=np.float64)

        query_ts_f = trades['timestamp'].values.astype(np.float64)
        query_ts_i = trades['timestamp'].values.astype(np.int64)

        out: dict[str, np.ndarray] = {}

        # --- EWMA (one merge pass per halflife) ---
        for hl_s in self.halflives_s:
            col   = f'liq_ewma_{self.venue}_{self.side}_{hl_s}s'
            hl_us = hl_s * 1_000_000.0
            out[col] = _ewma_event_state_at(query_ts_f, event_ts_f, event_vals, hl_us)

        # --- Count + time_since share the right boundary ---
        if self.count_windows_s or self.include_time_since:
            right = np.searchsorted(event_ts_i, query_ts_i, side='left')

            for w_s in self.count_windows_s:
                w_us = int(w_s * 1_000_000)
                left = np.searchsorted(event_ts_i, query_ts_i - w_us, side='left')
                out[f'liq_count_{self.venue}_{self.side}_{w_s}s'] = (right - left).astype(np.float64)

            if self.include_time_since:
                idx    = right - 1
                result = np.full(len(query_ts_i), np.nan)
                valid  = idx >= 0
                result[valid] = (query_ts_i[valid] - event_ts_i[idx[valid]]).astype(np.float64)
                out[f'liq_time_since_{self.venue}_{self.side}'] = result

        return pd.DataFrame(out, index=trades.index)

    @property
    def feature_names(self) -> list[str]:
        names = [f'liq_ewma_{self.venue}_{self.side}_{hl}s'   for hl in self.halflives_s]
        names += [f'liq_count_{self.venue}_{self.side}_{w}s'  for w  in self.count_windows_s]
        if self.include_time_since:
            names.append(f'liq_time_since_{self.venue}_{self.side}')
        return names
