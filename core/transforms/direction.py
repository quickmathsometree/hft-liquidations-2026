"""
Direction-relative feature transform.

Converts absolute BID/ASK and signed features into same-side / opposite-side
relative to the trade direction s_i (+1 taker buy, -1 taker sell).

Columns that are already direction-agnostic (spread_bps, vol, time, etc.)
pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.transforms.base import Transform


def _relativize_series(
    same_when_pos: pd.Series,
    same_when_neg: pd.Series,
    s: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """
    Given two absolute series (one for s=+1 side, one for s=-1 side),
    return (same_side, opp_side) relative to s.
    """
    same = pd.Series(
        np.where(s == 1, same_when_pos.values, same_when_neg.values),
        index=same_when_pos.index,
    )
    opp = pd.Series(
        np.where(s == 1, same_when_neg.values, same_when_pos.values),
        index=same_when_pos.index,
    )
    return same, opp


class DirectionRelativize(Transform):
    """
    Convert absolute features to direction-relative form using trade side.

    Rules applied (only to columns that are present):
      bid_amount, ask_amount
          → same_side_depth, opp_side_depth
      bid_amount_delta_*,  ask_amount_delta_*
          → same_side_depth_delta_*, opp_side_depth_delta_*
      signed_flow_*
          → same_side_flow_*  (= s * signed_flow)
      taker_imbalance_*
          → same_side_taker_imbalance_*  (= s * taker_imbalance)
      liq_*_{venue}_buy_* / liq_*_{venue}_buy
          → liq_*_{venue}_same_* / liq_*_{venue}_same
          (covers liq_ewma, liq_count, liq_time_since)

    All other columns pass through unchanged.
    """

    def apply(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        s   = trades['side'].map({'buy': 1, 'sell': -1}).values
        out = features.copy()

        # --- depth: bid_amount / ask_amount ---
        if 'bid_amount' in out.columns and 'ask_amount' in out.columns:
            # s=+1: taker bought → hit our ask → same side is ask
            same, opp = _relativize_series(out['ask_amount'], out['bid_amount'], s)
            out['same_side_depth'] = same
            out['opp_side_depth']  = opp
            out.drop(columns=['bid_amount', 'ask_amount'], inplace=True)

        # --- depth deltas ---
        bid_delta_cols = [c for c in out.columns if c.startswith('bid_amount_delta_')]
        ask_delta_cols = [c for c in out.columns if c.startswith('ask_amount_delta_')]
        for bid_col, ask_col in zip(sorted(bid_delta_cols), sorted(ask_delta_cols)):
            suffix = bid_col[len('bid_amount_delta_'):]
            same, opp = _relativize_series(out[ask_col], out[bid_col], s)
            out[f'same_side_depth_delta_{suffix}'] = same
            out[f'opp_side_depth_delta_{suffix}']  = opp
            out.drop(columns=[bid_col, ask_col], inplace=True)

        # --- signed flow ---
        for col in [c for c in out.columns if c.startswith('signed_flow_')]:
            suffix = col[len('signed_flow_'):]
            out[f'same_side_flow_{suffix}'] = s * out[col].values
            out.drop(columns=[col], inplace=True)

        # --- taker imbalance ---
        for col in [c for c in out.columns if c.startswith('taker_imbalance_')]:
            suffix = col[len('taker_imbalance_'):]
            out[f'same_side_taker_imbalance_{suffix}'] = s * out[col].values
            out.drop(columns=[col], inplace=True)

        # --- all liq features: buy/sell → same/opp ---
        # Handles _buy_ in the middle (liq_ewma_*, liq_count_*)
        # and _buy at the end (liq_time_since_*)
        buy_liq_cols = [
            c for c in out.columns
            if c.startswith('liq_') and ('_buy_' in c or c.endswith('_buy'))
        ]
        for buy_col in buy_liq_cols:
            if '_buy_' in buy_col:
                sell_col = buy_col.replace('_buy_', '_sell_')
                same_col = buy_col.replace('_buy_', '_same_')
                opp_col  = buy_col.replace('_buy_', '_opp_')
            else:                                    # ends with _buy
                stem     = buy_col[:-4]
                sell_col = stem + '_sell'
                same_col = stem + '_same'
                opp_col  = stem + '_opp'
            if sell_col not in out.columns:
                continue
            # s=+1 taker bought → maker on sell side → sell liqs are same-side pressure
            same, opp = _relativize_series(out[sell_col], out[buy_col], s)
            out[same_col] = same
            out[opp_col]  = opp
            out.drop(columns=[buy_col, sell_col], inplace=True)

        return out
