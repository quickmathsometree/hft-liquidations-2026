"""
Maker PnL computation in basis points.

Takes trades with markout columns (from compute_markout) and adds:
  s_i, notional, clipped weight w_i, and pnl_i(tau) for each horizon.
"""

from __future__ import annotations

import pandas as pd

NOTIONAL_CLIP: float = 100_000.0
MAKER_REBATE_BPS: float = 0.5
TAUS: tuple[int, ...] = (30, 120, 300)


def compute_pnl(
    trades: pd.DataFrame,
    taus: tuple[int, ...] = TAUS,
) -> pd.DataFrame:
    """
    Compute maker PnL and weights. Requires mid_{tau} columns from compute_markout.

    Adds columns:
        s         = +1 if side == 'buy' (taker buy, maker sell), else -1
        notional  = price * amount
        w         = min(notional, NOTIONAL_CLIP)
        pnl_{tau} = -s * (mid_{tau} - price) / price * 10_000 + MAKER_REBATE_BPS
                    NaN where edge_{tau} == True

    Example: price=100000, side='buy' => s=+1.
        mid_30=100050 => pnl_30 = -(+1)*(50/100000)*10000 + 0.5 = -4.5 bps (bad fill)
        mid_30= 99950 => pnl_30 = -(+1)*(-50/100000)*10000 + 0.5 = +5.5 bps (good fill)
    """
    trades = trades.copy()
    trades["s"]        = trades["side"].map({"buy": 1, "sell": -1}).astype("float64")
    trades["notional"] = trades["price"] * trades["amount"]
    trades["w"]        = trades["notional"].clip(upper=NOTIONAL_CLIP)

    for tau in taus:
        mid = trades[f"mid_{tau}"]
        pnl = -trades["s"] * (mid - trades["price"]) / trades["price"] * 10_000 + MAKER_REBATE_BPS
        pnl[trades[f"edge_{tau}"]] = float("nan")
        trades[f"pnl_{tau}"] = pnl

    return trades
