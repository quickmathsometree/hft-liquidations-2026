"""
Final submission function — called on the hidden test.

make_filter() is self-contained: it embeds pre-trained model/params
and calibrates the threshold on the test data via turnover (label-free).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import BYBIT_LAG_US, US_PER_SECOND, _prepare_frame, compute_num_days, detect_symbol
from core.features.book import compute_book_features
from core.features.flow import compute_flow_features
from core.features.liq import compute_liq_features
from core.features.time import compute_time_features
from core.features.vol import compute_vol_features
from core.model import predict
from core.targets.markout import add_mid
from core.targets.pnl import NOTIONAL_CLIP
from core.transforms.direction import direction_relativize
from core.transforms.normalize import fill_nan_inf

from strategies.liq_filter.config import TAUS, FittedPipeline
from strategies.liq_filter.strategy import strategy_raw_score
from strategies.liq_filter.threshold import fit_threshold, apply_filter


# Populated by run_train() before submission. Baked into module scope.
FITTED: FittedPipeline | None = None


def _side_to_s(trades: pd.DataFrame) -> np.ndarray:
    side = trades["side"].astype(str).str.lower()
    return np.where(side.eq("buy"), 1, -1).astype("int8")


def _build_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
    fitted: FittedPipeline,
) -> pd.DataFrame:
    cfg = fitted.feature_config

    X = pd.concat(
        [
            compute_liq_features(trades, liq_binance, liq_bybit, cfg.liq_halflives_s),
            compute_book_features(trades, bbo, cfg.book_windows_s),
            compute_flow_features(trades, cfg.flow_windows_s),
            compute_time_features(trades),
            compute_vol_features(trades, bbo),
        ],
        axis=1,
    )

    X = direction_relativize(X, _side_to_s(trades))
    X = fill_nan_inf(X)
    return X


def make_filter(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """
    FINAL submission function.

    Accepts 4 frames (same schemas as public files; liq_bybit arrives UNSHIFTED).
    Returns { 30: arr_30, 120: arr_120, 300: arr_300 },
    each arr is np.ndarray of length len(trades) with values 0 or 1.
    """
    if FITTED is None:
        raise RuntimeError("FITTED pipeline not set. Call run_train() first.")

    symbol = detect_symbol(trades)

    trades = _prepare_frame(trades)
    bbo = _prepare_frame(bbo)
    liq_binance = _prepare_frame(liq_binance)
    liq_bybit = _prepare_frame(liq_bybit)

    liq_bybit = liq_bybit.copy()
    liq_bybit["timestamp"] = liq_bybit["timestamp"].astype("int64") + BYBIT_LAG_US

    bbo = add_mid(bbo)

    X = _build_features(trades, bbo, liq_binance, liq_bybit, FITTED)
    scored = trades.loc[X.index].copy()

    w = (
        scored["price"].astype(float).to_numpy()
        * scored["amount"].astype(float).to_numpy()
    )
    w = np.minimum(np.nan_to_num(w, nan=0.0), NOTIONAL_CLIP)

    num_days = compute_num_days(scored)

    out: dict[int, np.ndarray] = {}

    max_bbo_ts = int(bbo["timestamp"].max()) if len(bbo) else -1
    trade_ts = scored["timestamp"].to_numpy(dtype="int64")

    for tau in TAUS:
        key = (symbol, int(tau))

        if FITTED.use_ml:
            model = FITTED.models.get(key)
            if model is None:
                raise ValueError(f"No fitted model for {key}")
            raw_score = predict(model, X)
        else:
            raw_score = strategy_raw_score(X, scored)

        threshold = fit_threshold(
            raw_score=raw_score,
            w=w,
            num_days=num_days,
            target_turnover_per_day=FITTED.target_turnover_per_day,
        )

        edge_mask = trade_ts + int(tau) * US_PER_SECOND > max_bbo_ts

        out[int(tau)] = apply_filter(
            raw_score=raw_score,
            threshold=threshold,
            edge_mask=edge_mask,
        )

    return out