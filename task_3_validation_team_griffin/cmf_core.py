"""
cmf_core
========

Shared, correctness-critical core library for the CMF HFT "liquidation filter"
project (Team Griffin). Used by all three task notebooks (EDA, Classes,
Validation) so that *every* number in the submission comes from one audited
implementation of the official spec.

The official spec is transcribed faithfully from the task skeleton
(github.com/cmf-team/hft-liquidations-2026):

    * Timestamps                int64 microseconds since UNIX epoch (UTC).
    * mid          = (bid_price + ask_price) / 2
    * microprice   = (bid_price*ask_amount + ask_price*bid_amount) / (bid_amount + ask_amount)
    * markout      mid_{tau} = forward-fill mid at t_i + tau   (last BBO mid with ts <= t_i+tau)
                   edge_{tau} = True where t_i+tau > max(bbo.ts)  -> excluded from scoring
    * s            = +1 if taker side == 'buy' (taker buy => maker SELL), else -1
    * notional     = price * amount
    * w            = min(notional, NOTIONAL_CLIP=100_000)
    * pnl_{tau}    = -s * (mid_{tau} - price) / price * 10_000 + MAKER_REBATE_BPS(0.5)   [bps]
    * Score(tau)   = PnL_kept(tau) - PnL_all(tau)
        PnL_all      = sum(w*pnl)       / sum(w)
        PnL_kept     = sum((1-f)*w*pnl) / sum((1-f)*w)
        PnL_filtered = sum(f*w*pnl)     / sum(f*w)
      where f_i = 1 means the trade is FILTERED OUT (removed); kept = (1 - f).
    * Constraint   kept turnover/day >= TURNOVER_FLOOR_PER_DAY (500_000) [the maker
                   must keep trading enough to matter].
    * Bybit liquidation timestamps are shifted by +200 ms before use.
    * Splits       train = [2025-12-01, 2026-02-01), validation = [2026-02-01, 2026-03-01).

All feature primitives are STRICTLY CAUSAL: a feature for trade i at time t_i is a
function of data with timestamp < t_i only (or <= t_i for the trade's own fields).
markout/pnl use FUTURE mids and are therefore TARGETS only, never features.

Author: Team Griffin
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

#  Official constants
US_PER_SECOND: int = 1_000_000
TAUS: tuple[int, ...] = (30, 120, 300)
NOTIONAL_CLIP: float = 100_000.0
MAKER_REBATE_BPS: float = 0.5
TURNOVER_FLOOR_PER_DAY: float = 500_000.0
BYBIT_LAG_US: int = 200_000  # +200 ms

SPLIT_RANGES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train":      (pd.Timestamp("2025-12-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")),
    "validation": (pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
}

# Binance USD-M perpetual funding occurs every 8h at 00/08/16 UTC.
FUNDING_HOURS_UTC: tuple[int, ...] = (0, 8, 16)

#  Column normalisation / data loading
# The raw parquet files may carry slightly different column names depending on
# the exporter. We normalise to a canonical schema here. The canonical schema is:
#   trades : timestamp, price, amount, side, ticker
#   bbo    : timestamp, bid_price, ask_price, bid_amount, ask_amount, [ticker]
#   liq    : timestamp, price, amount, side, ticker
_RENAME = {
    # timestamps
    "ts": "timestamp", "time": "timestamp", "timestamp_us": "timestamp",
    "transact_time": "timestamp", "trade_time": "timestamp", "T": "timestamp",
    "local_timestamp": "timestamp_local",
    # prices
    "p": "price", "px": "price",
    "bid": "bid_price", "best_bid": "bid_price", "bid_px": "bid_price", "b": "bid_price",
    "ask": "ask_price", "best_ask": "ask_price", "ask_px": "ask_price", "a": "ask_price",
    # sizes
    "q": "amount", "qty": "amount", "quantity": "amount", "size": "amount", "vol": "amount",
    "bid_qty": "bid_amount", "bid_size": "bid_amount", "bid_amt": "bid_amount", "B": "bid_amount",
    "ask_qty": "ask_amount", "ask_size": "ask_amount", "ask_amt": "ask_amount", "A": "ask_amount",
    # side / symbol
    "is_buyer_maker": "is_buyer_maker", "m": "is_buyer_maker",
    "symbol": "ticker", "instrument": "ticker", "pair": "ticker",
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _RENAME.get(c, c) for c in df.columns})
    # de-duplicate accidental collisions
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _to_us(ts: pd.Series) -> pd.Series:
    """Cast a timestamp column to int64 microseconds since epoch (UTC)."""
    if pd.api.types.is_datetime64_any_dtype(ts):     # handles tz-aware & tz-naive
        return (ts.view("int64") // 1_000).astype("int64")
    ts = pd.to_numeric(ts, errors="coerce")
    # Heuristic: detect the unit by magnitude of the median value.
    med = float(np.nanmedian(ts.values)) if len(ts) else 0.0
    if med == 0:
        return ts.astype("int64")
    digits = int(np.floor(np.log10(abs(med)))) + 1
    if digits <= 11:        # seconds (~1.7e9)
        factor = 1_000_000
    elif digits <= 14:      # milliseconds (~1.7e12)
        factor = 1_000
    elif digits <= 17:      # microseconds (~1.7e15)
        factor = 1
    else:                   # nanoseconds (~1.7e18)
        factor = 1 / 1_000
    return (ts * factor).round().astype("int64")

def _normalise_side(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a lower-case 'side' column in {'buy','sell'} (taker/aggressor side)."""
    if "side" in df.columns:
        s = df["side"].astype(str).str.lower().str.strip()
        s = s.replace({"b": "buy", "bid": "buy", "buyer": "buy", "long": "buy",
                       "a": "sell", "ask": "sell", "seller": "sell", "short": "sell",
                       "1": "buy", "0": "sell", "true": "buy", "false": "sell"})
        df["side"] = s
    elif "is_buyer_maker" in df.columns:
        # Binance aggTrades: is_buyer_maker=True => the buyer is the maker =>
        # the aggressor (taker) is the SELLER => taker side = 'sell'.
        ibm = df["is_buyer_maker"]
        if ibm.dtype == object:
            ibm = ibm.astype(str).str.lower().isin(["true", "1", "t", "yes"])
        df["side"] = np.where(ibm.astype(bool), "sell", "buy")
    return df

def read_parquet_dir(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a single parquet file OR a directory of parquet parts and concat."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        frames = [pd.read_parquet(f, columns=columns) for f in files]
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, columns=columns)

def prepare_frame(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Normalise a raw parquet frame to the canonical schema, cast timestamp to
    int64 microseconds, lower-case string columns, and sort by timestamp.
    `kind` in {'trades','bbo','liq'}. Idempotent.
    """
    df = _norm_cols(df).copy()
    if "timestamp" not in df.columns:
        # fall back: first datetime-like or *time* column
        cand = [c for c in df.columns if "time" in c.lower() or "ts" in c.lower()]
        if cand:
            df = df.rename(columns={cand[0]: "timestamp"})
    df["timestamp"] = _to_us(df["timestamp"])
    if kind in ("trades", "liq"):
        df = _normalise_side(df)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.lower()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df

@dataclass
class MarketData:
    """Bundle of the four causal frames for one symbol (Bybit already shifted)."""
    symbol: str
    trades: pd.DataFrame
    bbo: pd.DataFrame
    liq_binance: pd.DataFrame
    liq_bybit: pd.DataFrame

def _filter_range(df: pd.DataFrame, start_us: int, end_us: int) -> pd.DataFrame:
    m = (df["timestamp"] >= start_us) & (df["timestamp"] < end_us)
    return df.loc[m].reset_index(drop=True)

def load_symbol(
    paths: dict[str, str],
    symbol: str,
    split: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    liq_lookback_us: int = 5 * 60 * US_PER_SECOND,
) -> MarketData:
    """
    Load + preprocess the 4 frames for one symbol.

    `paths` maps {'trades','bbo','liq_binance','liq_bybit'} -> file/dir path.
    Range selection: pass either `split` (in SPLIT_RANGES) or explicit start/end.
    Liquidation frames include a `liq_lookback_us` buffer before the start so
    causal rolling/EWMA features do not cold-start at the split boundary.
    Bybit liquidation timestamps are shifted by +200 ms (after range filtering of
    the *real* range, before sorting).
    """
    trades = prepare_frame(read_parquet_dir(paths["trades"]), "trades")
    bbo = prepare_frame(read_parquet_dir(paths["bbo"]), "bbo")
    liq_b = prepare_frame(read_parquet_dir(paths["liq_binance"]), "liq")
    liq_y = prepare_frame(read_parquet_dir(paths["liq_bybit"]), "liq")

    # range
    if split is not None:
        start, end = SPLIT_RANGES[split]
    if start is not None or end is not None:
        s_us = int(start.value // 1_000) if start is not None else np.iinfo(np.int64).min
        e_us = int(end.value // 1_000) if end is not None else np.iinfo(np.int64).max
        trades = _filter_range(trades, s_us, e_us)
        bbo = _filter_range(bbo, s_us, e_us)
        # liq frames keep a lookback buffer
        liq_b = _filter_range(liq_b, s_us - liq_lookback_us, e_us)
        liq_y = _filter_range(liq_y, s_us - liq_lookback_us, e_us)

    # shift Bybit AFTER range filtering (matches official ordering: filter then shift)
    liq_y = liq_y.copy()
    liq_y["timestamp"] = liq_y["timestamp"] + BYBIT_LAG_US
    liq_y = liq_y.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    return MarketData(symbol, trades, bbo, liq_b, liq_y)

def compute_num_days(trades: pd.DataFrame) -> float:
    """Number of distinct UTC calendar dates spanned by `trades`."""
    days = (trades["timestamp"].to_numpy() // (86_400 * US_PER_SECOND))
    return float(np.unique(days).size)

def to_datetime(ts_us: np.ndarray | pd.Series) -> pd.DatetimeIndex:
    """int64 microseconds -> tz-aware UTC DatetimeIndex (for plotting/grouping)."""
    return pd.to_datetime(np.asarray(ts_us), unit="us", utc=True)

#  Mid / microprice / markout / PnL  (official targets)
def add_mid(bbo: pd.DataFrame) -> pd.DataFrame:
    """Add `mid` and `microprice` columns to a BBO frame (official formulas)."""
    bbo = bbo.copy()
    bp = bbo["bid_price"].to_numpy(float)
    ap = bbo["ask_price"].to_numpy(float)
    ba = bbo["bid_amount"].to_numpy(float)
    aa = bbo["ask_amount"].to_numpy(float)
    bbo["mid"] = (bp + ap) / 2.0
    denom = ba + aa
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = (bp * aa + ap * ba) / denom
    micro = np.where(denom > 0, micro, bbo["mid"].to_numpy())
    bbo["microprice"] = micro
    return bbo

def _asof_backward(query_ts: np.ndarray, ref_ts: np.ndarray, ref_val: np.ndarray) -> np.ndarray:
    """Value of ref at the last ref_ts <= query_ts (forward-fill). NaN if none."""
    idx = np.searchsorted(ref_ts, query_ts, side="right") - 1
    out = np.where(idx >= 0, ref_val[np.clip(idx, 0, len(ref_val) - 1)], np.nan)
    return out

def _asof_strict_before(query_ts: np.ndarray, ref_ts: np.ndarray, ref_val: np.ndarray) -> np.ndarray:
    """Value of ref at the last ref_ts STRICTLY < query_ts. NaN if none. (causal)"""
    idx = np.searchsorted(ref_ts, query_ts, side="left") - 1
    out = np.where(idx >= 0, ref_val[np.clip(idx, 0, len(ref_val) - 1)], np.nan)
    return out

def compute_markout(trades: pd.DataFrame, bbo: pd.DataFrame,
                    taus: tuple[int, ...] = TAUS) -> pd.DataFrame:
    """
    For each trade and tau, forward-fill mid at t_i + tau.
    Adds mid_{tau} (NaN on right-edge) and edge_{tau} (bool) to a COPY of trades.
    Requires `mid` on bbo (call add_mid first).
    """
    trades = trades.copy()
    bbo_ts = bbo["timestamp"].to_numpy()
    bbo_mid = bbo["mid"].to_numpy(float)
    t = trades["timestamp"].to_numpy()
    max_bbo = bbo_ts[-1] if len(bbo_ts) else -1
    for tau in taus:
        look = t + tau * US_PER_SECOND
        mid = _asof_backward(look, bbo_ts, bbo_mid)
        edge = look > max_bbo
        mid = np.where(edge, np.nan, mid)
        trades[f"mid_{tau}"] = mid
        trades[f"edge_{tau}"] = edge
    return trades

def compute_pnl(trades: pd.DataFrame, taus: tuple[int, ...] = TAUS) -> pd.DataFrame:
    """
    Maker markout-PnL in bps. Requires mid_{tau}/edge_{tau} (compute_markout).
    Adds s, notional, w, pnl_{tau}.
    """
    trades = trades.copy()
    side = trades["side"].astype(str).str.lower().to_numpy()
    s = np.where(side == "buy", 1.0, -1.0)
    price = trades["price"].to_numpy(float)
    amount = trades["amount"].to_numpy(float)
    notional = price * amount
    trades["s"] = s.astype(np.int8)
    trades["notional"] = notional
    trades["w"] = np.minimum(notional, NOTIONAL_CLIP)
    for tau in taus:
        mid = trades[f"mid_{tau}"].to_numpy(float)
        pnl = -s * (mid - price) / price * 10_000.0 + MAKER_REBATE_BPS
        pnl = np.where(trades[f"edge_{tau}"].to_numpy(bool), np.nan, pnl)
        trades[f"pnl_{tau}"] = pnl
    return trades

def build_targets(md: MarketData, taus: tuple[int, ...] = TAUS) -> pd.DataFrame:
    """Convenience: add_mid -> compute_markout -> compute_pnl on md.trades."""
    bbo = add_mid(md.bbo)
    tr = compute_markout(md.trades, bbo, taus)
    tr = compute_pnl(tr, taus)
    return tr

#  Official scoring
@dataclass
class ScoreReport:
    tau: int
    score: float
    pnl_all: float
    pnl_kept: float
    pnl_filtered: float
    kept_turnover_per_day: float
    filtered_turnover_per_day: float
    constraint_ok: bool
    n_trades: int
    n_valid: int
    n_kept: int
    n_filtered: int
    n_edge: int

def _wmean(w: np.ndarray, x: np.ndarray) -> float:
    sw = w.sum()
    return float((w * x).sum() / sw) if sw > 0 else np.nan

def score_one(pnl: np.ndarray, w: np.ndarray, f: np.ndarray,
              num_days: float, tau: int,
              turnover_floor: float = TURNOVER_FLOOR_PER_DAY) -> ScoreReport:
    """
    Official score for one tau. `f` is the FILTER-OUT indicator (1=remove,0=keep).
    Edge trades must already be set to f=0 and have pnl=NaN; they are dropped from
    the weighted averages (valid = ~isnan(pnl)). Turnover constraint applies to the
    KEPT set.
    """
    pnl = np.asarray(pnl, float)
    w = np.asarray(w, float)
    f = np.asarray(f, float)
    valid = ~np.isnan(pnl)
    n_trades = len(pnl)
    n_edge = int((~valid).sum())

    pv, wv, fv = pnl[valid], w[valid], f[valid]
    keep = (1.0 - fv)
    pnl_all = _wmean(wv, pv)
    pnl_kept = _wmean(wv * keep, pv)
    pnl_filtered = _wmean(wv * fv, pv)
    score = pnl_kept - pnl_all

    kept_to = float((wv * keep).sum() / num_days) if num_days > 0 else np.nan
    filt_to = float((wv * fv).sum() / num_days) if num_days > 0 else np.nan
    constraint_ok = bool(kept_to >= turnover_floor)

    return ScoreReport(
        tau=tau, score=score, pnl_all=pnl_all, pnl_kept=pnl_kept,
        pnl_filtered=pnl_filtered, kept_turnover_per_day=kept_to,
        filtered_turnover_per_day=filt_to, constraint_ok=constraint_ok,
        n_trades=n_trades, n_valid=int(valid.sum()),
        n_kept=int((keep > 0).sum()), n_filtered=int((fv > 0).sum()), n_edge=n_edge,
    )

def score_all(trades_with_pnl: pd.DataFrame, f_by_tau: dict[int, np.ndarray],
              num_days: float, taus: tuple[int, ...] = TAUS) -> dict[int, ScoreReport]:
    out = {}
    w = trades_with_pnl["w"].to_numpy(float)
    for tau in taus:
        pnl = trades_with_pnl[f"pnl_{tau}"].to_numpy(float)
        f = np.asarray(f_by_tau[tau], float)
        out[tau] = score_one(pnl, w, f, num_days, tau)
    return out

def reports_to_frame(reports: dict[int, ScoreReport], symbol: str = "",
                     experiment: str = "") -> pd.DataFrame:
    rows = []
    for tau, r in reports.items():
        d = r.__dict__.copy()
        d["symbol"] = symbol
        d["experiment"] = experiment
        rows.append(d)
    cols = ["experiment", "symbol", "tau", "score", "pnl_all", "pnl_kept",
            "pnl_filtered", "kept_turnover_per_day", "filtered_turnover_per_day",
            "constraint_ok", "n_trades", "n_valid", "n_kept", "n_filtered", "n_edge"]
    return pd.DataFrame(rows)[cols]

#  Threshold calibration under the turnover constraint (label-free)
def fit_threshold(raw_score: np.ndarray, w: np.ndarray, num_days: float,
                  target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY) -> float:
    """
    Pick t* so that kept trades (raw_score >= t*) have daily turnover >= target.
    Label-free: uses only w. Returns -inf if even keeping everything misses target.
    """
    raw_score = np.asarray(raw_score, float)
    w = np.asarray(w, float)
    order = np.argsort(-raw_score, kind="mergesort")  # descending
    s_sorted = raw_score[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted) / num_days
    hit = np.searchsorted(cum, target_turnover_per_day, side="left")
    if hit >= len(s_sorted):
        return -np.inf  # keep everything
    return float(s_sorted[hit])

def fit_threshold_fast(raw_score: np.ndarray, w: np.ndarray, num_days: float,
                       target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
                       topk: int = 60_000) -> float:
    """
    Exact-equivalent of fit_threshold but avoids a full argsort when the kept set
    is tiny (the usual case: floor is ~0.004% of turnover). Takes the top-`topk`
    trades by raw_score via argpartition (O(n)), sorts only those, and locates the
    marginal threshold. Falls back to the full sort if topk does not reach target.
    """
    raw_score = np.asarray(raw_score, float)
    w = np.asarray(w, float)
    n = len(raw_score)
    target = target_turnover_per_day * num_days
    if n == 0:
        return -np.inf
    if topk >= n:
        return fit_threshold(raw_score, w, num_days, target_turnover_per_day)
    part = np.argpartition(-raw_score, topk)[:topk]
    order = part[np.argsort(-raw_score[part], kind="mergesort")]
    cum = np.cumsum(w[order])
    if cum[-1] < target:                      # tiny topk didn't reach floor -> exact path
        return fit_threshold(raw_score, w, num_days, target_turnover_per_day)
    hit = int(np.searchsorted(cum, target, side="left"))
    return float(raw_score[order[hit]])

def apply_filter(raw_score: np.ndarray, threshold: float,
                 edge_mask: np.ndarray | None = None) -> np.ndarray:
    """f_i = 1 (remove) if raw_score < threshold else 0 (keep). Edge -> 0."""
    raw_score = np.asarray(raw_score, float)
    f = (raw_score < threshold).astype(np.int8)
    if edge_mask is not None:
        f = np.where(np.asarray(edge_mask, bool), 0, f).astype(np.int8)
    return f

def filter_from_score(raw_score: np.ndarray, w: np.ndarray, num_days: float,
                      edge_mask: np.ndarray | None = None,
                      target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY) -> np.ndarray:
    """Convenience: fit_threshold then apply_filter. Higher raw_score = keep."""
    t = fit_threshold(raw_score, w, num_days, target_turnover_per_day)
    return apply_filter(raw_score, t, edge_mask)

#  Causal feature primitives  (vectorised, strictly < query time)
def causal_window_sum(query_ts: np.ndarray, event_ts: np.ndarray,
                      event_vals: np.ndarray, window_us: int) -> np.ndarray:
    """Sum of event_vals for events in [query-window, query) (strictly before)."""
    query_ts = np.asarray(query_ts)
    if len(event_ts) == 0:
        return np.zeros(len(query_ts))
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(event_vals, float))])
    hi = np.searchsorted(event_ts, query_ts, side="left")          # strictly <
    lo = np.searchsorted(event_ts, query_ts - window_us, side="left")
    return cs[hi] - cs[lo]

def causal_window_count(query_ts: np.ndarray, event_ts: np.ndarray,
                        window_us: int) -> np.ndarray:
    """Count of events in [query-window, query) (strictly before query)."""
    query_ts = np.asarray(query_ts)
    if len(event_ts) == 0:
        return np.zeros(len(query_ts))
    hi = np.searchsorted(event_ts, query_ts, side="left")
    lo = np.searchsorted(event_ts, query_ts - window_us, side="left")
    return (hi - lo).astype(float)

def causal_time_since(query_ts: np.ndarray, event_ts: np.ndarray) -> np.ndarray:
    """Microseconds since last event strictly before query. inf if none."""
    query_ts = np.asarray(query_ts)
    if len(event_ts) == 0:
        return np.full(len(query_ts), np.inf)
    idx = np.searchsorted(event_ts, query_ts, side="left") - 1
    out = np.where(idx >= 0, query_ts - event_ts[np.clip(idx, 0, len(event_ts) - 1)], np.inf)
    return out.astype(float)

def causal_decaying_sum_at(query_ts: np.ndarray, event_ts: np.ndarray,
                           event_vals: np.ndarray, halflife_us: float,
                           prev_idx: np.ndarray | None = None,
                           post_state: np.ndarray | None = None) -> np.ndarray:
    """
    FAST event-anchored EWMA decaying sum, exact and strictly causal.

    Identical result to causal_decaying_sum but O(n_events) loop + O(n_query)
    vectorised lookup - ideal when events (e.g. liquidations) are SPARSE relative
    to queries (trades). The EWMA state only changes at event times; between events
    it decays deterministically, so we:
      1. compute the post-event state S_e at each event (small loop over events),
      2. for each query, find the last event strictly before it and decay S_e to
         the query time:  state(q) = S_e[k] * exp(-(q - t_k)/tau).
    """
    query_ts = np.asarray(query_ts, np.int64)
    event_ts = np.asarray(event_ts, np.int64)
    event_vals = np.asarray(event_vals, float)
    nq = len(query_ts)
    ne = len(event_ts)
    if ne == 0:
        return np.zeros(nq)
    tau = halflife_us / np.log(2.0)
    # post-event states (state immediately AFTER adding event k)
    if post_state is not None:
        post = post_state
    else:
        post = np.empty(ne)
        state = 0.0
        last_t = event_ts[0]
        for k in range(ne):
            dt = event_ts[k] - last_t
            if dt > 0:
                state *= np.exp(-dt / tau)
                last_t = event_ts[k]
            state += event_vals[k]
            post[k] = state
    # for each query, last event strictly before it
    if prev_idx is not None:
        idx = prev_idx
    else:
        idx = np.searchsorted(event_ts, query_ts, side="left") - 1
    has = idx >= 0
    out = np.zeros(nq)
    ii = idx[has]
    dt = (query_ts[has] - event_ts[ii]).astype(float)
    out[has] = post[ii] * np.exp(-dt / tau)
    return out

def causal_decaying_sum(query_ts: np.ndarray, event_ts: np.ndarray,
                        event_vals: np.ndarray, halflife_us: float) -> np.ndarray:
    """
    Time-decayed (EWMA-style) sum of event_vals using only events strictly before
    each query: state(q) = sum_{t_j < q} v_j * exp(-(q - t_j)/tau), tau=halflife/ln2.

    Numerically-stable, fully vectorised: processes the merged timeline in blocks
    of bounded log-range so exp() never overflows. O((n+m) log(n+m)).
    """
    query_ts = np.asarray(query_ts, np.int64)
    event_ts = np.asarray(event_ts, np.int64)
    event_vals = np.asarray(event_vals, float)
    nq = len(query_ts)
    if len(event_ts) == 0:
        return np.zeros(nq)
    tau = halflife_us / np.log(2.0)

    # Merge events and queries. At equal timestamps queries sort BEFORE events
    # (kind 0 < kind 1) so a query never sees a simultaneous event (strict <).
    times = np.concatenate([query_ts, event_ts])
    kind = np.concatenate([np.zeros(nq, np.int8), np.ones(len(event_ts), np.int8)])
    vals = np.concatenate([np.zeros(nq), event_vals])
    qpos = np.concatenate([np.arange(nq), -np.ones(len(event_ts), np.int64)])
    order = np.lexsort((kind, times))   # primary: time, secondary: kind
    times, kind, vals, qpos = times[order], kind[order], vals[order], qpos[order]

    out = np.zeros(nq)
    state = 0.0
    last_t = times[0]
    # iterate in blocks where (t - block_ref)/tau stays < 700 to avoid overflow.
    # We simply walk; the recurrence is sequential but cheap (one multiply/add).
    # Vectorisation across the whole array is impossible (varying decay + resets),
    # so we use a tight numpy-scalar loop which is fine for event streams.
    n = len(times)
    for i in range(n):
        dt = times[i] - last_t
        if dt > 0:
            state *= np.exp(-dt / tau)
            last_t = times[i]
        if kind[i] == 0:        # query: record state of events strictly before
            out[qpos[i]] = state
        else:                   # event: add to state
            state += vals[i]
    return out

#  Statistical validation utilities
def daily_score_series(trades: pd.DataFrame, f: np.ndarray, tau: int) -> pd.DataFrame:
    """
    Per-UTC-day decomposition of the Score for a filter f at horizon tau.
    Returns a frame indexed by date with columns:
        pnl_all, pnl_kept, score, kept_w, filt_w  (weighted, valid trades only)
    This is the unit of analysis for block bootstrap / DM / paired tests.
    """
    pnl = trades[f"pnl_{tau}"].to_numpy(float)
    w = trades["w"].to_numpy(float)
    f = np.asarray(f, float)
    valid = ~np.isnan(pnl)
    day = (trades["timestamp"].to_numpy() // (86_400 * US_PER_SECOND))
    df = pd.DataFrame({
        "day": day[valid], "pnl": pnl[valid], "w": w[valid], "f": f[valid],
    })
    g = df.groupby("day")
    keep = 1.0 - df["f"]
    df["wk"] = df["w"] * keep
    df["wf"] = df["w"] * df["f"]
    df["wp"] = df["w"] * df["pnl"]
    df["wkp"] = df["wk"] * df["pnl"]
    g = df.groupby("day")
    agg = g.agg(W=("w", "sum"), WK=("wk", "sum"), WF=("wf", "sum"),
                WP=("wp", "sum"), WKP=("wkp", "sum"))
    agg["pnl_all"] = agg["WP"] / agg["W"]
    agg["pnl_kept"] = agg["WKP"] / agg["WK"]
    agg["score"] = agg["pnl_kept"] - agg["pnl_all"]
    agg["kept_w"] = agg["WK"]
    agg["filt_w"] = agg["WF"]
    agg.index = pd.to_datetime(agg.index * 86_400 * US_PER_SECOND, unit="us", utc=True)
    return agg[["pnl_all", "pnl_kept", "score", "kept_w", "filt_w"]]

def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator,
                                 n_boot: int) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap: returns (n_boot, n) index matrix.
    Geometric block lengths with mean `mean_block` (p = 1/mean_block).
    """
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            # geometric run length
            L = rng.geometric(p)
            for k in range(L):
                if i >= n:
                    break
                idx[i] = (start + k) % n
                i += 1
        out[b] = idx
    return out

def block_bootstrap_mean_ci(x: np.ndarray, mean_block: float = 5.0,
                            n_boot: int = 5000, alpha: float = 0.05,
                            seed: int = 0) -> dict:
    """
    Stationary-bootstrap CI and p-value for H0: E[x] <= 0 (one-sided, x = per-day
    score or score-difference). Returns mean, CI, and bootstrap p-value.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    if n == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p_value=np.nan, n=0)
    idx = stationary_bootstrap_indices(n, mean_block, rng, n_boot)
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # one-sided p for mean>0: fraction of bootstrap means <= 0 around observed,
    # use the centred distribution (Hall): p = P(boot_mean - mean <= -mean)
    obs = x.mean()
    centred = means - obs
    p_value = float(np.mean(centred <= -obs)) if obs > 0 else float(np.mean(centred >= -obs))
    return dict(mean=float(obs), lo=float(lo), hi=float(hi),
                p_value=float(p_value), n=n, boot_means=means)

def diebold_mariano(d: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano test on a loss-differential series d_t (e.g. per-day score
    difference candidate - baseline). H0: E[d]=0. HAC (Newey-West) variance with
    lag h-1 (>=0). Returns DM stat, two-sided p (normal), and one-sided p (>0).
    Uses the Harvey-Leybourne-Newbold small-sample correction.
    """
    from scipy import stats
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return dict(dm=np.nan, p_two=np.nan, p_greater=np.nan, mean=np.nan, n=n)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    L = max(h - 1, 0)
    for k in range(1, L + 1):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / (L + 1)) * gk
    if not np.isfinite(var) or var <= 0:          # identical/degenerate series
        return dict(dm=np.nan, p_two=np.nan, p_greater=np.nan, mean=float(dbar), n=n)
    dm = dbar / np.sqrt(var / n)
    # HLN correction
    corr = np.sqrt((n + 1 - 2 * (L + 1) + (L + 1) * L / n) / n)
    dm_hln = dm * corr if np.isfinite(corr) and corr > 0 else dm
    p_two = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    p_greater = 1 - stats.t.cdf(dm_hln, df=n - 1)
    return dict(dm=float(dm_hln), p_two=float(p_two), p_greater=float(p_greater),
                mean=float(dbar), n=n)

def deflated_sharpe_pvalue(daily: np.ndarray, n_trials: int) -> dict:
    """
    Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado). Given a per-period
    return series, compute SR, its variance accounting for skew/kurtosis, and the
    p-value that the *best of n_trials* SRs exceeds 0 by chance (multiple-testing
    deflation). daily = per-day score series (the 'return' of the filter).
    """
    from scipy import stats
    x = np.asarray(daily, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return dict(sr=np.nan, psr=np.nan, dsr=np.nan, n=n)
    sr = x.mean() / x.std(ddof=1)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)  # non-excess
    # expected max SR under n_trials independent null trials (variance of SR est ~1)
    emc = 0.5772156649
    e_max = (np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(n_trials)))) if n_trials > 1 else 0.0
    sr0 = e_max / np.sqrt(n)            # threshold SR per-period scale (approx)
    sr_std = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1))
    psr = stats.norm.cdf((sr - 0.0) / sr_std)          # PSR vs 0
    dsr = stats.norm.cdf((sr - sr0) / sr_std)          # deflated vs expected-max
    return dict(sr=float(sr), psr=float(psr), dsr=float(dsr),
                sr_threshold=float(sr0), n=n)

#  Purged / embargoed CV split generators (Lopez de Prado)
def walk_forward_splits(days: np.ndarray, n_splits: int = 6, min_train: int = 10):
    """Expanding-window walk-forward: yield (train_day_idx, test_day_idx)."""
    udays = np.unique(days)
    nd = len(udays)
    fold = max((nd - min_train) // n_splits, 1)
    for k in range(n_splits):
        tr_end = min_train + k * fold
        te_end = min(tr_end + fold, nd)
        if tr_end >= nd:
            break
        yield udays[:tr_end], udays[tr_end:te_end]

def rolling_window_splits(days: np.ndarray, train_size: int = 20, test_size: int = 5):
    """Fixed rolling-window: yield (train_day_idx, test_day_idx)."""
    udays = np.unique(days)
    nd = len(udays)
    start = 0
    while start + train_size + test_size <= nd:
        yield (udays[start:start + train_size],
               udays[start + train_size:start + train_size + test_size])
        start += test_size

def purged_kfold_splits(days: np.ndarray, n_splits: int = 5, embargo: int = 1):
    """
    Purged + embargoed K-fold over days (Lopez de Prado, AFML ch.7). Test blocks
    are contiguous day ranges; training days within `embargo` of the test block on
    either side (and overlapping the label horizon) are purged.
    """
    udays = np.unique(days)
    nd = len(udays)
    folds = np.array_split(np.arange(nd), n_splits)
    for fold in folds:
        if len(fold) == 0:
            continue
        lo, hi = fold[0], fold[-1]
        test = udays[fold]
        left = max(lo - embargo, 0)
        right = min(hi + embargo, nd - 1)
        train_mask = np.ones(nd, bool)
        train_mask[left:right + 1] = False
        yield udays[train_mask], test

def combinatorial_purged_splits(days: np.ndarray, n_groups: int = 6,
                                k_test: int = 2, embargo: int = 1):
    """
    Combinatorial Purged CV (CPCV): split days into n_groups contiguous groups,
    test on every combination of k_test groups, purge+embargo the rest for train.
    Yields (train_days, test_days, test_group_tuple).
    """
    from itertools import combinations
    udays = np.unique(days)
    nd = len(udays)
    groups = np.array_split(np.arange(nd), n_groups)
    for combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_idx.sort()
        train_mask = np.ones(nd, bool)
        train_mask[test_idx] = False
        # embargo around each test group
        for g in combo:
            lo, hi = groups[g][0], groups[g][-1]
            left = max(lo - embargo, 0)
            right = min(hi + embargo, nd - 1)
            train_mask[left:right + 1] = False
        yield udays[train_mask], udays[test_idx], combo

__all__ = [k for k in dir() if not k.startswith("_")]
