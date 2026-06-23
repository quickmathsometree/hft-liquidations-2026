"""
Streaming feature/target/class factory for the 1.1-billion-row trade dataset.

The full trade tape (402M BTC + 706M ETH rows) does not fit in 32 GB RAM, so we
process the data day-by-day using parquet predicate-pushdown (the files are
timestamp-sorted, so a one-day read touches only the relevant row groups -
~0.1 s for 3.8 M rows).

For each (symbol, UTC day) `process_day` computes, with STRICT causality:
  * official markout + maker PnL (targets; use FUTURE mids -> never features)
  * book / flow / liquidation / time features (use only data strictly before t_i)
  * a battery of interpretable binary CLASSES
  * raw scores for candidate FILTERS (Task 3)
and returns compact accumulators (per-day aggregates, per-class weighted PnL,
per-day filter scores) plus a small stratified per-trade sample for plotting.

Everything heavy is vectorised (searchsorted / cumsum / asof). Liquidation EWMAs
use the event-anchored primitive (sparse events -> O(n_trades) lookup).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import cmf_core as cc

US = cc.US_PER_SECOND
DAY_US = 86_400 * US

# Window/halflife configuration (seconds)
FLOW_WINDOWS = (1, 5, 30)
LIQ_WINDOWS = (1, 5, 30)
LIQ_HALFLIVES = (1.0, 5.0, 30.0)
VOL_WINDOWS = (60, 300)
PREROLL_S = 600          # trades/liq/BBO context before day start (>= longest vol window 300s)
POST_S = max(cc.TAUS) + 60   # BBO tail after day end (> longest markout horizon, with slack)

def data_paths(base: str, symbol: str) -> dict[str, str]:
    return {
        "trades":      f"{base}/binance_trades/perp_{symbol}.parquet",
        "bbo":         f"{base}/binance_booktickers/perp_{symbol}.parquet",
        "liq_binance": f"{base}/binance_liquidations/perp_{symbol}.parquet",
        "liq_bybit":   f"{base}/bybit_liquidations/{symbol}.parquet",
    }

def _read(path: str, lo: int, hi: int, cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols,
                         filters=[("timestamp", ">=", lo), ("timestamp", "<", hi)])
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

def read_day(paths: dict[str, str], day_start_us: int,
             preroll_us: int = PREROLL_S * US, post_us: int = POST_S * US):
    """Load the four frames for one UTC day with the necessary context margins."""
    d0, d1 = day_start_us, day_start_us + DAY_US
    trades = _read(paths["trades"], d0 - preroll_us, d1)
    # BBO preroll must cover the longest vol window so absret_{w}s is not NaN at day start
    bbo = _read(paths["bbo"], d0 - preroll_us, d1 + post_us)
    liqB = _read(paths["liq_binance"], d0 - preroll_us, d1)
    liqY = _read(paths["liq_bybit"], d0 - preroll_us - cc.BYBIT_LAG_US, d1)
    liqY = liqY.copy()
    liqY["timestamp"] = liqY["timestamp"] + cc.BYBIT_LAG_US   # +200 ms cross-venue lag
    liqY = liqY.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return trades, bbo, liqB, liqY

def _signed_liq_vals(liq: pd.DataFrame) -> np.ndarray:
    """Signed liquidation notional: +notional for 'buy' (upward pressure), else -."""
    side = liq["side"].astype(str).str.lower().to_numpy()
    notional = (liq["price"].to_numpy(float) * liq["amount"].to_numpy(float))
    return np.where(side == "buy", 1.0, -1.0) * notional

def compute_features(trades: pd.DataFrame, bbo: pd.DataFrame,
                     liqB: pd.DataFrame, liqY: pd.DataFrame) -> pd.DataFrame:
    """
    Causal feature matrix aligned to `trades`. All values use data strictly before
    each trade timestamp (book/flow/liq), except the trade's own price/side/notional.
    """
    t = trades["timestamp"].to_numpy(np.int64)
    price = trades["price"].to_numpy(float)
    amount = trades["amount"].to_numpy(float)
    side = trades["side"].astype(str).str.lower().to_numpy()
    s = np.where(side == "buy", 1.0, -1.0)
    notional = price * amount
    w = np.minimum(notional, cc.NOTIONAL_CLIP)

    F = pd.DataFrame({"timestamp": t})
    F["dir"] = s.astype(np.int8)
    F["price"] = price
    F["notional"] = notional
    F["w"] = w

    # ---- Book features (last BBO strictly before trade); ONE searchsorted reused ----
    bb = cc.add_mid(bbo)
    bts = bb["timestamp"].to_numpy(np.int64)
    nbb = len(bts)
    bidx = np.searchsorted(bts, t, side="left") - 1     # last bbo strictly before
    bvalid = bidx >= 0
    bclip = np.clip(bidx, 0, nbb - 1)

    def _bk(col):
        v = bb[col].to_numpy(float)[bclip]
        return np.where(bvalid, v, np.nan)
    mid_prev = _bk("mid")
    micro_prev = _bk("microprice")
    bidp = _bk("bid_price"); askp = _bk("ask_price")
    bida = _bk("bid_amount"); aska = _bk("ask_amount")
    F["mid_prev"] = mid_prev
    F["spread_bps"] = (askp - bidp) / mid_prev * 1e4
    F["spread_ticks"] = np.round((askp - bidp) / _tick(price), 3)
    _dep = bida + aska
    F["imbalance"] = np.where(_dep > 0, (bida - aska) / np.where(_dep > 0, _dep, 1.0), 0.0)
    F["microprice_dev_bps"] = (micro_prev - mid_prev) / mid_prev * 1e4
    # direction-relative depth: same_side = side the maker rests on / taker hits
    F["same_side_depth"] = np.where(s > 0, aska, bida)   # taker buy hits ask
    F["opp_side_depth"] = np.where(s > 0, bida, aska)
    F["depth_imb_dirrel"] = s * F["imbalance"]           # >0: book leans toward trade dir
    # how far the fill is from mid, in maker's favour (bps): maker sells high / buys low
    F["edge_vs_mid_bps"] = s * (price - mid_prev) / mid_prev * 1e4
    F["micro_signal_bps"] = s * F["microprice_dev_bps"]  # >0: micro confirms trade dir

    # ---- Flow features (Binance trades, windowed, strictly before) ----
    # `hi` (count of trades strictly before each trade) is shared across all windows.
    tr_signed = (s * w)  # signed clipped notional of each trade
    hi_self = np.searchsorted(t, t, side="left")
    csS = np.concatenate([[0.0], np.cumsum(tr_signed)])
    csW = np.concatenate([[0.0], np.cumsum(w)])
    for win in FLOW_WINDOWS:
        lo = np.searchsorted(t, t - win * US, side="left")
        sv = csS[hi_self] - csS[lo]
        tv = csW[hi_self] - csW[lo]
        cnt = (hi_self - lo).astype(float)
        F[f"signed_flow_{win}s"] = sv
        F[f"total_flow_{win}s"] = tv
        with np.errstate(invalid="ignore", divide="ignore"):
            imb = np.where(tv > 0, sv / tv, 0.0)
        F[f"taker_imb_{win}s"] = imb
        F[f"same_side_flow_{win}s"] = s * sv          # >0: recent flow same dir as trade
        F[f"trade_count_{win}s"] = cnt

    # ---- Liquidation features (per venue + combined; windowed + EWMA) ----
    venues = {"bin": liqB, "byb": liqY}
    # combined = stack both venues' events
    allt = np.concatenate([liqB["timestamp"].to_numpy(np.int64),
                           liqY["timestamp"].to_numpy(np.int64)])
    allv = np.concatenate([_signed_liq_vals(liqB), _signed_liq_vals(liqY)])
    allabs = np.abs(allv)
    o = np.argsort(allt, kind="mergesort")
    allt, allv, allabs = allt[o], allv[o], allabs[o]
    venues_signed = {"bin": (liqB["timestamp"].to_numpy(np.int64), _signed_liq_vals(liqB)),
                     "byb": (liqY["timestamp"].to_numpy(np.int64), _signed_liq_vals(liqY)),
                     "all": (allt, allv)}
    for vn, (ets, evs) in venues_signed.items():
        ets = np.asarray(ets, np.int64)
        if len(ets) == 0:
            for win in LIQ_WINDOWS:
                F[f"liq_signed_{vn}_{win}s"] = 0.0
                F[f"liq_mag_{vn}_{win}s"] = 0.0
                F[f"liq_aligned_{vn}_{win}s"] = 0.0
            for hl in LIQ_HALFLIVES:
                F[f"liq_ewma_signed_{vn}_{int(hl)}s"] = 0.0
                F[f"liq_ewma_aligned_{vn}_{int(hl)}s"] = 0.0
            F[f"liq_count_{vn}_5s"] = 0.0
            F[f"time_since_liq_{vn}_s"] = np.inf
            continue
        eabs = np.abs(evs)
        hi = np.searchsorted(ets, t, side="left")     # events strictly before (shared)
        csSig = np.concatenate([[0.0], np.cumsum(evs)])
        csMag = np.concatenate([[0.0], np.cumsum(eabs)])
        lo5 = None
        for win in LIQ_WINDOWS:
            lo = np.searchsorted(ets, t - win * US, side="left")
            if win == 5:
                lo5 = lo
            sp = csSig[hi] - csSig[lo]
            mg = csMag[hi] - csMag[lo]
            F[f"liq_signed_{vn}_{win}s"] = sp
            F[f"liq_mag_{vn}_{win}s"] = mg
            F[f"liq_aligned_{vn}_{win}s"] = s * sp
        # EWMA: precompute post-states + shared prev_idx (= hi-1) across halflives.
        # Note: post_state depends on tau, so we still loop, but reuse prev_idx.
        prev_idx = hi - 1
        for hl in LIQ_HALFLIVES:
            spe = cc.causal_decaying_sum_at(t, ets, evs, hl * US, prev_idx=prev_idx)
            F[f"liq_ewma_signed_{vn}_{int(hl)}s"] = spe
            F[f"liq_ewma_aligned_{vn}_{int(hl)}s"] = s * spe
        F[f"liq_count_{vn}_5s"] = (hi - (lo5 if lo5 is not None
                                         else np.searchsorted(ets, t - 5 * US, side="left"))).astype(float)
        ts = np.where(prev_idx >= 0, (t - ets[np.clip(prev_idx, 0, len(ets) - 1)]).astype(float), np.inf)
        F[f"time_since_liq_{vn}_s"] = ts / US

    # ---- Volatility / regime (windowed realized move from mid path) ----
    for win in VOL_WINDOWS:
        mid_past = cc._asof_strict_before(t - win * US, bts, bb["mid"].to_numpy(float))
        F[f"ret_{win}s_bps"] = (mid_prev / mid_past - 1.0) * 1e4
        F[f"absret_{win}s_bps"] = np.abs(F[f"ret_{win}s_bps"])

    # ---- Time features ----
    dt = cc.to_datetime(t)
    F["hour_utc"] = dt.hour.astype(np.int16)
    F["minute_utc"] = dt.minute.astype(np.int16)
    F["dow"] = dt.dayofweek.astype(np.int16)
    sec_of_day = (t % DAY_US) // US
    # next funding boundary at 0/8/16/24h
    fh_next = np.array([0, 8, 16, 24]) * 3600
    nxt = fh_next[np.searchsorted(fh_next, sec_of_day, side="right")]
    F["time_to_funding_s"] = (nxt - sec_of_day).astype(np.int32)
    # last funding boundary at 0/8/16h
    fh_prev = np.array([0, 8, 16]) * 3600
    prev_idx = np.clip(np.searchsorted(fh_prev, sec_of_day, side="right") - 1, 0, 2)
    F["time_since_funding_s"] = (sec_of_day - fh_prev[prev_idx]).astype(np.int32)
    return F

def _tick(price: np.ndarray) -> np.ndarray:
    """Approximate price tick: 0.1 for BTC (>~1000), 0.01 for ETH-scale."""
    return np.where(price > 1000, 0.1, 0.01)

def add_targets(trades: pd.DataFrame, bbo: pd.DataFrame,
                taus=cc.TAUS) -> pd.DataFrame:
    bb = cc.add_mid(bbo)
    tr = cc.compute_markout(trades, bb, taus)
    tr = cc.compute_pnl(tr, taus)
    return tr

#  Interpretable CLASSES (Task 2) and candidate FILTERS (Task 3)
# Thresholds are calibrated on TRAIN data only (see fit_thresholds) and then
# frozen and applied unchanged to validation -> no look-ahead in class membership.
THRESH_KEYS = ("spread_q95", "notional_q90", "flow_q90", "vol_q90",
               "liq_align_q90", "liq_mag_q90", "micro_q90")

def fit_thresholds(F: pd.DataFrame) -> dict:
    """Empirical quantile thresholds from a (train) feature sample."""
    g = lambda col, q: float(np.nanquantile(F[col].to_numpy(float), q))
    return {
        "spread_q95":     g("spread_bps", 0.95),
        "notional_q90":   g("notional", 0.90),
        "flow_q90":       float(np.nanquantile(np.abs(F["same_side_flow_5s"]), 0.90)),
        "vol_q90":        g("absret_60s_bps", 0.90),
        "liq_align_q90":  float(np.nanquantile(np.abs(F["liq_aligned_all_5s"]), 0.90)),
        "liq_mag_q90":    g("liq_mag_all_30s", 0.90),
        "micro_q90":      float(np.nanquantile(np.abs(F["micro_signal_bps"]), 0.90)),
    }

def compute_classes(F: pd.DataFrame, TH: dict) -> pd.DataFrame:
    """Binary class membership for each trade (causal, frozen thresholds)."""
    out = pd.DataFrame(index=F.index)
    out["post_liq_aligned_5s"]  = (F["liq_aligned_all_5s"] >=  TH["liq_align_q90"]).to_numpy()
    out["against_liq_5s"]       = (F["liq_aligned_all_5s"] <= -TH["liq_align_q90"]).to_numpy()
    out["near_liq_5s"]          = (F["time_since_liq_all_s"] <= 5.0).to_numpy()
    out["big_liq_30s"]          = (F["liq_mag_all_30s"] >= TH["liq_mag_q90"]).to_numpy()
    out["wide_spread"]          = (F["spread_ticks"] >= 1.5).to_numpy()
    out["top5_spread"]          = (F["spread_bps"] >= TH["spread_q95"]).to_numpy()
    out["large_trade"]          = (F["notional"] >= TH["notional_q90"]).to_numpy()
    out["one_sided_flow_w_dir"] = (F["same_side_flow_5s"] >=  TH["flow_q90"]).to_numpy()
    out["against_flow"]         = (F["same_side_flow_5s"] <= -TH["flow_q90"]).to_numpy()
    out["high_vol"]             = (F["absret_60s_bps"] >= TH["vol_q90"]).to_numpy()
    out["micro_adverse"]        = (F["micro_signal_bps"] >=  TH["micro_q90"]).to_numpy()
    out["micro_favorable"]      = (F["micro_signal_bps"] <= -TH["micro_q90"]).to_numpy()
    out["funding_window"]       = ((F["time_to_funding_s"] <= 300) |
                                   (F["time_since_funding_s"] <= 300)).to_numpy()
    out["cascade_aligned"]      = (out["near_liq_5s"].to_numpy() &
                                   out["post_liq_aligned_5s"].to_numpy() &
                                   out["high_vol"].to_numpy())
    return out

CLASS_NAMES = ["post_liq_aligned_5s", "against_liq_5s", "near_liq_5s", "big_liq_30s",
               "wide_spread", "top5_spread", "large_trade", "one_sided_flow_w_dir",
               "against_flow", "high_vol", "micro_adverse", "micro_favorable",
               "funding_window", "cascade_aligned"]

def filter_raw_scores(F: pd.DataFrame, rng_seed: int = 0) -> dict[str, np.ndarray]:
    """
    Candidate FILTER raw scores. Convention: HIGHER raw score = KEEP (good trade).
    fit_threshold keeps the top trades by raw score until the turnover floor binds,
    so the lowest-scoring (most adverse) trades are filtered out.

    Signs are set by the empirical EDA (Task 1/2): the maker is COMPENSATED for
    absorbing liquidation cascades - trades ALIGNED with recent liquidation pressure
    mean-revert and earn POSITIVE markout - so we KEEP them (raw = +liq_aligned).
    We also keep micro-favourable fills, fills against the leaning book, and fills
    that printed far in the maker's favour vs the prevailing mid.
    """
    n = len(F)
    def z(x):
        x = np.asarray(x, float); m = np.nanmean(x); sd = np.nanstd(x) or 1.0
        return np.nan_to_num((x - m) / sd)
    scores = {}
    # 1) Liquidation cascade mean-reversion: keep trades absorbing the forced flow
    scores["liq_align"]   = F["liq_aligned_all_5s"].to_numpy(float)
    scores["liq_ewma"]    = F["liq_ewma_aligned_all_5s"].to_numpy(float)
    # 2) Flow / microstructure
    scores["same_flow"]   = F["same_side_flow_5s"].to_numpy(float)   # momentum-aligned fills
    scores["micro"]       = -F["micro_signal_bps"].to_numpy(float)   # microprice-favourable
    scores["depth_imb"]   = -F["depth_imb_dirrel"].to_numpy(float)   # fill against the leaning book
    scores["edge_vs_mid"] = F["edge_vs_mid_bps"].to_numpy(float)     # filled far in our favour vs mid
    # 3) Combined (equal-weight z-scores of the favourable signals)
    scores["combo"] = (z(F["liq_aligned_all_5s"]) + z(F["liq_ewma_aligned_all_5s"])
                       + z(-F["micro_signal_bps"]) + z(F["edge_vs_mid_bps"]))
    # 4) Sanity baseline
    rng = np.random.default_rng(rng_seed)
    scores["random"]      = rng.standard_normal(n)
    return scores

FILTER_NAMES = ["liq_align", "liq_ewma", "same_flow", "micro", "depth_imb",
                "combo", "random", "edge_vs_mid"]
