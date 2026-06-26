"""
Task 4 - Modelling core.

A curated liquidation/microstructure feature battery (Task 4.1), the expanding-window
W1-W5 protocol, and train/predict/score helpers built on the audited cmf_core scoring.

Scale note: the full 6-month tape is ~2.2B trades. Feature cost scales with the number
of QUERY trades, not the full tape, so compute_features_v4 accepts `q_idx` (a subset of
trades to evaluate) while using the full per-day context (BBO, trades, liquidations) for
the windowed aggregates. Training uses a sampled q_idx (cheap); exact OOS scoring passes
q_idx=None (all trades on the day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cmf_core as cc
import stream as st

US = cc.US_PER_SECOND

# Expanding-window CV (W1-W5): (name, train_start, train_end_excl, test_start, test_end_excl)
WINDOWS = [
    ("W1", "2025-11-01", "2025-12-01", "2025-12-08", "2026-01-01"),
    ("W2", "2025-11-01", "2026-01-01", "2026-01-08", "2026-02-01"),
    ("W3", "2025-11-01", "2026-02-01", "2026-02-08", "2026-03-01"),
    ("W4", "2025-11-01", "2026-03-01", "2026-03-08", "2026-04-01"),
    ("W5", "2025-11-01", "2026-04-01", "2026-04-08", "2026-04-29"),
]

def _us(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").value // 1000)

# Class thresholds for the binary Task-2 features are frozen on the earliest training
# month (Nov 2025) -> available for, and contained in, every window's training period.
THRESH_FALLBACK = dict(spread_q95=0.13, notional_q90=6300.0, flow_q90=3.9e6,
                       vol_q90=24.8, liq_align_q90=2.0e4, liq_mag_q90=2.76e5, micro_q90=0.0064)


def _signed_liq(liq):
    side = liq["side"].astype(str).str.lower().to_numpy()
    notional = liq["price"].to_numpy(float) * liq["amount"].to_numpy(float)
    sgn = np.where(side == "buy", 1.0, -1.0)
    return sgn * notional, notional, sgn


def _win_sum(cs, hi, lo):
    return cs[hi] - cs[lo]


def compute_features_v4(trades, bbo, liqB, liqY, q_idx=None, TH=None):
    """
    Causal Task-4 feature matrix. If `q_idx` is given, features are computed only for
    those trade rows (training subsample); otherwise for all trades (exact OOS scoring).
    Returns (F, meta) where meta has price/side/notional/w/timestamp for the query rows.
    """
    TH = TH or THRESH_FALLBACK
    t_all = trades["timestamp"].to_numpy(np.int64)
    price_all = trades["price"].to_numpy(float)
    amt_all = trades["amount"].to_numpy(float)
    side_all = trades["side"].astype(str).str.lower().to_numpy()
    s_all = np.where(side_all == "buy", 1.0, -1.0)
    notion_all = price_all * amt_all
    w_all = np.minimum(notion_all, cc.NOTIONAL_CLIP)

    if q_idx is None:
        q = np.arange(len(t_all))
    else:
        q = np.asarray(q_idx, np.int64)
    tq = t_all[q]; sq = s_all[q]; pq = price_all[q]; wq = w_all[q]; nq = notion_all[q]

    F = pd.DataFrame(index=np.arange(len(q)))
    F["dir"] = sq.astype(np.int8)
    F["log_notional"] = np.log1p(nq)

    # ---- Book / microstructure (last BBO strictly before each query trade) ----
    bb = cc.add_mid(bbo); bts = bb["timestamp"].to_numpy(np.int64); n_bb = len(bts)
    if n_bb == 0:
        bval = np.zeros(len(q), bool); bc = np.zeros(len(q), np.int64)
        bk = lambda col: np.full(len(q), np.nan)
    else:
        bidx = np.searchsorted(bts, tq, side="left") - 1
        bval = bidx >= 0; bc = np.clip(bidx, 0, n_bb - 1)
        bk = lambda col: np.where(bval, bb[col].to_numpy(float)[bc], np.nan)
    mid = bk("mid"); micro = bk("microprice")
    bidp = bk("bid_price"); askp = bk("ask_price"); bida = bk("bid_amount"); aska = bk("ask_amount")
    tick = np.where(pq > 1000, 0.1, 0.01)
    F["spread_bps"] = (askp - bidp) / mid * 1e4
    F["spread_ticks"] = np.round((askp - bidp) / tick, 2)
    dep = bida + aska
    imb = np.where(dep > 0, (bida - aska) / np.where(dep > 0, dep, 1.0), 0.0)
    F["depth_imb_dirrel"] = sq * imb
    F["micro_dev_bps"] = (micro - mid) / mid * 1e4
    F["micro_signal_bps"] = sq * F["micro_dev_bps"]
    F["edge_vs_mid_bps"] = sq * (pq - mid) / mid * 1e4
    F["same_side_depth_log"] = np.log1p(np.where(sq > 0, aska, bida))

    # ---- Volatility / regime (mid path) ----
    for wsec in (60, 300):
        mp = np.where(np.searchsorted(bts, tq - wsec * US, side="left") - 1 >= 0,
                      bb["mid"].to_numpy(float)[np.clip(np.searchsorted(bts, tq - wsec * US, side="left") - 1, 0, n_bb - 1)],
                      np.nan)
        F[f"absret_{wsec}s_bps"] = np.abs((mid / mp - 1.0) * 1e4)

    # ---- Signed taker flow (Binance trades; cumsum over full tape, query at points) ----
    hi = np.searchsorted(t_all, tq, side="left")
    csS = np.concatenate([[0.0], np.cumsum(s_all * w_all)])
    csW = np.concatenate([[0.0], np.cumsum(w_all)])
    for wsec in (5, 30):
        lo = np.searchsorted(t_all, tq - wsec * US, side="left")
        sv = _win_sum(csS, hi, lo); tv = _win_sum(csW, hi, lo)
        F[f"same_side_flow_{wsec}s"] = sq * sv
        with np.errstate(divide="ignore", invalid="ignore"):
            F[f"taker_imb_{wsec}s"] = np.where(tv > 0, sv / np.where(tv > 0, tv, 1.0), 0.0)
    F["trade_count_5s"] = (hi - np.searchsorted(t_all, tq - 5 * US, side="left")).astype(float)

    # ---- Liquidation battery (Binance + Bybit) ----
    venues = {}
    for nm, L in (("bin", liqB), ("byb", liqY)):
        if len(L):
            sgnval, mag, sgn = _signed_liq(L)
            venues[nm] = dict(ts=L["timestamp"].to_numpy(np.int64), sv=sgnval, mag=mag, sgn=sgn,
                              price=L["price"].to_numpy(float))
        else:
            venues[nm] = dict(ts=np.array([], np.int64), sv=np.array([]), mag=np.array([]),
                              sgn=np.array([]), price=np.array([]))
    # combined
    allts = np.concatenate([venues["bin"]["ts"], venues["byb"]["ts"]])
    allsv = np.concatenate([venues["bin"]["sv"], venues["byb"]["sv"]])
    allmag = np.concatenate([venues["bin"]["mag"], venues["byb"]["mag"]])
    allpr = np.concatenate([venues["bin"]["price"], venues["byb"]["price"]])
    if len(allts):
        o = np.argsort(allts, kind="mergesort")
        allts, allsv, allmag, allpr = allts[o], allsv[o], allmag[o], allpr[o]
    venues["all"] = dict(ts=allts, sv=allsv, mag=allmag, price=allpr)

    def liq_aligned(vn, wsec):
        d = venues[vn]
        if len(d["ts"]) == 0:
            return np.zeros(len(q)), np.zeros(len(q))
        hi_e = np.searchsorted(d["ts"], tq, side="left")
        lo_e = np.searchsorted(d["ts"], tq - wsec * US, side="left")
        csv = np.concatenate([[0.0], np.cumsum(d["sv"])])
        cmag = np.concatenate([[0.0], np.cumsum(d["mag"])])
        sp = _win_sum(csv, hi_e, lo_e); mg = _win_sum(cmag, hi_e, lo_e)
        return sq * sp, mg

    # net aligned pressure + magnitude over scales, all venues + per venue
    for wsec in (1, 5, 30):
        al, mg = liq_aligned("all", wsec)
        F[f"liq_aligned_all_{wsec}s"] = al
        F[f"liq_mag_all_{wsec}s"] = mg
    F["liq_aligned_bin_5s"] = liq_aligned("bin", 5)[0]
    F["liq_aligned_byb_5s"] = liq_aligned("byb", 5)[0]
    # EWMA aligned (event-anchored)
    if len(venues["all"]["ts"]):
        spe = cc.causal_decaying_sum_at(tq, venues["all"]["ts"], venues["all"]["sv"], 5.0 * US)
        F["liq_ewma_aligned_all_5s"] = sq * spe
    else:
        F["liq_ewma_aligned_all_5s"] = 0.0

    # buy/sell volume separately, count, avg & max size, concentration, acceleration (5s vs 30s)
    d = venues["all"]
    if len(d["ts"]):
        is_buy = (d["sv"] > 0).astype(float)
        cbuy = np.concatenate([[0.0], np.cumsum(d["mag"] * is_buy)])
        csell = np.concatenate([[0.0], np.cumsum(d["mag"] * (1 - is_buy))])
        ccnt = np.concatenate([[0.0], np.cumsum(np.ones(len(d["ts"])))])
        cmag = np.concatenate([[0.0], np.cumsum(d["mag"])])
        hi5 = np.searchsorted(d["ts"], tq, side="left")
        lo5 = np.searchsorted(d["ts"], tq - 5 * US, side="left")
        lo30 = np.searchsorted(d["ts"], tq - 30 * US, side="left")
        F["liq_buy_all_5s"] = _win_sum(cbuy, hi5, lo5)
        F["liq_sell_all_5s"] = _win_sum(csell, hi5, lo5)
        cnt5 = _win_sum(ccnt, hi5, lo5); mag5 = _win_sum(cmag, hi5, lo5)
        mag30 = _win_sum(cmag, hi5, lo30)
        F["liq_count_all_5s"] = cnt5
        F["liq_count_all_30s"] = _win_sum(ccnt, hi5, lo30)
        F["liq_avg_size_all_5s"] = np.where(cnt5 > 0, mag5 / np.maximum(cnt5, 1), 0.0)
        # acceleration: recent 5s rate vs the trailing 30s average rate
        F["liq_accel_5v30"] = mag5 - mag30 / 6.0
        # max single liq in 30s via running max over events (block-free: per-query loop avoided
        # by a sparse-events approach -> events are few, so use a simple windowed max)
        F["liq_max_size_all_30s"] = _windowed_max(tq, d["ts"], d["mag"], 30 * US)
        F["liq_concentration_30s"] = np.where(mag30 > 0, F["liq_max_size_all_30s"] / mag30, 0.0)
        # recent liq price vs mid (last liq), direction-relative
        last = hi5 - 1
        has = last >= 0
        lp = np.where(has, d["price"][np.clip(last, 0, len(d["price"]) - 1)], np.nan)
        F["liq_price_dev_bps"] = np.where(has, sq * (lp - mid) / mid * 1e4, 0.0)
    else:
        for c in ["liq_buy_all_5s", "liq_sell_all_5s", "liq_count_all_5s", "liq_count_all_30s",
                  "liq_avg_size_all_5s", "liq_accel_5v30", "liq_max_size_all_30s",
                  "liq_concentration_30s", "liq_price_dev_bps"]:
            F[c] = 0.0

    # time since last liq (any / buy / sell), seconds; bin-byb pressure agreement
    F["time_since_liq_all_s"] = _time_since(tq, venues["all"]["ts"]) / US
    if len(venues["all"]["ts"]):
        buy_ts = venues["all"]["ts"][venues["all"]["sv"] > 0]
        sell_ts = venues["all"]["ts"][venues["all"]["sv"] < 0]
    else:
        buy_ts = sell_ts = np.array([], np.int64)
    F["time_since_buyliq_s"] = np.minimum(_time_since(tq, buy_ts) / US, 1e6)
    F["time_since_sellliq_s"] = np.minimum(_time_since(tq, sell_ts) / US, 1e6)
    # agreement between Binance and Bybit signed pressure over 5s (sign product, normalised)
    spb = liq_aligned("bin", 5)[0]; spy = liq_aligned("byb", 5)[0]
    F["liq_binbyb_agree"] = np.sign(spb) * np.sign(spy) * np.minimum(np.abs(spb), np.abs(spy))

    # ---- Time features ----
    sec = (tq % (86_400 * US)) // US
    F["hour_utc"] = (sec // 3600).astype(np.int16)
    fh = np.array([0, 8, 16, 24]) * 3600
    F["time_to_funding_s"] = (fh[np.searchsorted(fh, sec, side="right")] - sec).astype(np.int32)

    # ---- Task-2 classes (binary) + interactions with liquidation pressure ----
    F["cls_post_liq_aligned"] = (F["liq_aligned_all_5s"] >= TH["liq_align_q90"]).astype(np.int8)
    F["cls_against_liq"] = (F["liq_aligned_all_5s"] <= -TH["liq_align_q90"]).astype(np.int8)
    F["cls_near_liq_5s"] = (F["time_since_liq_all_s"] <= 5.0).astype(np.int8)
    F["cls_high_vol"] = (F["absret_60s_bps"] >= TH["vol_q90"]).astype(np.int8)
    F["cls_wide_spread"] = (F["spread_ticks"] >= 1.5).astype(np.int8)
    F["x_liqalign_highvol"] = F["liq_aligned_all_5s"] * F["cls_high_vol"]
    F["x_liqalign_widespread"] = F["liq_aligned_all_5s"] * F["cls_wide_spread"]
    F["x_liqalign_nearliq"] = F["liq_aligned_all_5s"] * F["cls_near_liq_5s"]

    meta = pd.DataFrame(dict(timestamp=tq, price=pq, side=np.where(sq > 0, "buy", "sell"),
                            notional=nq, w=wq, s=sq.astype(np.int8)))
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return F, meta


def _time_since(query_ts, event_ts):
    if len(event_ts) == 0:
        return np.full(len(query_ts), 1e6 * US, float)
    idx = np.searchsorted(event_ts, query_ts, side="left") - 1
    return np.where(idx >= 0, (query_ts - event_ts[np.clip(idx, 0, len(event_ts) - 1)]).astype(float),
                    1e6 * US)


def _windowed_max(query_ts, event_ts, event_vals, window_us):
    """Max event value in [q-window, q) for each query. Events are sparse -> O((n+m)) merge."""
    nq = len(query_ts)
    if len(event_ts) == 0:
        return np.zeros(nq)
    lo = np.searchsorted(event_ts, query_ts - window_us, side="left")
    hi = np.searchsorted(event_ts, query_ts, side="left")
    out = np.zeros(nq)
    # group queries by (lo,hi) span is irregular; do a light loop only where there are events
    nz = np.where(hi > lo)[0]
    for i in nz:
        out[i] = event_vals[lo[i]:hi[i]].max()
    return out


FEATURE_COLS = None  # set after first computation


def compute_targets_v4(trades, bbo, q_idx, taus=cc.TAUS):
    """Maker PnL (bps) per horizon for the query trades. Edge -> NaN."""
    bb = cc.add_mid(bbo); bts = bb["timestamp"].to_numpy(np.int64); bmid = bb["mid"].to_numpy(float)
    t_all = trades["timestamp"].to_numpy(np.int64)
    side_all = trades["side"].astype(str).str.lower().to_numpy()
    s_all = np.where(side_all == "buy", 1.0, -1.0)
    price_all = trades["price"].to_numpy(float)
    q = np.asarray(q_idx, np.int64) if q_idx is not None else np.arange(len(t_all))
    tq = t_all[q]; sq = s_all[q]; pq = price_all[q]
    if len(bts) == 0:
        return {f"pnl_{tau}": np.full(len(q), np.nan) for tau in taus}
    maxb = bts[-1]
    out = {}
    for tau in taus:
        look = tq + tau * US
        idx = np.searchsorted(bts, look, side="right") - 1
        mid = np.where(idx >= 0, bmid[np.clip(idx, 0, len(bmid) - 1)], np.nan)
        edge = look > maxb
        pnl = -sq * (mid - pq) / pq * 1e4 + cc.MAKER_REBATE_BPS
        out[f"pnl_{tau}"] = np.where(edge, np.nan, pnl)
    return out
