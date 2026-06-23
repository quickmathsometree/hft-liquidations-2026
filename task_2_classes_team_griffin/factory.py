"""
Streaming feature/target/class/filter FACTORY.

Runs ONE causal pass over the full 1.1-billion-row tape (day-by-day, predicate
pushdown) and writes compact artifacts that the three task notebooks consume:

    artifacts/thresholds.json        frozen class thresholds (TRAIN only)
    artifacts/daily_agg.parquet      per (symbol, day) market + markout summary   [Task 1 & 3]
    artifacts/class_daily.parquet    per (symbol, day, class) weighted-PnL sums    [Task 2]
    artifacts/filter_daily.parquet   per (symbol, day, filter, tau) Score series   [Task 3]
    artifacts/sample_trades_*.parquet stratified per-trade sample w/ features+pnl   [Task 1 & 2]
    artifacts/meta.json              run metadata

Design notes
------------
* Threshold per filter is fit PER DAY to the 500k/day turnover floor (label-free,
  uses only w). This guarantees the constraint every day and yields a clean
  per-day Score series with no cross-day look-ahead - the right unit for the
  time-series statistical tests in Task 3. (A window-level single-threshold variant
  is reproduced in the Task 3 notebook for the official submission semantics.)
* Class thresholds are calibrated on a TRAIN subsample and frozen -> applied
  unchanged to validation (no leakage in class membership).
"""
from __future__ import annotations

import os, sys, json, time, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import cmf_core as cc
import stream as st

# Repo-relative paths (override with the HFT_ROOT env var). lib/ -> repo root is one level up.
ROOT = os.environ.get("HFT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "liquidation_task", "data")   # raw parquet tape
ART = os.path.join(ROOT, "artifacts")                           # generated outputs
os.makedirs(ART, exist_ok=True)

US = cc.US_PER_SECOND
DAY_US = st.DAY_US
SYMBOLS = ("btcusdt", "ethusdt")
TAUS = cc.TAUS
SAMPLE_PER_DAY = 8000          # systematic per-day sample rows -> ~1.4M total/symbol
SAMPLE_COLS = (["timestamp", "dir", "price", "notional", "w", "spread_bps", "spread_ticks",
                "imbalance", "depth_imb_dirrel", "microprice_dev_bps", "micro_signal_bps",
                "edge_vs_mid_bps", "same_side_flow_5s", "taker_imb_5s", "taker_imb_30s",
                "signed_flow_30s", "liq_aligned_all_5s", "liq_ewma_aligned_all_5s",
                "liq_mag_all_30s", "liq_signed_all_5s", "time_since_liq_all_s",
                "liq_count_all_5s", "absret_60s_bps", "absret_300s_bps", "hour_utc",
                "dow", "time_to_funding_s", "time_since_funding_s"])

def day_range(split_or_all="all"):
    if split_or_all == "all":
        lo = pd.Timestamp("2025-12-01", tz="UTC"); hi = pd.Timestamp("2026-03-01", tz="UTC")
    else:
        lo, hi = cc.SPLIT_RANGES[split_or_all]
    days = pd.date_range(lo, hi, freq="D", tz="UTC", inclusive="left")
    return [int(d.value // 1000) for d in days]

def split_of(day_us: int) -> str:
    d = pd.Timestamp(day_us, unit="us", tz="UTC")
    if cc.SPLIT_RANGES["train"][0] <= d < cc.SPLIT_RANGES["train"][1]:
        return "train"
    if cc.SPLIT_RANGES["validation"][0] <= d < cc.SPLIT_RANGES["validation"][1]:
        return "validation"
    return "other"

def fit_global_thresholds(symbol="btcusdt", n_days=8, seed=0):
    """Freeze class thresholds from a spread-out TRAIN subsample."""
    paths = st.data_paths(BASE, symbol)
    tdays = day_range("train")
    pick = np.linspace(0, len(tdays) - 1, n_days).astype(int)
    frames = []
    for di in pick:
        day = tdays[di]
        tr, bbo, lB, lY = st.read_day(paths, day)
        F = st.compute_features(tr, bbo, lB, lY)
        m = (F["timestamp"].to_numpy() >= day) & (F["timestamp"].to_numpy() < day + DAY_US)
        frames.append(F[m].iloc[::50])     # thin
    samp = pd.concat(frames, ignore_index=True)
    return st.fit_thresholds(samp)

def process_day(symbol, day, paths, TH, rng):
    tr_raw, bbo, lB, lY = st.read_day(paths, day)
    F = st.compute_features(tr_raw, bbo, lB, lY)
    tr = st.add_targets(tr_raw, bbo)
    t = F["timestamp"].to_numpy()
    m = (t >= day) & (t < day + DAY_US)
    F = F[m].reset_index(drop=True)
    tr = tr[m].reset_index(drop=True)
    w = F["w"].to_numpy(float)
    num_days = 1.0
    spl = split_of(day)

    # ---- daily aggregate ----
    agg = dict(symbol=symbol, day=pd.Timestamp(day, unit="us", tz="UTC"), split=spl,
               n_trades=int(m.sum()), turnover=float(w.sum()),
               notional=float(F["notional"].sum()),
               frac_buy=float((F["dir"].to_numpy() > 0).mean()),
               spread_bps_med=float(F["spread_bps"].median()),
               spread_bps_p95=float(F["spread_bps"].quantile(0.95)),
               spread_gt1tick=float((F["spread_ticks"].to_numpy() >= 1.5).mean()),
               absret60_med=float(F["absret_60s_bps"].median()),
               liq_n_bin=int(len(lB)), liq_n_byb=int(len(lY)),
               liq_notional_bin=float((lB["price"] * lB["amount"]).sum()) if len(lB) else 0.0,
               liq_notional_byb=float((lY["price"] * lY["amount"]).sum()) if len(lY) else 0.0)
    for tau in TAUS:
        pnl = tr[f"pnl_{tau}"].to_numpy(float)
        valid = ~np.isnan(pnl)
        agg[f"edge_{tau}"] = float((~valid).mean())
        agg[f"pnl_all_{tau}"] = float((w[valid] * pnl[valid]).sum() / w[valid].sum()) if valid.any() else np.nan
        agg[f"markout_med_{tau}"] = float(np.nanmedian(pnl))

    # ---- per-class daily sums (Task 2) ----
    classes = st.compute_classes(F, TH)
    class_rows = []
    pnl_by_tau = {tau: tr[f"pnl_{tau}"].to_numpy(float) for tau in TAUS}
    valid_by_tau = {tau: ~np.isnan(pnl_by_tau[tau]) for tau in TAUS}
    for cname in st.CLASS_NAMES:
        cm = classes[cname].to_numpy(bool)
        row = dict(symbol=symbol, day=agg["day"], split=spl, cls=cname,
                   n_member=int(cm.sum()), w_member=float(w[cm].sum()))
        for tau in TAUS:
            v = valid_by_tau[tau]
            mv = cm & v
            row[f"wpnl_member_{tau}"] = float((w[mv] * pnl_by_tau[tau][mv]).sum())
            row[f"w_member_v_{tau}"] = float(w[mv].sum())
            row[f"n_member_v_{tau}"] = int(mv.sum())
        class_rows.append(row)

    # ---- per-filter daily Score series (Task 3) ----
    raw_scores = st.filter_raw_scores(F, rng_seed=int(day % 2_000_000_000))
    filt_rows = []
    for fname, raw in raw_scores.items():
        # raw score is tau-independent -> fit ONE threshold, reuse across taus
        thr = cc.fit_threshold_fast(raw, w, num_days,
                                    target_turnover_per_day=cc.TURNOVER_FLOOR_PER_DAY)
        for tau in TAUS:
            pnl = pnl_by_tau[tau]; v = valid_by_tau[tau]
            edge = ~v
            f = cc.apply_filter(raw, thr, edge_mask=edge)
            rep = cc.score_one(pnl, w, f, num_days, tau)
            filt_rows.append(dict(symbol=symbol, day=agg["day"], split=spl, filt=fname, tau=tau,
                                  score=rep.score, pnl_all=rep.pnl_all, pnl_kept=rep.pnl_kept,
                                  pnl_filtered=rep.pnl_filtered,
                                  kept_turnover=rep.kept_turnover_per_day,
                                  n_kept=rep.n_kept, constraint_ok=rep.constraint_ok))

    # ---- stratified per-trade sample ----
    n = len(F)
    step = max(n // SAMPLE_PER_DAY, 1)
    idx = np.arange(0, n, step)
    samp = F.iloc[idx][[c for c in SAMPLE_COLS if c in F.columns]].copy()
    for tau in TAUS:
        samp[f"pnl_{tau}"] = pnl_by_tau[tau][idx]
        samp[f"edge_{tau}"] = (~valid_by_tau[tau])[idx]
    for cname in st.CLASS_NAMES:
        samp[cname] = classes[cname].to_numpy(bool)[idx]
    samp["symbol"] = symbol
    samp["split"] = spl

    return agg, class_rows, filt_rows, samp

def run(symbols=SYMBOLS, max_days=None, verbose=True):
    """Process the given symbols and write PER-SYMBOL artifacts (parallel-friendly)."""
    t_start = time.time()
    # thresholds are computed once on BTC train and shared; write only if absent
    if not os.path.exists(f"{ART}/thresholds.json"):
        TH = fit_global_thresholds("btcusdt")
        json.dump(TH, open(f"{ART}/thresholds.json", "w"), indent=2)
    TH = json.load(open(f"{ART}/thresholds.json"))
    if verbose:
        print("Frozen TRAIN thresholds:", json.dumps(TH, indent=2))

    rng = np.random.default_rng(12345)
    for symbol in symbols:
        paths = st.data_paths(BASE, symbol)
        days = day_range("all")
        if max_days:
            days = days[:max_days]
        daily, classd, filtd, samp_frames = [], [], [], []
        for k, day in enumerate(days):
            t0 = time.time()
            agg, crows, frows, samp = process_day(symbol, day, paths, TH, rng)
            daily.append(agg); classd.extend(crows); filtd.extend(frows)
            samp_frames.append(samp)
            if verbose:
                print(f"[{symbol}] {agg['day'].date()} ({agg['split'][:3]}) "
                      f"n={agg['n_trades']:,} pnl30={agg['pnl_all_30']:+.3f} "
                      f"liqB={agg['liq_n_bin']} {time.time()-t0:.1f}s", flush=True)
        pd.concat(samp_frames, ignore_index=True).to_parquet(f"{ART}/sample_trades_{symbol}.parquet", index=False)
        pd.DataFrame(daily).to_parquet(f"{ART}/daily_agg_{symbol}.parquet", index=False)
        pd.DataFrame(classd).to_parquet(f"{ART}/class_daily_{symbol}.parquet", index=False)
        pd.DataFrame(filtd).to_parquet(f"{ART}/filter_daily_{symbol}.parquet", index=False)
        if verbose:
            print(f"  -> wrote per-symbol artifacts for {symbol} "
                  f"({len(daily)} days, {time.time()-t_start:.0f}s elapsed)", flush=True)

def merge(symbols=SYMBOLS):
    """Concatenate per-symbol artifacts into unified files + write meta.json."""
    for base in ["daily_agg", "class_daily", "filter_daily"]:
        parts = [pd.read_parquet(f"{ART}/{base}_{s}.parquet")
                 for s in symbols if os.path.exists(f"{ART}/{base}_{s}.parquet")]
        pd.concat(parts, ignore_index=True).to_parquet(f"{ART}/{base}.parquet", index=False)
    TH = json.load(open(f"{ART}/thresholds.json"))
    meta = dict(symbols=list(symbols), taus=list(TAUS),
                turnover_floor=cc.TURNOVER_FLOOR_PER_DAY,
                maker_rebate_bps=cc.MAKER_REBATE_BPS, notional_clip=cc.NOTIONAL_CLIP,
                n_days=len(day_range("all")), thresholds=TH)
    json.dump(meta, open(f"{ART}/meta.json", "w"), indent=2, default=str)
    print("Merged unified artifacts:", [f"{b}.parquet" for b in ["daily_agg","class_daily","filter_daily"]])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--symbols", type=str, default="btcusdt,ethusdt")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    syms = tuple(args.symbols.split(","))
    if args.merge:
        merge(syms)
    else:
        run(symbols=syms, max_days=args.max_days)
