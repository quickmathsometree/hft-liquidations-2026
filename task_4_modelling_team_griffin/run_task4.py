"""
Task 4 pipeline: thresholds -> training sample -> per-window models -> exact OOS scoring.

Stages (CLI):
  python run_task4.py thresholds      # freeze class thresholds on Nov (W1 train), -> artifacts4/thresholds_v4.json
  python run_task4.py sample          # build sampled train features+targets (Nov-Mar) -> artifacts4/train_sample.parquet
  python run_task4.py oos             # train per-window models + exact streaming OOS scoring -> artifacts4/*

Models per (window, horizon): Ridge (interpretable), LightGBM (nonlinear), rule-based
(liq_aligned cascade signal), plus random & keep-all baselines. Pooled over symbols, scored
per symbol. Per-day turnover-floor threshold (label-free) -> binary filter f(tau).
"""
from __future__ import annotations
import os, sys, json, time, glob, pickle
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_core as cc, stream as st, model4 as m4

ROOT = os.environ.get("HFT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "full", "liquidation_task", "data")
ART = os.path.join(ROOT, "artifacts4")
os.makedirs(ART, exist_ok=True)
US = cc.US_PER_SECOND; DAY = st.DAY_US
SYMBOLS = ("btcusdt", "ethusdt")
TAUS = cc.TAUS
SAMPLE_PER_DAY = 10_000         # training sample per day-symbol
OOS_SCORE_SAMPLE = 500_000      # scoring sample per test day-symbol (representation-weighted)
OOS_STORE_SAMPLE = 4_000        # stored per test day-symbol for error analysis
FLOOR = cc.TURNOVER_FLOOR_PER_DAY


def daterange(s, e):
    d = pd.Timestamp(s, tz="UTC")
    out = []
    while d < pd.Timestamp(e, tz="UTC"):
        out.append(int(d.value // 1000)); d += pd.Timedelta(days=1)
    return out


def inday_idx(trades, day):
    t = trades["timestamp"].to_numpy()
    return np.where((t >= day) & (t < day + DAY))[0]


# thresholds
def stage_thresholds(n_days=8):
    days = daterange("2025-11-01", "2025-12-01")
    pick = np.linspace(0, len(days) - 1, n_days).astype(int)
    rng = np.random.default_rng(0); frames = []
    for di in pick:
        day = days[di]
        for sym in SYMBOLS:
            tr, bbo, lB, lY = st.read_day(st.data_paths(BASE, sym), day)
            ii = inday_idx(tr, day)
            if len(ii) == 0: continue
            samp = np.sort(rng.choice(ii, size=min(6000, len(ii)), replace=False))
            F, _ = m4.compute_features_v4(tr, bbo, lB, lY, q_idx=samp, TH=m4.THRESH_FALLBACK)
            frames.append(F[["liq_aligned_all_5s", "absret_60s_bps", "spread_bps"]])
    s = pd.concat(frames, ignore_index=True)
    TH = dict(m4.THRESH_FALLBACK)
    TH["liq_align_q90"] = float(np.nanquantile(np.abs(s["liq_aligned_all_5s"]), 0.90))
    TH["vol_q90"] = float(np.nanquantile(s["absret_60s_bps"], 0.90))
    TH["spread_q95"] = float(np.nanquantile(s["spread_bps"], 0.95))
    json.dump(TH, open(f"{ART}/thresholds_v4.json", "w"), indent=2)
    print("frozen Nov thresholds:", json.dumps(TH, indent=2))


# train sample
def stage_sample():
    TH = json.load(open(f"{ART}/thresholds_v4.json"))
    days = daterange("2025-11-01", "2026-04-01")     # union of all train periods
    rng = np.random.default_rng(123); frames = []
    t0 = time.time()
    for k, day in enumerate(days):
        for sym in SYMBOLS:
            tr, bbo, lB, lY = st.read_day(st.data_paths(BASE, sym), day)
            ii = inday_idx(tr, day)
            if len(ii) == 0: continue
            samp = np.sort(rng.choice(ii, size=min(SAMPLE_PER_DAY, len(ii)), replace=False))
            F, meta = m4.compute_features_v4(tr, bbo, lB, lY, q_idx=samp, TH=TH)
            tg = m4.compute_targets_v4(tr, bbo, samp)
            for tau in TAUS:
                F[f"pnl_{tau}"] = tg[f"pnl_{tau}"].astype(np.float32)
            F["w"] = meta["w"].to_numpy(np.float32); F["symbol"] = sym
            F["day"] = pd.Timestamp(day, unit="us", tz="UTC")
            frames.append(F)
        if k % 20 == 0:
            print(f"  {pd.Timestamp(day,unit='us').date()}  rows~{sum(len(f) for f in frames):,}  {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(f"{ART}/train_sample.parquet", index=False)
    print(f"train_sample: {df.shape}  in {(time.time()-t0)/60:.1f} min")


# models
FEATCOLS = None
def feature_cols(df):
    drop = {"w", "symbol", "day"} | {f"pnl_{t}" for t in TAUS}
    return [c for c in df.columns if c not in drop]


def train_window_models(train_df, tau, seed=0):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    cols = feature_cols(train_df)
    d = train_df[np.isfinite(train_df[f"pnl_{tau}"])]
    X = d[cols].to_numpy(np.float32); y = d[f"pnl_{tau}"].to_numpy(np.float64)
    w = d["w"].to_numpy(np.float64)
    lo, hi = np.quantile(y, [0.005, 0.995]); yw = np.clip(y, lo, hi)   # winsorise target
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ridge = Ridge(alpha=10.0).fit(Xs, yw, sample_weight=w)
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                            min_child_samples=200, reg_lambda=5.0, n_jobs=-1,
                            random_state=seed, verbose=-1)
    gbm.fit(X, yw, sample_weight=w)
    return dict(cols=cols, mu=mu, sd=sd, ridge=ridge, gbm=gbm)


def predict_models(F, models):
    cols = models["cols"]; X = F[cols].to_numpy(np.float32)
    Xs = (X - models["mu"]) / models["sd"]
    out = {}
    out["ridge"] = models["ridge"].predict(Xs)
    out["lgbm"] = models["gbm"].predict(X)
    out["rule_liq"] = F["liq_aligned_all_5s"].to_numpy(float)            # Task-2 cascade rule
    return out


# OOS scoring
def _acc_init():
    return dict(n=0.0, nflt=0.0, W=0.0, Wp=0.0, KW=0.0, KWp=0.0, FW=0.0, FWp=0.0, days=set())


def _acc_add(a, pnl, w, rep, f, day):
    """Representation-weighted accumulation. w is already w*rep; rep weights the counts."""
    valid = np.isfinite(pnl)
    pv, wv, rv, fv = pnl[valid], w[valid], rep[valid], f[valid].astype(float)
    keep = 1.0 - fv
    a["n"] += rv.sum(); a["nflt"] += (fv * rv).sum()
    a["W"] += wv.sum(); a["Wp"] += (wv * pv).sum()
    a["KW"] += (wv * keep).sum(); a["KWp"] += (wv * keep * pv).sum()
    a["FW"] += (wv * fv).sum(); a["FWp"] += (wv * fv * pv).sum()
    a["days"].add(day)


def _acc_report(a):
    pnl_all = a["Wp"] / a["W"] if a["W"] > 0 else np.nan
    pnl_kept = a["KWp"] / a["KW"] if a["KW"] > 0 else np.nan
    pnl_filt = a["FWp"] / a["FW"] if a["FW"] > 0 else np.nan
    nd = max(len(a["days"]), 1)
    return dict(n_trades=int(round(a["n"])), clipped_turnover=a["W"],
                filt_pct_trades=100.0 * a["nflt"] / max(a["n"], 1),
                filt_pct_turnover=100.0 * a["FW"] / max(a["W"], 1e-9),
                pnl_all=pnl_all, pnl_kept=pnl_kept, pnl_filtered=pnl_filt,
                score=pnl_kept - pnl_all, kept_turnover_per_day=a["KW"] / nd, n_days=nd)


def stage_oos():
    TH = json.load(open(f"{ART}/thresholds_v4.json"))
    train_df = pd.read_parquet(f"{ART}/train_sample.parquet")
    train_df["day"] = pd.to_datetime(train_df["day"], utc=True)
    rng = np.random.default_rng(7)
    MODELS = ["keep_all", "random", "rule_liq", "ridge", "lgbm"]
    reports = []; oos_samples = []; fitted_store = {}
    t_start = time.time()
    for (wn, tr_s, tr_e, te_s, te_e) in m4.WINDOWS:
        tr_mask = (train_df["day"] >= pd.Timestamp(tr_s, tz="UTC")) & (train_df["day"] < pd.Timestamp(tr_e, tz="UTC"))
        sub = train_df[tr_mask]
        print(f"\n[{wn}] train rows={len(sub):,}  ({tr_s}..{tr_e})  fitting models...", flush=True)
        wmodels = {tau: train_window_models(sub, tau) for tau in TAUS}
        # store compact importances
        fitted_store[wn] = {tau: dict(cols=wmodels[tau]["cols"],
                                      ridge_coef=wmodels[tau]["ridge"].coef_.tolist(),
                                      lgbm_imp=wmodels[tau]["gbm"].feature_importances_.tolist())
                            for tau in TAUS}
        acc = {(sym, mdl, tau): _acc_init() for sym in SYMBOLS for mdl in MODELS for tau in TAUS}
        for day in daterange(te_s, te_e):
            for sym in SYMBOLS:
                tr, bbo, lB, lY = st.read_day(st.data_paths(BASE, sym), day)
                ii = inday_idx(tr, day)
                if len(ii) == 0: continue
                # representation-weighted scoring sample (exact at the 2.2B-row scale is intractable)
                ns = min(OOS_SCORE_SAMPLE, len(ii))
                qi = np.sort(rng.choice(ii, size=ns, replace=False))
                rep_factor = len(ii) / ns
                F, meta = m4.compute_features_v4(tr, bbo, lB, lY, q_idx=qi, TH=TH)
                tg = m4.compute_targets_v4(tr, bbo, qi)
                w = meta["w"].to_numpy(float)
                w_rep = w * rep_factor                       # full-tape-equivalent weight
                rep = np.full(len(F), rep_factor)
                preds = {tau: predict_models(F, wmodels[tau]) for tau in TAUS}
                rand = rng.standard_normal(len(F))
                for tau in TAUS:
                    pnl = tg[f"pnl_{tau}"]; edge = ~np.isfinite(pnl)
                    for mdl in MODELS:
                        if mdl == "keep_all":
                            f = np.zeros(len(F), np.int8)
                        else:
                            sc = rand if mdl == "random" else preds[tau][mdl]
                            nz = ~edge      # edge trades are kept for free & out of Score -> exclude from the floor fit
                            thr = cc.fit_threshold_fast(sc[nz], w_rep[nz], 1.0, FLOOR) if nz.any() else -np.inf
                            f = cc.apply_filter(sc, thr, edge_mask=edge)
                        _acc_add(acc[(sym, mdl, tau)], pnl, w_rep, rep, f, day)
                # store a small scored sub-sample for error analysis
                si = np.sort(rng.choice(len(F), size=min(OOS_STORE_SAMPLE, len(F)), replace=False))
                smp = F.iloc[si].copy()
                for tau in TAUS:
                    smp[f"pnl_{tau}"] = tg[f"pnl_{tau}"][si]
                    smp[f"score_lgbm_{tau}"] = preds[tau]["lgbm"][si]
                    smp[f"score_ridge_{tau}"] = preds[tau]["ridge"][si]
                smp["w"] = w[si]; smp["symbol"] = sym; smp["window"] = wn
                smp["timestamp"] = meta["timestamp"].to_numpy()[si]
                oos_samples.append(smp)
            print(f"  [{wn}] {pd.Timestamp(day,unit='us').date()} done  ({time.time()-t_start:.0f}s)", flush=True)
        for (sym, mdl, tau), a in acc.items():
            r = _acc_report(a); r.update(window=wn, symbol=sym, model=mdl, tau=tau); reports.append(r)
    pd.DataFrame(reports).to_parquet(f"{ART}/oos_reports.parquet", index=False)
    pd.concat(oos_samples, ignore_index=True).to_parquet(f"{ART}/oos_sample.parquet", index=False)
    pickle.dump(fitted_store, open(f"{ART}/fitted_importances.pkl", "wb"))
    json.dump(dict(taus=list(TAUS), symbols=list(SYMBOLS), floor=FLOOR,
                   windows=[w[0] for w in m4.WINDOWS], features=feature_cols(train_df),
                   elapsed_min=round((time.time()-t_start)/60, 1)),
              open(f"{ART}/meta4.json", "w"), indent=2)
    print(f"\nOOS done in {(time.time()-t_start)/60:.1f} min -> {ART}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("thresholds", "all"): stage_thresholds()
    if stage in ("sample", "all"): stage_sample()
    if stage in ("oos", "all"): stage_oos()
