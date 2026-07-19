# pilot9.py - cross-instrument lead-lag block (Andrey's cross-instrument idea).
#
# H-09: the OTHER instrument's recent state (BTC for ETH, ETH for BTC) adds OOS
# signal beyond a single-instrument advanced-40. Rationale: BTC and ETH perp flow
# is tightly coupled and often lead-lags; a maker order's toxicity on ETH may be
# foreseeable from BTC's just-happened move / liquidation cascade, and vice versa.
#
#   python3 pilot9.py                    # target BTC, source ETH
#   INSTRUMENT=eth python3 pilot9.py     # target ETH, source BTC
#   SMOKE=1 python3 pilot9.py            # 1 window, 1 member, *_smoke files
#
# Cross features are attached by a BACKWARD as-of join on timestamp (only source
# state at or before the target trade, with a 120 s staleness tolerance), so the
# block is strictly causal. Same recipe as pilots 7-8 (4x LGBM-Huber, rank agg,
# exact threshold sweep, W1-W5) for a clean attribution vs the advanced control.
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import add_all_derived_features, get_feature_set
from models import make_lgbm_huber_train_fn, make_lgbm_predict_fn
from scoring import (
    compute_daily_scores_from_arrays,
    make_filter_from_threshold,
    score_from_arrays,
    to_toxicity_score,
    tune_threshold_from_toxicity,
)
from splits import iter_window_splits

INSTRUMENT = os.environ.get("INSTRUMENT", "btc").lower()
SOURCE = "eth" if INSTRUMENT == "btc" else "btc"
SMOKE = os.environ.get("SMOKE", "0") == "1"

ROOT = Path(__file__).parent
DATA = ROOT / f"data/enriched/{INSTRUMENT}_sub40.parquet"
SRC_DATA = ROOT / f"data/enriched/{SOURCE}_sub40.parquet"
OUT_DIR = ROOT / "artifacts/pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = f"{INSTRUMENT}_pilot9"
SUFFIX = "_smoke" if SMOKE else ""
XNEW = ["x_mid_ret_10s", "x_mid_ret_30s", "x_vpin_120s", "x_rv_120s", "x_net_liq_120s"]
TOLERANCE_US = 120_000_000   # 120 s: drop stale source state

GRID = [
    (30, "advanced"),
    (30, "xasset"),
    (120, "advanced"),
    (120, "xasset"),
]

MAX_FRS = [0.10, 0.30]
TURNOVER = 500_000.0

MEMBER_CONFIGS = [
    dict(huber_alpha=0.80, max_depth=5, num_leaves=31, learning_rate=0.04,
         n_estimators=1200, min_child_samples=700, reg_lambda=3.0,
         subsample=0.85, colsample_bytree=0.80),
    dict(huber_alpha=0.85, max_depth=6, num_leaves=63, learning_rate=0.04,
         n_estimators=1200, min_child_samples=600, reg_lambda=2.0,
         subsample=0.80, colsample_bytree=0.80),
    dict(huber_alpha=0.90, max_depth=-1, num_leaves=63, learning_rate=0.05,
         n_estimators=1000, min_child_samples=500, reg_lambda=1.0,
         subsample=0.80, colsample_bytree=0.80),
    dict(huber_alpha=0.95, max_depth=-1, num_leaves=127, learning_rate=0.03,
         n_estimators=1500, min_child_samples=800, reg_lambda=2.0,
         subsample=0.75, colsample_bytree=0.90),
]

if SMOKE:
    MEMBER_CONFIGS = [dict(MEMBER_CONFIGS[0], n_estimators=40)]


def rank01(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (order + 1.0) / len(x)


def add_cross_features(df: pl.DataFrame, source_path: Path) -> pl.DataFrame:
    """
    Attach the source instrument's recent state by a causal backward as-of join.
    Only source rows at or before each target timestamp (within TOLERANCE_US) are
    used; source features are themselves backward-windowed, so no lookahead.
    """
    src = pl.read_parquet(
        source_path,
        columns=["timestamp", "mid_ret_10s", "mid_ret_30s", "vpin_120s",
                 "rv_120s", "liq_binance_120s", "liq_bybit_120s"],
    )
    if SMOKE:
        src = src.gather_every(20)
    src = (src
           .with_columns((pl.col("liq_binance_120s") + pl.col("liq_bybit_120s"))
                         .cast(pl.Float32).alias("x_net_liq_120s"))
           .select([
               "timestamp",
               pl.col("mid_ret_10s").alias("x_mid_ret_10s"),
               pl.col("mid_ret_30s").alias("x_mid_ret_30s"),
               pl.col("vpin_120s").alias("x_vpin_120s"),
               pl.col("rv_120s").alias("x_rv_120s"),
               "x_net_liq_120s",
           ])
           .sort("timestamp"))
    df = df.sort("timestamp")
    df = df.join_asof(src, on="timestamp", strategy="backward",
                      tolerance=TOLERANCE_US)
    return df


def extract_Xyw(df, tau, features):
    from features import extract_Xyw as _extract
    return _extract(df, tau, features)


df = pl.read_parquet(DATA)
if SMOKE:
    df = df.gather_every(20)
df = add_all_derived_features(df)
df = add_cross_features(df, SRC_DATA)

feature_sets = {
    "advanced": list(get_feature_set("advanced")),
    "xasset": list(get_feature_set("advanced")) + XNEW,
}

cov = 1.0 - df.select(pl.col("x_vpin_120s").is_null().mean()).item()
predict_fn = make_lgbm_predict_fn()
print(
    f"target {INSTRUMENT} rows {df.height:,} | source {SOURCE} | "
    f"cross coverage {cov:.1%} | "
    + " | ".join(f"{k}={len(v)}" for k, v in feature_sets.items())
    + f" | new={XNEW}",
    flush=True,
)

rows, oos = [], {}
importance_acc: dict[tuple, list] = {}


def _save(final):
    """Serialize the accumulated results. Called after every window with a
    `_partial` suffix (crash safety — the run already died once mid-W5), and
    once at the end with the real suffix. Final artifacts are byte-identical to
    the original single-shot save; the partials are just extra."""
    suf = SUFFIX if final else SUFFIX + "_partial"
    tab = pl.DataFrame(rows)
    tab.write_parquet(OUT_DIR / f"{PREFIX}_results{suf}.parquet")
    oos_meta, oos_arrays = [], {}
    for (variant, tau, max_fr), parts in sorted(oos.items()):
        tag = f"{variant}|{tau}|{max_fr}"
        oos_meta.append({"tag": tag, "variant": variant, "tau": tau, "max_fr": max_fr})
        oos_arrays[f"{tag}/y"] = np.concatenate([p[0] for p in parts])
        oos_arrays[f"{tag}/w"] = np.concatenate([p[1] for p in parts])
        oos_arrays[f"{tag}/ts"] = np.concatenate([p[2] for p in parts])
        oos_arrays[f"{tag}/f"] = np.concatenate([p[3] for p in parts])
    pl.DataFrame(oos_meta).write_parquet(OUT_DIR / f"{PREFIX}_oos_meta{suf}.parquet")
    np.savez_compressed(OUT_DIR / f"{PREFIX}_oos{suf}.npz", **oos_arrays)
    imp_arrays = {}
    for (variant, tau), per_window in importance_acc.items():
        imp_arrays[f"{variant}|{tau}/gain"] = np.mean(per_window, axis=0)
        imp_arrays[f"{variant}|{tau}/features"] = np.array(
            feature_sets[variant], dtype=object)
    np.savez_compressed(
        OUT_DIR / f"{PREFIX}_importance{suf}.npz", **imp_arrays, allow_pickle=True)
    return tab, imp_arrays


for split in iter_window_splits(df, calib_frac=0.20):
    wid = split.window_id
    print(f"\n=== {wid} ===", flush=True)

    for tau, variant in GRID:
        features = feature_sets[variant]
        t_var = time.time()

        X_tr, y_tr, w_tr, _ = extract_Xyw(split.train_model, tau, features)
        X_ca, y_ca, w_ca, ts_ca = extract_Xyw(split.calib, tau, features)
        X_te, y_te, w_te, ts_te = extract_Xyw(split.test, tau, features)

        preds_ca, preds_te = [], []
        gain_sum = np.zeros(len(features), dtype=np.float64)
        for k, cfg in enumerate(MEMBER_CONFIGS):
            train_fn = make_lgbm_huber_train_fn(winsorize=None, **cfg)
            model = train_fn(X_tr, y_tr, w_tr)
            preds_ca.append(predict_fn(model, X_ca))
            preds_te.append(predict_fn(model, X_te))
            g = model.booster_.feature_importance(importance_type="gain")
            gain_sum += g / max(g.sum(), 1e-12)
        importance_acc.setdefault((variant, tau), []).append(
            gain_sum / len(MEMBER_CONFIGS)
        )

        P_ca, P_te = np.column_stack(preds_ca), np.column_stack(preds_te)
        ens_rank_ca = np.column_stack(
            [rank01(P_ca[:, k]) for k in range(P_ca.shape[1])]
        ).mean(axis=1)
        ens_rank_te = np.column_stack(
            [rank01(P_te[:, k]) for k in range(P_te.shape[1])]
        ).mean(axis=1)

        tox_ca = to_toxicity_score(ens_rank_ca, "pnl")
        tox_te = to_toxicity_score(ens_rank_te, "pnl")

        for max_fr in MAX_FRS:
            r = tune_threshold_from_toxicity(
                tox_ca, y_ca, w_ca, ts_ca, max_filter_rate=max_fr,
                turnover_constraint=TURNOVER, n_blocks=1, robust_lambda=0.0)
            f_te = make_filter_from_threshold(tox_te, r.threshold)
            s = score_from_arrays(
                y=y_te, w=w_te, ts=ts_te, f=f_te,
                turnover_constraint=TURNOVER,
            )
            rows.append({
                "variant": variant, "tau": tau, "window": wid,
                "max_fr": max_fr, "test_score": s.score, "test_fr": s.filter_rate,
            })
            oos.setdefault((variant, tau, max_fr), []).append(
                (y_te, w_te, ts_te, f_te)
            )

        print(f"  {variant} tau={tau}s done in {(time.time()-t_var)/60:.1f} min",
              flush=True)

    _save(final=False)
    print(f"  [checkpoint saved after {wid}]", flush=True)

    if SMOKE:
        break

tab, imp_arrays = _save(final=True)

print("\nMEAN TEST SCORE ACROSS WINDOWS (bps), ens_rank")
print(
    tab.group_by(["variant", "tau", "max_fr"])
    .agg(
        pl.col("test_score").mean().alias("mean_score"),
        pl.col("test_score").min().alias("worst_window"),
        pl.col("test_fr").mean().alias("mean_fr"),
    )
    .sort(["max_fr", "tau", "variant"])
    .to_pandas().to_string(index=False)
)

print("\nDELTA xasset vs advanced at the same tau (mean score, bps):")
base = (
    tab.group_by(["tau", "max_fr", "variant"])
    .agg(pl.col("test_score").mean().alias("mean_score"))
)
adv = base.filter(pl.col("variant") == "advanced").rename(
    {"mean_score": "advanced_score"}).drop("variant")
chal = base.filter(pl.col("variant") == "xasset")
delta = chal.join(adv, on=["tau", "max_fr"]).with_columns(
    (pl.col("mean_score") - pl.col("advanced_score")).alias("delta_vs_advanced"))
print(delta.sort(["max_fr", "tau"]).to_pandas().to_string(index=False))

print("\nPAIRED DAILY t (xasset - advanced), ens_rank:")
for tau in sorted({t for t, _ in GRID}):
    for max_fr in MAX_FRS:
        pa = oos.get(("advanced", tau, max_fr))
        px = oos.get(("xasset", tau, max_fr))
        if not pa or not px:
            continue
        ya, wa, tsa, fa = (np.concatenate([p[i] for p in pa]) for i in range(4))
        yx, wx, tsx, fx = (np.concatenate([p[i] for p in px]) for i in range(4))
        da = compute_daily_scores_from_arrays(y=ya, w=wa, ts=tsa, f=fa)
        dx = compute_daily_scores_from_arrays(y=yx, w=wx, ts=tsx, f=fx)
        n = min(len(da), len(dx))
        diff = dx[:n] - da[:n]
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std() else 0.0
        print(f"  tau={tau}s fr={max_fr}: mean daily diff {diff.mean():+.4f}, "
              f"t = {t:+.2f}, days = {len(diff)}")

# cross-block gain share at tau=120 (does the model actually use the source state?)
adv_set = set(feature_sets["advanced"])
if "xasset|120/gain" in imp_arrays:
    g = imp_arrays["xasset|120/gain"]; feats = list(imp_arrays["xasset|120/features"])
    g = g / g.sum()
    xshare = sum(v for f, v in zip(feats, g) if f not in adv_set)
    top = sorted(zip(feats, g), key=lambda t: -t[1])[:10]
    print(f"\nxasset|120 cross-block gain share: {xshare:.0%}")
    for f, v in top:
        mark = "*" if f not in adv_set else " "
        print(f"  {mark} {f:<26} {v:.3f}")

print(f"\n{INSTRUMENT.upper()} PILOT9 DONE", flush=True)
