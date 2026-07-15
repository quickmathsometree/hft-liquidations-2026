# pilot7.py - horizon-matched feature blocks (run from p34_prod/).
#
# Pilot 6 lesson: feature windows must match the markout horizon. Every
# challenger here is ADVANCED (production control) plus ONE focused block,
# so any win attributes cleanly:
#
#   tau=30 : fast30       = advanced + 1s/10s rolling + 5/10/30s momentum
#   tau=120: adv_momentum = advanced + 30/120/300s momentum
#   tau=300: adv_momentum = advanced + momentum only
#            roll300      = advanced + 300s rolling + 300s derived (no momentum)
#            slow300      = roll300 + 600s rolling + 120/300/600s momentum
#
# Usage:
#   python3 pilot7.py                    # BTC, full grid
#   INSTRUMENT=eth python3 pilot7.py     # ETH, confirmation subset
#   SMOKE=1 python3 pilot7.py            # 1 window, 1 member, *_smoke files
#
# Expects data/enriched/{instrument}_sub40.parquet with the full window set
# (1/5/10/30/120/300/600s rolling, 5-600s momentum) - build it with the
# BUILD_DATASET cell of backtest.ipynb pointed at the raw files.
#
# Saves per-window scores, concatenated OOS trade arrays (daily stats in the
# notebook) and per-config mean gain importances to artifacts/pilot/.
import os, sys, time
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
SMOKE = os.environ.get("SMOKE", "0") == "1"

ROOT = Path(__file__).parent
DATA = ROOT / f"data/enriched/{INSTRUMENT}_sub40.parquet"
OUT_DIR = ROOT / "artifacts/pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = f"{INSTRUMENT}_pilot7"
SUFFIX = "_smoke" if SMOKE else ""

# (tau, feature_set); "advanced" is the production control at every tau.
# ETH runs the confirmation subset only (the claims worth testing off-BTC).
GRIDS = {
    "btc": [
        (30, "advanced"),
        (30, "fast30"),
        (120, "advanced"),
        (120, "adv_momentum"),
        (300, "advanced"),
        (300, "adv_momentum"),
        (300, "roll300"),
        (300, "slow300"),
    ],
    "eth": [
        (30, "advanced"),
        (30, "fast30"),
        (120, "advanced"),
        (300, "advanced"),
        (300, "roll300"),
    ],
}
GRID = GRIDS[INSTRUMENT]

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


def extract_Xyw(df, tau, features):
    from features import extract_Xyw as _extract
    return _extract(df, tau, features)


df = pl.read_parquet(DATA)
missing = [c for c in ("vpin_1s", "vpin_600s", "mid_ret_5s") if c not in df.columns]
if missing:
    raise SystemExit(
        f"{DATA} lacks the full window set ({missing}). Rebuild it with the "
        "BUILD_DATASET cell of backtest.ipynb."
    )
if SMOKE:
    df = df.gather_every(20)
df = add_all_derived_features(df)

predict_fn = make_lgbm_predict_fn()
feature_sets = {name: list(get_feature_set(name)) for _, name in GRID}
print(
    f"{INSTRUMENT} rows {df.height:,} | "
    + " | ".join(f"{k}={len(v)}" for k, v in feature_sets.items()),
    flush=True,
)

rows, oos = [], {}
importance_acc: dict[tuple, np.ndarray] = {}

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
            t1 = time.time()
            train_fn = make_lgbm_huber_train_fn(winsorize=None, **cfg)
            model = train_fn(X_tr, y_tr, w_tr)
            preds_ca.append(predict_fn(model, X_ca))
            preds_te.append(predict_fn(model, X_te))
            g = model.booster_.feature_importance(importance_type="gain")
            gain_sum += g / max(g.sum(), 1e-12)
            print(
                f"  {variant} tau={tau}s m{k}: {model.best_iteration_} iters, "
                f"{time.time()-t1:.0f}s",
                flush=True,
            )
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
                "variant": variant,
                "tau": tau,
                "window": wid,
                "max_fr": max_fr,
                "test_score": s.score,
                "test_fr": s.filter_rate,
            })
            oos.setdefault((variant, tau, max_fr), []).append(
                (y_te, w_te, ts_te, f_te)
            )

        print(f"  {variant} tau={tau}s done in {(time.time()-t_var)/60:.1f} min",
              flush=True)

    if SMOKE:
        break

tab = pl.DataFrame(rows)
tab.write_parquet(OUT_DIR / f"{PREFIX}_results{SUFFIX}.parquet")

# OOS trade arrays for notebook daily / per-trade plots (utilities.load_pilot_oos).
oos_meta, oos_arrays = [], {}
for (variant, tau, max_fr), parts in sorted(oos.items()):
    tag = f"{variant}|{tau}|{max_fr}"
    oos_meta.append({"tag": tag, "variant": variant, "tau": tau, "max_fr": max_fr})
    oos_arrays[f"{tag}/y"] = np.concatenate([p[0] for p in parts])
    oos_arrays[f"{tag}/w"] = np.concatenate([p[1] for p in parts])
    oos_arrays[f"{tag}/ts"] = np.concatenate([p[2] for p in parts])
    oos_arrays[f"{tag}/f"] = np.concatenate([p[3] for p in parts])
pl.DataFrame(oos_meta).write_parquet(OUT_DIR / f"{PREFIX}_oos_meta{SUFFIX}.parquet")
np.savez_compressed(OUT_DIR / f"{PREFIX}_oos{SUFFIX}.npz", **oos_arrays)

# Mean gain importance per config (over windows), aligned with feature names.
imp_arrays = {}
for (variant, tau), per_window in importance_acc.items():
    imp_arrays[f"{variant}|{tau}/gain"] = np.mean(per_window, axis=0)
    imp_arrays[f"{variant}|{tau}/features"] = np.array(
        feature_sets[variant], dtype=object
    )
np.savez_compressed(
    OUT_DIR / f"{PREFIX}_importance{SUFFIX}.npz", **imp_arrays, allow_pickle=True
)

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

for max_fr in MAX_FRS:
    print(f"\nPER-WINDOW (fr={max_fr}):")
    print(
        tab.filter(pl.col("max_fr") == max_fr)
        .with_columns(
            (pl.col("variant") + "_tau" + pl.col("tau").cast(pl.Utf8)).alias("key")
        )
        .pivot(values="test_score", index="window", on="key")
        .sort("window")
        .to_pandas().to_string(index=False)
    )

print("\nDELTA vs advanced control at the same tau (mean score, bps):")
base = (
    tab.group_by(["tau", "max_fr", "variant"])
    .agg(pl.col("test_score").mean().alias("mean_score"))
)
adv = base.filter(pl.col("variant") == "advanced").rename(
    {"mean_score": "advanced_score"}
).drop("variant")
challengers = base.filter(pl.col("variant") != "advanced")
delta = challengers.join(adv, on=["tau", "max_fr"]).with_columns(
    (pl.col("mean_score") - pl.col("advanced_score")).alias("delta_vs_advanced")
)
print(delta.sort(["max_fr", "tau", "variant"]).to_pandas().to_string(index=False))

print("\nDAILY t-stats (all configs, ens_rank):")
for (variant, tau, max_fr), parts in sorted(oos.items()):
    y = np.concatenate([p[0] for p in parts])
    w = np.concatenate([p[1] for p in parts])
    ts = np.concatenate([p[2] for p in parts])
    f = np.concatenate([p[3] for p in parts])
    d = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  {variant:>13} tau={tau}s fr={max_fr}: "
          f"mean daily {d.mean():+.4f}, t = {t:+.2f}, days = {len(d)}")

print(f"\n{INSTRUMENT.upper()} PILOT7 DONE")
