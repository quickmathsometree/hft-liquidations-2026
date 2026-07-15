# pilot6.py (run from p34_prod/) - multi-horizon sweep with extended features:
#   * _300s rolling columns (vpin / rv / liq)
#   * mid-price momentum (mid_ret / signed / abs at 30/120/300s)
# Compares "advanced" (legacy 40 features) vs "extended" (+22 new cols)
# on tau in {30, 120, 300}.
import sys, time
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

DATA = str(Path(__file__).parent / "data/enriched/btc_sub40.parquet")
OUT_DIR = Path(__file__).parent / "artifacts/pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAUS = (30, 120, 300)
VARIANTS = ("advanced", "extended")
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


def rank01(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (order + 1.0) / len(x)


def extract_Xyw(df, tau, features):
    from features import extract_Xyw as _extract
    return _extract(df, tau, features)


def require_extended_columns(df: pl.DataFrame) -> None:
    missing = [c for c in ("vpin_300s", "mid_ret_300s") if c not in df.columns]
    if missing:
        raise SystemExit(
            "Extended columns missing from parquet. Rebuild first:\n"
            "  rebuild via the BUILD_DATASET cell of backtest.ipynb"
        )


df = pl.read_parquet(DATA)
require_extended_columns(df)
df = add_all_derived_features(df)
predict_fn = make_lgbm_predict_fn()
feature_sets = {v: list(get_feature_set(v)) for v in VARIANTS}
print(
    f"rows {df.height:,} | "
    f"advanced={len(feature_sets['advanced'])} feats | "
    f"extended={len(feature_sets['extended'])} feats",
    flush=True,
)

rows, oos = [], {}
for split in iter_window_splits(df, calib_frac=0.20):
    wid = split.window_id
    print(f"\n=== {wid} ===", flush=True)

    for variant in VARIANTS:
        features = feature_sets[variant]
        t_var = time.time()

        for tau in TAUS:
            X_tr, y_tr, w_tr, _ = extract_Xyw(split.train_model, tau, features)
            X_ca, y_ca, w_ca, ts_ca = extract_Xyw(split.calib, tau, features)
            X_te, y_te, w_te, ts_te = extract_Xyw(split.test, tau, features)

            preds_ca, preds_te = [], []
            for k, cfg in enumerate(MEMBER_CONFIGS):
                t1 = time.time()
                train_fn = make_lgbm_huber_train_fn(winsorize=None, **cfg)
                model = train_fn(X_tr, y_tr, w_tr)
                preds_ca.append(predict_fn(model, X_ca))
                preds_te.append(predict_fn(model, X_te))
                print(
                    f"  {variant} tau={tau}s m{k}: {model.best_iteration_} iters, "
                    f"{time.time()-t1:.0f}s",
                    flush=True,
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

        print(f"  {variant} done in {(time.time()-t_var)/60:.1f} min", flush=True)

tab = pl.DataFrame(rows)
tab.write_parquet(OUT_DIR / "pilot6_results.parquet")

# OOS trade arrays for notebook daily / per-trade plots (see utilities.load_pilot_oos).
oos_meta, oos_arrays = [], {}
for (variant, tau, max_fr), parts in sorted(oos.items()):
    tag = f"{variant}|{tau}|{max_fr}"
    oos_meta.append({"tag": tag, "variant": variant, "tau": tau, "max_fr": max_fr})
    oos_arrays[f"{tag}/y"] = np.concatenate([p[0] for p in parts])
    oos_arrays[f"{tag}/w"] = np.concatenate([p[1] for p in parts])
    oos_arrays[f"{tag}/ts"] = np.concatenate([p[2] for p in parts])
    oos_arrays[f"{tag}/f"] = np.concatenate([p[3] for p in parts])
pl.DataFrame(oos_meta).write_parquet(OUT_DIR / "pilot6_oos_meta.parquet")
np.savez_compressed(OUT_DIR / "pilot6_oos.npz", **oos_arrays)

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

print("\nDELTA extended - advanced (mean score, bps):")
base = (
    tab.group_by(["tau", "max_fr", "variant"])
    .agg(pl.col("test_score").mean().alias("mean_score"))
)
pivot = base.pivot(values="mean_score", index=["tau", "max_fr"], on="variant")
if "extended" in pivot.columns and "advanced" in pivot.columns:
    pivot = pivot.with_columns(
        (pl.col("extended") - pl.col("advanced")).alias("delta_ext_minus_adv")
    )
    print(pivot.sort(["max_fr", "tau"]).to_pandas().to_string(index=False))

print("\nDAILY t-stats (extended, ens_rank):")
for (variant, tau, max_fr), parts in sorted(oos.items()):
    if variant != "extended":
        continue
    y = np.concatenate([p[0] for p in parts])
    w = np.concatenate([p[1] for p in parts])
    ts = np.concatenate([p[2] for p in parts])
    f = np.concatenate([p[3] for p in parts])
    d = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  tau={tau}s fr={max_fr}: mean daily {d.mean():+.4f}, t = {t:+.2f}")

print("\nPILOT6 DONE")
