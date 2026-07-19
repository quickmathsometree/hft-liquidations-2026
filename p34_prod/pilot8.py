# pilot8.py - signed order-flow block (Andrey's feature list, Tier A).
#
# H-08: short-window SIGNED order-flow imbalance and a flow-vs-book divergence
# add OOS signal beyond advanced-40 at the production horizon. advanced already
# carries vpin (UNSIGNED flow) and obi (book), but no signed short-window OFI and
# no explicit flow/book divergence - the two cheap-to-build items on Andrey's
# list. opposing_cascade (Tier B) needs directional liquidation volume, which is
# not in the enriched data (liq_* is total, "signed_liq" = sign_of_trade * total)
# - it requires a raw rebuild and is deliberately NOT here.
#
#   python3 pilot8.py                    # BTC, tau 30 + 120, control vs flow8
#   INSTRUMENT=eth python3 pilot8.py     # ETH confirmation
#   SMOKE=1 python3 pilot8.py            # 1 window, 1 member, *_smoke files
#
# Same recipe as pilot7 (4x LGBM-Huber, rank aggregation, exact threshold sweep,
# W1-W5) so numbers are directly comparable to the advanced control. Saves the
# pilot7-compatible artifacts (results / oos / oos_meta / importance).
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
SMOKE = os.environ.get("SMOKE", "0") == "1"

ROOT = Path(__file__).parent
DATA = ROOT / f"data/enriched/{INSTRUMENT}_sub40.parquet"
OUT_DIR = ROOT / "artifacts/pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = f"{INSTRUMENT}_pilot8"
SUFFIX = "_smoke" if SMOKE else ""
FLOW8_NEW = ["trade_imb_5s", "trade_imb_10s", "div_z"]

GRID = [
    (30, "advanced"),
    (30, "flow8"),
    (120, "advanced"),
    (120, "flow8"),
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


def add_flow8_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Signed short-window order-flow imbalance and flow-vs-book divergence.
    All backward-looking (rolling on the trade stream), so causal and
    leakage-free when computed on the full frame before splitting.

      trade_imb_w = sum(sign*notional, w) / sum(notional, w)   in [-1, 1]
      div_z       = z(trade_imb_10s) - z(obi)   over a trailing 300s window
    """
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).alias("_ts_dt")
    ).sort("_ts_dt")

    for w in ("5s", "10s"):
        df = df.with_columns([
            (pl.col("sign") * pl.col("notional"))
            .rolling_sum_by("_ts_dt", window_size=w).alias("_sgn"),
            pl.col("notional").rolling_sum_by("_ts_dt", window_size=w).alias("_tot"),
        ]).with_columns(
            (pl.col("_sgn") / (pl.col("_tot") + 1.0))
            .clip(-1.0, 1.0).cast(pl.Float32).alias(f"trade_imb_{w}")
        ).drop("_sgn", "_tot")

    df = df.with_columns([
        ((pl.col("trade_imb_10s")
          - pl.col("trade_imb_10s").rolling_mean_by("_ts_dt", "300s"))
         / (pl.col("trade_imb_10s").rolling_std_by("_ts_dt", "300s") + 1e-6)
         ).alias("_ti_z"),
        ((pl.col("obi") - pl.col("obi").rolling_mean_by("_ts_dt", "300s"))
         / (pl.col("obi").rolling_std_by("_ts_dt", "300s") + 1e-6)
         ).alias("_bi_z"),
    ]).with_columns(
        (pl.col("_ti_z") - pl.col("_bi_z"))
        .fill_nan(0.0).cast(pl.Float32).alias("div_z")
    ).drop("_ts_dt", "_ti_z", "_bi_z")
    return df


def extract_Xyw(df, tau, features):
    from features import extract_Xyw as _extract
    return _extract(df, tau, features)


df = pl.read_parquet(DATA)
if SMOKE:
    df = df.gather_every(20)
df = add_all_derived_features(df)
df = add_flow8_features(df)

feature_sets = {
    "advanced": list(get_feature_set("advanced")),
    "flow8": list(get_feature_set("advanced")) + FLOW8_NEW,
}

predict_fn = make_lgbm_predict_fn()
print(
    f"{INSTRUMENT} rows {df.height:,} | "
    + " | ".join(f"{k}={len(v)}" for k, v in feature_sets.items())
    + f" | new={FLOW8_NEW}",
    flush=True,
)

rows, oos = [], {}
importance_acc: dict[tuple, list] = {}

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

    if SMOKE:
        break

tab = pl.DataFrame(rows)
tab.write_parquet(OUT_DIR / f"{PREFIX}_results{SUFFIX}.parquet")

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

print("\nDELTA flow8 vs advanced at the same tau (mean score, bps):")
base = (
    tab.group_by(["tau", "max_fr", "variant"])
    .agg(pl.col("test_score").mean().alias("mean_score"))
)
adv = base.filter(pl.col("variant") == "advanced").rename(
    {"mean_score": "advanced_score"}).drop("variant")
chal = base.filter(pl.col("variant") == "flow8")
delta = chal.join(adv, on=["tau", "max_fr"]).with_columns(
    (pl.col("mean_score") - pl.col("advanced_score")).alias("delta_vs_advanced"))
print(delta.sort(["max_fr", "tau"]).to_pandas().to_string(index=False))

print("\nPAIRED DAILY t (flow8 - advanced), ens_rank:")
for tau in sorted({t for t, _ in GRID}):
    for max_fr in MAX_FRS:
        pa = oos.get(("advanced", tau, max_fr))
        pf = oos.get(("flow8", tau, max_fr))
        if not pa or not pf:
            continue
        ya, wa, tsa, fa = (np.concatenate([p[i] for p in pa]) for i in range(4))
        yf, wf, tsf, ff = (np.concatenate([p[i] for p in pf]) for i in range(4))
        da = compute_daily_scores_from_arrays(y=ya, w=wa, ts=tsa, f=fa)
        dfl = compute_daily_scores_from_arrays(y=yf, w=wf, ts=tsf, f=ff)
        n = min(len(da), len(dfl))
        diff = dfl[:n] - da[:n]
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std() else 0.0
        print(f"  tau={tau}s fr={max_fr}: mean daily diff {diff.mean():+.4f}, "
              f"t = {t:+.2f}, days = {len(diff)}")

print(f"\n{INSTRUMENT.upper()} PILOT8 DONE", flush=True)
