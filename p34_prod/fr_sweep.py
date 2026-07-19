# fr_sweep.py - production ensemble_rank, filter-rate sweep across horizons.
#
# Klim's request (2026-07-17): keep the report's model (advanced + 4x LGBM-Huber,
# rank aggregation) and walk-forward exactly as on the front page, but sweep the
# filter-rate cap over a finer ladder so the score-vs-aggressiveness curve is
# visible instead of just the 0.30 / 0.90 endpoints.
#
#   INSTRUMENT=btc python3 fr_sweep.py     # BTC, tau 30/120/300
#   INSTRUMENT=eth python3 fr_sweep.py     # ETH, same
#
# The filter-rate cap only enters threshold tuning, not training: one walk-forward
# fit per (tau, window) is scored at every fr in FR_GRID. Per-window scores go to
# {inst}_frsweep_results.parquet; the per-window OOS toxicity is cached to
# {inst}_frsweep_oos.npz so any future fr ladder is a re-tune with no retrain.
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment import WalkForwardConfig, run_experiment_from_df, results_to_frame
from features import add_all_derived_features, get_feature_set

INSTRUMENT = os.environ.get("INSTRUMENT", "btc").lower()
SMOKE = os.environ.get("SMOKE", "0") == "1"

ROOT = Path(__file__).parent
DATA = ROOT / f"data/enriched/{INSTRUMENT}_sub40.parquet"
OUT_DIR = ROOT / "artifacts/pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# TAG keeps a horizon-sensitivity bracket (TAG=taubracket) from clobbering the
# main sweep's {inst}_frsweep_* artifacts.
TAG = os.environ.get("TAG", "frsweep")
PREFIX = f"{INSTRUMENT}_{TAG}"

# TAUS override, e.g. TAUS=90,180 for the horizon bracket.
if os.environ.get("TAUS"):
    TAUS = [int(t) for t in os.environ["TAUS"].split(",")]
else:
    TAUS = [30, 120, 300]
# 0.30 / 0.90 already in the report; 0.40-0.75 fill the middle, 0.99 / 0.999 the tail.
FR_GRID = [0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 0.99, 0.999]

if SMOKE:
    TAUS = [120]
    FR_GRID = [0.30, 0.90]

df = pl.read_parquet(DATA)
if SMOKE:
    df = df.gather_every(20)
# Derive once so the per-window runner finds the advanced columns already present.
df = add_all_derived_features(df)
advanced = list(get_feature_set("advanced"))
print(f"{INSTRUMENT} rows {df.height:,} | advanced={len(advanced)} | "
      f"taus={TAUS} | fr={FR_GRID}", flush=True)

frames = []
oos_arrays = {}

for tau in TAUS:
    t0 = time.time()
    print(f"\n=== tau={tau}s ===", flush=True)
    cfg = WalkForwardConfig(
        instrument       = INSTRUMENT,
        tau              = tau,
        model_name       = "lgbm_huber_ensemble_rank",
        max_filter_rates = FR_GRID,
        verbose          = False,
        use_tqdm         = False,
    )
    res = run_experiment_from_df(df, cfg, advanced)
    fr = results_to_frame(res)
    frames.append(fr)

    # One toxicity vector per window (identical across fr) -> cache the first fr's.
    for w in res.windows:
        if not np.isclose(w.max_filter_rate, FR_GRID[0]):
            continue
        k = f"{tau}|{w.window_id}"
        oos_arrays[f"{k}/tox"] = w.toxicity_test
        oos_arrays[f"{k}/y"] = w.y_test
        oos_arrays[f"{k}/w"] = w.w_test
        oos_arrays[f"{k}/ts"] = w.ts_test

    # Progress: mean score and kept turnover per fr for this tau.
    summ = (
        fr.group_by("max_filter_rate")
        .agg(
            pl.col("test_score").mean().alias("mean_score"),
            pl.col("test_score").min().alias("worst"),
            (pl.col("test_score") > 0).sum().alias("pos_windows"),
            pl.col("test_kept_usd/day").mean().alias("kept_usd_day"),
        )
        .sort("max_filter_rate")
    )
    for row in summ.iter_rows(named=True):
        print(f"  fr<={row['max_filter_rate']:.3f}: "
              f"mean {row['mean_score']:+.4f}  worst {row['worst']:+.4f}  "
              f"pos {row['pos_windows']}/5  kept ${row['kept_usd_day']/1e6:.1f}M/day",
              flush=True)
    print(f"  tau={tau}s done in {(time.time()-t0)/60:.1f} min", flush=True)

all_fr = pl.concat(frames)
all_fr.write_parquet(OUT_DIR / f"{PREFIX}_results.parquet")
np.savez_compressed(OUT_DIR / f"{PREFIX}_oos.npz", **oos_arrays)

print(f"\nSAVED {PREFIX}_results.parquet ({all_fr.height} rows) + _oos.npz")
print(f"{INSTRUMENT.upper()} FR SWEEP DONE", flush=True)
