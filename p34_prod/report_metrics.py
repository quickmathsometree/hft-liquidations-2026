# report_metrics.py - strategy-economics + stability numbers for the report
# (Andrey's headings 3 and 5). Reads the fr-sweep artifacts (production
# advanced ensemble_rank) and prints, per instrument, for the production
# operating point advanced @ tau=120, fr<=0.30:
#   window-by-window score / filter-rate / kept turnover,
#   across-window median / range / fraction positive,
#   max drawdown of cumulative daily kept-PnL,
#   PnL concentration (share of kept PnL in the top 1% of kept trades),
#   turnover headroom vs the $500k/day floor,
# and the fr ladder economics at tau=120.
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scoring import compute_daily_scores_from_arrays

PILOT = Path("artifacts/pilot")
TURN_FLOOR = 500_000.0
PROD_TAU, PROD_FR = 120, 0.30


def prod_window_table(inst):
    res = pl.read_parquet(PILOT / f"{inst}_frsweep_results.parquet")
    r = (res.filter((pl.col("tau") == f"{PROD_TAU}s")
                    & (np.isclose(res["max_filter_rate"], PROD_FR)))
         .sort("window"))
    return r


def daily_drawdown_and_conc(inst):
    """Reconstruct the kept set per window (kept = tox <= threshold), pool, and
    compute daily-score max drawdown + top-1% kept-PnL concentration."""
    res = prod_window_table(inst)
    thr = {row["window"]: row["threshold"] for row in res.iter_rows(named=True)}
    z = np.load(PILOT / f"{inst}_frsweep_oos.npz")
    ys, ws, tss, fs = [], [], [], []
    kept_wy = []          # weighted pnl of kept trades (for concentration)
    for w in sorted(thr):
        key = f"{PROD_TAU}|{w}"
        tox, y, wt, ts = z[f"{key}/tox"], z[f"{key}/y"], z[f"{key}/w"], z[f"{key}/ts"]
        t = thr[w]
        f = (tox > t).astype(np.int8) if np.isfinite(t) else np.zeros(len(tox), np.int8)
        ys.append(y); ws.append(wt); tss.append(ts); fs.append(f)
        kept = f == 0
        kept_wy.append((y[kept] * wt[kept]))
    y = np.concatenate(ys); w = np.concatenate(ws)
    ts = np.concatenate(tss); f = np.concatenate(fs)

    daily = compute_daily_scores_from_arrays(y=y, w=w, ts=ts, f=f)
    cum = np.cumsum(daily)
    dd = cum - np.maximum.accumulate(cum)
    max_dd = dd.min()

    # Net kept PnL is near-zero (kept set ~ the full stream), so express tail
    # concentration against gross positive kept PnL - the meaningful fragility
    # measure: how much of the winning PnL rides on the top 1% of trades.
    wy = np.concatenate(kept_wy)
    pos = wy[wy > 0]
    total_pos = pos.sum()
    conc_pos = (np.sort(pos)[-max(1, int(np.ceil(0.01 * len(pos)))):].sum() / total_pos
                if total_pos > 0 else float("nan"))
    return max_dd, conc_pos, len(daily)


for inst in ("btc", "eth"):
    print(f"\n{'='*66}\n{inst.upper()} - production advanced @ tau={PROD_TAU}s, fr<={PROD_FR}\n{'='*66}")
    r = prod_window_table(inst)
    sc = r["test_score"].to_numpy()
    fr_pct = r["test_filter_rate (%)"].to_numpy()
    kept = r["test_kept_usd/day"].to_numpy()
    print(f"{'window':>7} {'score(bps)':>11} {'filter_rate':>12} {'kept $/day':>14}")
    for w, s, fp, kp in zip(r["window"], sc, fr_pct, kept):
        print(f"{w:>7} {s:>+11.4f} {fp:>11.2f}% {kp:>13,.0f}")
    print(f"  median {np.median(sc):+.4f} | range [{sc.min():+.4f}, {sc.max():+.4f}] "
          f"| positive {int((sc>0).sum())}/{len(sc)}")
    print(f"  kept turnover min ${kept.min():,.0f}/day = {kept.min()/TURN_FLOOR:.0f}x the "
          f"${TURN_FLOOR:,.0f} floor (never binds)")
    max_dd, conc_pos, ndays = daily_drawdown_and_conc(inst)
    print(f"  max drawdown of cumulative daily kept-PnL: {max_dd:+.3f} bps over {ndays} days")
    print(f"  PnL concentration: top 1% of kept trades = {conc_pos:.1%} of gross positive kept PnL")

    print(f"\n  fr ladder economics (tau={PROD_TAU}s):")
    res = pl.read_parquet(PILOT / f"{inst}_frsweep_results.parquet")
    lad = (res.filter(pl.col("tau") == f"{PROD_TAU}s")
           .group_by("max_filter_rate")
           .agg(pl.col("test_score").mean().alias("mean"),
                pl.col("test_score").min().alias("worst"),
                (pl.col("test_score") > 0).sum().alias("pos"),
                pl.col("test_kept_usd/day").mean().alias("kept"))
           .sort("max_filter_rate"))
    print(f"  {'fr<=':>6} {'mean':>9} {'worst':>9} {'pos':>5} {'kept $M/day':>13}")
    for row in lad.iter_rows(named=True):
        print(f"  {row['max_filter_rate']:>6.3f} {row['mean']:>+9.3f} {row['worst']:>+9.3f} "
              f"{row['pos']:>4}/5 {row['kept']/1e6:>12.1f}")
