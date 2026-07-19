# fr_sweep_charts.py - report-style charts + score/turnover curves for the
# filter-rate sweep. Reads {inst}_frsweep_results.parquet (fr_sweep.py output).
#
#   INSTRUMENT=btc python3 fr_sweep_charts.py
#
# Produces, in artifacts/pilot/frsweep/:
#   {inst}_frsweep_curve.png      mean OOS score vs fr, one line per tau
#   {inst}_frsweep_turnover.png   kept $/day vs fr (log), one line per tau
#   {inst}_frsweep_bywindow.png   report-style grouped bars, W1-W5 x 3 tau,
#                                 one panel per fr
#   {inst}_frsweep_summary.csv    tau x fr table (mean/med/min/max/pos/turnover)
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTRUMENT = os.environ.get("INSTRUMENT", "btc").lower()
ROOT = Path(__file__).parent
PILOT = ROOT / "artifacts/pilot"
OUT = PILOT / "frsweep"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = PILOT / f"{INSTRUMENT}_frsweep_results.parquet"

# Report palette: tau=30 blue, tau=120 teal, tau=300 orange.
TAU_COLOR = {30: "#2b6cb0", 120: "#2f9e8f", 300: "#e08a1e"}
TAUS = [30, 120, 300]

df = pl.read_parquet(RESULTS).to_pandas()
df["tau_i"] = df["tau"].str.replace("s", "").astype(int)
df["fr"] = df["max_filter_rate"].astype(float)
df["kept_musd"] = df["test_kept_usd/day"] / 1e6
FRS = sorted(df["fr"].unique())
WINDOWS = sorted(df["window"].unique())

# ---- summary table: tau x fr ---------------------------------------------
rows = []
for tau in TAUS:
    for fr in FRS:
        s = df[(df["tau_i"] == tau) & (np.isclose(df["fr"], fr))]
        sc = s["test_score"].values
        rows.append({
            "tau": tau, "fr": fr,
            "mean": sc.mean(), "median": np.median(sc),
            "min": sc.min(), "max": sc.max(),
            "pos_windows": int((sc > 0).sum()), "n": len(sc),
            "kept_musd_day": s["kept_musd"].mean(),
        })
summ = pl.DataFrame(rows)
summ.write_csv(OUT / f"{INSTRUMENT}_frsweep_summary.csv")
print(f"{INSTRUMENT.upper()} summary (mean OOS score, bps):")
piv = summ.to_pandas().pivot(index="fr", columns="tau", values="mean")
print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))


def _fmt_fr_axis(ax):
    ax.set_xscale("logit") if False else None
    ax.set_xticks(range(len(FRS)))
    ax.set_xticklabels([f"{f:g}" for f in FRS])
    ax.set_xlabel("filter-rate cap (fr ≤)")


# ---- chart 1: score vs fr, two panels --------------------------------------
# The fr>=0.99 tail explodes the y-axis (a handful of kept trades) and hides the
# deployable range, so split: left = fr in [0.3,0.9] linear (usable), right =
# full ladder with mean + worst window so the tail's instability is visible.
x = np.arange(len(FRS))
DEPLOY = [fr for fr in FRS if fr <= 0.90]
xd = np.arange(len(DEPLOY))


def _mean(tau, fr):
    return summ.filter((pl.col("tau") == tau) & (pl.col("fr") == fr))["mean"][0]


def _worst(tau, fr):
    return summ.filter((pl.col("tau") == tau) & (pl.col("fr") == fr))["min"][0]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))

for tau in TAUS:
    axL.plot(xd, [_mean(tau, fr) for fr in DEPLOY], "o-",
             color=TAU_COLOR[tau], lw=2, ms=6, label=f"τ = {tau}s")
axL.axhline(0, color="#999", lw=0.8, ls="--")
axL.set_xticks(xd)
axL.set_xticklabels([f"{f:g}" for f in DEPLOY])
axL.set_xlabel("filter-rate cap (fr ≤)")
axL.set_ylabel("mean OOS score W1-W5, bps")
axL.set_title("deployable range: fr ≤ 0.90 (5/5 windows positive)")
axL.legend(frameon=False)
axL.grid(True, alpha=0.25)

for tau in TAUS:
    axR.plot(x, [_mean(tau, fr) for fr in FRS], "o-",
             color=TAU_COLOR[tau], lw=2, ms=6, label=f"τ={tau}s mean")
    axR.plot(x, [_worst(tau, fr) for fr in FRS], "o--",
             color=TAU_COLOR[tau], lw=1.2, ms=4, alpha=0.55,
             label=f"τ={tau}s worst window")
axR.axhline(0, color="#999", lw=0.8, ls="--")
axR.set_xticks(x)
axR.set_xticklabels([f"{f:g}" for f in FRS])
axR.set_xlabel("filter-rate cap (fr ≤)")
axR.set_ylabel("OOS score, bps")
axR.set_title("full ladder: tail mean is a mirage — worst window goes negative")
axR.legend(frameon=False, fontsize=8, ncol=2)
axR.grid(True, alpha=0.25)

fig.suptitle(f"{INSTRUMENT.upper()} · score vs filter aggressiveness · "
             f"ensemble_rank", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / f"{INSTRUMENT}_frsweep_curve.png", dpi=140)
plt.close(fig)

# ---- chart 1b: efficient frontier — score vs kept turnover -----------------
# The capacity/score tradeoff made explicit: filtering harder moves up-and-left.
# You read the score at your turnover target; there is no fr to "optimize".
fig, ax = plt.subplots(figsize=(9, 5.6))
for tau in TAUS:
    xs = [summ.filter((pl.col("tau") == tau) & (pl.col("fr") == fr))["kept_musd_day"][0]
          for fr in DEPLOY]
    ys = [_mean(tau, fr) for fr in DEPLOY]
    ax.plot(xs, ys, "o-", color=TAU_COLOR[tau], lw=2, ms=6, label=f"τ = {tau}s")
    for fr, xv, yv in zip(DEPLOY, xs, ys):
        if fr in (0.30, 0.60, 0.90):
            ax.annotate(f"{fr:g}", (xv, yv), fontsize=8, color=TAU_COLOR[tau],
                        xytext=(3, 4), textcoords="offset points")
ax.axhline(0, color="#999", lw=0.8, ls="--")
ax.set_xscale("log")
ax.invert_xaxis()  # more filtering (less turnover) to the right
ax.set_xlabel("kept turnover, $M/day (test 1/40) — filtering increases →")
ax.set_ylabel("mean OOS score W1-W5, bps")
ax.set_title(f"{INSTRUMENT.upper()} · capacity/score frontier (labels = fr cap)")
ax.legend(frameon=False)
ax.grid(True, alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT / f"{INSTRUMENT}_frsweep_frontier.png", dpi=140)
plt.close(fig)

# ---- chart 2: kept turnover vs fr (log) ----------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
for tau in TAUS:
    y = [summ.filter((pl.col("tau") == tau) & (pl.col("fr") == fr))["kept_musd_day"][0]
         for fr in FRS]
    ax.plot(x, y, "o-", color=TAU_COLOR[tau], lw=2, ms=6, label=f"τ = {tau}s")
ax.set_yscale("log")
_fmt_fr_axis(ax)
ax.set_ylabel("kept turnover, $M/day (test 1/40)")
ax.set_title(f"{INSTRUMENT.upper()} · capacity collapses as fr rises")
ax.legend(frameon=False)
ax.grid(True, alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT / f"{INSTRUMENT}_frsweep_turnover.png", dpi=140)
plt.close(fig)

# ---- chart 3: report-style grouped bars per fr ---------------------------
ncol = 4
nrow = int(np.ceil(len(FRS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                         squeeze=False)
bw = 0.26
xw = np.arange(len(WINDOWS))
for idx, fr in enumerate(FRS):
    ax = axes[idx // ncol][idx % ncol]
    for j, tau in enumerate(TAUS):
        s = df[(df["tau_i"] == tau) & (np.isclose(df["fr"], fr))]
        s = s.set_index("window").reindex(WINDOWS)
        ax.bar(xw + (j - 1) * bw, s["test_score"].values, bw,
               color=TAU_COLOR[tau], label=f"τ={tau}s")
    ax.axhline(0, color="#666", lw=0.7)
    ax.set_xticks(xw)
    ax.set_xticklabels(WINDOWS, fontsize=8)
    ax.set_title(f"fr ≤ {fr:g}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    if idx % ncol == 0:
        ax.set_ylabel("OOS score, bps", fontsize=9)
for idx in range(len(FRS), nrow * ncol):
    axes[idx // ncol][idx % ncol].axis("off")
axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(f"{INSTRUMENT.upper()} · OOS score by window, three τ "
             f"· ensemble_rank · filter-rate ladder", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / f"{INSTRUMENT}_frsweep_bywindow.png", dpi=130)
plt.close(fig)

print(f"\nwrote 4 charts (curve, frontier, turnover, bywindow) + summary.csv to {OUT}")
