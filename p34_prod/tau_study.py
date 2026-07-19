# tau_study.py - horizon-sensitivity view. Combines the main fr sweep
# (tau 30/120/300) with the {90,180} bracket (and {60,240} if present) and asks
# one question: is tau=120 a genuine peak, or does some other horizon beat it
# out-of-family? Framed as robustness, NOT "pick the best tau" - tau is the
# markout target, so maximizing backtest score over it is the same overfit trap
# we flagged for fr.
#
#   python3 tau_study.py            # reads btc + eth, all available tau tags
#
# Reads {inst}_{tag}_results.parquet for tag in TAGS; writes tau_study_score.png
# and prints a verdict line the report/loop can act on.
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
PILOT = ROOT / "artifacts/pilot"
OUT = PILOT / "frsweep"
OUT.mkdir(parents=True, exist_ok=True)

TAGS = ["frsweep", "taubracket", "tauextra"]
INSTRUMENTS = ["btc", "eth"]
FR_SHOW = [0.30, 0.90]
FR_COLOR = {0.30: "#2b6cb0", 0.90: "#e08a1e"}
CHAMPION = 120
MATERIAL = 0.15


def load(inst):
    frames = []
    for tag in TAGS:
        p = PILOT / f"{inst}_{tag}_results.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return None
    d = pl.concat(frames).to_pandas()
    d["tau_i"] = d["tau"].str.replace("s", "").astype(int)
    d["fr"] = d["max_filter_rate"].astype(float)
    return d


def agg(d, tau, fr):
    s = d[(d["tau_i"] == tau) & (np.isclose(d["fr"], fr))]["test_score"].values
    if len(s) == 0:
        return None
    return {"mean": s.mean(), "worst": s.min(), "pos": int((s > 0).sum()), "n": len(s)}


data = {i: load(i) for i in INSTRUMENTS}
data = {i: d for i, d in data.items() if d is not None}

verdict_rows = []
fig, axes = plt.subplots(1, len(data), figsize=(7 * len(data), 5.2), squeeze=False)
for col, (inst, d) in enumerate(data.items()):
    taus = sorted(d["tau_i"].unique())
    ax = axes[0][col]
    print(f"\n=== {inst.upper()} · mean OOS score by horizon (bps) ===")
    header = "  tau  " + "  ".join(f"fr{fr:g}".rjust(8) for fr in FR_SHOW) + "   pos@0.3"
    print(header)
    for tau in taus:
        cells = []
        for fr in FR_SHOW:
            a = agg(d, tau, fr)
            cells.append(f"{a['mean']:+.3f}" if a else "   -   ")
        a03 = agg(d, tau, 0.30)
        star = " *120" if tau == CHAMPION else ""
        print(f"  {tau:>4}  " + "  ".join(c.rjust(8) for c in cells)
              + f"   {a03['pos']}/{a03['n']}{star}")
    for fr in FR_SHOW:
        ys = [agg(d, tau, fr)["mean"] for tau in taus]
        ax.plot(taus, ys, "o-", color=FR_COLOR[fr], lw=2, ms=6, label=f"fr ≤ {fr:g}")
    ax.axvline(CHAMPION, color="#999", lw=1, ls=":", label="production τ=120")
    ax.axhline(0, color="#ccc", lw=0.8, ls="--")
    ax.set_xlabel("markout horizon τ, s")
    ax.set_ylabel("mean OOS score W1-W5, bps")
    ax.set_title(f"{inst.upper()} · score vs horizon")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    base = agg(d, CHAMPION, 0.30)
    for tau in taus:
        if tau == CHAMPION:
            continue
        a = agg(d, tau, 0.30)
        if a is None:
            continue
        beats = (a["mean"] > base["mean"] * (1 + MATERIAL)) and (a["pos"] == a["n"])
        verdict_rows.append({"inst": inst, "tau": tau, "mean": a["mean"],
                             "base120": base["mean"], "pos": a["pos"], "n": a["n"],
                             "beats": beats})

fig.suptitle("Horizon sensitivity · advanced-40 · ensemble_rank "
             "(robustness check, not tau optimization)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "tau_study_score.png", dpi=140)
plt.close(fig)
print("\n=== VERDICT ===")
challengers = {}
for r in verdict_rows:
    challengers.setdefault(r["tau"], []).append(r)
signal = False
for tau, rows in sorted(challengers.items()):
    insts_beaten = [r["inst"] for r in rows if r["beats"]]
    if len(insts_beaten) == len(data) and len(data) >= 2:
        signal = True
        print(f"  tau={tau}: BEATS 120 materially on both {insts_beaten}")
for tau, rows in sorted(challengers.items()):
    detail = ", ".join(
        f"{r['inst']}={r['mean']:+.3f}(base{r['base120']:+.3f},{r['pos']}/{r['n']})"
        for r in rows
    )
    print(f"  tau={tau}: {detail}")
verdict = ("YES - run fuller {60,90,180,240} grid" if signal
           else "NO - tau=120 holds, proceed to report")
print(f"\n  SIGNAL={verdict}")
print(f"\nwrote tau_study_score.png to {OUT}")
