# w2_regime.py - characterize the W2 (Jan 2026) regime (diagnostics only).
#
# W2 is the strong-score outlier in every walk-forward run. This does NOT try to
# fix it - it quantifies HOW the January regime differs from the other test
# windows, so the report can say what "regime" means here rather than hand-wave.
#
# Per instrument, over each window's TEST period [train_end, test_end), it means
# the microstructure state variables already in the enriched frame and shows each
# window relative to the cross-window median.
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from splits import get_window_ids, get_window_spec

OUT = Path("artifacts/pilot/frsweep")
OUT.mkdir(parents=True, exist_ok=True)

# state variables to profile (all already causal features in the enriched frame)
METRICS = {
    "rv_120s": "realized vol 120s",
    "vpin_120s": "vpin 120s (informed flow)",
    "spread": "spread",
    "liq_total_120s": "liquidation vol 120s (binance+bybit)",
    "abs_mid_ret_120s": "|mid return| 120s",
    "notional": "trade notional",
}
COLS = ["timestamp", "rv_120s", "vpin_120s", "spread",
        "liq_binance_120s", "liq_bybit_120s", "abs_mid_ret_120s", "notional"]
WIN = get_window_ids()


def window_table(inst):
    lf = pl.scan_parquet(f"data/enriched/{inst}_sub40.parquet").select(COLS)
    rows = []
    for w in WIN:
        spec = get_window_spec(w)
        t0, t1 = spec.train_end, spec.test_end          # test period
        sub = (lf.filter((pl.col("timestamp") >= t0) & (pl.col("timestamp") < t1))
               .with_columns((pl.col("liq_binance_120s") + pl.col("liq_bybit_120s"))
                             .alias("liq_total_120s")))
        agg = sub.select(
            [pl.col(m).fill_nan(None).mean().alias(m) for m in METRICS]
            + [pl.len().alias("n_trades")]
        ).collect()
        d = agg.to_dicts()[0]
        d["window"] = w
        d["days"] = (t1 - t0) / 1e6 / 86400
        d["trades_per_day"] = d["n_trades"] / d["days"]
        rows.append(d)
    return pl.DataFrame(rows)


results = {}
for inst in ("btc", "eth"):
    t = window_table(inst)
    results[inst] = t
    print(f"\n{'='*74}\n{inst.upper()} - regime state by test window (mean over trades)\n{'='*74}")
    show = ["window"] + list(METRICS) + ["trades_per_day"]
    pdf = t.select(show).to_pandas().set_index("window")
    # ratio to cross-window median, to expose W2's elevation
    med = pdf.median()
    ratio = (pdf / med)
    print("absolute:")
    print(pdf.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\nx cross-window median (W2 is the outlier row):")
    print(ratio.to_string(float_format=lambda x: f"{x:.2f}"))
    t.write_csv(OUT / f"{inst}_w2_regime.csv")

# chart: each metric as ratio-to-median, windows on x, W2 highlighted
fig, axes = plt.subplots(2, len(METRICS), figsize=(3.0 * len(METRICS), 6.2),
                         squeeze=False)
for r, inst in enumerate(("btc", "eth")):
    pdf = results[inst].select(["window"] + list(METRICS)).to_pandas().set_index("window")
    ratio = pdf / pdf.median()
    for c, m in enumerate(METRICS):
        ax = axes[r][c]
        colors = ["#e08a1e" if w == "W2" else "#2f6f9f" for w in ratio.index]
        ax.bar(ratio.index, ratio[m].values, color=colors)
        ax.axhline(1.0, color="#999", lw=0.8, ls="--")
        ax.set_title(METRICS[m] if r == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel(f"{inst.upper()}\nx median", fontsize=9)
fig.suptitle("W2 (Jan 2026) regime vs other test windows - state x cross-window median",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "w2_regime.png", dpi=140)
plt.close(fig)
print(f"\nwrote w2_regime.png + {{btc,eth}}_w2_regime.csv to {OUT}")
