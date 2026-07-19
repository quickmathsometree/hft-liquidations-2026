# Toxic maker-flow filter — filter-rate sweep & signed order-flow block

**Repo / version.** `cmf-team/hft-liquidations-2026`, PR
[#35](https://github.com/cmf-team/hft-liquidations-2026/pull/35), branch
`p34-prod`. Numbers in this report were produced by the scripts committed at
`e2389e4` (`fr_sweep.py`, `pilot8.py`, `report_metrics.py`). The production
recipe under test was frozen earlier at `e7c7d0e` (pilots 6–7 + ETH
confirmation, notebook §4b–4c).

**Instruments / data.** BTC-PERP and ETH-PERP, test window Dec 2025 – Apr 2026,
walk-forward W1–W5, evaluated on the 1/40 trade subsample (`*_sub40.parquet`,
v2 schema). Score is `PnL_kept − PnL_all` weighted by clipped notional, in bps;
positive means the filter earns over the unfiltered stream.

This report covers two experiments on top of the frozen production model:

- **Exp A — filter-rate sweep.** No model change; the filter-rate cap is swept
  over a full ladder instead of the two reported endpoints, to show whether the
  score is a tunable optimum or a point on a capacity curve.
- **Exp B — pilot 8 / H-08.** One feature block added (signed short-window
  order-flow imbalance + flow-vs-book divergence), attributed cleanly against
  the advanced control.

---

## 1. Diff from the previous version

Baseline (unchanged, `e7c7d0e`): **advanced-40 features, τ = 120 s, 4× LGBM-Huber
ensemble with rank aggregation, exact threshold sweep**, walk-forward W1–W5.
Established as champion across pilots 6–7 and confirmed out-of-family on ETH.

| Exp | Substantive change | Hypothesis |
|-----|--------------------|------------|
| A | filter-rate cap `{0.10, 0.30}` → `{0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 0.99, 0.999}`; nothing else | The reported +0.19 / +1.39 endpoints are points on a monotone capacity curve, not a filter-rate tuned for score. |
| B (H-08) | advanced-40 **+ 3 features**: `trade_imb_5s`, `trade_imb_10s` (signed short-window OFI), `div_z` (flow-vs-book divergence z) | Signed short-window order flow and a flow/book divergence add OOS signal beyond advanced-40 (which carries only *unsigned* vpin and instantaneous book imbalance). |

Both are single-lever changes so the effect isolates cleanly.

---

## 2. Results & strategy economics

### 2.1 Production operating point — advanced @ τ=120 s, fr ≤ 0.30

Window-by-window, both instruments (`report_metrics.py`):

**BTC**

| window | score, bps | filter rate | kept turnover, $/day |
|--------|-----------:|------------:|---------------------:|
| W1 | +0.166 | 29.6% | 177.0 M |
| W2 | +0.589 | 31.4% | 178.3 M |
| W3 | +0.041 | 26.5% | 214.8 M |
| W4 | +0.052 | 14.6% | 235.0 M |
| W5 | +0.110 | 3.6% | 214.3 M |

median +0.110, range [+0.041, +0.589], **positive 5/5**. Max drawdown of
cumulative daily kept-PnL: **−1.56 bps over 114 days**. Kept turnover never falls
below **$177 M/day = 354× the $500 k/day floor** — the turnover constraint never
binds. Top 1% of kept trades carry 36.6% of gross positive kept PnL.

**ETH**

| window | score, bps | filter rate | kept turnover, $/day |
|--------|-----------:|------------:|---------------------:|
| W1 | +0.037 | 2.1% | 241.2 M |
| W2 | +0.918 | 29.1% | 174.1 M |
| W3 | +0.171 | 31.1% | 154.0 M |
| W4 | +0.186 | 30.6% | 166.8 M |
| W5 | +0.170 | 23.8% | 150.6 M |

median +0.171, range [+0.037, +0.918], **positive 5/5**. Max drawdown **−3.10 bps
over 114 days**. Kept turnover ≥ $150.6 M/day = 301× the floor. Top 1% of kept
trades carry 45.5% of gross positive kept PnL.

Both instruments confirm the recipe holds across every OOS window; W2 is the
strong-regime outlier on both (see §6, Limitations).

### 2.2 Filter-rate ladder (Exp A) — τ=120 s

**BTC**

| fr ≤ | mean, bps | worst window | pos | kept, $M/day |
|------|----------:|-------------:|:---:|-------------:|
| 0.30 | +0.191 | +0.042 | 5/5 | 203.9 |
| 0.60 | +0.430 | +0.100 | 5/5 | 115.6 |
| 0.90 | +1.386 | +0.207 | 5/5 | 35.8 |
| 0.99 | +5.828 | −2.645 | 4/5 | 7.0 |
| 0.999 | +22.918 | −4.285 | 4/5 | 0.5 |

**ETH**

| fr ≤ | mean, bps | worst window | pos | kept, $M/day |
|------|----------:|-------------:|:---:|-------------:|
| 0.30 | +0.297 | +0.037 | 5/5 | 177.4 |
| 0.60 | +0.623 | +0.037 | 5/5 | 126.0 |
| 0.90 | +2.451 | +0.865 | 5/5 | 32.5 |
| 0.99 | +10.822 | +2.939 | 5/5 | 8.3 |
| 0.999 | +42.506 | +4.680 | 5/5 | 1.1 |

Reading of Exp A (three facts that settle the "is 0.6 an overfit?" question):

1. **Score rises monotonically with fr on both instruments and every τ** — fr is
   an aggressiveness/capacity dial, not a parameter with an interior optimum to
   fit. There is no "best fr" to select post-hoc.
2. **The rise is bought with capacity.** BTC τ=120: fr0.3 = +0.19 bps on $204 M/day,
   fr0.9 = +1.39 on $36 M/day, fr0.999 = +22.9 on **$0.5 M/day**. You pick fr for a
   turnover target and read the score off the frontier, you do not maximize it.
3. **The tail fr ≥ 0.99 is a mirage.** The large means ride a handful of kept
   trades; the worst window turns sharply negative (BTC −2.6 / −4.3 bps) and
   window robustness breaks from 5/5 to 4/5. Not deployable.

Deployable range is **fr ≤ 0.90**, where τ=120 dominates all τ on both
instruments and stays positive in all five windows.

Charts: `report_assets/{btc,eth}_frsweep_curve.png` (deployable panel + full-ladder
mirage with per-window worst overlaid), `..._frontier.png` (capacity/score
frontier), `..._turnover.png` (capacity collapse).

### 2.3 H-08 result (Exp B) — flow8 vs advanced, BTC

Window-by-window delta (flow8 − advanced) and pooled paired daily t (114 days):

**τ = 120 s, fr ≤ 0.30**

| window | advanced | flow8 | Δ |
|--------|---------:|------:|---:|
| W1 | +0.162 | +0.134 | −0.028 |
| W2 | +0.587 | +0.616 | +0.029 |
| W3 | +0.041 | +0.046 | +0.004 |
| W4 | +0.052 | +0.114 | +0.062 |
| W5 | +0.110 | +0.104 | −0.006 |

win rate 3/5, mean Δ +0.012 bps, **paired daily t = +0.55** (not significant).

| τ, fr | Δ mean, bps | paired daily t |
|-------|------------:|---------------:|
| 120, 0.10 | −0.013 | **−2.42** |
| 120, 0.30 | +0.012 | +0.55 |
| 30, 0.10 | −0.002 | −1.51 |
| 30, 0.30 | +0.008 | +2.07 |

**Verdict: H-08 rejected.** flow8 does not beat advanced-40 at the production
horizon — at the tight budget (fr0.1) it is *significantly worse* (t = −2.42),
at fr0.3 it is a coin-flip (t = +0.55, 3/5 windows). The only positive cell is
τ=30 / fr0.3 (t = +2.07, +0.008 bps), and its win is concentrated in W1 (Δ+0.041,
the other four windows ≈ 0). This is the same weak τ=30 spot pilot 7's `fast30`
found, which washed out on ETH (paired t +0.57) — so the ETH confirmation run
was **not** spent. The model does assign the new columns gain importance and
still loses OOS score: importance ≠ value, consistent with pilots 6–7.

---

## 3. Data & training pipeline (exact configuration)

- **Dataset:** `{inst}_sub40.parquet`, 1/40 of the trade stream (BTC 20.1 M rows,
  ETH 34.3 M). v2 build (rolling 1/5/10/30/120/300/600 s, momentum 5–600 s);
  markout targets `pnl_{30,120,300}s`.
- **Validation:** walk-forward, 5 expanding windows W1–W5, `calib_frac = 0.20`
  (per window: train → calibrate threshold on the last 20% → test on the next
  block). No shuffling; splits are chronological.
- **Model:** 4-member LightGBM Huber ensemble, aggregated by **rank mean**
  (scale-free, matches the rank-based downstream filter). Member hyperparameters
  are **frozen** since the pilot-3 tuning (`ENSEMBLE_MEMBER_CONFIGS`,
  `random_state=42`); they are not re-tuned per window:

  | member | huber_α | max_depth | leaves | lr | n_est | min_child |
  |--------|--------:|----------:|-------:|----:|------:|----------:|
  | 1 | 0.80 | 5 | 31 | 0.04 | 1200 | 700 |
  | 2 | 0.85 | 6 | 63 | 0.04 | 1200 | 600 |
  | 3 | 0.90 | −1 | 63 | 0.05 | 1000 | 500 |
  | 4 | 0.95 | −1 | 127 | 0.03 | 1500 | 800 |

- **Threshold:** exact vectorized sweep (sort + prefix sums) picking the argmax
  calibration score subject to `max_filter_rate` and the $500 k/day turnover
  floor; applied to test as a fixed threshold. `apply_mode="rate_daily"` (daily
  re-anchoring) is available and score-neutral but not used here.
- **Features:** advanced-40 (microstructure + derived: vpin, rv, Binance/Bybit
  liquidation flow over ≤120 s, signed/relative versions, liquidation-alignment).
  All derived features are backward-looking and computed on the full stream
  before splitting (causal, no lookahead). Exp B adds the three flow8 columns,
  built the same way (rolling `sign*notional` / rolling `notional`; 300 s
  z-scores).

**Leakage note (honest):** hyperparameters are frozen (no per-window tuning, no
leakage there). Feature-*set* selection (advanced vs extended/momentum/etc.) was
done by comparing on the same W1–W5 — a mild selection-on-full-data effect.
Nothing in the pipeline peeks at test-window targets to fit thresholds or models.

---

## 4. Walk-forward results & retraining stability

Across-window summary for the production point (§2.1): median +0.110 (BTC) /
+0.171 (ETH), positive **5/5** on both, worst window +0.041 / +0.037. The score
is not carried by a single split — W2 is the best on both instruments but every
window is positive.

Top gain importance, production advanced @ τ=120 (mean over windows):
`vpin_120s` 12.0%, `signed_liq_bybit_120s` 10.5%, `liq_bybit_120s` 7.8%,
`hour` 6.2%, `signed_liq_binance_30s` 5.7%, `liq_binance_30s` 5.5%,
`liq_aligned_5s` 5.2%, `rv_120s` 5.0%. The signal sits in liquidation flow (signed
and unsigned), consistent with every prior pilot.

**Not yet computed (see Limitations):** prediction correlation between adjacent
windows, and per-window top-20 overlap. The pilots save mean importance over
windows, not per-window, so retraining stability is currently argued only from
the 5/5 positive-window rate and the stable top-feature list, not from an
adjacent-window correlation number.

---

## 5. Hypothesis log

Append-only; deltas are vs the advanced-40 control at the same τ / fr unless noted.

| id | change | metric | win rate | decision | motivation |
|----|--------|--------|----------|----------|------------|
| H-06 | extended-63 feature set | τ120 loses, τ300 helps but unstable | — | reject | feature windows must match markout horizon |
| H-07a | `fast30` block @ τ30 | +0.075→+0.091, paired t +2.22 (BTC) | 4/5 | **reject** | small, and washes on ETH (t +0.57) |
| H-07b | momentum block @ τ120 | −0.056, paired t −1.1 | — | reject | momentum overfits at 120 s despite 29% gain importance |
| H-07c | `roll300` block @ τ300 | +0.258, t +3.07 (BTC) | 5/5 | **reject** | does not transfer — ETH τ300 negative, W4 −1.24 |
| H-ETH | advanced @ τ120 on ETH | +0.304, daily t +2.35 | 5/5 | **keep** | production recipe confirmed out-of-family |
| H-fr | filter-rate ladder (Exp A) | monotone; tail fr≥0.99 mirage | — | informational | fr is a capacity dial; deploy fr ≤ 0.90 |
| H-08 | signed OFI + div_z (Exp B) | τ120 fr0.1 t −2.42; fr0.3 t +0.55 | 3/5 | **reject** | no gain at production horizon; τ30 blip is W1-only |

Net: nothing promoted since `e7c7d0e`. Production stays **advanced-40 @ τ=120,
rank ensemble, exact threshold sweep**.

---

## 6. Limitations

- **Same five windows** have served every comparison in this project; repeated
  reuse for model/feature selection is a mild multiple-comparisons / leakage risk
  (§3). Feature-set choice touched all windows.
- **W1 / τ=300 is degenerate** — every ensemble member early-stops at ~1 tree on
  the shortest train slice; τ=300 conclusions rest on W2–W5.
- **W2 is a distinct high-score regime** on both instruments (BTC +0.59, ETH +0.92
  at the production point). Regime handling (sample weights / a regime feature)
  is not yet done — flagged as the next honest improvement, not claimed here.
- **fr ≥ 0.99 is not deployable** (kept turnover $0.5–8 M/day, worst window
  negative on BTC). Included only to show where the capacity curve breaks.
- **Retraining stability** is argued from positive-window rate + stable top
  features, not from adjacent-window prediction correlation (not computed).
- **Untested and potentially material:** cross-instrument features (BTC↔ETH
  lead-lag) — needs time-alignment of the two datasets; `opposing_cascade` —
  needs directional liquidation volume (buy-liq vs sell-liq), absent from the
  enriched data (only total `liq_*` is stored; `signed_liq` = trade-sign × total),
  so a rebuild from raw is required. Andrey's momentum-style features
  (`impact_momentum`, `volatility_breakout`) are expected to fail at τ=120 given
  H-07b.
- **Horizon grid is coarse** ({30, 120, 300}); a 90/180 bracket would confirm 120
  is a genuine peak but needs new `pnl_{90,180}s` targets (rebuild) and risks the
  same select-by-score critique, so it is parked.
- Everything is on the 1/40 subsample; full-stream turnover is ≈ ×40 the numbers
  above.
