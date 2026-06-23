# Task 1 — Exploratory Data Analysis · Team Griffin

Self-contained folder. `task_1_eda_team_griffin.ipynb` (executed; `.html` alongside) profiles the
Binance/Bybit perpetual tape (trades, BBO, liquidations) across three tiers: data quality, microstructure &
regimes, and advanced exploration (price impact, liquidation heatmap, cross-venue lead–lag).

**Library (bundled):** `cmf_core.py` (official markout/PnL/Score + causal feature primitives + stats),
`stream.py` (streaming features/classes/filters), `factory.py` (one causal pass → `../artifacts/`),
`test_cmf_core.py` (spec + causality unit tests).

**Run**
```bash
# 1) place the parquet tape at  <repo-root>/data/liquidation_task/data/   (or set HFT_ROOT)
# 2) build artifacts once (writes <repo-root>/artifacts/):
python factory.py --symbols btcusdt && python factory.py --symbols ethusdt && python factory.py --merge
# 3) open the notebook from THIS folder and run top-to-bottom
```
Data and `artifacts/` live at the repo root (shared across tasks, git-ignored). The notebook locates the
bundled library in this folder and the data/artifacts at the repo root automatically.
