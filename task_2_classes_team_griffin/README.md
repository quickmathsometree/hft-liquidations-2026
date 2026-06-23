# Task 2 — Interpretable Market Classes · Team Griffin

Self-contained folder. `task_2_classes_team_griffin.ipynb` (executed; `.html` alongside) discovers
causal, at-or-before-trade conditions that separate Binance trades by expected maker **markout** and
**turnover**. Headline finding: the maker is *paid to absorb liquidation cascades* — trades aligned with
recent cross-venue liquidation pressure mean-revert and earn positive markout. Class thresholds are frozen
on TRAIN and applied unchanged to validation (no leakage).

**Library (bundled):** `cmf_core.py`, `stream.py`, `factory.py`, `test_cmf_core.py`.

**Run**
```bash
# 1) place the parquet tape at  <repo-root>/data/liquidation_task/data/   (or set HFT_ROOT)
# 2) build artifacts once (writes <repo-root>/artifacts/):
python factory.py --symbols btcusdt && python factory.py --symbols ethusdt && python factory.py --merge
# 3) open the notebook from THIS folder and run top-to-bottom
```
Data and `artifacts/` live at the repo root (shared across tasks, git-ignored).
