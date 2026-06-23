# Task 3 — Validation · Team Griffin

Self-contained folder. `task_3_validation_team_griffin.ipynb` (executed; `.html` alongside) implements the
official markout/PnL/Score, builds baselines (incl. a **random** filter sanity check), and evaluates whether
the signal filters are *statistically* superior: walk-forward / rolling-window / purged-K-fold /
combinatorial-purged CV, Diebold–Mariano (HAC) tests, stationary block bootstrap, the Model Confidence Set,
and the Deflated Sharpe Ratio. Verdict is computed from the actual results (no hardcoded outcomes).

**Library (bundled):** `cmf_core.py`, `stream.py`, `factory.py`, `test_cmf_core.py`.

**Run**
```bash
# 1) place the parquet tape at  <repo-root>/data/liquidation_task/data/   (or set HFT_ROOT)
# 2) build artifacts once (writes <repo-root>/artifacts/):
python factory.py --symbols btcusdt && python factory.py --symbols ethusdt && python factory.py --merge
# 3) open the notebook from THIS folder and run top-to-bottom
python test_cmf_core.py   # optional: spec + causality unit tests
```
Data and `artifacts/` live at the repo root (shared across tasks, git-ignored).
