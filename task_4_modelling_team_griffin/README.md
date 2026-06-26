# Task 4 - Modelling · Team Griffin

Self-contained folder. `task_4_modelling_team_griffin.ipynb` (executed; `.html` alongside) builds a
per-horizon ML maker-fill filter (binary 0 keep / 1 filter for each tau in {30,120,300}s) from the Task-2
market classes, Binance + Bybit liquidation data, and Task-1 microstructure, validated with the Task-3
official scoring on an **expanding-window** scheme (W1-W5, one-month step, 7-day embargo) over the full
6-month tape (2025-11 -> 2026-04, ~2.2B trades).

Models per horizon: **Ridge** (interpretable), **LightGBM** (nonlinear), and the **best Task-2 rule**
(cascade-absorption), plus **random** and **keep-all** baselines. The objective is the official OOS Score
subject to the 500k USD/day kept-turnover floor; AUC/F1 are diagnostics only.

**Library (bundled):** `cmf_core.py`, `stream.py`, `model4.py` (Task-4.1 feature battery), `run_task4.py`
(pipeline), plus `factory.py` / `test_cmf_core.py`.

**Run**
```bash
pip install -r ../requirements.txt   # incl. lightgbm
# place the 6-month tape at  <repo-root>/data/full/liquidation_task/data/   (or set HFT_ROOT)
python run_task4.py thresholds        # freeze class thresholds on Nov  -> <repo-root>/artifacts4/
python run_task4.py sample            # build weighted training sample (Nov-Mar)
python run_task4.py oos               # train per-window models + OOS scoring (W1-W5)
# then open the notebook from THIS folder and run top-to-bottom
```
Data and `artifacts4/` live at the repo root (git-ignored). Exact per-trade scoring over 2.2B trades is
intractable, so OOS uses a large representation-weighted scoring sample per test day (documented in the
notebook); the model *ranking* and the `random`~0 calibration are robust to it.
