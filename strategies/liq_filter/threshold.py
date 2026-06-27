"""
Threshold calibration under the turnover constraint, and filter application.

fit_threshold is label-free (uses only w_i), so it works on the hidden test.
"""

from __future__ import annotations

import numpy as np

from strategies.liq_filter.config import TURNOVER_FLOOR_PER_DAY


def fit_threshold(
    raw_score: np.ndarray,
    w: np.ndarray,
    num_days: float,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
) -> float:
    """
    Pick threshold t* so kept trades (raw_score >= t*) have daily turnover >= target.

    Algorithm:
      1. Sort by raw_score descending.
      2. Accumulate w from top.
      3. t* = score of the marginal trade where cumsum/num_days >= target.
      4. If all trades together don't meet target, return -inf (keep everything).

    Label-free: requires only w_i, not pnl. Valid on hidden test.
    """
    score = np.asarray(raw_score, dtype="float64")
    weight = np.asarray(w, dtype="float64")

    ok = np.isfinite(score) & np.isfinite(weight) & (weight > 0)

    if not np.any(ok):
        return -np.inf

    days = max(float(num_days), 1e-12)
    need = float(target_turnover_per_day) * days

    total = float(weight[ok].sum())
    if total < need:
        return -np.inf

    score_ok = score[ok]
    weight_ok = weight[ok]

    order = np.argsort(-score_ok, kind="mergesort")
    score_sorted = score_ok[order]
    weight_sorted = weight_ok[order]

    csum = np.cumsum(weight_sorted)
    k = int(np.searchsorted(csum, need, side="left"))

    return float(score_sorted[min(k, len(score_sorted) - 1)])


def apply_filter(
    raw_score: np.ndarray,
    threshold: float,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    f_i = 1 (filter) if raw_score < threshold, else 0 (keep).
    Edge trades (edge_mask=True) forced to f_i = 0 (free turnover, out of score).
    Returns int array of 0/1.
    """
    score = np.asarray(raw_score, dtype="float64")

    f = (score < float(threshold)).astype(np.int8)

    bad = ~np.isfinite(score)
    f[bad] = 0

    if edge_mask is not None:
        edge = np.asarray(edge_mask, dtype=bool)
        f[edge] = 0

    return f