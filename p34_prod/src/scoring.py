# scoring.py - compute PNL for windows / days / trades.


# imports.
from dataclasses import dataclass
import numpy as np


# constants.
US_PER_SEC = 1_000_000
US_PER_DAY = 86_400 * US_PER_SEC
DEFAULT_TURNOVER_CONSTRAINT = 500_000


# dataclasses.
@dataclass
class ScoreResult:
    pnl_all: float
    pnl_kept: float
    pnl_filtered: float
    score: float

    kept_turnover_total: float
    kept_turnover_per_day: float
    n_days: float

    n_total: int
    n_kept: int
    n_filtered: int
    filter_rate: float

    turnover_ok: bool

@dataclass
class ThresholdResult:
    threshold: float | None
    score: float
    pnl_all: float
    pnl_kept: float
    filter_rate: float
    kept_turnover_per_day: float

    turnover_ok: bool
    filter_rate_ok: bool
    valid: bool

    # Robust-selection diagnostics (defaults keep old call sites working).
    objective: float = 0.0
    n_blocks: int = 1
    block_scores: tuple[float, ...] = ()


# helper functions.
def _valid_mask(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray | None = None,
    f: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)

    if ts is not None:
        mask &= np.isfinite(ts)

    if f is not None:
        mask &= np.isfinite(f)

    return mask

def _weighted_mean(
    x: np.ndarray,
    w: np.ndarray,
) -> float:
    s = np.sum(w)
    if s <= 0:
        return 0.0
    return float(np.sum(w * x) / s)

def make_filter_from_threshold(
    toxicity: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    if threshold is None or not np.isfinite(threshold):
        return np.zeros(len(toxicity), dtype=np.int8)

    return (toxicity > threshold).astype(np.int8)


def make_filter_from_rate_daily(
    toxicity: np.ndarray,
    ts: np.ndarray,
    filter_rate: float,
    warmup_toxicity: np.ndarray | None = None,
    lookback_days: int = 7,
    min_obs: int = 1_000,
) -> np.ndarray:
    """
    Causal rate-transfer filter with daily re-anchoring.

    Carries the target filter rate instead of a fixed threshold value: each
    day d is filtered at the (1 - filter_rate) quantile of toxicity observed
    over the trailing `lookback_days` days before d, seeded with
    `warmup_toxicity` (calibration predictions) while the trailing window has
    fewer than `min_obs` observations. Only past information enters each
    day's threshold, so the rule is implementable online and the achieved
    filter rate stays near the target under prediction-distribution drift.
    """
    toxicity = np.asarray(toxicity, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)

    f = np.zeros(len(toxicity), dtype=np.int8)

    if filter_rate <= 0.0 or len(toxicity) == 0:
        return f

    q = 1.0 - filter_rate
    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    warmup = (
        np.asarray(warmup_toxicity, dtype=np.float64)
        if warmup_toxicity is not None
        else np.empty(0)
    )
    warmup = warmup[np.isfinite(warmup)]

    for d in unique_days:
        in_day = day_ids == d

        trailing = (day_ids < d) & (day_ids >= d - lookback_days)
        pool = toxicity[trailing]
        pool = pool[np.isfinite(pool)]

        if len(pool) < min_obs:
            pool = np.concatenate([warmup, pool])

        if len(pool) == 0:
            continue

        threshold_d = np.quantile(pool, q)
        f[in_day] = toxicity[in_day] > threshold_d

    return f


# main scoring.
def score_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray,
    turnover_constraint: float = DEFAULT_TURNOVER_CONSTRAINT
) -> ScoreResult:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y = y[valid]
    w = w[valid]
    ts = ts[valid]
    f = f[valid].astype(np.int8)

    kept = f == 0
    filt = f == 1

    if len(ts) > 1:
        n_days = max((int(ts.max()) - int(ts.min())) / US_PER_DAY, 1.0)
    else:
        n_days = 1.0

    pnl_all = _weighted_mean(y, w)

    if kept.any():
        pnl_kept = _weighted_mean(y[kept], w[kept])
        kept_turnover_total = float(np.sum(w[kept]))
    else:
        pnl_kept = 0.0
        kept_turnover_total = 0.0

    if filt.any():
        pnl_filtered = _weighted_mean(y[filt], w[filt])
    else:
        pnl_filtered = 0.0

    kept_turnover_per_day = kept_turnover_total / n_days

    return ScoreResult(
        pnl_all                 = round(pnl_all, 6),
        pnl_kept                = round(pnl_kept, 6),
        pnl_filtered            = round(pnl_filtered, 6),
        score                   = round(pnl_kept - pnl_all, 6),

        kept_turnover_total     = round(kept_turnover_total, 2),
        kept_turnover_per_day   = round(kept_turnover_per_day, 2),
        n_days                  = round(float(n_days), 2),

        n_total                 = int(len(y)),
        n_kept                  = int(kept.sum()),
        n_filtered              = int(filt.sum()),
        filter_rate             = round(float(filt.mean()), 4),

        turnover_ok=kept_turnover_per_day >= turnover_constraint
    )

def compute_daily_scores_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y   = y[valid]
    w   = w[valid]
    ts  = ts[valid]
    f   = f[valid].astype(np.int8)


    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_scores = []

    for day in unique_days:
        mask = day_ids == day

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        pnl_all = _weighted_mean(y_d, w_d)

        kept = f_d == 0
        if kept.any():
            pnl_kept = _weighted_mean(y_d[kept], w_d[kept])
        else:
            pnl_kept = 0.0

        daily_scores.append(pnl_kept - pnl_all)

    return np.asarray(daily_scores, dtype=np.float64)

def compute_daily_pnl_kept_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y   = y[valid]
    w   = w[valid]
    ts  = ts[valid]
    f   = f[valid].astype(np.int8)

    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_pnl_kept = []

    for day in unique_days:
        mask = day_ids == day

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        kept = f_d == 0

        if kept.any():
            pnl_kept = _weighted_mean(y_d[kept], w_d[kept])
        else:
            pnl_kept = 0.0

        daily_pnl_kept.append(pnl_kept)

    return np.asarray(daily_pnl_kept, dtype=np.float64)

def compute_trade_diagnostics_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute trade-level diagnostics for score decomposition.

    Parameters
    ----------
    y:
        Trade-level pnl, for example pnl_30s.

    w:
        Trade weights.

    ts:
        Timestamps in microseconds.

    f:
        Binary filter:
            1 = filtered / removed,
            0 = kept.

    Returns
    -------
    diagnostics:
        Dictionary with trade-level arrays.

    Notes
    -----
    The column `score_contribution` satisfies:

        score_contribution.sum() == pnl_kept - pnl_all

    up to floating point error.
    """

    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    f = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y = y[valid]
    w = w[valid]
    ts = ts[valid]
    f = f[valid].astype(np.int8)

    n = len(y)

    is_kept = f == 0
    is_filtered = f == 1

    total_weight = float(np.sum(w))
    kept_weight = float(np.sum(w[is_kept]))

    pnl_all_contribution = np.zeros(n, dtype=np.float64)
    pnl_kept_contribution = np.zeros(n, dtype=np.float64)

    if total_weight > 0:
        pnl_all_contribution = w * y / total_weight

    if kept_weight > 0:
        pnl_kept_contribution[is_kept] = w[is_kept] * y[is_kept] / kept_weight

    score_contribution = pnl_kept_contribution - pnl_all_contribution

    kept_pnl = np.full(n, np.nan, dtype=np.float64)
    kept_pnl[is_kept] = y[is_kept]

    filtered_pnl = np.full(n, np.nan, dtype=np.float64)
    filtered_pnl[is_filtered] = y[is_filtered]

    return {
        "timestamp": ts,
        "pnl": y,
        "weight": w,
        "filter": f,
        "is_kept": is_kept,
        "is_filtered": is_filtered,

        # Contributions to weighted aggregate values.
        "pnl_all_contribution": pnl_all_contribution,
        "pnl_kept_contribution": pnl_kept_contribution,
        "score_contribution": score_contribution,

        # Raw per-trade pnl separated by decision.
        "kept_pnl": kept_pnl,
        "filtered_pnl": filtered_pnl
    }


# prediction conversion.
def to_toxicity_score(
    pred: np.ndarray,
    prediction_type: str
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)

    if prediction_type == "pnl":
        return -pred

    if prediction_type in {"toxicity", "probability"}:
        return pred

    raise ValueError(
        "Unknown prediction_type. "
        "Expected one of: 'pnl', 'toxicity', 'probability'."
    )


# threshold tuning.
def _candidate_thresholds(
    toxicity_sorted_desc: np.ndarray,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact candidate set: every unique toxicity value is a distinct filter
    "toxicity > v". Returns (thresholds, n_filtered) where n_filtered[i] is
    the number of trades strictly above thresholds[i] (= position of the
    first occurrence of the value in the descending sort).

    If the exact set is larger than max_candidates, it is thinned uniformly
    in rank space, which preserves filter-rate resolution.
    """
    n = len(toxicity_sorted_desc)

    # Positions where a new (smaller) value starts in the descending sort.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(
        toxicity_sorted_desc[1:], toxicity_sorted_desc[:-1], out=is_new[1:]
    )
    cut_positions = np.flatnonzero(is_new)          # n_filtered for each value

    if len(cut_positions) > max_candidates:
        idx = np.linspace(0, len(cut_positions) - 1, max_candidates)
        cut_positions = cut_positions[np.round(idx).astype(np.int64)]
        cut_positions = np.unique(cut_positions)

    thresholds = toxicity_sorted_desc[cut_positions]
    return thresholds, cut_positions


def _block_scores_at_thresholds(
    toxicity: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    thresholds: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """
    Score of filter "toxicity > threshold" inside each of n_blocks contiguous
    time blocks, for every candidate threshold at once.

    Returns a (n_valid_blocks, n_thresholds) matrix. Blocks with no trades
    are dropped. Same conventions as score_from_arrays: pnl_kept = 0 when
    nothing is kept.
    """
    edges = np.quantile(ts, np.linspace(0.0, 1.0, n_blocks + 1))
    edges[-1] = np.inf

    rows = []
    for b in range(n_blocks):
        in_block = (ts >= edges[b]) & (ts < edges[b + 1])
        if not in_block.any():
            continue

        tox_b = toxicity[in_block]
        wy_b = (w[in_block] * y[in_block]).astype(np.float64)
        w_b = w[in_block].astype(np.float64)

        order = np.argsort(tox_b, kind="stable")
        tox_b = tox_b[order]

        pre_wy = np.concatenate(([0.0], np.cumsum(wy_b[order])))
        pre_w = np.concatenate(([0.0], np.cumsum(w_b[order])))

        total_wy = pre_wy[-1]
        total_w = pre_w[-1]
        pnl_all_b = total_wy / total_w if total_w > 0 else 0.0

        # kept = toxicity <= threshold; ascending sort => first idx elements.
        idx = np.searchsorted(tox_b, thresholds, side="right")
        kept_wy = pre_wy[idx]
        kept_w = pre_w[idx]

        pnl_kept_b = np.where(kept_w > 0, kept_wy / np.maximum(kept_w, 1e-300), 0.0)
        rows.append(pnl_kept_b - pnl_all_b)

    return np.vstack(rows) if rows else np.zeros((0, len(thresholds)))


def tune_threshold_from_toxicity(
    toxicity: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    max_filter_rate: float,
    min_filter_rate: float | None = None,
    n_grid: int = 200,                      # kept for API compat; superseded by max_candidates
    turnover_constraint: float = DEFAULT_TURNOVER_CONSTRAINT,
    n_blocks: int = 1,
    robust_lambda: float = 0.0,
    objective: str = "mean",
    max_candidates: int = 20_000,
) -> ThresholdResult:
    """
    Choose the filter threshold on the calibration set.

    The sweep is exact and vectorized: trades are sorted by toxicity once and
    every achievable cut is scored via prefix sums in O(n log n). With the
    default n_blocks=1, robust_lambda=0.0 the selection is the argmax of the
    aggregate calibration score.

    Optional robust selection (n_blocks > 1): the calibration period is cut
    into n_blocks contiguous time blocks and each candidate threshold is
    scored inside every block; the objective is

        objective = mean_b(score_b) - robust_lambda * std_b(score_b)   ("mean")
        objective = median_b(score_b)                                  ("median")

    Constraints (filter-rate window, turnover) are evaluated on the full
    calibration set. Ties in the objective are broken toward the smaller
    filter rate.
    """
    toxicity = np.asarray(toxicity, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)

    valid = (
        np.isfinite(toxicity)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(ts)
    )

    toxicity = toxicity[valid]
    y = y[valid]
    w = w[valid]
    ts = ts[valid]

    invalid_result = ThresholdResult(
        threshold               = None,
        score                   = 0.0,
        pnl_all                 = 0.0,
        pnl_kept                = 0.0,
        filter_rate             = 0.0,
        kept_turnover_per_day   = 0.0,
        turnover_ok             = False,
        filter_rate_ok          = False,
        valid                   = False,
        objective               = 0.0,
        n_blocks                = n_blocks,
    )

    n = len(toxicity)
    if n == 0:
        return invalid_result

    if min_filter_rate is None:
        min_filter_rate = 0.0

    # --- exact aggregate sweep via one descending sort + prefix sums. ---
    order = np.argsort(-toxicity, kind="stable")
    tox_sorted = toxicity[order]
    wy_sorted = (w * y)[order]
    w_sorted = w[order]

    pre_wy = np.concatenate(([0.0], np.cumsum(wy_sorted)))
    pre_w = np.concatenate(([0.0], np.cumsum(w_sorted)))
    total_wy = pre_wy[-1]
    total_w = pre_w[-1]

    if total_w <= 0:
        return invalid_result

    pnl_all = total_wy / total_w

    if len(ts) > 1:
        n_days = max((int(ts.max()) - int(ts.min())) / US_PER_DAY, 1.0)
    else:
        n_days = 1.0

    thresholds, n_filtered = _candidate_thresholds(tox_sorted, max_candidates)

    kept_wy = total_wy - pre_wy[n_filtered]
    kept_w = total_w - pre_w[n_filtered]

    pnl_kept = np.where(kept_w > 0, kept_wy / np.maximum(kept_w, 1e-300), 0.0)
    agg_score = pnl_kept - pnl_all
    filter_rate = n_filtered / n
    kept_turnover_per_day = kept_w / n_days

    feasible = (
        (filter_rate <= max_filter_rate)
        & (filter_rate >= min_filter_rate)
        & (kept_turnover_per_day >= turnover_constraint)
    )

    if not feasible.any():
        return invalid_result

    # --- selection objective. ---
    if n_blocks > 1:
        block_matrix = _block_scores_at_thresholds(
            toxicity, y, w, ts, thresholds, n_blocks
        )
    else:
        block_matrix = agg_score[None, :]

    if block_matrix.shape[0] == 0:
        block_matrix = agg_score[None, :]

    if objective == "median":
        obj = np.median(block_matrix, axis=0)
    elif objective == "mean":
        obj = block_matrix.mean(axis=0)
        if robust_lambda > 0.0 and block_matrix.shape[0] > 1:
            obj = obj - robust_lambda * block_matrix.std(axis=0, ddof=1)
    else:
        raise ValueError("objective must be 'mean' or 'median'.")

    # Argmax over feasible candidates; ties -> smaller filter rate.
    obj_feasible = np.where(feasible, obj, -np.inf)
    best_obj = obj_feasible.max()
    ties = np.flatnonzero(obj_feasible == best_obj)
    best = ties[np.argmin(filter_rate[ties])]

    return ThresholdResult(
        threshold               = float(thresholds[best]),
        score                   = round(float(agg_score[best]), 6),
        pnl_all                 = round(float(pnl_all), 6),
        pnl_kept                = round(float(pnl_kept[best]), 6),
        filter_rate             = round(float(filter_rate[best]), 4),
        kept_turnover_per_day   = round(float(kept_turnover_per_day[best]), 2),
        turnover_ok             = True,
        filter_rate_ok          = True,
        valid                   = True,
        objective               = round(float(obj[best]), 6),
        n_blocks                = int(n_blocks),
        block_scores            = tuple(
            round(float(s), 6) for s in block_matrix[:, best]
        ),
    )