# experiment.py - unified walk-forward runner.

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from features import DERIVED_FEATURES, add_derived_features, extract_Xyw, validate_columns
from meta import iter_split_datasets
from models import ModelSpec, get_model
from scoring import (
    DEFAULT_TURNOVER_CONSTRAINT,
    make_filter_from_rate_daily,
    make_filter_from_threshold,
    score_from_arrays,
    to_toxicity_score,
    tune_threshold_from_toxicity
)
from splits import SplitData, get_window_ids, iter_window_splits


# Config / result objects
@dataclass
class WalkForwardConfig:
    """
    Configuration for final walk-forward experiment.
    """
    instrument: str
    tau: int
    model_name: str

    max_filter_rates: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])
    window_ids: list[str] | None = None

    calib_frac: float = 0.20
    min_filter_rate: float | None = None
    n_threshold_grid: int = 200
    turnover_constraint: float = DEFAULT_TURNOVER_CONSTRAINT

    # Threshold selection (see scoring.tune_threshold_from_toxicity).
    # Defaults select the argmax of the aggregate calibration score via the
    # exact sweep. Block-robust selection (threshold_blocks > 1 with
    # threshold_robust_lambda > 0) penalizes cross-block score variance.
    threshold_blocks: int = 1
    threshold_robust_lambda: float = 0.0
    threshold_objective: str = "mean"
    threshold_max_candidates: int = 20_000

    # How the tuned filter is applied to the test period:
    #   "threshold"  - carry the calibration threshold value.
    #   "rate_daily" - carry the calibration filter rate; the threshold is
    #                  re-anchored each test day on trailing toxicity (see
    #                  scoring.make_filter_from_rate_daily).
    apply_mode: str = "threshold"
    rate_lookback_days: int = 7

    # Legacy behavior preserved: target NaNs are always dropped; feature NaNs are
    # preserved unless this flag is True. LightGBM can handle NaNs, linear models
    # in models.py impute NaNs to zero.
    drop_feature_nans: bool = False

    # When using meta columns, rows without meta predictions are usually a window
    # edge / invalid-target artifact. Legacy meta runner dropped such rows.
    drop_meta_nans: bool = True

    # If requested features include derived columns that are absent, compute them.
    add_derived_if_needed: bool = True

    verbose: bool = True
    use_tqdm: bool = True

@dataclass
class WindowResult:
    window_id: str
    tau: int
    max_filter_rate: float

    threshold: float | None
    threshold_valid: bool

    calib_score: float
    test_score: float

    test_pnl_all: float
    test_pnl_kept: float
    test_pnl_filtered: float
    test_filter_rate: float
    test_kept_turnover_per_day: float
    test_turnover_ok: bool

    n_train_model: int
    n_calib: int
    n_test: int

    y_test: np.ndarray
    w_test: np.ndarray
    ts_test: np.ndarray
    pred_test: np.ndarray
    toxicity_test: np.ndarray
    f_test: np.ndarray

@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    features: list[str]
    windows: list[WindowResult]
    models: list[dict]


# Public API
def walk_forward(
    config: WalkForwardConfig,
    features: list[str],
    df: pl.DataFrame | None = None,
    split_dir: str | Path | None = None
) -> WalkForwardResult:
    """
    Unified walk-forward runner.

    Use cases
    ---------
    1. Full enriched dataset, no precomputed split files:

        result = walk_forward(config, features, df=df)

    2. Pre-split dataset from 02_meta_dataset.ipynb:

        result = walk_forward(config, features, split_dir="data/meta/btc")

    Meta vs no-meta is determined only by `features`.
    """
    if (df is None) == (split_dir is None):
        raise ValueError("Pass exactly one of `df` or `split_dir`.")

    features = list(features)
    window_ids = config.window_ids or get_window_ids()

    model_spec = get_model(
        name            = config.model_name,
        feature_names   = features
    )

    split_iter = _make_split_iterator(
        df          = df,
        split_dir   = split_dir,
        config      = config,
        window_ids  = window_ids
    )

    if config.use_tqdm:
        split_iter = tqdm(
            list(split_iter),
            desc=f"WF {config.instrument} {config.model_name} tau={config.tau}s"
        )

    all_results: list[WindowResult] = []
    models: list[dict] = []

    for split in split_iter:
        window_results, model_info = run_one_split(
            split       = split,
            config      = config,
            features    = features,
            model_spec  = model_spec
        )
        all_results.extend(window_results)
        models.append(model_info)

        gc.collect()

    return WalkForwardResult(
        config      = config,
        features    = features,
        windows     = all_results,
        models      = models
    )

def run_experiment_from_df(
    df: pl.DataFrame,
    config: WalkForwardConfig,
    features: list[str]
) -> WalkForwardResult:
    """
    Convenience wrapper around walk_forward(..., df=df).
    """
    return walk_forward(config=config, features=features, df=df)

def run_experiment_from_split_dir(
    split_dir: str | Path,
    config: WalkForwardConfig,
    features: list[str]
) -> WalkForwardResult:
    """
    Convenience wrapper around walk_forward(..., split_dir=split_dir).
    """
    return walk_forward(config=config, features=features, split_dir=split_dir)

def run_one_split(
    split: SplitData,
    config: WalkForwardConfig,
    features: list[str],
    model_spec: ModelSpec
) -> tuple[list[WindowResult], dict]:
    """
    Train one final model on split.train_model, tune thresholds on split.calib,
    evaluate on split.test.
    """
    split = _prepare_split(split, features, config)

    if config.verbose:
        print(f"\n=== WF {config.instrument} {split.window_id} ===")
        print(f"train_model: {split.train_model.height:,}")
        print(f"calib      : {split.calib.height:,}")
        print(f"test       : {split.test.height:,}")
        print(f"features   : {len(features)}")

    (X_tr, y_tr, w_tr, ts_tr), (X_ca, y_ca, w_ca, ts_ca), (X_te, y_te, w_te, ts_te) = _extract_split_arrays(
        split               = split,
        tau                 = config.tau,
        features            = features,
        drop_feature_nans   = config.drop_feature_nans
    )

    _validate_non_empty_arrays(split.window_id, X_tr, X_ca, X_te)

    model       = model_spec.train_fn(X_tr, y_tr, w_tr)
    pred_calib  = model_spec.predict_fn(model, X_ca)
    pred_test   = model_spec.predict_fn(model, X_te)

    toxicity_calib = to_toxicity_score(
        pred            = pred_calib,
        prediction_type = model_spec.prediction_type
    )
    toxicity_test = to_toxicity_score(
        pred            = pred_test,
        prediction_type = model_spec.prediction_type
    )

    window_results: list[WindowResult] = []

    for max_fr in config.max_filter_rates:
        threshold = tune_threshold_from_toxicity(
            toxicity            = toxicity_calib,
            y                   = y_ca,
            w                   = w_ca,
            ts                  = ts_ca,
            max_filter_rate     = max_fr,
            min_filter_rate     = config.min_filter_rate,
            n_grid              = config.n_threshold_grid,
            turnover_constraint = config.turnover_constraint,
            n_blocks            = config.threshold_blocks,
            robust_lambda       = config.threshold_robust_lambda,
            objective           = config.threshold_objective,
            max_candidates      = config.threshold_max_candidates
        )

        if not threshold.valid:
            f_test = np.zeros(len(toxicity_test), dtype=np.int8)
        elif config.apply_mode == "rate_daily":
            f_test = make_filter_from_rate_daily(
                toxicity        = toxicity_test,
                ts              = ts_te,
                filter_rate     = threshold.filter_rate,
                warmup_toxicity = toxicity_calib,
                lookback_days   = config.rate_lookback_days
            )
        elif config.apply_mode == "threshold":
            f_test = make_filter_from_threshold(toxicity_test, threshold.threshold)
        else:
            raise ValueError(
                f"Unknown apply_mode='{config.apply_mode}'. "
                "Expected 'threshold' or 'rate_daily'."
            )

        test_score = score_from_arrays(
            y   = y_te,
            w   = w_te,
            ts  = ts_te,
            f   = f_test,
            turnover_constraint = config.turnover_constraint
        )

        window_results.append(
            WindowResult(
                window_id                   = split.window_id,
                tau                         = config.tau,
                max_filter_rate             = float(max_fr),

                threshold                   = threshold.threshold,
                threshold_valid             = bool(threshold.valid),

                calib_score                 = float(threshold.score),
                test_score                  = float(test_score.score),

                test_pnl_all                = float(test_score.pnl_all),
                test_pnl_kept               = float(test_score.pnl_kept),
                test_pnl_filtered           = float(test_score.pnl_filtered),
                test_filter_rate            = float(test_score.filter_rate),
                test_kept_turnover_per_day  = float(test_score.kept_turnover_per_day),
                test_turnover_ok            = bool(test_score.turnover_ok),

                n_train_model               = int(X_tr.shape[0]),
                n_calib                     = int(X_ca.shape[0]),
                n_test                      = int(X_te.shape[0]),

                y_test                      = y_te,
                w_test                      = w_te,
                ts_test                     = ts_te,
                pred_test                   = pred_test,
                toxicity_test               = toxicity_test,
                f_test                      = f_test
            )
        )

    model_info = {
        "window": split.window_id,
        "tau": config.tau,
        "instrument": config.instrument,
        "model_name": config.model_name,
        "prediction_type": model_spec.prediction_type,
        "model": model,
        "features": list(features),
        "n_train_model": int(X_tr.shape[0]),
        "n_calib": int(X_ca.shape[0]),
        "n_test": int(X_te.shape[0])
    }

    return window_results, model_info

def results_to_frame(result: WalkForwardResult) -> pl.DataFrame:
    """
    Convert WalkForwardResult to a compact Polars summary table.
    """
    rows = []

    for r in result.windows:
        rows.append({
            "instrument": result.config.instrument,
            "model_name": result.config.model_name,
            "window": r.window_id,
            "tau": f"{r.tau}s",
            "max_filter_rate": r.max_filter_rate,
            "n_train_model": r.n_train_model,
            "n_calib": r.n_calib,
            "n_test": r.n_test,
            "threshold": r.threshold,
            "threshold_valid": r.threshold_valid,
            "calib_score": r.calib_score,
            "test_score": r.test_score,
            "test_pnl_all": r.test_pnl_all,
            "test_pnl_kept": r.test_pnl_kept,
            "test_pnl_filtered": r.test_pnl_filtered,
            "test_filter_rate (%)": round(100.0 * r.test_filter_rate, 2),
            "test_kept_usd/day": r.test_kept_turnover_per_day,
            "test_turnover_ok": r.test_turnover_ok
        })

    return pl.DataFrame(rows)

def collect_oos_arrays(
    result: WalkForwardResult,
    max_filter_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate OOS arrays across windows for a selected max_filter_rate.

    Returns:
        y_oos, w_oos, ts_oos, f_oos
    """
    selected = [
        r for r in result.windows
        if np.isclose(r.max_filter_rate, max_filter_rate)
    ]

    if not selected:
        raise ValueError(f"No window results for max_filter_rate={max_filter_rate}.")

    y = np.concatenate([r.y_test for r in selected])
    w = np.concatenate([r.w_test for r in selected])
    ts = np.concatenate([r.ts_test for r in selected])
    f = np.concatenate([r.f_test for r in selected])

    return y, w, ts, f

def collect_oos_predictions(
    result: WalkForwardResult,
    max_filter_rate: float,
) -> pl.DataFrame:
    """
    Return OOS predictions/filters as a Polars DataFrame.
    """
    selected = [
        r for r in result.windows
        if np.isclose(r.max_filter_rate, max_filter_rate)
    ]

    if not selected:
        raise ValueError(f"No window results for max_filter_rate={max_filter_rate}.")

    frames = []
    for r in selected:
        frames.append(
            pl.DataFrame({
                "window": [r.window_id] * len(r.y_test),
                "tau": [r.tau] * len(r.y_test),
                "max_filter_rate": [r.max_filter_rate] * len(r.y_test),
                "timestamp": r.ts_test,
                "y": r.y_test,
                "w": r.w_test,
                "pred": r.pred_test,
                "toxicity": r.toxicity_test,
                "filter": r.f_test
            })
        )

    return pl.concat(frames, how="vertical_relaxed")



# Internal helpers
def _make_split_iterator(
    df: pl.DataFrame | None,
    split_dir: str | Path | None,
    config: WalkForwardConfig,
    window_ids: list[str]
):
    if split_dir is not None:
        return iter_split_datasets(
            split_dir   = split_dir,
            instrument  = config.instrument,
            window_ids  = window_ids
        )

    assert df is not None
    return iter_window_splits(
        df          = df,
        window_ids  = window_ids,
        calib_frac  = config.calib_frac
    )

def _prepare_split(
    split: SplitData,
    features: list[str],
    config: WalkForwardConfig,
) -> SplitData:
    train_model = split.train_model
    calib       = split.calib
    test        = split.test

    if config.add_derived_if_needed:
        train_model = _ensure_derived_features_if_needed(train_model, features)
        calib       = _ensure_derived_features_if_needed(calib, features)
        test        = _ensure_derived_features_if_needed(test, features)

    if config.drop_meta_nans:
        meta_cols = [c for c in features if c.startswith("meta_")]
        if meta_cols:
            validate_columns(train_model, meta_cols, context=f"{split.window_id}/train_model meta")
            validate_columns(calib, meta_cols, context=f"{split.window_id}/calib meta")
            validate_columns(test, meta_cols, context=f"{split.window_id}/test meta")

            train_model = train_model.drop_nulls(meta_cols).drop_nans(meta_cols)
            calib       = calib.drop_nulls(meta_cols).drop_nans(meta_cols)
            test        = test.drop_nulls(meta_cols).drop_nans(meta_cols)

    train_full = pl.concat([train_model, calib], how="vertical_relaxed")

    return SplitData(
        window_id   = split.window_id,
        train_full  = train_full,
        train_model = train_model,
        calib       = calib,
        test        = test
    )

def _ensure_derived_features_if_needed(
    df: pl.DataFrame,
    features: list[str]
) -> pl.DataFrame:
    required_derived = [c for c in DERIVED_FEATURES if c in features]
    missing_derived = [c for c in required_derived if c not in df.columns]

    if missing_derived:
        df = add_derived_features(df)

    return df

def _extract_split_arrays(
    split: SplitData,
    tau: int,
    features: list[str],
    drop_feature_nans: bool
):
    X_tr, y_tr, w_tr, ts_tr = extract_Xyw(
        split.train_model,
        tau         = tau,
        features    = features,
        drop_feature_nans = drop_feature_nans
    )
    X_ca, y_ca, w_ca, ts_ca = extract_Xyw(
        split.calib,
        tau         = tau,
        features    = features,
        drop_feature_nans = drop_feature_nans
    )
    X_te, y_te, w_te, ts_te = extract_Xyw(
        split.test,
        tau         = tau,
        features    = features,
        drop_feature_nans = drop_feature_nans
    )

    return (
        (X_tr, y_tr, w_tr, ts_tr),
        (X_ca, y_ca, w_ca, ts_ca),
        (X_te, y_te, w_te, ts_te)
    )

def _validate_non_empty_arrays(
    window_id: str,
    X_tr: np.ndarray,
    X_ca: np.ndarray,
    X_te: np.ndarray
) -> None:
    if len(X_tr) == 0:
        raise ValueError(f"{window_id}: train_model is empty after filtering.")
    if len(X_ca) == 0:
        raise ValueError(f"{window_id}: calib is empty after filtering.")
    if len(X_te) == 0:
        raise ValueError(f"{window_id}: test is empty after filtering.")
