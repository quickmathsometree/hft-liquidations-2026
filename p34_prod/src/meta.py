# meta.py - build per-window datasets enriched with meta-features.

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

from features import (
    DERIVED_FEATURES,
    add_derived_features,
    get_feature_set,
    get_meta_features,
    validate_columns,
)
from splits import SplitData, get_window_ids, iter_window_splits


# Config / result objects
@dataclass
class MetaDatasetConfig:
    """
    Configuration for building a split meta-feature dataset.

    Current implemented meta kind:
        adverse:
            meta_adv_{tau}s = P(pnl_{tau}s is adverse)

    The final files are already split into train_model / calib / test for every
    walk-forward window. This is important: final training should not re-split
    these files.
    """
    instrument: str

    horizons_s: tuple[int, ...] = (30, 120, 300)
    window_ids: list[str] | None = None

    base_feature_set: str = "base_plus_derived"
    meta_kind: str = "adverse"

    calib_frac: float = 0.20
    n_folds: int = 5

    adverse_mode: str = "sign"          # "sign" -> pnl < 0; "quantile" -> pnl < q
    adverse_quantile: float = 0.25       # used only if adverse_mode == "quantile"

    lgbm_params: dict = field(default_factory=lambda: dict(
        objective           = "binary",
        boosting_type       = "gbdt",
        n_estimators        = 400,
        learning_rate       = 0.03,
        num_leaves          = 31,
        max_depth           = 6,
        min_child_samples   = 1000,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        reg_lambda          = 5.0,
        random_state        = 42,
        n_jobs              = -1,
        force_col_wise      = True,
        verbosity           = -1
    ))

    overwrite: bool = False
    verbose: bool = True
    use_tqdm: bool = True

@dataclass
class MetaWindowResult:
    """
    One window with meta-features attached.
    """
    window_id: str
    train_model: pl.DataFrame
    calib: pl.DataFrame
    test: pl.DataFrame
    meta_cols: list[str]


# File naming / split load-save helpers
def split_file_path(
    split_dir: str | Path,
    instrument: str,
    window_id: str,
    part: str
) -> Path:
    """
    Return path for a split parquet part.

    Expected names:
        btc_W1_train_model.parquet
        btc_W1_calib.parquet
        btc_W1_test.parquet
    """
    if part not in {"train_model", "calib", "test"}:
        raise ValueError("part must be one of: 'train_model', 'calib', 'test'.")

    return Path(split_dir) / f"{instrument}_{window_id}_{part}.parquet"

def split_files_exist(
    split_dir: str | Path,
    instrument: str,
    window_id: str
) -> bool:
    """
    Check whether all three split files exist and are non-empty.
    """
    return all(
        split_file_path(split_dir, instrument, window_id, part).exists()
        and split_file_path(split_dir, instrument, window_id, part).stat().st_size > 0
        for part in ("train_model", "calib", "test")
    )

def save_split_dataset(
    result: MetaWindowResult | SplitData,
    output_dir: str | Path,
    instrument: str,
    compression: str = "zstd"
) -> None:
    """
    Save train_model / calib / test split files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_id = result.window_id

    parts = {
        "train_model": result.train_model,
        "calib": result.calib,
        "test": result.test
    }

    for part, df in parts.items():
        path = split_file_path(output_dir, instrument, window_id, part)
        df.write_parquet(path, compression=compression, statistics=True)
        print(f"saved {path.name:>30s}  rows={df.height:,}")

def load_split_dataset(
    split_dir: str | Path,
    instrument: str,
    window_id: str
) -> SplitData:
    """
    Load train_model / calib / test split files and return SplitData.

    train_full is reconstructed as vertical concat of train_model and calib.
    It is kept only for completeness; final experiments use train_model, calib,
    and test directly.
    """
    split_dir = Path(split_dir)

    train_model_path    = split_file_path(split_dir, instrument, window_id, "train_model")
    calib_path          = split_file_path(split_dir, instrument, window_id, "calib")
    test_path           = split_file_path(split_dir, instrument, window_id, "test")

    missing = [
        str(p) for p in (train_model_path, calib_path, test_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing))

    train_model     = pl.read_parquet(train_model_path)
    calib           = pl.read_parquet(calib_path)
    test            = pl.read_parquet(test_path)

    train_full = pl.concat([train_model, calib], how="vertical_relaxed")

    return SplitData(
        window_id   = window_id,
        train_full  = train_full,
        train_model = train_model,
        calib       = calib,
        test        = test
    )

def iter_split_datasets(
    split_dir: str | Path,
    instrument: str,
    window_ids: list[str] | None = None
):
    """
    Iterate over saved split datasets.
    """
    if window_ids is None:
        window_ids = get_window_ids()

    for window_id in window_ids:
        yield load_split_dataset(split_dir, instrument, window_id)


# Public meta dataset builder
def build_meta_dataset(
    df: pl.DataFrame,
    output_dir: str | Path,
    config: MetaDatasetConfig
) -> list[str]:
    """
    Build and save a meta-feature dataset split by walk-forward windows.

    This function is intended for 02_meta_dataset.ipynb.

    Output files:
        {instrument}_{window_id}_train_model.parquet
        {instrument}_{window_id}_calib.parquet
        {instrument}_{window_id}_test.parquet
    """
    if config.meta_kind != "adverse":
        raise NotImplementedError(
            "Only meta_kind='adverse' is implemented now. "
            "The feature-name helpers already support other names, but the "
            "training logic is currently adverse-markout only."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_ids = config.window_ids or get_window_ids()
    meta_cols = get_meta_features(
        kind        = config.meta_kind,
        horizons_s  = config.horizons_s
    )

    iterator = iter_window_splits(
        df          = df,
        window_ids  = window_ids,
        calib_frac  = config.calib_frac
    )

    if config.use_tqdm:
        iterator = tqdm(list(iterator), desc=f"meta dataset ({config.instrument})")

    for split in iterator:
        if (not config.overwrite) and split_files_exist(
            output_dir,
            config.instrument,
            split.window_id,
        ):
            if config.verbose:
                print(f"skip {split.window_id}: split files already exist")
            continue

        result = build_meta_for_split(split, config)
        save_split_dataset(result, output_dir, config.instrument)

        del result
        gc.collect()

    return meta_cols

def build_meta_for_split(
    split: SplitData,
    config: MetaDatasetConfig
) -> MetaWindowResult:
    """
    Build meta-features for one already constructed SplitData.

    Leakage rule:
        train_model -> OOF predictions
        calib       -> predictions from model refit on full train_model
        test        -> predictions from model refit on full train_model
    """
    feature_names = get_feature_set(
        name            = config.base_feature_set,
        horizons_s      = config.horizons_s,
        meta_kind       = config.meta_kind,
    )

    # Meta models must not use meta features as their own inputs.
    feature_names = [c for c in feature_names if not c.startswith("meta_")]

    train_model = _ensure_derived_features_if_needed(split.train_model, feature_names)
    calib       = _ensure_derived_features_if_needed(split.calib, feature_names)
    test        = _ensure_derived_features_if_needed(split.test, feature_names)

    if config.verbose:
        print(f"\n=== meta {config.instrument} {split.window_id} ===")
        print(f"train_model: {train_model.height:,}")
        print(f"calib      : {calib.height:,}")
        print(f"test       : {test.height:,}")
        print(f"features   : {len(feature_names)}")

    meta_train: dict[str, np.ndarray] = {}
    meta_calib: dict[str, np.ndarray] = {}
    meta_test: dict[str, np.ndarray] = {}
    meta_cols: list[str] = []

    for tau in config.horizons_s:
        col = meta_col_name(tau, kind=config.meta_kind)
        meta_cols.append(col)

        X_tm, pnl_tm, idx_tm = _extract_meta_arrays(train_model, tau, feature_names)
        X_ca, pnl_ca, idx_ca = _extract_meta_arrays(calib, tau, feature_names)
        X_te, pnl_te, idx_te = _extract_meta_arrays(test, tau, feature_names)

        y_tm, threshold_value = _make_adverse_target(
            pnl         = pnl_tm,
            mode        = config.adverse_mode,
            quantile    = config.adverse_quantile,
            quantile_value = None
        )

        # y_ca/y_te are not used for fitting, but computing them here makes the
        # threshold policy explicit and keeps the pipeline easy to audit.
        _y_ca, _ = _make_adverse_target(
            pnl             = pnl_ca,
            mode            = config.adverse_mode,
            quantile        = config.adverse_quantile,
            quantile_value  = threshold_value
        )
        _y_te, _ = _make_adverse_target(
            pnl             = pnl_te,
            mode            = config.adverse_mode,
            quantile        = config.adverse_quantile,
            quantile_value  = threshold_value
        )

        oof_tm = _oof_predict(
            X           = X_tm,
            y           = y_tm,
            n_folds     = config.n_folds,
            lgbm_params = config.lgbm_params
        )
        proba_ca = _refit_predict(X_tm, y_tm, X_ca, config.lgbm_params)
        proba_te = _refit_predict(X_tm, y_tm, X_te, config.lgbm_params)

        arr_tm = np.full(train_model.height, np.nan, dtype=np.float32)
        arr_ca = np.full(calib.height, np.nan, dtype=np.float32)
        arr_te = np.full(test.height, np.nan, dtype=np.float32)

        arr_tm[idx_tm] = oof_tm
        arr_ca[idx_ca] = proba_ca
        arr_te[idx_te] = proba_te

        meta_train[col] = arr_tm
        meta_calib[col] = arr_ca
        meta_test[col] = arr_te

        if config.verbose:
            thr_str = "n/a" if threshold_value is None else f"{threshold_value:.6f}"
            pos_rate = float(np.mean(y_tm)) if len(y_tm) else float("nan")
            print(
                f"tau={tau:>3}s | {col:<15s} | "
                f"valid_train={len(idx_tm):>10,} | "
                f"pos_rate={pos_rate:.4f} | threshold={thr_str}"
            )

        del X_tm, X_ca, X_te, pnl_tm, pnl_ca, pnl_te, y_tm
        del _y_ca, _y_te, oof_tm, proba_ca, proba_te
        gc.collect()

    train_model_out = train_model.with_columns([
        pl.Series(name, values) for name, values in meta_train.items()
    ])
    calib_out = calib.with_columns([
        pl.Series(name, values) for name, values in meta_calib.items()
    ])
    test_out = test.with_columns([
        pl.Series(name, values) for name, values in meta_test.items()
    ])

    return MetaWindowResult(
        window_id   = split.window_id,
        train_model = train_model_out,
        calib       = calib_out,
        test        = test_out,
        meta_cols   = meta_cols
    )


# Internal helpers
def meta_col_name(tau: int, kind: str = "adverse") -> str:
    if kind == "adverse":
        return f"meta_adv_{tau}s"
    if kind == "vpin":
        return f"meta_vpin_up_{tau}s"
    if kind == "liq":
        return f"meta_liq_spike_{tau}s"
    raise ValueError(f"Unknown meta kind: {kind}")

def _ensure_derived_features_if_needed(
    df: pl.DataFrame,
    feature_names: list[str]
) -> pl.DataFrame:
    """
    Compute derived columns if they are requested but absent.
    """
    required_derived = [c for c in DERIVED_FEATURES if c in feature_names]
    missing_derived = [c for c in required_derived if c not in df.columns]

    if missing_derived:
        df = add_derived_features(df)

    return df

def _make_adverse_target(
    pnl: np.ndarray,
    mode: str,
    quantile: float,
    quantile_value: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """
    Build binary adverse target from pnl values.
    """
    pnl = np.asarray(pnl, dtype=np.float32)

    if mode == "sign":
        return (pnl < 0.0).astype(np.int8), None

    if mode == "quantile":
        if quantile_value is None:
            threshold = float(np.nanquantile(pnl, quantile))
        else:
            threshold = float(quantile_value)
        return (pnl < threshold).astype(np.int8), threshold

    raise ValueError("adverse_mode must be one of: 'sign', 'quantile'.")

def _extract_meta_arrays(
    df: pl.DataFrame,
    tau: int,
    feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, pnl and row indices for meta-model fitting/prediction.

    Rows are valid only if:
        - pnl_{tau}s is finite;
        - all meta-model input features are finite.

    Row indices are needed to scatter predictions back into the original split
    DataFrame and leave NaN where meta-prediction is not computable.
    """
    pnl_col = f"pnl_{tau}s"
    required_cols = feature_names + [pnl_col]
    validate_columns(df, required_cols, context="_extract_meta_arrays")

    df_idx = df.with_row_index(name="__row_idx__")

    valid_expr = (
        pl.col(pnl_col).is_not_null()
        & pl.col(pnl_col).is_not_nan()
        & pl.all_horizontal([
            pl.col(c).is_not_null() & pl.col(c).is_not_nan()
            for c in feature_names
        ])
    )

    valid = df_idx.filter(valid_expr)

    X = valid.select(feature_names).to_numpy().astype(np.float32)
    pnl = valid[pnl_col].to_numpy().astype(np.float32)
    idx = valid["__row_idx__"].to_numpy().astype(np.int64)

    return X, pnl, idx

def _constant_proba(y: np.ndarray, n: int) -> np.ndarray:
    """
    Safe fallback when a classifier cannot be trained.
    """
    if len(y) == 0:
        p = 0.0
    else:
        p = float(np.mean(y))
    p = float(np.clip(p, 0.0, 1.0))
    return np.full(n, p, dtype=np.float32)

def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    lgbm_params: dict
) -> LGBMClassifier | None:
    """
    Fit LightGBM classifier or return None if fitting is impossible.
    """
    if len(y) == 0:
        return None

    if np.unique(y).size < 2:
        return None

    clf = LGBMClassifier(**lgbm_params)
    clf.fit(X, y)
    return clf

def _oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    lgbm_params: dict
) -> np.ndarray:
    """
    Contiguous-fold OOF predictions for train_model.

    This preserves the legacy design: train_model gets out-of-fold predictions,
    while calib/test get predictions from the model refit on full train_model.
    """
    n = len(y)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    n_folds = max(2, min(int(n_folds), n))
    fold_size = n // n_folds

    oof = np.full(n, np.nan, dtype=np.float32)

    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n

        if start >= end:
            continue

        mask_val = np.zeros(n, dtype=bool)
        mask_val[start:end] = True
        mask_train = ~mask_val

        y_train = y[mask_train]

        clf = _fit_classifier(X[mask_train], y_train, lgbm_params)

        if clf is None:
            oof[start:end] = _constant_proba(y_train, end - start)
        else:
            oof[start:end] = clf.predict_proba(X[start:end])[:, 1].astype(np.float32)
            del clf

        gc.collect()

    # This should normally not happen, but makes the function robust.
    if np.isnan(oof).any():
        oof[np.isnan(oof)] = float(np.mean(y)) if len(y) else 0.0

    return oof

def _refit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_new: np.ndarray,
    lgbm_params: dict
) -> np.ndarray:
    """
    Refit classifier on full train_model and predict probabilities for calib/test.
    """
    if len(X_new) == 0:
        return np.empty(0, dtype=np.float32)

    clf = _fit_classifier(X_train, y_train, lgbm_params)

    if clf is None:
        return _constant_proba(y_train, len(X_new))

    proba = clf.predict_proba(X_new)[:, 1].astype(np.float32)
    del clf
    gc.collect()

    return proba
