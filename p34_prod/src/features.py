# features.py - feature construction / feature lists / model array extraction.
#
# Contain two independent parts:
#
# 1. Feature construction
#   add_bbo_and_markout
#   add_rolling_features
#   add_liquidation_features
#   add_derived_features
# 
# 2. Feature selection
#   BBO_FEATURES
#   VPIN_FEATURES
#   LIQUIDATION_FEATURES
#   FEATURE_SETS
#   get_feature_set
#   get_meta_features


# imports.
import numpy as np
import polars as pl


# constants.
US_PER_SEC = 1_000_000


# base model features / sets.

BBO_FEATURES = [
    "spread",
    "obi",
    "dist_to_mid",
]

TIME_FEATURES = [
    "hour",
]

# sign is kept as a technical column in the dataframe,
# but it is not used directly as a model feature.
TECHNICAL_FEATURES = [
    "sign",
]

# Raw sign is intentionally excluded from model features.
TRADE_FEATURES = []

VPIN_FEATURES = [
    "vpin_5s",
    "vpin_30s",
    "vpin_120s",
]

RV_FEATURES = [
    "rv_5s",
    "rv_30s",
    "rv_120s",
]

LIQUIDATION_FEATURES = [
    "liq_binance_5s",
    "liq_binance_30s",
    "liq_binance_120s",
    "liq_bybit_5s",
    "liq_bybit_30s",
    "liq_bybit_120s",
]

# Base features WITHOUT raw sign.
BASE_FEATURES = (
    BBO_FEATURES
    + TIME_FEATURES
    + VPIN_FEATURES
    + RV_FEATURES
    + LIQUIDATION_FEATURES
)

# Old derived features from the previous pipeline.
OLD_DERIVED_FEATURES = [
    "log_notional",
    "net_liq_5s",
    "net_liq_30s",
    "net_liq_120s",
    "liq_agreement_30s",
]

# Category 1: sign-aware / sign-invariant features.
SIGN_INVARIANT_FEATURES = [
    "signed_dist_to_mid",
    "signed_obi",

    "signed_liq_binance_5s",
    "signed_liq_binance_30s",
    "signed_liq_binance_120s",

    "signed_liq_bybit_5s",
    "signed_liq_bybit_30s",
    "signed_liq_bybit_120s",
]

# Category 2: relative / dimensionless features.
RELATIVE_FEATURES = [
    "vpin_ratio_5_120",
    "rv_ratio_5_120",
    "liq_binance_ratio_5_120",
    "liq_bybit_ratio_5_120",
    "dist_to_mid_over_spread",
    "notional_over_median",
]

# Category 3: Griffin liquidation battery.
EXTRA_LIQUIDATION_FEATURES = [
    "liq_aligned_5s",
    "liq_aligned_30s",
    "liq_aligned_120s",
    "liq_direction_agreement_30s",
    "liq_accel_5_30",
]

NEW_ENGINEERED_FEATURES = (
    SIGN_INVARIANT_FEATURES
    + RELATIVE_FEATURES
    + EXTRA_LIQUIDATION_FEATURES
)

# IMPORTANT:
# DERIVED_FEATURES must contain every column created by add_derived_features().
# experiment.py uses this list to decide whether add_derived_features() should be called.
DERIVED_FEATURES = (
    OLD_DERIVED_FEATURES
    + NEW_ENGINEERED_FEATURES
)

ADVANCED_FEATURES = (
    BASE_FEATURES
    + DERIVED_FEATURES
)

FEATURE_SETS = {
    "base": BASE_FEATURES,

    "bbo": BBO_FEATURES,
    "vpin": VPIN_FEATURES,
    "rv": RV_FEATURES,
    "liq": LIQUIDATION_FEATURES,

    "bbo_plus_vpin": BBO_FEATURES + VPIN_FEATURES,
    "bbo_plus_liq": BBO_FEATURES + LIQUIDATION_FEATURES,
    "vpin_plus_liq": VPIN_FEATURES + LIQUIDATION_FEATURES,

    # Old feature set: base + old derived only.
    "base_plus_derived": BASE_FEATURES + OLD_DERIVED_FEATURES,

    # New advanced feature sets.
    "base_plus_engineered": ADVANCED_FEATURES,
    "advanced": ADVANCED_FEATURES,
}



# utilities.
def get_meta_adverse_features(
    horizons_s: list[int] | tuple[int, ...] = (30, 120, 300)
) -> list[str]:
    """
    Return adverse meta-feature names.

    Example:
        horizons_s = (30, 120, 300)

    Returns:
        ["meta_adv_30s", "meta_adv_120s", "meta_adv_300s"]
    """
    return [f"meta_adv_{tau}s" for tau in horizons_s]

def get_meta_vpin_features(
    horizons_s: list[int] | tuple[int, ...] = (30, 120, 300)
) -> list[str]:
    return [f"meta_vpin_up_{tau}s" for tau in horizons_s]

def get_meta_liq_features(
    horizons_s: list[int] | tuple[int, ...] = (30, 120, 300)
) -> list[str]:
    return [f"meta_liq_spike_{tau}s" for tau in horizons_s]

def get_meta_features(
    kind: str = "adverse",
    horizons_s: list[int] | tuple[int, ...] = (30, 120, 300)
) -> list[str]:

    if kind == "adverse":
        return get_meta_adverse_features(horizons_s)

    if kind == "vpin":
        return get_meta_vpin_features(horizons_s)

    if kind == "liq":
        return get_meta_liq_features(horizons_s)

    if kind == "all":
        return (
            get_meta_adverse_features(horizons_s)
            + get_meta_vpin_features(horizons_s)
            + get_meta_liq_features(horizons_s)
        )

    raise ValueError(f"Unknown meta feature kind: {kind}")

def get_feature_set(
    name: str,
    horizons_s: list[int] | tuple[int, ...] = (30, 120, 300),
    meta_kind: str = "adverse"
) -> list[str]:

    meta_features = get_meta_features(
        kind        = meta_kind,
        horizons_s  = horizons_s
    )

    if name in FEATURE_SETS:
        return FEATURE_SETS[name].copy()

    if name == "base_plus_meta":
        return BASE_FEATURES + meta_features

    if name == "base_plus_derived_plus_meta":
        return BASE_FEATURES + OLD_DERIVED_FEATURES + meta_features

    if name == "advanced_plus_meta":
        return ADVANCED_FEATURES + meta_features

    if name == "all":
        return ADVANCED_FEATURES + meta_features

    raise ValueError(
        f"Unknown feature set: {name}. "
        f"Available sets: {list(FEATURE_SETS.keys()) + ['base_plus_meta', 'base_plus_derived_plus_meta', 'advanced_plus_meta', 'all']}"
    )

def validate_columns(
    df: pl.DataFrame,
    columns: list[str] | tuple[str, ...],
    context: str = "DataFrame"
) -> None:
    """
    Check that all required columns exist in df.
    """
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"{context}: missing columns: {missing}")


# helper functions.
def _rolling_count_and_sum(
    ts: np.ndarray,
    values: np.ndarray,
    window_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return count and sum over [t - window_us, t) for every timestamp t.

    The current row is excluded.
    """
    left    = np.searchsorted(ts, ts - window_us, side="left")
    right   = np.searchsorted(ts, ts, side="left")

    count   = right - left

    values64 = values.astype(np.float64, copy=False)

    prefix = np.empty(len(values64) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values64, out=prefix[1:])

    rolling_sum = prefix[right] - prefix[left]

    return count, rolling_sum

def _rolling_sum_sq(
    ts: np.ndarray,
    values: np.ndarray,
    window_us: int,
) -> np.ndarray:
    """
    Return sum of squares over [t - window_us, t).
    """
    left    = np.searchsorted(ts, ts - window_us, side="left")
    right   = np.searchsorted(ts, ts, side="left")

    values64 = values.astype(np.float64, copy=False)

    prefix2 = np.empty(len(values64) + 1, dtype=np.float64)
    prefix2[0] = 0.0
    np.cumsum(values64 * values64, out=prefix2[1:])

    return prefix2[right] - prefix2[left]

def _liq_window_sum(
    trade_ts: np.ndarray,
    liq_ts: np.ndarray,
    cum_prefix: np.ndarray,
    window_us: int
) -> np.ndarray:
    """
    Sum signed liquidations over [t - window_us, t) for every trade timestamp t.
    """
    if len(liq_ts) == 0:
        return np.zeros(len(trade_ts), dtype=np.float32)

    left = np.searchsorted(liq_ts, trade_ts - window_us, side="left")
    right = np.searchsorted(liq_ts, trade_ts, side="left")

    out = cum_prefix[right] - cum_prefix[left]

    return out.astype(np.float32)


# functions for data.py.
def add_bbo_and_markout(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,
    horizons_s: list[int],
    rebate_bps: float,
    max_bbo_ts: int
) -> pl.DataFrame:
    """
    Add BBO features at trade time and markout PnL at t + tau.

    Adds:
        mid_price
        spread
        obi
        dist_to_mid
        pnl_{tau}s for every tau in horizons_s

    Parameters
    ----------
    trades:
        Trade-level DataFrame.

    bbo:
        BBO DataFrame with timestamp, mid_price, spread, obi.

    horizons_s:
        Markout horizons in seconds.

    rebate_bps:
        Maker rebate in basis points.

    max_bbo_ts:
        Maximum available BBO timestamp.
        If t + tau is beyond this timestamp, pnl_{tau}s is set to NaN.
    """

    # validations and check.
    validate_columns(
        trades,
        ["timestamp", "price", "sign"],
        context="add_bbo_and_markout/trades"
    )

    validate_columns(
        bbo,
        ["timestamp", "mid_price", "spread", "obi"],
        context="add_bbo_and_markout/bbo"
    )

    if trades.height == 0:
        return trades

    if bbo.height == 0:
        raise ValueError("add_bbo_and_markout: bbo is empty.")
    

    # current BBO at trade time.
    df = trades.join_asof(
        bbo,
        on      = "timestamp",
        strategy= "backward"
    )

    # trade distance from mid in bps.
    df = df.with_columns([
        (
            (pl.col("price") - pl.col("mid_price"))/ pl.col("mid_price")
            * 10_000.0
        )
        .cast(pl.Float32)
        .alias("dist_to_mid")
    ])


    # markout PnL for every horizon.
    for tau in horizons_s:
        tau_us = tau * US_PER_SEC

        lookup_ts_col   = f"_lookup_ts_{tau}"
        bbo_ts_col      = f"_bbo_ts_{tau}"
        mid_col         = f"_mid_{tau}"
        pnl_col         = f"pnl_{tau}s"

        right = bbo.select([
            pl.col("timestamp").alias(bbo_ts_col),
            pl.col("mid_price").alias(mid_col)
        ])

        df = (
            df
            .with_columns(
                (pl.col("timestamp") + tau_us).alias(lookup_ts_col)
            )
            .join_asof(
                right,
                left_on     = lookup_ts_col,
                right_on    = bbo_ts_col,
                strategy    = "backward"
            )
            .with_columns([
                pl.when(pl.col(lookup_ts_col) <= max_bbo_ts)
                .then(
                    -1.0
                    * pl.col("sign")
                    * (pl.col(mid_col) - pl.col("price"))
                    / pl.col("price")
                    * 10_000.0
                    + rebate_bps
                )
                .otherwise(float("nan"))
                .cast(pl.Float32)
                .alias(pnl_col)
            ])
            .drop([lookup_ts_col, bbo_ts_col, mid_col])
        )

    return df.with_columns([
        pl.col("mid_price").cast(pl.Float32),
        pl.col("spread").cast(pl.Float32),
        pl.col("obi").cast(pl.Float32),
    ])

def add_rolling_features(
    df: pl.DataFrame,
    windows_s: list[int],
    n_min_obs: int = 3
) -> pl.DataFrame:
    """
    Add rolling trade features.

    For every window W in windows_s adds:
        vpin_{W}s
        rv_{W}s

    Definitions
    -----------
    vpin_W:
        abs(sum signed notional over [t-W, t)) / sum notional over [t-W, t)

    rv_W:
        sample standard deviation of log(trade price) over [t-W, t).
        NOTE: matches the legacy pipeline (utilities_preprocess.py) —
        stddev of log price level, computed on trade price (not mid).

    Rows with fewer than n_min_obs observations in the window receive NaN.
    """
    validate_columns(
        df,
        ["timestamp", "sign", "notional", "price"],
        context="add_rolling_features"
    )

    if df.height == 0:
        return df

    ts = df["timestamp"].to_numpy().astype(np.int64, copy=False)

    sign = df["sign"].to_numpy().astype(np.float64, copy=False)
    notional = df["notional"].to_numpy().astype(np.float64, copy=False)

    signed_notional = sign * notional

    # RV is computed from log TRADE price (legacy formula).
    price = df["price"].to_numpy().astype(np.float64, copy=False)
    log_price = np.log(price)

    new_cols: list[pl.Series] = []

    for w in windows_s:
        window_us = w * US_PER_SEC

        # VPIN inputs.
        count, abs_vol = _rolling_count_and_sum(
            ts          = ts,
            values      = notional,
            window_us   = window_us
        )

        _, signed_vol = _rolling_count_and_sum(
            ts          = ts,
            values      = signed_notional,
            window_us   = window_us
        )

        # RV inputs: rolling sum and sum of squares of log price.
        count_rv, sum_lp = _rolling_count_and_sum(
            ts          = ts,
            values      = log_price,
            window_us   = window_us
        )
        sum_lp2 = _rolling_sum_sq(
            ts          = ts,
            values      = log_price,
            window_us   = window_us
        )

        vpin = np.full(len(ts), np.nan, dtype=np.float32)
        rv   = np.full(len(ts), np.nan, dtype=np.float32)

        # VPIN validity: enough obs AND non-zero volume.
        valid_vpin = (count >= n_min_obs) & (abs_vol > 0)
        vpin[valid_vpin] = (
            np.abs(signed_vol[valid_vpin]) / abs_vol[valid_vpin]
        ).astype(np.float32)

        # RV validity: enough obs only (legacy behavior).
        ok_rv = count_rv >= n_min_obs
        c = count_rv[ok_rv].astype(np.float64)
        var = (sum_lp2[ok_rv] - sum_lp[ok_rv] ** 2 / c) / (c - 1.0)
        var = np.maximum(var, 0.0)  # numerical floor.
        rv[ok_rv] = np.sqrt(var).astype(np.float32)

        new_cols.append(pl.Series(f"vpin_{w}s", vpin))
        new_cols.append(pl.Series(f"rv_{w}s", rv))

    return df.with_columns(new_cols)

def add_liquidation_features(
    df: pl.DataFrame,
    bin_ts: np.ndarray,
    bin_cum: np.ndarray,
    byb_ts: np.ndarray,
    byb_cum: np.ndarray,
    windows_s: list[int]
) -> pl.DataFrame:
    """
    Add rolling signed liquidation features.

    For every window W in windows_s adds:
        liq_binance_{W}s
        liq_bybit_{W}s
    """

    # validations and check.
    validate_columns(
        df,
        ["timestamp"],
        context="add_liquidation_features"
    )

    if df.height == 0:
        return df

    trade_ts = df["timestamp"].to_numpy().astype(np.int64, copy=False)

    new_cols: list[pl.Series] = []

    for w in windows_s:
        window_us = w * US_PER_SEC

        liq_binance = _liq_window_sum(
            trade_ts    = trade_ts,
            liq_ts      = bin_ts,
            cum_prefix  = bin_cum,
            window_us   = window_us
        )

        liq_bybit = _liq_window_sum(
            trade_ts    = trade_ts,
            liq_ts      = byb_ts,
            cum_prefix  = byb_cum,
            window_us   = window_us
        )

        new_cols.append(pl.Series(f"liq_binance_{w}s", liq_binance))
        new_cols.append(pl.Series(f"liq_bybit_{w}s", liq_bybit))

    return df.with_columns(new_cols)


# derived features after enriched dataset is built.
def _add_notional_rolling_median_120s(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling median of notional over [t - 120s, t).

    This is used for:
        notional_over_median = notional / rolling_median(notional, 120s)

    The current row is excluded via closed="left".
    """

    validate_columns(
        df,
        ["timestamp", "notional"],
        context="_add_notional_rolling_median_120s",
    )

    if df.height == 0:
        return df

    # The dataframe should already be sorted by timestamp in the pipeline,
    # but sorting here makes the rolling operation safe.
    df = df.sort("timestamp")

    tmp_ts_col = "_timestamp_dt"
    tmp_med_col = "_notional_median_120s"

    df = (
        df
        .with_columns(
            pl.from_epoch(
                pl.col("timestamp"),
                time_unit="us",
            ).alias(tmp_ts_col)
        )
        .with_columns(
            pl.col("notional")
            .rolling_median_by(
                by=tmp_ts_col,
                window_size="120s",
                closed="left",
            )
            .cast(pl.Float32)
            .alias(tmp_med_col)
        )
        .drop(tmp_ts_col)
    )

    return df

def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add cheap derived features computed from existing enriched columns.

    Old features:
        log_notional
        net_liq_5s
        net_liq_30s
        net_liq_120s
        liq_agreement_30s

    New features:
        Category 1: sign-invariant features.
        Category 2: relative / dimensionless features.
        Category 3: liquidation battery.

    Important:
        Raw sign is used only as a technical column.
        It should not be included directly in the final model feature list.
    """

    eps = 1e-6

    validate_columns(
        df,
        [
            "timestamp",
            "sign",
            "notional",

            "spread",
            "obi",
            "dist_to_mid",

            "vpin_5s",
            "vpin_120s",

            "rv_5s",
            "rv_120s",

            "liq_binance_5s",
            "liq_binance_30s",
            "liq_binance_120s",

            "liq_bybit_5s",
            "liq_bybit_30s",
            "liq_bybit_120s",
        ],
        context="add_derived_features",
    )

    df = _add_notional_rolling_median_120s(df)

    df = df.with_columns([

        # Old derived features.
        pl.col("notional")
        .log1p()
        .cast(pl.Float32)
        .alias("log_notional"),

        (
            pl.col("liq_binance_5s") + pl.col("liq_bybit_5s")
        )
        .cast(pl.Float32)
        .alias("net_liq_5s"),

        (
            pl.col("liq_binance_30s") + pl.col("liq_bybit_30s")
        )
        .cast(pl.Float32)
        .alias("net_liq_30s"),

        (
            pl.col("liq_binance_120s") + pl.col("liq_bybit_120s")
        )
        .cast(pl.Float32)
        .alias("net_liq_120s"),

        (
            pl.col("liq_binance_30s").sign() * pl.col("liq_bybit_30s").sign()
        )
        .cast(pl.Float32)
        .alias("liq_agreement_30s"),


        # Category 1: sign-invariant features.
        (
            pl.col("sign") * pl.col("dist_to_mid")
        )
        .cast(pl.Float32)
        .alias("signed_dist_to_mid"),

        (
            pl.col("sign") * pl.col("obi")
        )
        .cast(pl.Float32)
        .alias("signed_obi"),

        (
            pl.col("sign") * pl.col("liq_binance_5s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_binance_5s"),

        (
            pl.col("sign") * pl.col("liq_binance_30s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_binance_30s"),

        (
            pl.col("sign") * pl.col("liq_binance_120s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_binance_120s"),

        (
            pl.col("sign") * pl.col("liq_bybit_5s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_bybit_5s"),

        (
            pl.col("sign") * pl.col("liq_bybit_30s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_bybit_30s"),

        (
            pl.col("sign") * pl.col("liq_bybit_120s")
        )
        .cast(pl.Float32)
        .alias("signed_liq_bybit_120s"),


        # Category 2: relative / dimensionless features.
        (
            pl.col("vpin_5s") / (pl.col("vpin_120s") + eps)
        )
        .cast(pl.Float32)
        .alias("vpin_ratio_5_120"),

        (
            pl.col("rv_5s") / (pl.col("rv_120s") + eps)
        )
        .cast(pl.Float32)
        .alias("rv_ratio_5_120"),

        # Liquidation columns are signed, so for concentration ratios
        # we use absolute values.
        (
            pl.col("liq_binance_5s").abs()
            / (pl.col("liq_binance_120s").abs() + eps)
        )
        .cast(pl.Float32)
        .alias("liq_binance_ratio_5_120"),

        (
            pl.col("liq_bybit_5s").abs()
            / (pl.col("liq_bybit_120s").abs() + eps)
        )
        .cast(pl.Float32)
        .alias("liq_bybit_ratio_5_120"),

        (
            pl.col("dist_to_mid") / (pl.col("spread") + eps)
        )
        .cast(pl.Float32)
        .alias("dist_to_mid_over_spread"),

        (
            pl.col("notional") / (pl.col("_notional_median_120s") + eps)
        )
        .cast(pl.Float32)
        .alias("notional_over_median"),


        # Category 3: liquidation battery.
        (
            pl.col("sign") * (
                pl.col("liq_binance_5s") + pl.col("liq_bybit_5s")
            )
        )
        .cast(pl.Float32)
        .alias("liq_aligned_5s"),

        (
            pl.col("sign") * (
                pl.col("liq_binance_30s") + pl.col("liq_bybit_30s")
            )
        )
        .cast(pl.Float32)
        .alias("liq_aligned_30s"),

        (
            pl.col("sign") * (
                pl.col("liq_binance_120s") + pl.col("liq_bybit_120s")
            )
        )
        .cast(pl.Float32)
        .alias("liq_aligned_120s"),

        # 1 if Binance and Bybit liquidation pressure have the same non-zero sign.
        pl.when(
            (
                pl.col("liq_binance_30s").sign()
                * pl.col("liq_bybit_30s").sign()
            ) > 0
        )
        .then(1.0)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("liq_direction_agreement_30s"),

        (
            pl.col("liq_binance_5s").abs()
            / (pl.col("liq_binance_30s").abs() + eps)
        )
        .cast(pl.Float32)
        .alias("liq_accel_5_30"),
    ])

    return df.drop("_notional_median_120s")

# model matrix extraction.
def drop_invalid_rows_for_target(
    df: pl.DataFrame,
    tau: int
) -> pl.DataFrame:
    """
    Drop rows where pnl_{tau}s is null or NaN.
    """
    pnl_col = f"pnl_{tau}s"

    validate_columns(
        df,
        [pnl_col],
        context="drop_invalid_rows_for_target"
    )

    return df.filter(pl.col(pnl_col).is_not_null() & pl.col(pnl_col).is_not_nan())

def drop_invalid_rows_for_features(
    df: pl.DataFrame,
    features: list[str]
) -> pl.DataFrame:
    """
    Drop rows where at least one feature is null or NaN.
    """
    validate_columns(
        df,
        features,
        context="drop_invalid_rows_for_features",
    )

    return df.filter(
        pl.all_horizontal([
            pl.col(f).is_not_null() & pl.col(f).is_not_nan()
            for f in features
        ])
    )

def extract_Xyw(
    df: pl.DataFrame,
    tau: int,
    features: list[str],
    drop_feature_nans: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, y, w, ts from a feature-enriched DataFrame.

    Target:
        y = pnl_{tau}s

    Weight:
        w = weight

    Timestamp:
        ts = timestamp
    """
    pnl_col = f"pnl_{tau}s"

    required_cols = features + [pnl_col, "weight", "timestamp"]

    validate_columns(
        df,
        required_cols,
        context="extract_Xyw",
    )

    valid = drop_invalid_rows_for_target(df, tau)

    if drop_feature_nans:
        valid = drop_invalid_rows_for_features(valid, features)

    X   = valid.select(features).to_numpy().astype(np.float32)
    y   = valid[pnl_col].to_numpy().astype(np.float32)
    w   = valid["weight"].to_numpy().astype(np.float32)
    ts  = valid["timestamp"].to_numpy().astype(np.int64)

    return X, y, w, ts
