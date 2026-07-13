# splits.py - how we cut data.


# imports.
import datetime
import polars as pl
from dataclasses import dataclass


# Constants.
US_PER_SEC = 1_000_000
US_PER_DAY = 86_400 * US_PER_SEC


def date_to_us(year: int, month: int, day: int) -> int:
    """
    Convert UTC date to microseconds since Unix epoch.
    """
    dt = datetime.datetime(
        year, month, day,
        tzinfo=datetime.timezone.utc
    )
    return int(dt.timestamp() * US_PER_SEC)


@dataclass(frozen=True)
class WindowSpec:
    """
    One walk-forward window.
    """
    train_start: int
    train_end: int
    embargo_end: int
    test_end: int


@dataclass
class SplitData:
    """
    DataFrames for one walk-forward split.
    """
    window_id: str
    train_full: pl.DataFrame
    train_model: pl.DataFrame
    calib: pl.DataFrame
    test: pl.DataFrame


WINDOWS: dict[str, WindowSpec] = {
    "W1": WindowSpec(
        train_start=date_to_us(2025, 11, 1),
        train_end=date_to_us(2025, 12, 1),
        embargo_end=date_to_us(2025, 12, 8),
        test_end=date_to_us(2026, 1, 1)
    ),
    "W2": WindowSpec(
        train_start=date_to_us(2025, 11, 1),
        train_end=date_to_us(2026, 1, 1),
        embargo_end=date_to_us(2026, 1, 8),
        test_end=date_to_us(2026, 2, 1)
    ),
    "W3": WindowSpec(
        train_start=date_to_us(2025, 11, 1),
        train_end=date_to_us(2026, 2, 1),
        embargo_end=date_to_us(2026, 2, 8),
        test_end=date_to_us(2026, 3, 1)
    ),
    "W4": WindowSpec(
        train_start=date_to_us(2025, 11, 1),
        train_end=date_to_us(2026, 3, 1),
        embargo_end=date_to_us(2026, 3, 8),
        test_end=date_to_us(2026, 4, 1)
    ),
    "W5": WindowSpec(
        train_start=date_to_us(2025, 11, 1),
        train_end=date_to_us(2026, 4, 1),
        embargo_end=date_to_us(2026, 4, 8),
        test_end=date_to_us(2026, 4, 29)
    )
}


def get_window_ids() -> list[str]:
    """
    Return available walk-forward window ids.
    """
    return list(WINDOWS.keys())


def get_window_spec(window_id: str) -> WindowSpec:
    """
    Return WindowSpec by window id.
    """
    if window_id not in WINDOWS:
        raise ValueError(
            f"Unknown window_id={window_id}. "
            f"Expected one of {list(WINDOWS)}."
        )

    return WINDOWS[window_id]


def filter_time_range(
    df: pl.DataFrame,
    start_us: int,
    end_us: int,
    timestamp_col: str = "timestamp"
) -> pl.DataFrame:
    """
    Filter DataFrame by half-open time interval [start_us, end_us).
    """
    return df.filter(
        (pl.col(timestamp_col) >= start_us)  & (pl.col(timestamp_col) < end_us)
    )


def get_outer_window_split(
    df: pl.DataFrame,
    window_id: str,
    timestamp_col: str = "timestamp"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Return outer train and outer test for one walk-forward window.

    Embargo interval is skipped.
    """
    spec = get_window_spec(window_id)

    train_full = filter_time_range(
        df,
        spec.train_start,
        spec.train_end,
        timestamp_col=timestamp_col
    )

    test = filter_time_range(
        df,
        spec.embargo_end,
        spec.test_end,
        timestamp_col=timestamp_col
    )

    return train_full, test


def split_train_calib_by_time(
    df_train: pl.DataFrame,
    calib_frac: float = 0.20,
    timestamp_col: str = "timestamp"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split outer train into model-train and calibration by time.
    """
    if not (0.0 < calib_frac < 1.0):
        raise ValueError("calib_frac must be between 0 and 1.")

    if df_train.height == 0:
        return df_train, df_train

    ts_min = int(df_train[timestamp_col].min())
    ts_max = int(df_train[timestamp_col].max())

    split_ts = int(ts_min + (1.0 - calib_frac) * (ts_max - ts_min))

    train_model = df_train.filter(pl.col(timestamp_col) < split_ts)
    calib = df_train.filter(pl.col(timestamp_col) >= split_ts)

    return train_model, calib


def get_window_split(
    df: pl.DataFrame,
    window_id: str,
    calib_frac: float = 0.20,
    timestamp_col: str = "timestamp"
) -> SplitData:
    """
    Return train_full, train_model, calib and test for one window.
    """
    train_full, test = get_outer_window_split(
        df,
        window_id       = window_id,
        timestamp_col   = timestamp_col,
    )

    train_model, calib = split_train_calib_by_time(
        train_full,
        calib_frac      = calib_frac,
        timestamp_col   = timestamp_col,
    )

    return SplitData(
        window_id       = window_id,
        train_full      = train_full,
        train_model     = train_model,
        calib=calib,
        test=test
    )


def iter_window_splits(
    df: pl.DataFrame,
    window_ids: list[str] | None = None,
    calib_frac: float = 0.20,
    timestamp_col: str = "timestamp"
):
    """
    Iterate over walk-forward splits.
    """
    if window_ids is None:
        window_ids = get_window_ids()

    for window_id in window_ids:
        yield get_window_split(
            df,
            window_id       = window_id,
            calib_frac      = calib_frac,
            timestamp_col   = timestamp_col
        )


def summarize_window_splits(
    df: pl.DataFrame,
    window_ids: list[str] | None = None,
    calib_frac: float = 0.20,
    timestamp_col: str = "timestamp"
) -> pl.DataFrame:
    """
    Build summary table with row counts for each split.
    """
    if window_ids is None:
        window_ids = get_window_ids()

    rows = []

    for window_id in window_ids:
        split = get_window_split(
            df,
            window_id       = window_id,
            calib_frac      = calib_frac,
            timestamp_col   = timestamp_col
        )

        rows.append({
            "window_id": window_id,
            "n_train_full": split.train_full.height,
            "n_train_model": split.train_model.height,
            "n_calib": split.calib.height,
            "n_test": split.test.height
        })

    return pl.DataFrame(rows)