"""
Data loading and mandatory preprocessing.

Responsible for:
  - reading parquet files per symbol
  - casting timestamps to int64 microseconds
  - shifting Bybit liquidation timestamps by +200ms
  - sorting all frames by timestamp
  - optional date-range filtering for train/val splits
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

Symbol = Literal["btcusdt", "ethusdt"]

US_PER_SECOND: int = 1_000_000
BYBIT_LAG_US: int = 200_000  # +200 ms

SPLIT_RANGES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train":        (pd.Timestamp("2025-12-01",          tz="UTC"), pd.Timestamp("2026-02-01",          tz="UTC")),
    "validation":   (pd.Timestamp("2026-02-01",          tz="UTC"), pd.Timestamp("2026-03-01",          tz="UTC")),

    # small sample splits for quick tests
    "sample_train": (pd.Timestamp("2025-12-01 00:00:00", tz="UTC"), pd.Timestamp("2025-12-01 01:00:00", tz="UTC")),
    "sample_val":   (pd.Timestamp("2025-12-01 01:00:00", tz="UTC"), pd.Timestamp("2025-12-01 02:00:00", tz="UTC")),
}


_TICKER_TO_SYMBOL: dict[str, Symbol] = {
    "perp:btcusdt": "btcusdt",
    "perp:ethusdt": "ethusdt",
    "btcusdt":      "btcusdt",
    "ethusdt":      "ethusdt",
}

LIQ_BUFFER_US: int = 5 * 60 * US_PER_SECOND  # 5-minute cold-start buffer for liq EWMA


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw parquet frame:
      - ensure timestamp is int64
      - lowercase string columns (side, ticker)
      - sort by timestamp ascending
    Idempotent.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("side", "ticker"):
        if col in df.columns:
            df[col] = df[col].str.lower()
    # stable sort: ties (same-microsecond timestamps) must keep on-disk order
    # regardless of which subset of rows is being sorted, so that chunked
    # reads produce the same row order as a single whole-split read.
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _read_parquet_filtered(path: Path, start_us: int | None, end_us: int | None) -> pd.DataFrame:
    """
    Read a parquet file, pushing a timestamp range filter down to PyArrow so
    row groups entirely outside [start_us, end_us) are skipped without being
    decompressed into memory. Falls back to a full read when no range is given.
    """
    dataset = ds.dataset(path, format="parquet")
    filt = None
    if start_us is not None:
        filt = (pc.field("timestamp") >= start_us) & (pc.field("timestamp") < end_us)
    return dataset.to_table(filter=filt).to_pandas()


def load_data_range(
    data_dir: str,
    symbol: Symbol,
    start_us: int | None,
    end_us: int | None,
    liq_buffer_us: int = LIQ_BUFFER_US,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 4 frames for one symbol over an explicit [start_us, end_us)
    range, pushing the range down to PyArrow row-group filtering so
    out-of-range data is never materialized in memory.

    liq frames get an extra `liq_buffer_us` lookback before start_us so liq
    EWMA features don't cold-start at the window edge. Pass
    start_us=end_us=None to load every frame in full (no filtering).

    Returns (trades, bbo, liq_binance, liq_bybit). liq_bybit timestamps are
    ALREADY shifted by BYBIT_LAG_US.
    """
    trades, bbo = load_trades_bbo_range(data_dir, symbol, start_us, end_us)

    liq_start_us = None if start_us is None else start_us - liq_buffer_us
    liq_bin, liq_bybit = load_liq_data(data_dir, symbol, liq_start_us, end_us)

    return trades, bbo, liq_bin, liq_bybit


def load_data_with_required_preprocess(
    data_dir: str,
    symbol: Symbol,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 4 frames for one symbol and run mandatory preprocessing.

    If split is given, filters to that split's date range (with a liq
    lookback buffer) via load_data_range so that we don't load the whole dataset into memory. 
    
    Returns (trades, bbo, liq_binance, liq_bybit). liq_bybit timestamps are ALREADY shifted.
    """
    if split is None:
        return load_data_range(data_dir, symbol, start_us=None, end_us=None)

    start_ts, end_ts = SPLIT_RANGES[split]
    start_us = int(start_ts.timestamp() * US_PER_SECOND)
    end_us   = int(end_ts.timestamp()   * US_PER_SECOND)
    return load_data_range(data_dir, symbol, start_us, end_us)


def compute_num_days(trades: pd.DataFrame) -> float:
    """
    Number of distinct calendar dates (UTC) spanned by the trades frame.
    Uses trades["timestamp"] (int64 microseconds).
    """
    dates = pd.to_datetime(trades["timestamp"], unit="us", utc=True).dt.date
    return float(dates.nunique())


def detect_symbol(trades: pd.DataFrame) -> Symbol:
    """
    Infer symbol from the ticker column.
    'perp:btcusdt' -> 'btcusdt', 'perp:ethusdt' -> 'ethusdt'.
    Raises ValueError on mixed or unknown tickers.
    """
    tickers = trades["ticker"].unique()
    symbols = {_TICKER_TO_SYMBOL[t] for t in tickers if t in _TICKER_TO_SYMBOL}
    unknown = [t for t in tickers if t not in _TICKER_TO_SYMBOL]
    if unknown or len(symbols) != 1:
        raise ValueError(f"Mixed or unknown tickers: {list(tickers)}")
    return symbols.pop()
