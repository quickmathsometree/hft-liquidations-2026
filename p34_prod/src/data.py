# data.py - file for data reading / chanking / saving / preprocessing.


# imports.
import gc
import shutil
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from dataclasses import dataclass

from features import (
    MOMENTUM_WINDOWS_S,
    ROLLING_WINDOWS_S,
    add_bbo_and_markout,
    add_liquidation_features,
    add_mid_momentum_features,
    add_rolling_features,
    get_momentum_column_names,
)


# constants.
US_PER_SEC = 1_000_000
US_PER_DAY = 86_400 * US_PER_SEC


# data config.
@dataclass
class DataConfig:
    chunk_days: int = 1

    windows_s: tuple[int, ...] = ROLLING_WINDOWS_S
    horizons_s: tuple[int, ...] = (30, 120, 300)

    rebate_bps: float = 0.5
    clipped_notional: float = 50_000.0

    bbo_lookback_s: int = 60
    bybit_shift_ms: int = 200

    n_min_obs: int = 3

    @property
    def effective_bbo_lookback_s(self) -> int:
        if self.bbo_lookback_s is None:
            return max(self.windows_s)

        return max(self.bbo_lookback_s, max(self.windows_s))

    @property
    def output_columns(self) -> tuple[str, ...]:
        base_cols = (
            "timestamp",
            "price",
            "mid_price",
            "spread",
            "obi",
            "dist_to_mid",
            "sign",
            "notional",
            "weight",
            "hour",
        )

        vpin_cols = tuple(f"vpin_{w}s" for w in self.windows_s)
        rv_cols = tuple(f"rv_{w}s" for w in self.windows_s)

        liq_cols = tuple(
            col
            for w in self.windows_s
            for col in (f"liq_binance_{w}s", f"liq_bybit_{w}s")
        )

        pnl_cols = tuple(f"pnl_{tau}s" for tau in self.horizons_s)
        momentum_cols = get_momentum_column_names()

        return base_cols + vpin_cols + rv_cols + liq_cols + momentum_cols + pnl_cols


# parquet support functions.
def collect_lazy(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect pl.DataFrame from pl.LazyFrame.
    """
    return lf.collect(engine='streaming')

def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """
    Create .parquet file from pl.Dataframe.
    """
    df.write_parquet(path, compression="zstd", statistics=True)

def count_rows_parquet(path: str | Path) -> int:
    """
    Return number of rows in parquet file without loading data into memory.

    Uses parquet metadata only.

    Can be used for one file: `".parquet"` or chunks: `"*.parquet"`.
    """

    path = Path(path)

    if path.is_file():
        return pq.ParquetFile(path).metadata.num_rows

    if path.is_dir():

        parquet_files = sorted(path.glob("*.parquet"))

        return sum(
            pq.ParquetFile(p).metadata.num_rows
            for p in parquet_files
        )
    
    raise ValueError(f"Unsupported path type: {path}")

def safe_unlink(path: Path) -> None:
    """
    Deleate file.
    """
    if path.exists():
        path.unlink()
    else:
        raise ValueError("Can't find file.")


# chunks building.
def build_chunk_plan(
    trades_path: str | Path,
    chunk_days: int,
    align_to_utc: bool = True
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Build time chunks from trades parquet.

    Parametrs
    ---------
    trades_path: 
        - Path to raw trades parquet.

    chunk_days: 
        - Chunk width in calendar days.

    align_to_utc:
        - If True, the first chunk starts at UTC day boundary.
        - If False, the first chunk starts exactly at min(timestamp).


    Returns
    -------
    chunks:
        - List of half-open intervals [chunk_start_us, chunk_end_us).

    raw_ts_min:
        - Minimum timestamp in the file.

    raw_ts_max:
        - Maximum timestamp in the file.
    """

    trades_path = Path(trades_path)


    # values check.
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive.")

    if not trades_path.exists():
        raise FileNotFoundError(f"File not found: {trades_path}")
    

    # Find min/max timestamp.
    range_df = collect_lazy(
        pl.scan_parquet(trades_path).select([
            pl.col("timestamp").min().alias("ts_min"),
            pl.col("timestamp").max().alias("ts_max")
        ])
    )

    ts_min = range_df[0, "ts_min"]
    ts_max = range_df[0, "ts_max"]


    chunk_width_us = int(chunk_days * US_PER_DAY)

    if align_to_utc:
        first_chunk_start = (ts_min // US_PER_DAY) * US_PER_DAY
    else:
        first_chunk_start = ts_min

    
    # timestamp is integer microseconds.
    # Since later we filter as timestamp < chunk_end,
    # we need last_exclusive = max_timestamp + 1.
    last_exclusive = ts_max + 1

    chunks: list[tuple[int, int]] = []
    chunk_start = first_chunk_start
    while chunk_start < last_exclusive:
        chunk_end = min(chunk_start + chunk_width_us, last_exclusive)
        chunks.append((chunk_start, chunk_end))
        chunk_start += chunk_width_us

    return chunks, ts_min, ts_max


# trades / BBO / liquidations reading.
def read_trades_for_chunk(
    path: str | Path,
    chunk_start_us: int,
    chunk_end_us: int,
    max_window_us: int,
    clipped_notional: float
) -> pl.DataFrame:
    """
    Read trades for one time chunk with left overlap.

    Read intervals:
        [chunk_start_us - max_window_us, chunk_end_us)
    
    The overlap is needed because rolling features at the beginning of the
    chunk require historical trades before chunk_start_us.


    Output columns:

        - timestamp : int64
        - price     : float64
        - sign      : int8, +1 for buy, -1 otherwise
        - notional  : float32, price * amount
        - weight    : float32, clipped notional
        - hour      : int8, UTC hour
    """

    # values check.
    if chunk_end_us <= chunk_start_us:
        raise ValueError("chunk_end_us must be greater than chunk_start_us.")
    
    # chunk size.
    read_start_us = chunk_start_us - max_window_us
    read_end_us = chunk_end_us

    # create lazyframe.
    lf = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("timestamp") >= read_start_us) & (pl.col("timestamp") < read_end_us)
        )
        .select([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("price").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
            pl.col("side")
        ])
        .with_columns([
            pl.when(pl.col("side") == "buy")
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .cast(pl.Int8)
            .alias("sign"),

            (pl.col("price") * pl.col("amount"))
            .cast(pl.Float32)
            .alias("notional"),

            (pl.col("price") * pl.col("amount"))
            .clip(upper_bound=clipped_notional)
            .cast(pl.Float32)
            .alias("weight"),

            pl.from_epoch(pl.col("timestamp"), time_unit="us")
            .dt.hour()
            .cast(pl.Int8)
            .alias("hour")
        ])
        .select([
            "timestamp",
            "price",
            "sign",
            "notional",
            "weight",
            "hour"
        ])
        .sort("timestamp")
    )

    return collect_lazy(lf)

def read_bbo_for_chunk(
    path: str | Path,
    chunk_start_us: int,
    chunk_end_us: int,
    bbo_lookback_us: int,
    max_horizon_us: int  
) -> pl.DataFrame:
    """
    Read BBO data for one chunk with left lookback and right lookahead.

    Reads interval:
        [chunk_start_us - bbo_lookback_us, chunk_end_us   + max_horizon_us)

    Left lookback is needed for ASOF join at trade time.
    Right lookahead is needed for markout pnl at t + tau.

    Returns columns:
        timestamp  : Int64
        mid_price  : Float64
        spread     : Float32
        obi        : Float32

    where:
        mid_price = (bid_price + ask_price) / 2
        spread    = ask_price - bid_price
        obi       = bid_amount / (bid_amount + ask_amount)
    """

    path = Path(path)

    read_start  = chunk_start_us - bbo_lookback_us
    read_end    = chunk_end_us + max_horizon_us

    # create lazyframe.
    lf = (
        pl.scan_parquet(str(path))
        .filter(
            (pl.col("timestamp") >= read_start) & (pl.col("timestamp") < read_end)
        )
        .select([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("bid_amount").cast(pl.Float64),
            pl.col("ask_amount").cast(pl.Float64)
        ])
        .with_columns([
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0)
            .cast(pl.Float64)
            .alias("mid_price"),

            (pl.col("ask_price") - pl.col("bid_price"))
            .cast(pl.Float32)
            .alias("spread"),

            pl.when((pl.col("bid_amount") + pl.col("ask_amount")) > 0)
            .then(
                pl.col("bid_amount") / (pl.col("bid_amount") + pl.col("ask_amount"))
            )
            .otherwise(None)
            .cast(pl.Float32)
            .alias("obi")
        ])
        .select([
            "timestamp",
            "mid_price",
            "spread",
            "obi"
        ])
        .sort("timestamp")
    )

    return collect_lazy(lf)

def read_liquidations_for_chunk(
    path: str | Path,
    chunk_start_us: int,
    chunk_end_us: int,
    max_window_us: int,
    timestamp_shift_us: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read signed liquidation amounts for one chunk.

    The returned timestamps are already shifted by timestamp_shift_us.

    Parameters
    ----------
    path:
        Path to liquidation parquet.

    chunk_start_us, chunk_end_us:
        Main chunk interval in microseconds:
            [chunk_start_us, chunk_end_us)

    max_window_us:
        Left overlap needed for rolling liquidation features.
        We need liquidations from:
            [chunk_start_us - max_window_us, chunk_end_us)

    timestamp_shift_us:
        Timestamp shift applied after reading.

        Binance:
            timestamp_shift_us = 0

        Bybit:
            timestamp_shift_us = 200_000  # 200 ms

    Returns
    -------
    liq_ts:
        Sorted int64 timestamps after shift.

    cum_prefix:
        Prefix sum of signed liquidation amount.
        Shape: len(liq_ts) + 1
        cum_prefix[0] = 0.
    """

    path = Path(path)

    # Desired interval AFTER timestamp shift.
    shifted_read_start  = chunk_start_us - max_window_us
    shifted_read_end    = chunk_end_us

    # Convert desired shifted interval to RAW timestamp interval.
    # We need : raw_timestamp + timestamp_shift_us in [shifted_read_start, shifted_read_end)
    raw_read_start = shifted_read_start - timestamp_shift_us
    raw_read_end = shifted_read_end - timestamp_shift_us

    # create lazyframe.
    lf = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("timestamp") >= raw_read_start)& (pl.col("timestamp") < raw_read_end)
        )
        .select([
            (
                pl.col("timestamp").cast(pl.Int64) + pl.lit(timestamp_shift_us, dtype=pl.Int64)
            ).alias("timestamp"),

            pl.when(pl.col("side") == "buy")
              .then(pl.col("amount"))
              .otherwise(-pl.col("amount"))
              .cast(pl.Float32)
              .alias("signed_amount"),
        ])
        .drop_nulls(["timestamp", "signed_amount"])
        .sort("timestamp")
    )

    df = collect_lazy(lf)

    liq_ts = df["timestamp"].to_numpy().astype(np.int64, copy=False)
    signed_amount = df["signed_amount"].to_numpy().astype(np.float64, copy=False)

    cum_prefix = np.empty(len(signed_amount) + 1, dtype=np.float64)
    cum_prefix[0] = 0.0
    np.cumsum(signed_amount, out=cum_prefix[1:])

    return liq_ts, cum_prefix


# one chunk building.
def compute_chunk(
    trades_path: str | Path,
    bbo_path: str | Path,
    liq_binance_path: str | Path,
    liq_bybit_path: str | Path,
    chunk_start_us: int,
    chunk_end_us: int,
    config: DataConfig
) -> pl.DataFrame:
    """
    Compute one enriched chunk.

    The function reads raw data with the required left/right overlaps,
    computes all trade-level features, then returns only rows from [chunk_start_us, chunk_end_us)

    Parameters
    ----------
    trades_path:
        Path to raw trades parquet.

    bbo_path:
        Path to raw BBO parquet.

    liq_binance_path:
        Path to Binance liquidations parquet.

    liq_bybit_path:
        Path to Bybit liquidations parquet.

    chunk_start_us:
        Chunk start timestamp in microseconds.

    chunk_end_us:
        Chunk end timestamp in microseconds.

    config:
        DataConfig-like object with fields:
            horizons_s
            windows_s
            bbo_lookback_s
            bybit_shift_ms
            clipped_notional
            rebate_bps
            output_columns
            n_min_obs
    """

    # 0. paths.
    bbo_path            = Path(bbo_path)
    trades_path         = Path(trades_path)
    liq_bybit_path      = Path(liq_bybit_path)
    liq_binance_path    = Path(liq_binance_path)

    # 1. time overlaps.
    max_horizon_us = max(config.horizons_s) * US_PER_SEC
    max_window_us = max(config.windows_s) * US_PER_SEC

    bbo_lookback_us = config.effective_bbo_lookback_s * US_PER_SEC
    bybit_delay_us = config.bybit_shift_ms * 1000

    # 2. read trades.
    trades = read_trades_for_chunk(
        path            = trades_path,
        chunk_start_us  = chunk_start_us,
        chunk_end_us    = chunk_end_us,
        max_window_us   = max_window_us,
        clipped_notional= config.clipped_notional
    )

    if 0 == trades.height:
        raise ValueError(
            f"No trades found for chunk "
            f"[{chunk_start_us}, {chunk_end_us}) "
            f"with left overlap {max_window_us} us."
        )
    
    # 3. read BBO with left lookback and right lookahead.
    bbo = read_bbo_for_chunk(
            path            = bbo_path,
            chunk_start_us  = chunk_start_us,
            chunk_end_us    = chunk_end_us,
            bbo_lookback_us = bbo_lookback_us,
            max_horizon_us  = max_horizon_us
        )

    if bbo.height == 0:
        raise ValueError(
            f"No BBO found for chunk "
            f"[{chunk_start_us}, {chunk_end_us}) "
            f"with lookback {bbo_lookback_us} us "
            f"and lookahead {max_horizon_us} us."
        )

    max_bbo_ts = int(bbo["timestamp"].max())

    # 4. Read liquidations.
    binance_ts, binance_cum = read_liquidations_for_chunk(
        path                = liq_binance_path,
        chunk_start_us      = chunk_start_us,
        chunk_end_us        = chunk_end_us,
        max_window_us       = max_window_us,
        timestamp_shift_us  = 0
    )

    bybit_ts, bybit_cum = read_liquidations_for_chunk(
        path                = liq_bybit_path,
        chunk_start_us      = chunk_start_us,
        chunk_end_us        = chunk_end_us,
        max_window_us       = max_window_us,
        timestamp_shift_us  = bybit_delay_us
    )

    # 5. Add BBO features and markout PnL.
    df = add_bbo_and_markout(
        trades      = trades,
        bbo         = bbo,
        horizons_s  = list(config.horizons_s),
        rebate_bps  = config.rebate_bps,
        max_bbo_ts  = max_bbo_ts
    )

    # 6. Add rolling trade features: VPIN, RV, etc.
    df = add_rolling_features(
        df          = df,
        windows_s   = list(config.windows_s),
        n_min_obs   = config.n_min_obs
    )

    # 7. Add rolling liquidation features.
    df = add_liquidation_features(
        df          = df,
        bin_ts      = binance_ts,
        bin_cum     = binance_cum,
        byb_ts      = bybit_ts,
        byb_cum     = bybit_cum,
        windows_s   = list(config.windows_s)
    )

    # 8. Mid-price momentum from the BBO path.
    df = add_mid_momentum_features(
        df          = df,
        bbo         = bbo,
        windows_s   = list(MOMENTUM_WINDOWS_S),
    )

    # 9. Drop left overlap.
    df = df.filter(
        (pl.col("timestamp") >= chunk_start_us) & (pl.col("timestamp") < chunk_end_us)
    )

    if df.height == 0:
        raise ValueError(
            f"Chunk became empty after dropping overlap: "
            f"[{chunk_start_us}, {chunk_end_us})."
        )

    # 10. Fixed output schema/order.
    df = (
        df
        .select(list(config.output_columns))
        .sort("timestamp")
    )

    return df


# dataset building.
def make_subsample_for_chunk(
    df: pl.DataFrame,
    global_row_offset: int,
    subsample_rate: int | None
) -> pl.DataFrame:
    """
    Deterministic chronological subsample for one chunk.

    Keeps rows satisfying: (global_row_offset + local_row_index) % subsample_rate == 0

    Parameters
    ----------
    df:
        Chunk DataFrame already sorted chronologically.

    global_row_offset:
        Number of rows in all previous chunks before this chunk.

    subsample_rate:
        If 40, keep every 40th row globally.
        If None or 1, return df unchanged.

    Returns
    -------
    pl.DataFrame
        Subsampled chunk.
    """

    if (subsample_rate is None) or (1 == subsample_rate):
        return df
    
    local_idx = np.arange(df.height, dtype=np.int64)
    global_idx = global_row_offset + local_idx

    mask = (global_idx % subsample_rate) == 0

    return df.filter(pl.Series(mask))

def build_dataset(
    trades_path: str | Path,
    bbo_path: str | Path,
    liq_binance_path: str | Path,
    liq_bybit_path: str | Path,
    output_path: str | Path,
    config: DataConfig,
    *,
    subsample_rate: int | None = None,
    overwrite: bool = False,
    keep_chunk_files: bool = False,
    align_to_utc: bool = True
) -> Path:
    """
    Build full enriched dataset from raw trades, BBO and liquidation files.

    The function:
        1. Builds chronological chunk plan from trades.
        2. Computes enriched data for each chunk using compute_chunk(...).
        3. Optionally applies deterministic chronological subsampling.
        4. Saves temporary chunk parquet files.
        5. Concatenates all chunks into final output_path.

    Parameters
    ----------
    trades_path:
        Path to raw trades parquet.

    bbo_path:
        Path to raw BBO parquet.

    liq_binance_path:
        Path to Binance liquidations parquet.

    liq_bybit_path:
        Path to Bybit liquidations parquet.

    output_path:
        Path to final enriched parquet.

    config:
        DataConfig object.

    subsample_rate:
        If None or 1, no subsampling.
        If 40, keeps every 40th row globally across all chunks.

    overwrite:
        If True, removes existing output file and temporary chunk directory.

    keep_chunk_files:
        If True, keeps temporary chunk parquet files.

    align_to_utc:
        If True, chunks are aligned to UTC day boundaries.

    Returns
    -------
    Path
        Path to final enriched parquet.
    """

    # 1. Convert paths.
    output_path         = Path(output_path)
    bbo_path            = Path(bbo_path)
    trades_path         = Path(trades_path)
    liq_bybit_path      = Path(liq_bybit_path)
    liq_binance_path    = Path(liq_binance_path)

    # 2. Validate input files.
    for path in [trades_path, bbo_path, liq_binance_path, liq_bybit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    # 3. Validate parameters.
    if config.chunk_days <= 0:
        raise ValueError("config.chunk_days must be positive.")

    if len(config.windows_s) == 0:
        raise ValueError("config.windows_s must be non-empty.")

    if len(config.horizons_s) == 0:
        raise ValueError("config.horizons_s must be non-empty.")

    if subsample_rate is not None and subsample_rate <= 0:
        raise ValueError("subsample_rate must be positive or None.")

    # 4. Prepare output paths.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_path.parent / f".{output_path.stem}_chunks"

    if output_path.exists():
        if overwrite:
            safe_unlink(output_path)
        else:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                f"Use overwrite=True to overwrite it."
            )

    if tmp_dir.exists():
        if overwrite:
            shutil.rmtree(tmp_dir)
        else:
            raise FileExistsError(
                f"Temporary chunk directory already exists: {tmp_dir}. "
                f"Use overwrite=True to remove it."
            )

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build chunk plan.
    chunks, _, _ = build_chunk_plan(
        trades_path     = trades_path,
        chunk_days      = config.chunk_days,
        align_to_utc    = align_to_utc
    )

    if len(chunks) == 0:
        raise RuntimeError(
            "No chunks were created. Check trades timestamp range."
        )

    chunk_files: list[Path] = []
    global_row_offset = 0

    # 6. Process chunks.
    for chunk_idx, (chunk_start_us, chunk_end_us) in enumerate(
        tqdm(chunks, desc="Building enriched dataset")
    ):
        chunk_path = tmp_dir / f"chunk_{chunk_idx:05d}.parquet"


        df_chunk = compute_chunk(
            trades_path         = trades_path,
            bbo_path            = bbo_path,
            liq_binance_path    = liq_binance_path,
            liq_bybit_path      = liq_bybit_path,
            chunk_start_us      = chunk_start_us,
            chunk_end_us        = chunk_end_us,
            config              = config
        )


        # Number of rows before subsampling.
        # This is important: offset must be updated by full chunk size,
        # otherwise subsampling will depend on chunk boundaries.
        full_chunk_rows = df_chunk.height

        if subsample_rate is not None and subsample_rate > 1:
            df_chunk = make_subsample_for_chunk(
                df                  = df_chunk,
                global_row_offset   = global_row_offset,
                subsample_rate      = subsample_rate
            )

        global_row_offset += full_chunk_rows

        if df_chunk.height == 0:
            del df_chunk
            gc.collect()
            continue

        write_parquet(df_chunk, chunk_path)
        chunk_files.append(chunk_path)

        del df_chunk
        gc.collect()

    if len(chunk_files) == 0:
        raise RuntimeError(
            "All chunks are empty. Check raw files, timestamp ranges and BBO coverage."
        )

    # 7. Concatenate temporary chunks into final parquet.
    lf = (
        pl.scan_parquet([str(path) for path in chunk_files])
        .select(list(config.output_columns))
        .sort("timestamp")
    )

    try:
        lf.sink_parquet(
            str(output_path),
            compression="zstd",
            statistics=True
        )
    except Exception:
        df_final = collect_lazy(lf)
        write_parquet(df_final, output_path)
        del df_final

    # 8. Remove temporary chunk files if needed.
    if not keep_chunk_files:
        shutil.rmtree(tmp_dir)

    gc.collect()

    return output_path

