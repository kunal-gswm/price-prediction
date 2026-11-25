"""Indian Stock Market Data Collection Module.

Collects historical daily OHLCV market data for Indian equities (NSE & BSE)
via Yahoo Finance (`yfinance`) with multi-index normalization, rate-limit resilience,
metadata auditing, and local CSV upload support.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import io
import json
import pandas as pd
import yfinance as yf

from src.utils.config import (
    RAW_DATA_DIR,
    DEFAULT_TICKER,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
    MARKET_CURRENCY,
    CURRENCY_SYMBOL,
    MARKET_TIMEZONE,
    get_stock_display_info,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataCollectionError(Exception):
    """Exception raised when stock market data collection fails."""
    pass


def _flatten_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens MultiIndex columns if returned by yfinance and standardizes OHLCV column names."""
    if df.empty:
        return df

    # Handle MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        df.columns = first_level

    # Normalize column capitalization
    col_mapping = {}
    for col in df.columns:
        c_str = str(col).strip().lower()
        if c_str == "open":
            col_mapping[col] = "Open"
        elif c_str == "high":
            col_mapping[col] = "High"
        elif c_str == "low":
            col_mapping[col] = "Low"
        elif c_str == "close":
            col_mapping[col] = "Close"
        elif c_str in ["adj close", "adjclose"]:
            col_mapping[col] = "Adj Close"
        elif c_str == "volume":
            col_mapping[col] = "Volume"

    df = df.rename(columns=col_mapping)
    return df


def validate_ohlcv_dataframe(df: pd.DataFrame) -> None:
    """Validates that DataFrame contains required OHLCV columns and is non-empty.

    Args:
        df: Input pandas DataFrame to validate.

    Raises:
        DataCollectionError: If DataFrame is empty, missing columns, or invalid.
    """
    if df is None or df.empty:
        raise DataCollectionError("The retrieved Indian equity dataset is empty.")

    missing_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataCollectionError(
            f"Missing required OHLCV columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataCollectionError("DataFrame index must be a DatetimeIndex.")

    # Deduplicate index
    if df.index.duplicated().any():
        logger.warning("Duplicate timestamps found in dataset. Deduplicating by keeping last.")
        df = df[~df.index.duplicated(keep="last")]

    # Check monotonic ascending order
    if not df.index.is_monotonic_increasing:
        logger.warning("Data index is not monotonically increasing. Sorting chronologically.")
        df.sort_index(inplace=True)


def fetch_stock_data(
    ticker: str = DEFAULT_TICKER,
    period: str = DEFAULT_PERIOD,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = DEFAULT_INTERVAL,
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetches historical Indian equity OHLCV data from Yahoo Finance via yfinance.

    Default Instrument: Reliance Industries Limited (RELIANCE.NS) on NSE.

    Args:
        ticker: Indian stock ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.BO').
        period: Historical period ('1y', '2y', '5y', 'max').
        start: Optional start date string ('YYYY-MM-DD').
        end: Optional end date string ('YYYY-MM-DD').
        interval: Data interval ('1d', '1wk').
        save_raw: Whether to save raw CSV and metadata to disk.

    Returns:
        Tuple of (pd.DataFrame, metadata dict).

    Raises:
        DataCollectionError: If fetching fails, ticker is invalid, or data is empty.
    """
    ticker_clean = str(ticker).strip().upper()
    if not ticker_clean:
        raise DataCollectionError("Indian stock ticker symbol cannot be empty.")

    stock_info = get_stock_display_info(ticker_clean)
    logger.info(
        f"Fetching historical market data for '{ticker_clean}' ({stock_info['name']} on {stock_info['exchange']}) "
        f"[Period={period}, Start={start}, End={end}, Interval={interval}]."
    )

    try:
        yf_ticker = yf.Ticker(ticker_clean)
        if start and end:
            df = yf_ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        elif start:
            df = yf_ticker.history(start=start, interval=interval, auto_adjust=False)
        else:
            df = yf_ticker.history(period=period, interval=interval, auto_adjust=False)

        # Fallback to yf.download if history returns empty
        if df is None or df.empty:
            logger.info(f"yf.Ticker history returned empty for {ticker_clean}; trying yf.download fallback.")
            if start and end:
                df = yf.download(ticker_clean, start=start, end=end, interval=interval, progress=False)
            else:
                df = yf.download(ticker_clean, period=period, interval=interval, progress=False)

    except Exception as e:
        logger.error(f"Network or yfinance error while retrieving {ticker_clean}: {e}")
        raise DataCollectionError(
            f"Failed to fetch market data for '{ticker_clean}' from Yahoo Finance: {str(e)}"
        ) from e

    if df is None or df.empty:
        raise DataCollectionError(
            f"No data returned for Indian ticker symbol '{ticker_clean}'. "
            "Please verify that the ticker includes the exchange suffix (e.g. '.NS' for NSE or '.BO' for BSE)."
        )

    # Flatten and normalize columns
    df = _flatten_yfinance_dataframe(df)

    # Convert DatetimeIndex to Indian Standard Time (Asia/Kolkata) then naive date index
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(MARKET_TIMEZONE).tz_localize(None)

    # Validate
    validate_ohlcv_dataframe(df)

    # Sort chronologically and drop duplicate dates
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Retain core columns plus Adj Close if available
    desired_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[desired_cols]

    metadata = {
        "ticker": ticker_clean,
        "company_name": stock_info["name"],
        "exchange": stock_info["exchange"],
        "sector": stock_info["sector"],
        "currency": MARKET_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOL,
        "timezone": MARKET_TIMEZONE,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "period": period if not (start and end) else None,
        "start_date": str(df.index.min().date()) if len(df) > 0 else start,
        "end_date": str(df.index.max().date()) if len(df) > 0 else end,
        "interval": interval,
        "source": "Yahoo Finance (yfinance)",
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "notes": (
            "Historical daily OHLCV data from Indian equity exchanges (NSE/BSE). "
            "All prices are quoted in Indian Rupees (INR, ₹)."
        ),
    }

    if save_raw:
        safe_ticker = ticker_clean.replace(".", "_").replace("^", "")
        raw_csv_path = RAW_DATA_DIR / f"{safe_ticker}_raw.csv"
        raw_meta_path = RAW_DATA_DIR / f"{safe_ticker}_metadata.json"

        df.to_csv(raw_csv_path)
        with open(raw_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved raw Indian equity data ({len(df)} rows) to {raw_csv_path}")

    return df, metadata


def load_csv_data(
    file_source: Union[str, Path, io.BytesIO, io.StringIO],
    ticker: str = "CUSTOM.NS",
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads historical OHLCV data from an uploaded or local CSV file for Indian stocks.

    Args:
        file_source: Path to CSV or file-like buffer.
        ticker: Identifier label for the dataset (default includes .NS).
        save_raw: Whether to save to raw data directory.

    Returns:
        Tuple of (pd.DataFrame, metadata dict).

    Raises:
        DataCollectionError: If CSV is invalid or missing required columns.
    """
    try:
        df = pd.read_csv(file_source)
    except Exception as e:
        raise DataCollectionError(f"Failed to parse CSV file: {str(e)}") from e

    if df.empty:
        raise DataCollectionError("Uploaded CSV file is empty.")

    # Locate date column
    date_col = None
    for candidate in ["Date", "date", "Timestamp", "timestamp", "Datetime", "datetime", "Trading Date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise DataCollectionError(
            f"CSV must contain a 'Date' column. Available columns: {list(df.columns)}"
        )

    # Parse dates and set as index (supports mixed formats: ISO YYYY-MM-DD or Indian DD-MM-YYYY)
    try:
        df[date_col] = pd.to_datetime(df[date_col], format="mixed")
        df = df.set_index(date_col)
    except Exception:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        except Exception as e:
            raise DataCollectionError(f"Failed to parse dates in column '{date_col}': {e}") from e

    # Flatten & capitalize columns
    df = _flatten_yfinance_dataframe(df)

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(MARKET_TIMEZONE).tz_localize(None)

    # Validate
    validate_ohlcv_dataframe(df)

    # Sort index and drop duplicate dates
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Convert numeric columns
    for col in REQUIRED_OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where any OHLCV is NaN
    df = df.dropna(subset=REQUIRED_OHLCV_COLUMNS)

    ticker_clean = str(ticker).strip().upper()
    stock_info = get_stock_display_info(ticker_clean)

    metadata = {
        "ticker": ticker_clean,
        "company_name": stock_info["name"],
        "exchange": stock_info["exchange"],
        "currency": MARKET_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOL,
        "timezone": MARKET_TIMEZONE,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "period": "custom_upload",
        "start_date": str(df.index.min().date()) if len(df) > 0 else None,
        "end_date": str(df.index.max().date()) if len(df) > 0 else None,
        "interval": "custom",
        "source": "CSV Upload",
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "notes": "User-supplied Indian equity OHLCV dataset.",
    }

    if save_raw:
        safe_ticker = ticker_clean.replace(".", "_").replace("^", "")
        raw_csv_path = RAW_DATA_DIR / f"{safe_ticker}_upload_raw.csv"
        raw_meta_path = RAW_DATA_DIR / f"{safe_ticker}_upload_metadata.json"
        df.to_csv(raw_csv_path)
        with open(raw_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return df, metadata
