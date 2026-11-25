"""Unit tests for Indian stock data collection module."""

import io
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.collector import (
    fetch_stock_data,
    load_csv_data,
    validate_ohlcv_dataframe,
    _flatten_yfinance_dataframe,
    DataCollectionError,
)


def test_validate_ohlcv_dataframe_valid(synthetic_ohlcv_df):
    """Validates that a correctly formed DataFrame passes validation."""
    validate_ohlcv_dataframe(synthetic_ohlcv_df)


def test_validate_ohlcv_dataframe_missing_column(synthetic_ohlcv_df):
    """Checks that missing required OHLCV columns raises DataCollectionError."""
    bad_df = synthetic_ohlcv_df.drop(columns=["Volume"])
    with pytest.raises(DataCollectionError, match="Missing required OHLCV columns"):
        validate_ohlcv_dataframe(bad_df)


def test_validate_ohlcv_dataframe_empty():
    """Checks that empty DataFrame raises DataCollectionError."""
    with pytest.raises(DataCollectionError, match="dataset is empty"):
        validate_ohlcv_dataframe(pd.DataFrame())


def test_flatten_yfinance_multiindex():
    """Tests flattening of yfinance MultiIndex column headers for Indian equities."""
    cols = pd.MultiIndex.from_tuples([
        ("Close", "RELIANCE.NS"),
        ("Open", "RELIANCE.NS"),
        ("High", "RELIANCE.NS"),
        ("Low", "RELIANCE.NS"),
        ("Volume", "RELIANCE.NS"),
    ])
    df = pd.DataFrame([[1350, 1340, 1360, 1335, 15000000]], columns=cols)
    flattened = _flatten_yfinance_dataframe(df)
    assert list(flattened.columns) == ["Close", "Open", "High", "Low", "Volume"]


@patch("src.data.collector.yf.Ticker")
def test_fetch_stock_data_success_nse(mock_ticker, synthetic_ohlcv_df):
    """Tests successful fetch and metadata creation for NSE stock with mocked yfinance."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = synthetic_ohlcv_df.copy()
    mock_ticker.return_value = mock_instance

    df, meta = fetch_stock_data(ticker="RELIANCE.NS", period="1y", save_raw=False)

    assert len(df) == len(synthetic_ohlcv_df)
    assert meta["ticker"] == "RELIANCE.NS"
    assert meta["exchange"] == "NSE"
    assert meta["currency"] == "INR"
    assert meta["currency_symbol"] == "₹"
    assert meta["timezone"] == "Asia/Kolkata"
    assert meta["source"] == "Yahoo Finance (yfinance)"
    assert meta["row_count"] == len(synthetic_ohlcv_df)
    assert "Close" in df.columns


@patch("src.data.collector.yf.Ticker")
def test_fetch_stock_data_success_bse(mock_ticker, synthetic_ohlcv_df):
    """Tests successful fetch and metadata creation for BSE stock with mocked yfinance."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = synthetic_ohlcv_df.copy()
    mock_ticker.return_value = mock_instance

    df, meta = fetch_stock_data(ticker="TCS.BO", period="1y", save_raw=False)

    assert len(df) == len(synthetic_ohlcv_df)
    assert meta["ticker"] == "TCS.BO"
    assert meta["exchange"] == "BSE"
    assert meta["currency"] == "INR"


@patch("src.data.collector.yf.Ticker")
def test_fetch_stock_data_empty_raises(mock_ticker):
    """Tests that empty return from yfinance raises DataCollectionError."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance

    with patch("src.data.collector.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(DataCollectionError, match="No data returned for Indian ticker"):
            fetch_stock_data(ticker="INVALID_TICKER.NS", save_raw=False)


def test_fetch_stock_data_empty_ticker_string():
    """Tests that empty ticker string raises error."""
    with pytest.raises(DataCollectionError, match="ticker symbol cannot be empty"):
        fetch_stock_data(ticker="   ", save_raw=False)


def test_load_csv_data_valid(synthetic_ohlcv_df):
    """Tests loading from in-memory CSV buffer for Indian equities."""
    csv_buf = io.StringIO()
    synthetic_ohlcv_df.to_csv(csv_buf)
    csv_buf.seek(0)

    df, meta = load_csv_data(csv_buf, ticker="INFY.NS", save_raw=False)
    assert len(df) == len(synthetic_ohlcv_df)
    assert meta["ticker"] == "INFY.NS"
    assert meta["currency"] == "INR"
    assert meta["source"] == "CSV Upload"
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_csv_data_missing_date_column():
    """Tests loading CSV missing a Date column raises error."""
    csv_content = "Open,High,Low,Close,Volume\n1300,1350,1280,1320,15000000\n"
    csv_buf = io.StringIO(csv_content)

    with pytest.raises(DataCollectionError, match="must contain a 'Date' column"):
        load_csv_data(csv_buf, save_raw=False)
