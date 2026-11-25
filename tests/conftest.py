"""Pytest fixtures and deterministic synthetic test data for Indian stock market.

NOTE: All synthetic datasets created here are strictly TEST FIXTURES ONLY
and are never mixed into production datasets or real model training results.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generates 120 days of deterministic daily OHLCV data in INR (₹) for testing.

    LABEL: TEST FIXTURE ONLY.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=120, freq="B")  # Business days
    n = len(dates)

    # Realistic Indian large-cap stock price levels (₹1,200 to ₹1,500)
    t = np.linspace(0, 10, n)
    trend = 1250.0 + 20.0 * t + 15.0 * np.sin(t)
    noise = np.random.normal(0, 5.0, n)
    close = np.round(trend + noise, 2)

    open_p = np.round(close + np.random.normal(0, 2.0, n), 2)
    high_p = np.round(np.maximum(close, open_p) + np.abs(np.random.normal(4.0, 1.0, n)), 2)
    low_p = np.round(np.minimum(close, open_p) - np.abs(np.random.normal(4.0, 1.0, n)), 2)
    volume = np.random.randint(5_000_000, 25_000_000, size=n)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def short_ohlcv_df() -> pd.DataFrame:
    """Generates 20 days of OHLCV data (insufficient for 50-day SMA).

    LABEL: TEST FIXTURE ONLY.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=20, freq="B")
    n = len(dates)
    close = np.linspace(1000, 1100, n)

    df = pd.DataFrame(
        {
            "Open": close - 2.0,
            "High": close + 5.0,
            "Low": close - 5.0,
            "Close": close,
            "Volume": np.full(n, 10000000),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
