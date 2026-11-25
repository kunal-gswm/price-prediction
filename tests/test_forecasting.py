"""Unit tests for Indian equity next-trading-day forecasting engine."""

import pytest
import pandas as pd
import numpy as np

from src.models.forecasting import predict_next_trading_day
from src.utils.indian_calendar import get_next_indian_trading_session, is_indian_market_trading_day
from src.models.linear_regression import StockPriceModel
from src.features.engineering import prepare_training_dataset
from src.data.preprocessing import chronological_train_val_test_split
from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN


def test_indian_calendar_weekend_handling():
    """Verifies that Saturday and Sunday are marked as non-trading days."""
    saturday = pd.Timestamp("2026-11-21")  # Saturday
    sunday = pd.Timestamp("2026-11-22")    # Sunday
    monday = pd.Timestamp("2026-11-23")    # Monday

    assert not is_indian_market_trading_day(saturday)
    assert not is_indian_market_trading_day(sunday)
    assert is_indian_market_trading_day(monday)


def test_indian_calendar_next_session_weekend_roll():
    """Verifies that Friday rolls to Monday (skipping Saturday and Sunday)."""
    friday = pd.Timestamp("2026-11-20")  # Friday
    next_session, method = get_next_indian_trading_session(friday)

    assert next_session == pd.Timestamp("2026-11-23")  # Monday
    assert "Asia/Kolkata" in method


def test_indian_calendar_holiday_handling():
    """Verifies that Republic Day (Jan 26) and Independence Day (Aug 15) are recognized as holidays."""
    republic_day = pd.Timestamp("2026-01-26")  # Monday
    independence_day = pd.Timestamp("2026-08-15")  # Saturday / Holiday

    assert not is_indian_market_trading_day(republic_day)
    assert not is_indian_market_trading_day(independence_day)


def test_predict_next_trading_day_end_to_end_inr(synthetic_ohlcv_df):
    """Tests full next-session prediction generation in INR for Indian equities."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    model = StockPriceModel()
    model.fit(splits["X_train"], splits["y_train"], ticker="RELIANCE.NS")

    forecast = predict_next_trading_day(
        model=model,
        df_history=synthetic_ohlcv_df,
        conformal_margin=18.75,
        confidence_level=0.95,
    )

    assert "predicted_next_close" in forecast
    assert forecast["currency"] == "INR"
    assert forecast["currency_symbol"] == "₹"
    assert "₹" in forecast["latest_observed_close_formatted"]
    assert "₹" in forecast["predicted_next_close_formatted"]
    assert "uncertainty_interval" in forecast
    assert forecast["uncertainty_interval"]["is_conformal"] is True
    assert forecast["uncertainty_interval"]["margin"] == 18.75
    assert forecast["uncertainty_interval"]["lower_bound"] < forecast["predicted_next_close"]
    assert forecast["uncertainty_interval"]["upper_bound"] > forecast["predicted_next_close"]
    assert "Split Conformal" in forecast["uncertainty_interval"]["methodology"]
    assert "ACADEMIC RESEARCH PURPOSES ONLY" in forecast["disclaimer"]
