"""Unit tests for feature engineering and leakage prevention."""

import pytest
import numpy as np
import pandas as pd

from src.features.engineering import (
    compute_technical_features,
    prepare_training_dataset,
    extract_latest_features_for_inference,
    FeatureEngineeringError,
)
from src.utils.config import TARGET_COLUMN, FEATURE_COLUMNS


def test_technical_features_generation(synthetic_ohlcv_df):
    """Verifies that all required moving averages, lags, and volatility features are created."""
    feat_df = compute_technical_features(synthetic_ohlcv_df)

    expected = [
        "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_10",
        "Prev_Close", "Prev_Open", "Prev_High", "Prev_Low", "Prev_Volume",
        "Daily_Return", "Rolling_Volatility_10", "Volume_Change",
        TARGET_COLUMN,
    ]
    for col in expected:
        assert col in feat_df.columns, f"Expected feature {col} missing in output DataFrame."


def test_target_leakage_prevention(synthetic_ohlcv_df):
    """Rigorous test to confirm no target leakage.

    Feature at time t must depend ONLY on data at index <= t.
    Target at time t is strictly Close at t+1.
    """
    feat_df = compute_technical_features(synthetic_ohlcv_df)

    for i in range(10, len(feat_df) - 1):
        # Target at day i must match Close at day i+1
        assert np.isclose(feat_df[TARGET_COLUMN].iloc[i], synthetic_ohlcv_df["Close"].iloc[i + 1])

        # Prev_Close at day i must match Close at day i-1
        assert np.isclose(feat_df["Prev_Close"].iloc[i], synthetic_ohlcv_df["Close"].iloc[i - 1])

        # SMA_5 at day i must match mean of Close from day i-4 to day i
        expected_sma5 = synthetic_ohlcv_df["Close"].iloc[i - 4 : i + 1].mean()
        assert np.isclose(feat_df["SMA_5"].iloc[i], expected_sma5)


def test_prepare_training_dataset_nan_dropping(synthetic_ohlcv_df):
    """Verifies that rolling window warmup rows and unobserved target row are dropped."""
    clean_df, audit = prepare_training_dataset(synthetic_ohlcv_df)

    # Clean dataset should have no NaNs across feature columns and target
    assert not clean_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()
    # At least 50 rows dropped due to SMA_50 warmup and target shift
    assert audit["dropped_rows"] >= 50
    assert audit["clean_rows"] == len(synthetic_ohlcv_df) - audit["dropped_rows"]


def test_extract_latest_features_for_inference(synthetic_ohlcv_df):
    """Verifies that latest row for inference is extracted from the latest timestamp."""
    latest_features, latest_ts = extract_latest_features_for_inference(synthetic_ohlcv_df)

    assert len(latest_features) == 1
    assert latest_ts == synthetic_ohlcv_df.index[-1]
    # Check that all required features exist and are not NaN
    for feat in FEATURE_COLUMNS:
        assert feat in latest_features.columns
        assert not np.isnan(latest_features[feat].iloc[0])


def test_extract_latest_features_insufficient_history(short_ohlcv_df):
    """Verifies that extracting inference features on short dataset raises error because SMA_50 is NaN."""
    with pytest.raises(FeatureEngineeringError, match="contains NaN in features"):
        extract_latest_features_for_inference(short_ohlcv_df)
