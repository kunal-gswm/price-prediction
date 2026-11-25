"""Unit tests for preprocessing and data splitting module."""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import (
    clean_raw_data,
    chronological_train_val_test_split,
    scale_features,
    PreprocessingError,
)
from src.features.engineering import prepare_training_dataset
from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN


def test_clean_raw_data_deduplication(synthetic_ohlcv_df):
    """Verifies that duplicate dates are deduplicated keeping the latest record."""
    df_with_dups = pd.concat([synthetic_ohlcv_df, synthetic_ohlcv_df.iloc[[0]]])
    cleaned, summary = clean_raw_data(df_with_dups)
    assert summary["duplicates_removed"] == 1
    assert len(cleaned) == len(synthetic_ohlcv_df)


def test_clean_raw_data_invalid_prices(synthetic_ohlcv_df):
    """Verifies that non-positive prices are transparently filtered."""
    df_bad = synthetic_ohlcv_df.copy()
    df_bad.iloc[5, df_bad.columns.get_loc("Close")] = -10.0
    cleaned, summary = clean_raw_data(df_bad)
    assert summary["invalid_price_rows_removed"] == 1
    assert len(cleaned) == len(synthetic_ohlcv_df) - 1


def test_chronological_splitting_order(synthetic_ohlcv_df):
    """Verifies that train/val/test splits maintain strict chronological ordering with no shuffle."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    train_idx = splits["train_index"]
    val_idx = splits["val_index"]
    test_idx = splits["test_index"]

    # Maximum date of train must be strictly earlier than minimum date of val
    assert train_idx.max() < val_idx.min()
    # Maximum date of val must be strictly earlier than minimum date of test
    assert val_idx.max() < test_idx.min()

    # Sum of counts equals total
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(clean_df)


def test_chronological_splitting_insufficient_samples(short_ohlcv_df):
    """Verifies that attempting to split insufficient samples raises error."""
    with pytest.raises(PreprocessingError, match="Insufficient samples"):
        chronological_train_val_test_split(short_ohlcv_df, feature_cols=["Close"], target_col="Close")


def test_scale_features_no_leakage():
    """Verifies that scaler learns parameters strictly from training data."""
    X_train = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})
    X_val = pd.DataFrame({"feat": [40.0, 50.0]})
    X_test = pd.DataFrame({"feat": [60.0, 70.0]})

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # Scaler mean must match training mean (20.0) exactly, unaffected by val/test
    assert np.isclose(scaler.mean_[0], 20.0)
    assert np.isclose(X_train_scaled.mean(), 0.0)
