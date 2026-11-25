"""Unit tests for machine learning model implementations."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.linear_regression import StockPriceModel
from src.features.engineering import prepare_training_dataset
from src.data.preprocessing import chronological_train_val_test_split
from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN


def test_linear_regression_fit_predict(synthetic_ohlcv_df):
    """Tests fitting and predicting with LinearRegression."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    model = StockPriceModel(model_type="LinearRegression")
    model.fit(splits["X_train"], splits["y_train"], ticker="TEST_TICKER")

    assert model.is_fitted
    assert len(model.feature_names) == len(FEATURE_COLUMNS)

    preds = model.predict(splits["X_test"])
    assert len(preds) == len(splits["X_test"])
    assert isinstance(preds, np.ndarray)


def test_ridge_regression_fit_predict(synthetic_ohlcv_df):
    """Tests regularized Ridge regression alternative."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    model = StockPriceModel(model_type="Ridge", alpha=2.0)
    model.fit(splits["X_train"], splits["y_train"], ticker="TEST_RIDGE")

    assert model.is_fitted
    preds = model.predict(splits["X_test"])
    assert len(preds) == len(splits["X_test"])


def test_model_feature_mismatch(synthetic_ohlcv_df):
    """Verifies that predicting on DataFrame missing expected features raises ValueError."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    model = StockPriceModel()
    model.fit(splits["X_train"], splits["y_train"])

    bad_X = splits["X_test"].drop(columns=["SMA_5"])
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(bad_X)


def test_model_save_and_load(synthetic_ohlcv_df, tmp_path):
    """Tests serialization and deserialization roundtrip with joblib."""
    clean_df, _ = prepare_training_dataset(synthetic_ohlcv_df)
    splits = chronological_train_val_test_split(clean_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)

    model = StockPriceModel()
    model.fit(splits["X_train"], splits["y_train"], ticker="SAVE_TEST")
    initial_preds = model.predict(splits["X_test"])

    save_path = tmp_path / "test_model.joblib"
    model.save(filepath=save_path)
    assert save_path.exists()

    loaded_model = StockPriceModel.load(save_path)
    assert loaded_model.is_fitted
    loaded_preds = loaded_model.predict(splits["X_test"])

    np.testing.assert_allclose(initial_preds, loaded_preds)
