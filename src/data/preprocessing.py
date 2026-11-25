"""Data preprocessing module for Indian equity time-series modeling.

Handles chronological data cleaning, duplicate date removal, missing value resolution,
chronological splitting, and leakage-free feature standardization.
"""

from typing import Tuple, Dict, Any, List
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    PROCESSED_DATA_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TARGET_COLUMN,
    MARKET_TIMEZONE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Exception raised during data preprocessing."""
    pass


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs raw market data validation and initial cleanup.

    Steps:
    1. Sorts index chronologically.
    2. Converts timezone to Indian Standard Time (Asia/Kolkata) then naive dates.
    3. Drops duplicate timestamps keeping the latest record.
    4. Verifies numeric conversion of OHLCV columns.
    5. Handles missing values transparently (forward-fill up to 2 days for market halts, then drops remaining).
    6. Checks and filters zero/negative prices.

    Args:
        df: Input raw OHLCV DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, cleanup summary dict).
    """
    if df is None or df.empty:
        raise PreprocessingError("Input DataFrame for cleaning is empty.")

    initial_rows = len(df)
    cleaned = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        try:
            cleaned.index = pd.to_datetime(cleaned.index)
        except Exception as e:
            raise PreprocessingError(f"Failed to convert DataFrame index to DatetimeIndex: {e}")

    # Remove or normalize timezone to IST
    if hasattr(cleaned.index, "tz") and cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_convert(MARKET_TIMEZONE).tz_localize(None)

    # Chronological sort
    cleaned = cleaned.sort_index()

    # Deduplicate
    duplicates_count = int(cleaned.index.duplicated().sum())
    if duplicates_count > 0:
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    # Numeric check
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in cleaned.columns]
    for col in ohlcv_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Missing values count before handling
    missing_before = cleaned[ohlcv_cols].isna().sum().to_dict()

    # Forward-fill minor gaps (market holidays/short gaps up to 2 trading days)
    cleaned[ohlcv_cols] = cleaned[ohlcv_cols].ffill(limit=2)

    # Drop any remaining unfillable NaN rows
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])

    # Ensure non-zero, positive prices
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in cleaned.columns]
    valid_prices_mask = (cleaned[price_cols] > 0).all(axis=1)
    invalid_price_count = int((~valid_prices_mask).sum())
    cleaned = cleaned[valid_prices_mask]

    final_rows = len(cleaned)

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "duplicates_removed": duplicates_count,
        "missing_values_imputed": missing_before,
        "invalid_price_rows_removed": invalid_price_count,
        "date_range_start": str(cleaned.index.min().date()) if final_rows > 0 else None,
        "date_range_end": str(cleaned.index.max().date()) if final_rows > 0 else None,
    }

    logger.info(f"Cleaned raw Indian stock data: {initial_rows} -> {final_rows} rows (removed {initial_rows - final_rows}).")
    return cleaned, summary


def chronological_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COLUMN,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Dict[str, Any]:
    """Splits Indian equity data strictly chronologically into train, validation, and test subsets.

    STRICT TIME-SERIES INTEGRITY:
    No random shuffling is applied. Future observations are strictly isolated to
    validation and test partitions to prevent forward-information leakage.

    Args:
        df: Feature-engineered DataFrame with feature_cols and target_col.
        feature_cols: List of predictor feature column names.
        target_col: Prediction target column name.
        train_ratio: Fraction for training (0.70).
        val_ratio: Fraction for validation (0.15).
        test_ratio: Fraction for testing (0.15).

    Returns:
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test,
        date indices, and split metadata.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise PreprocessingError(f"Split ratios must sum to 1.0, got {total_ratio:.3f}")

    n_samples = len(df)
    if n_samples < 50:
        raise PreprocessingError(
            f"Insufficient samples for train/val/test splitting: {n_samples} (minimum required is 50)."
        )

    # Missing check in selected features and target
    subset_cols = feature_cols + [target_col]
    missing_in_data = df[subset_cols].isna().sum()
    if missing_in_data.any():
        bad_cols = missing_in_data[missing_in_data > 0].to_dict()
        raise PreprocessingError(f"NaN values found in features/target prior to splitting: {bad_cols}")

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if n_train < 10 or n_val < 5 or n_test < 5:
        raise PreprocessingError(
            f"Split resulting in too few samples: train={n_train}, val={n_val}, test={n_test}"
        )

    # Chronological partition slices
    train_df = df.iloc[0:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    split_info = {
        "total_samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_dates": (str(train_df.index.min().date()), str(train_df.index.max().date())),
        "val_dates": (str(val_df.index.min().date()), str(val_df.index.max().date())),
        "test_dates": (str(test_df.index.min().date()), str(test_df.index.max().date())),
        "feature_names": feature_cols,
        "target_name": target_col,
    }

    logger.info(
        f"Chronological split complete: Train={n_train} ({split_info['train_dates']}), "
        f"Val={n_val} ({split_info['val_dates']}), Test={n_test} ({split_info['test_dates']})."
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_index": train_df.index,
        "val_index": val_df.index,
        "test_index": test_df.index,
        "split_info": split_info,
    }


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Applies feature scaling while strictly preventing data leakage.

    The scaler is fitted ONLY on training observations (`X_train`), and subsequently
    transforms validation (`X_val`) and testing (`X_test`) sets.

    Args:
        X_train: Training predictor DataFrame.
        X_val: Validation predictor DataFrame.
        X_test: Test predictor DataFrame.
        scaler: Optional custom scaler instance (defaults to StandardScaler).

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler).
    """
    if scaler is None:
        scaler = StandardScaler()

    # Fit strictly on train
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform val and test using parameters learned from train
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(df: pd.DataFrame, ticker: str, metadata: Dict[str, Any] = None) -> Path:
    """Saves the engineered and cleaned Indian stock dataset to data/processed directory."""
    safe_ticker = str(ticker).replace(".", "_").replace("^", "")
    processed_path = PROCESSED_DATA_DIR / f"{safe_ticker}_processed.csv"
    meta_path = PROCESSED_DATA_DIR / f"{safe_ticker}_processed_metadata.json"

    df.to_csv(processed_path)
    if metadata:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"Saved processed dataset ({len(df)} rows) to {processed_path}")
    return processed_path
