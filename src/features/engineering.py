"""Feature engineering module for Indian stock price forecasting.

Calculates technical indicators, moving averages, lagged values, and rolling metrics.
Enforces strict chronological feature generation with zero future-lookahead leakage.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineeringError(Exception):
    """Exception raised during feature computation."""
    pass


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical and lag features strictly using historical information (t <= current timestamp).

    Features generated:
    - Moving Averages: SMA_5, SMA_10, SMA_20, SMA_50
    - Exponential Moving Average: EMA_10
    - Returns: Daily_Return
    - Lagged OHLCV: Prev_Close, Prev_Open, Prev_High, Prev_Low, Prev_Volume
    - Volatility: Rolling_Volatility_10 (10-day rolling standard deviation of daily returns)
    - Volume Dynamics: Volume_Change (percentage change from previous trading session's volume)

    Target:
    - Target_Next_Close: Next trading session closing price (Close shifted by -1).

    Args:
        df: Input DataFrame with DatetimeIndex and standard OHLCV columns.

    Returns:
        pd.DataFrame: New DataFrame containing original OHLCV, engineered features, and target.
    """
    if df is None or df.empty:
        raise FeatureEngineeringError("Input DataFrame is empty.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise FeatureEngineeringError(f"Missing required columns for feature generation: {missing}")

    feat_df = df.copy()

    # Ensure chronological order
    if not feat_df.index.is_monotonic_increasing:
        feat_df = feat_df.sort_index()

    # 1. Simple Moving Averages of Close
    feat_df["SMA_5"] = feat_df["Close"].rolling(window=5, min_periods=5).mean()
    feat_df["SMA_10"] = feat_df["Close"].rolling(window=10, min_periods=10).mean()
    feat_df["SMA_20"] = feat_df["Close"].rolling(window=20, min_periods=20).mean()
    feat_df["SMA_50"] = feat_df["Close"].rolling(window=50, min_periods=50).mean()

    # 2. Exponential Moving Average of Close
    feat_df["EMA_10"] = feat_df["Close"].ewm(span=10, adjust=False).mean()

    # 3. Previous day OHLCV (t-1 lag)
    feat_df["Prev_Close"] = feat_df["Close"].shift(1)
    feat_df["Prev_Open"] = feat_df["Open"].shift(1)
    feat_df["Prev_High"] = feat_df["High"].shift(1)
    feat_df["Prev_Low"] = feat_df["Low"].shift(1)
    feat_df["Prev_Volume"] = feat_df["Volume"].shift(1)

    # 4. Daily Return of Close
    feat_df["Daily_Return"] = feat_df["Close"].pct_change()

    # 5. Rolling Volatility (10-day rolling std of returns)
    feat_df["Rolling_Volatility_10"] = feat_df["Daily_Return"].rolling(window=10, min_periods=10).std()

    # 6. Volume Change
    feat_df["Volume_Change"] = feat_df["Volume"].pct_change()

    # Replace infinite values resulting from zero volume changes
    feat_df["Daily_Return"] = feat_df["Daily_Return"].replace([np.inf, -np.inf], np.nan)
    feat_df["Volume_Change"] = feat_df["Volume_Change"].replace([np.inf, -np.inf], np.nan)

    # 7. Next-Trading-Day Close Target (t+1 shifted by -1)
    # Strictly for training and evaluation
    feat_df[TARGET_COLUMN] = feat_df["Close"].shift(-1)

    return feat_df


def prepare_training_dataset(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates features and filters rows with NaN values resulting from rolling windows and target shift.

    Rows removed:
    - Initial 49-50 rows: NaN due to SMA_50 and rolling window warmup.
    - Final row: NaN for target because future session t+1 has not yet occurred.

    Args:
        df: Raw cleaned OHLCV DataFrame.
        feature_cols: List of features to keep (defaults to config.FEATURE_COLUMNS).
        target_col: Name of target column (defaults to config.TARGET_COLUMN).

    Returns:
        Tuple of (clean DataFrame ready for splitting, removal audit dict).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    feat_df = compute_technical_features(df)
    initial_count = len(feat_df)

    eval_cols = list(feature_cols) + [target_col]

    nan_mask = feat_df[eval_cols].isna().any(axis=1)
    nan_count = int(nan_mask.sum())

    clean_df = feat_df[~nan_mask].copy()
    final_count = len(clean_df)

    audit = {
        "initial_rows": initial_count,
        "clean_rows": final_count,
        "dropped_rows": nan_count,
        "warmup_rows_dropped": initial_count - final_count - 1 if initial_count > final_count else 0,
        "last_target_row_dropped": 1 if initial_count > 0 else 0,
        "start_date": str(clean_df.index.min().date()) if final_count > 0 else None,
        "end_date": str(clean_df.index.max().date()) if final_count > 0 else None,
        "features": feature_cols,
        "target": target_col,
    }

    logger.info(
        f"Feature engineering dataset prepared: {initial_count} -> {final_count} rows. "
        f"Dropped {nan_count} rows ({audit['warmup_rows_dropped']} rolling window warmup, "
        f"{audit['last_target_row_dropped']} unobserved target at t+1)."
    )

    return clean_df, audit


def extract_latest_features_for_inference(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Extracts the feature vector corresponding to the latest observed trading session (T)

    to predict the closing price of the next Indian trading session (T+1).

    Args:
        df: Historical OHLCV DataFrame.
        feature_cols: Feature columns expected by the trained model.

    Returns:
        Tuple of (single-row DataFrame with features at session T, timestamp of session T).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    feat_df = compute_technical_features(df)
    if feat_df.empty:
        raise FeatureEngineeringError("Cannot extract inference features from empty DataFrame.")

    latest_row = feat_df.iloc[[-1]][feature_cols]
    latest_timestamp = feat_df.index[-1]

    if latest_row.isna().any().any():
        nan_features = latest_row.columns[latest_row.isna().any()].tolist()
        raise FeatureEngineeringError(
            f"The latest data row at {latest_timestamp.date()} contains NaN in features: {nan_features}. "
            "Ensure at least 60 historical sessions are available for rolling technical indicators."
        )

    return latest_row, latest_timestamp
