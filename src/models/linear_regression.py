"""Linear Regression model implementation for Indian Stock Price Prediction.

Provides model training, regularized Ridge alternative, artifact persistence,
and feature coefficient inspection.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.config import (
    MODELS_DIR,
    RANDOM_SEED,
    DEFAULT_TICKER,
    MARKET_CURRENCY,
    CURRENCY_SYMBOL,
    get_stock_display_info,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockPriceModel:
    """Encapsulates a scikit-learn regressor with a fitted StandardScaler, feature definitions, and metadata."""

    def __init__(
        self,
        model_type: str = "LinearRegression",
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ):
        """Initializes the Indian stock price predictor.

        Args:
            model_type: 'LinearRegression' (primary) or 'Ridge' (regularized secondary).
            alpha: L2 regularization strength for Ridge.
            fit_intercept: Whether to calculate the intercept for this model.
        """
        self.model_type = model_type
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        if model_type == "Ridge":
            self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=RANDOM_SEED)
        elif model_type == "LinearRegression":
            self.model = LinearRegression(fit_intercept=fit_intercept)
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'LinearRegression' or 'Ridge'.")

        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        ticker: str = DEFAULT_TICKER,
        train_dates: Optional[tuple] = None,
    ) -> "StockPriceModel":
        """Fits the scaler and linear regression model strictly on training observations.

        Args:
            X_train: Predictor training DataFrame.
            y_train: Target training Series.
            ticker: Indian stock ticker identifier (e.g. RELIANCE.NS).
            train_dates: Tuple of (start_date_str, end_date_str).

        Returns:
            self
        """
        if X_train.empty or len(y_train) == 0:
            raise ValueError("Training data cannot be empty.")

        self.feature_names = list(X_train.columns)
        self.target_name = str(y_train.name) if y_train.name else "Target_Next_Close"

        # Fit scaler ONLY on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_arr = np.asarray(y_train).ravel()

        # Fit model
        self.model.fit(X_train_scaled, y_train_arr)
        self.is_fitted = True

        coef_dict = dict(zip(self.feature_names, self.model.coef_.tolist()))
        intercept = float(self.model.intercept_)

        stock_info = get_stock_display_info(ticker)
        self.metadata = {
            "ticker": ticker,
            "company_name": stock_info["name"],
            "exchange": stock_info["exchange"],
            "currency": MARKET_CURRENCY,
            "currency_symbol": CURRENCY_SYMBOL,
            "model_type": self.model_type,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": len(X_train),
            "train_dates": train_dates,
            "features": self.feature_names,
            "target": self.target_name,
            "intercept": intercept,
            "coefficients": coef_dict,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        }

        logger.info(
            f"Fitted {self.model_type} model on {len(X_train)} samples for Indian stock {ticker}. "
            f"Intercept: {intercept:.4f} (INR)"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms predictors using the fitted scaler and returns predictions in INR.

        Args:
            X: DataFrame containing the expected feature columns.

        Returns:
            np.ndarray: 1D array of predicted closing prices in INR.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() before predict().")

        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            raise ValueError(
                f"Feature mismatch: input DataFrame is missing required features {missing_features}"
            )

        X_ordered = X[self.feature_names]
        X_scaled = self.scaler.transform(X_ordered)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns normalized feature coefficients for linear interpretability."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance.")

        df_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.model.coef_,
            "Abs_Coefficient": np.abs(self.model.coef_),
        }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

        return df_importance

    def save(self, filepath: Optional[Path] = None, filename: Optional[str] = None) -> Path:
        """Serializes the model, scaler, and metadata to disk using joblib.

        Args:
            filepath: Direct path to save file.
            filename: Alternative filename under MODELS_DIR.

        Returns:
            Path: Path to saved artifact.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        if filepath is None:
            if filename is None:
                ticker_clean = self.metadata.get("ticker", "MODEL").replace(".", "_")
                filename = f"{ticker_clean}_{self.model_type.lower()}.joblib"
            filepath = MODELS_DIR / filename

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "model_type": self.model_type,
            "alpha": self.alpha,
            "metadata": self.metadata,
        }

        joblib.dump(payload, filepath)
        logger.info(f"Model saved successfully to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "StockPriceModel":
        """Loads a saved model artifact from disk.

        Args:
            filepath: Path to joblib model file.

        Returns:
            StockPriceModel: Restored model instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")

        payload = joblib.load(filepath)
        instance = cls(
            model_type=payload.get("model_type", "LinearRegression"),
            alpha=payload.get("alpha", 1.0),
        )
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.target_name = payload["target_name"]
        instance.metadata = payload.get("metadata", {})
        instance.is_fitted = True

        logger.info(f"Loaded {instance.model_type} model from {filepath}")
        return instance
