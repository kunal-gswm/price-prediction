"""Next-trading-day forecasting engine for Indian equity markets.

Computes point estimates, calendar session rolls (aware of NSE/BSE holidays and weekends),
percentage movements, and leakage-free Split Conformal Prediction intervals in INR (₹).
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.models.linear_regression import StockPriceModel
from src.features.engineering import extract_latest_features_for_inference
from src.utils.indian_calendar import get_next_indian_trading_session
from src.utils.currency import format_inr
from src.utils.config import CURRENCY_SYMBOL, MARKET_CURRENCY, MARKET_TIMEZONE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_next_trading_day(
    model: StockPriceModel,
    df_history: pd.DataFrame,
    conformal_margin: Optional[float] = None,
    residual_rmse: Optional[float] = None,
    confidence_level: float = 0.95,
    calibration_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates the estimated closing price for the next Indian market trading session.

    CRITICAL SAFETY & INTEGRITY:
    - Strictly uses historical observations up to session T to forecast session T+1.
    - Resolves next trading day using the NSE/BSE Indian exchange calendar (skipping weekends and market holidays).
    - Expresses all price estimates in Indian Rupees (INR, ₹).
    - Uncertainty intervals default to distribution-free Split Conformal Prediction calibrated on validation residuals.

    Args:
        model: Trained StockPriceModel instance.
        df_history: Full historical OHLCV DataFrame (at least 60 trading sessions).
        conformal_margin: Frozen interval margin from validation conformal calibration in INR.
        residual_rmse: Fallback RMSE for uncalibrated Gaussian intervals if conformal_margin is omitted.
        confidence_level: Confidence level for prediction interval (default 0.95).
        calibration_info: Optional metadata from conformal calibration.

    Returns:
        Dictionary containing forecast details, metrics, interval info, and disclaimer.
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before running next-day inference.")

    # Extract feature row for latest observed session
    latest_features_row, latest_timestamp = extract_latest_features_for_inference(
        df_history, feature_cols=model.feature_names
    )

    # Latest observed close in INR
    latest_close = float(df_history.loc[latest_timestamp, "Close"])

    # Predict next session close
    predicted_close = float(model.predict(latest_features_row)[0])

    # Next Indian trading session date using calendar awareness
    next_session_date, cal_methodology = get_next_indian_trading_session(latest_timestamp)

    # Difference metrics
    price_change = predicted_close - latest_close
    pct_change = (price_change / latest_close) * 100.0

    # Uncertainty interval calculation
    interval_info = None

    if conformal_margin is not None and conformal_margin > 0:
        margin = float(conformal_margin)
        lower_bound = max(0.0, predicted_close - margin)
        upper_bound = predicted_close + margin
        width = 2.0 * margin
        interval_info = {
            "confidence_level": confidence_level,
            "margin": margin,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean_interval_width": width,
            "median_interval_width": width,
            "interval_text": f"{format_inr(lower_bound)} - {format_inr(upper_bound)}",
            "is_conformal": True,
            "calibration_samples": calibration_info.get("n_calibration_samples") if calibration_info else None,
            "methodology": (
                f"Distribution-free Split Conformal Prediction {int(confidence_level * 100)}% interval "
                f"(±{format_inr(margin)} calibrated on validation set residuals). "
                f"Does not assume normal distribution of residuals."
            ),
        }
    elif residual_rmse is not None and residual_rmse > 0:
        z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)
        margin = z * residual_rmse
        lower_bound = max(0.0, predicted_close - margin)
        upper_bound = predicted_close + margin
        width = 2.0 * margin
        interval_info = {
            "confidence_level": confidence_level,
            "margin": margin,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean_interval_width": width,
            "median_interval_width": width,
            "interval_text": f"{format_inr(lower_bound)} - {format_inr(upper_bound)}",
            "is_conformal": False,
            "calibration_samples": None,
            "methodology": (
                f"Uncalibrated Gaussian {int(confidence_level * 100)}% prediction interval based on "
                f"RMSE ({format_inr(residual_rmse)}) assuming normally distributed errors."
            ),
        }

    direction = "UPWARD" if price_change > 0 else ("DOWNWARD" if price_change < 0 else "FLAT")

    forecast_result = {
        "ticker": model.metadata.get("ticker", "UNKNOWN"),
        "company_name": model.metadata.get("company_name", "Indian Equity"),
        "exchange": model.metadata.get("exchange", "NSE"),
        "currency": MARKET_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOL,
        "timezone": MARKET_TIMEZONE,
        "latest_observed_date": str(latest_timestamp.date()),
        "latest_observed_close": latest_close,
        "latest_observed_close_formatted": format_inr(latest_close),
        "next_trading_session_date": str(next_session_date.date()),
        "calendar_methodology": cal_methodology,
        "predicted_next_close": round(predicted_close, 4),
        "predicted_next_close_formatted": format_inr(predicted_close),
        "predicted_price_change": round(price_change, 4),
        "predicted_pct_change": round(pct_change, 4),
        "predicted_direction": direction,
        "uncertainty_interval": interval_info,
        "input_features": latest_features_row.to_dict(orient="records")[0],
        "disclaimer": (
            "ACADEMIC RESEARCH PURPOSES ONLY. This point forecast is an output of an academic "
            "Linear Regression model fitted on historical Indian equity market dynamics (NSE/BSE). "
            "Stock markets are non-stationary, influenced by macroeconomic factors, RBI monetary policy, "
            "global market sentiment, and domestic liquidity. This prediction is NOT investment advice, "
            "trading recommendation, or a guaranteed future market quote."
        ),
    }

    logger.info(
        f"Next-day forecast for {forecast_result['ticker']} ({forecast_result['exchange']}): "
        f"Last Close {format_inr(latest_close)} ({latest_timestamp.date()}) -> "
        f"Pred Close {format_inr(predicted_close)} ({next_session_date.date()}), "
        f"Diff: {pct_change:+.2f}%"
    )

    return forecast_result
