"""Training and evaluation pipeline script for Indian Stock Price Prediction.

Runs end-to-end: data collection from NSE/BSE, cleaning, feature engineering,
chronological splitting, model training, evaluation against naive baseline in INR,
figure generation, and next-day forecasting with Indian calendar awareness.
"""

import argparse
import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DEFAULT_TICKER,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    REPORTS_DIR,
    CURRENCY_SYMBOL,
    MARKET_CURRENCY,
    get_stock_display_info,
)
from src.utils.logger import get_logger
from src.data.collector import fetch_stock_data
from src.data.preprocessing import clean_raw_data, chronological_train_val_test_split, save_processed_data
from src.features.engineering import prepare_training_dataset
from src.models.linear_regression import StockPriceModel
from src.models.evaluation import (
    evaluate_model_with_baseline,
    format_metrics_summary,
    compute_split_conformal_quantile,
)
from src.models.forecasting import predict_next_trading_day
from src.visualization.charts import (
    plot_historical_price_matplotlib,
    plot_moving_averages_matplotlib,
    plot_returns_distribution_matplotlib,
    plot_correlation_heatmap_matplotlib,
    plot_rolling_volatility_matplotlib,
    plot_actual_vs_predicted_matplotlib,
    plot_predicted_vs_actual_scatter_matplotlib,
    plot_residuals_matplotlib,
    plot_baseline_comparison_matplotlib,
    plot_feature_importance_matplotlib,
)

logger = get_logger("train_pipeline")


def run_pipeline(
    ticker: str = DEFAULT_TICKER,
    period: str = DEFAULT_PERIOD,
    start: str = None,
    end: str = None,
    interval: str = DEFAULT_INTERVAL,
    model_type: str = "LinearRegression",
    alpha: float = 1.0,
    generate_figures: bool = True,
) -> dict:
    """Executes the full machine learning training and evaluation pipeline for an Indian equity."""
    stock_info = get_stock_display_info(ticker)
    logger.info(
        f"=== Starting Indian Stock Pipeline for {ticker} ({stock_info['name']} on {stock_info['exchange']}) "
        f"[Model: {model_type}, Period: {period}] ==="
    )

    # 1. Data Collection
    raw_df, col_meta = fetch_stock_data(
        ticker=ticker, period=period, start=start, end=end, interval=interval, save_raw=True
    )

    # 2. Preprocessing & Cleaning (IST dates, deduplication, valid prices)
    cleaned_df, clean_summary = clean_raw_data(raw_df)

    # 3. Feature Engineering
    features_df, feat_audit = prepare_training_dataset(
        cleaned_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN
    )

    # Save processed dataset
    save_processed_data(features_df, ticker=ticker, metadata={"clean_summary": clean_summary, "feat_audit": feat_audit})

    # 4. Chronological Splitting (70% train, 15% val, 15% test)
    split_data = chronological_train_val_test_split(
        features_df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN
    )

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]

    # 5. Model Fitting
    model = StockPriceModel(model_type=model_type, alpha=alpha)
    model.fit(
        X_train=X_train,
        y_train=y_train,
        ticker=ticker,
        train_dates=split_data["split_info"]["train_dates"],
    )

    # 6. Model Evaluation on Validation Set & Split Conformal Calibration
    val_pred = model.predict(X_val)
    cal_info = compute_split_conformal_quantile(
        y_cal=y_val,
        y_cal_pred=val_pred,
        alpha=0.05,
    )
    conformal_margin = cal_info["conformal_quantile"]
    logger.info(
        f"Split Conformal Calibration on Validation Partition (N={cal_info['n_calibration_samples']}): "
        f"k={cal_info['finite_sample_k']}, Margin={conformal_margin:.4f} INR, Width={cal_info['interval_width']:.4f} INR"
    )

    val_eval = evaluate_model_with_baseline(
        y_true=y_val,
        y_pred=val_pred,
        current_close=X_val["Close"],
        prev_close=X_val["Prev_Close"] if "Prev_Close" in X_val.columns else None,
        conformal_margin=conformal_margin,
        calibration_info=cal_info,
    )

    # 7. Model Evaluation on Untouched Test Set (Using Frozen Validation Conformal Margin)
    test_pred = model.predict(X_test)
    test_eval = evaluate_model_with_baseline(
        y_true=y_test,
        y_pred=test_pred,
        current_close=X_test["Close"],
        prev_close=X_test["Prev_Close"] if "Prev_Close" in X_test.columns else None,
        conformal_margin=conformal_margin,
        calibration_info=cal_info,
    )

    logger.info("\n" + format_metrics_summary(test_eval))

    # 8. Feature Importance
    importance_df = model.get_feature_importance()

    # 9. Save Model Artifact
    model_path = model.save()

    # 10. Next-Trading-Day Forecast with Indian Calendar Awareness & Split Conformal Intervals
    forecast = predict_next_trading_day(
        model=model,
        df_history=cleaned_df,
        conformal_margin=conformal_margin,
        residual_rmse=val_eval["model_metrics"]["RMSE"],
        confidence_level=0.95,
        calibration_info=cal_info,
    )

    # 11. Generate Reproducible Academic Figures in INR
    figure_paths = {}
    if generate_figures:
        logger.info("Generating static academic figures for Indian stock report...")
        figure_paths["price_history"] = str(plot_historical_price_matplotlib(cleaned_df, ticker=ticker))
        figure_paths["moving_averages"] = str(plot_moving_averages_matplotlib(features_df, ticker=ticker))
        figure_paths["returns_distribution"] = str(plot_returns_distribution_matplotlib(cleaned_df, ticker=ticker))
        figure_paths["correlation_heatmap"] = str(plot_correlation_heatmap_matplotlib(features_df, FEATURE_COLUMNS + [TARGET_COLUMN]))
        figure_paths["rolling_volatility"] = str(plot_rolling_volatility_matplotlib(cleaned_df, ticker=ticker))
        figure_paths["actual_vs_predicted"] = str(plot_actual_vs_predicted_matplotlib(y_test, test_pred, split_data["test_index"], ticker=ticker))
        figure_paths["scatter"] = str(plot_predicted_vs_actual_scatter_matplotlib(y_test, test_pred, ticker=ticker))
        figure_paths["residuals"] = str(plot_residuals_matplotlib(y_test, test_pred, split_data["test_index"], ticker=ticker))
        figure_paths["baseline_comparison"] = str(plot_baseline_comparison_matplotlib(test_eval["model_metrics"], test_eval["naive_metrics"], ticker=ticker))
        figure_paths["feature_importance"] = str(plot_feature_importance_matplotlib(importance_df, ticker=ticker))
        logger.info(f"Generated {len(figure_paths)} static figures in reports/figures/")

    # Save run summary metrics JSON for report generator and audits
    run_results = {
        "ticker": ticker,
        "company_name": stock_info["name"],
        "exchange": stock_info["exchange"],
        "sector": stock_info["sector"],
        "currency": MARKET_CURRENCY,
        "currency_symbol": CURRENCY_SYMBOL,
        "model_type": model_type,
        "data_metadata": col_meta,
        "cleaning_summary": clean_summary,
        "feature_audit": feat_audit,
        "split_info": split_data["split_info"],
        "calibration_info": cal_info,
        "val_metrics": val_eval["model_metrics"],
        "test_metrics": test_eval["model_metrics"],
        "test_naive_metrics": test_eval["naive_metrics"],
        "test_comparison": {
            "rmse_improvement_pct": test_eval["rmse_improvement_pct"],
            "mae_improvement_pct": test_eval["mae_improvement_pct"],
            "directional_accuracy_pct": test_eval["directional_accuracy_pct"],
            "naive_directional_accuracy_pct": test_eval["naive_directional_accuracy_pct"],
            "no_change_directional_accuracy_pct": test_eval["no_change_directional_accuracy_pct"],
            "random_walk_expected_pct": test_eval["random_walk_expected_pct"],
            "valid_directional_sessions": test_eval["valid_directional_sessions"],
            "zero_movement_sessions": test_eval["zero_movement_sessions"],
            "total_test_sessions": test_eval["total_test_sessions"],
            "residual_stats": test_eval["residual_stats"],
            "interval_stats": test_eval["interval_stats"],
            "diebold_mariano": test_eval["diebold_mariano"],
        },
        "forecast": forecast,
        "model_path": str(model_path),
        "figure_paths": figure_paths,
    }

    safe_t = ticker.replace(".", "_")
    results_path = REPORTS_DIR / f"{safe_t}_latest_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(run_results, f, indent=2)

    logger.info(f"Pipeline finished successfully. Results saved to {results_path}")
    return run_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Stock Price Prediction ML Pipeline (NSE/BSE)")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Indian ticker symbol (default: RELIANCE.NS)")
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD, help="Historical period (default: 5y)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="LinearRegression", choices=["LinearRegression", "Ridge"], help="Model type")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization alpha")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")

    args = parser.parse_args()

    results = run_pipeline(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        model_type=args.model,
        alpha=args.alpha,
        generate_figures=not args.no_figures,
    )
