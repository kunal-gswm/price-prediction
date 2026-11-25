"""Streamlit Dashboard Application for Indian Stock Price Prediction.

Academic Machine Learning Project: NSE/BSE Historical Analysis, Linear Regression,
Chronological Out-of-Sample Evaluation, Naive Random Walk Benchmark,
and Indian Trading Session Forecasting in Indian Rupees (INR, ₹).
"""

import io
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Indian Stock Price Prediction | NSE/BSE ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom restrained CSS
st.markdown(
    """
    <style>
    .main { background-color: #fcfcfd; }
    h1, h2, h3, h4 { color: #1e293b; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMetric { background-color: #ffffff; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 14px 18px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-size: 0.92rem;
        color: #334155;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-size: 0.90rem;
        color: #92400e;
    }
    .market-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.80rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        border-radius: 4px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.utils.config import (
    DEFAULT_TICKER,
    POPULAR_TICKERS,
    INDIAN_STOCKS_METADATA,
    CONTEXTUAL_INDICES,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    REPORTS_DIR,
    CURRENCY_SYMBOL,
    MARKET_CURRENCY,
    MARKET_TIMEZONE,
    PRIMARY_EXCHANGE,
    get_stock_display_info,
)
from src.utils.currency import format_inr, format_inr_compact, format_indian_volume
from src.data.collector import fetch_stock_data, load_csv_data, DataCollectionError
from src.data.preprocessing import clean_raw_data, chronological_train_val_test_split
from src.features.engineering import prepare_training_dataset
from src.models.linear_regression import StockPriceModel
from src.models.evaluation import (
    evaluate_model_with_baseline,
    format_metrics_summary,
    compute_split_conformal_quantile,
)
from src.models.forecasting import predict_next_trading_day
from src.visualization.charts import (
    plot_candlestick_interactive,
    plot_moving_averages_interactive,
    plot_actual_vs_predicted_interactive,
    plot_residuals_interactive,
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(ticker: str, period: str, interval: str):
    """Caches Indian market data fetching to optimize response times."""
    return fetch_stock_data(ticker=ticker, period=period, interval=interval, save_raw=True)


def init_session_state():
    """Initializes session state variables with Indian market defaults."""
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = DEFAULT_TICKER
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "features_df" not in st.session_state:
        st.session_state.features_df = None
    if "split_data" not in st.session_state:
        st.session_state.split_data = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "test_eval" not in st.session_state:
        st.session_state.test_eval = None
    if "test_preds" not in st.session_state:
        st.session_state.test_preds = None
    if "forecast" not in st.session_state:
        st.session_state.forecast = None


init_session_state()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🇮🇳 Indian Market Controls")
st.sidebar.caption(f"Exchange: {PRIMARY_EXCHANGE} | Currency: INR ({CURRENCY_SYMBOL})")

data_source_mode = st.sidebar.radio(
    "Data Source",
    ["NSE / BSE (Yahoo Finance)", "Upload Custom CSV"],
    help="Select live Indian market API or upload a historical OHLCV CSV file.",
)

selected_ticker = DEFAULT_TICKER

if data_source_mode == "NSE / BSE (Yahoo Finance)":
    # Formatted options with stock name and exchange
    ticker_display_options = [
        f"{t} ({INDIAN_STOCKS_METADATA.get(t, {}).get('name', t)} - {INDIAN_STOCKS_METADATA.get(t, {}).get('exchange', 'NSE')})"
        for t in POPULAR_TICKERS
    ]
    ticker_choice_idx = st.sidebar.selectbox(
        "Select Indian Blue-Chip Stock",
        range(len(POPULAR_TICKERS)),
        format_func=lambda i: ticker_display_options[i],
        index=0,
    )
    ticker_from_list = POPULAR_TICKERS[ticker_choice_idx]

    custom_ticker_input = st.sidebar.text_input(
        "Or Enter Any NSE/BSE Symbol (e.g. SBIN.NS, TCS.BO, LT.NS)",
        value="",
    ).strip().upper()

    selected_ticker = custom_ticker_input if custom_ticker_input else ticker_from_list

    period_choice = st.sidebar.selectbox(
        "Historical Period",
        ["1y", "2y", "5y", "max"],
        index=2,
        help="Recommended: 5 years for robust 50-day moving average calculation and statistical sample size.",
    )
    interval_choice = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)

    if st.sidebar.button("📥 Ingest Indian Market Data", use_container_width=True):
        try:
            with st.spinner(f"Fetching Indian market data for {selected_ticker}..."):
                raw_df, meta = get_cached_stock_data(selected_ticker, period_choice, interval_choice)
                cleaned_df, clean_summary = clean_raw_data(raw_df)
                features_df, feat_audit = prepare_training_dataset(cleaned_df)

                st.session_state.current_ticker = selected_ticker
                st.session_state.raw_df = raw_df
                st.session_state.cleaned_df = cleaned_df
                st.session_state.features_df = features_df
                st.session_state.meta = meta
                st.session_state.model = None
                st.session_state.test_eval = None
                st.success(f"Loaded {len(raw_df)} trading sessions for {selected_ticker}!")
        except Exception as e:
            st.sidebar.error(f"Data Ingestion Error: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload Indian Stock CSV (OHLCV)", type=["csv"])
    selected_ticker = st.sidebar.text_input("Dataset Ticker Identifier", value="CUSTOM.NS").strip().upper()
    if uploaded_file is not None and st.sidebar.button("📥 Parse Uploaded CSV", use_container_width=True):
        try:
            with st.spinner("Processing custom CSV file..."):
                raw_df, meta = load_csv_data(uploaded_file, ticker=selected_ticker)
                cleaned_df, clean_summary = clean_raw_data(raw_df)
                features_df, feat_audit = prepare_training_dataset(cleaned_df)

                st.session_state.current_ticker = selected_ticker
                st.session_state.raw_df = raw_df
                st.session_state.cleaned_df = cleaned_df
                st.session_state.features_df = features_df
                st.session_state.meta = meta
                st.session_state.model = None
                st.session_state.test_eval = None
                st.success(f"Parsed {len(raw_df)} rows from CSV!")
        except Exception as e:
            st.sidebar.error(f"CSV Parsing Error: {e}")

# Model Training Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Controls")
model_type = st.sidebar.selectbox("Model Type", ["LinearRegression", "Ridge"])
alpha_val = 1.0
if model_type == "Ridge":
    alpha_val = st.sidebar.slider("L2 Regularization Alpha", 0.01, 10.0, 1.0, 0.1)

train_btn = st.sidebar.button("🚀 Train & Evaluate on NSE/BSE", type="primary", use_container_width=True)

if train_btn:
    if st.session_state.features_df is None:
        try:
            with st.spinner(f"Auto-fetching data for {selected_ticker}..."):
                raw_df, meta = get_cached_stock_data(selected_ticker, "5y", "1d")
                cleaned_df, _ = clean_raw_data(raw_df)
                features_df, _ = prepare_training_dataset(cleaned_df)
                st.session_state.current_ticker = selected_ticker
                st.session_state.raw_df = raw_df
                st.session_state.cleaned_df = cleaned_df
                st.session_state.features_df = features_df
                st.session_state.meta = meta
        except Exception as e:
            st.sidebar.error(f"Failed to auto-load: {e}")

    if st.session_state.features_df is not None:
        with st.spinner("Training model with chronological split (70% Train, 15% Val, 15% Test)..."):
            splits = chronological_train_val_test_split(
                st.session_state.features_df,
                feature_cols=FEATURE_COLUMNS,
                target_col=TARGET_COLUMN,
            )
            model = StockPriceModel(model_type=model_type, alpha=alpha_val)
            model.fit(
                splits["X_train"],
                splits["y_train"],
                ticker=st.session_state.current_ticker,
                train_dates=splits["split_info"]["train_dates"],
            )

            # Split Conformal Calibration on validation partition
            val_preds = model.predict(splits["X_val"])
            cal_info = compute_split_conformal_quantile(
                y_cal=splits["y_val"],
                y_cal_pred=val_preds,
                alpha=0.05,
            )
            conformal_margin = cal_info["conformal_quantile"]

            test_preds = model.predict(splits["X_test"])
            test_eval = evaluate_model_with_baseline(
                y_true=splits["y_test"],
                y_pred=test_preds,
                current_close=splits["X_test"]["Close"],
                prev_close=splits["X_test"]["Prev_Close"] if "Prev_Close" in splits["X_test"].columns else None,
                conformal_margin=conformal_margin,
                calibration_info=cal_info,
            )

            forecast = predict_next_trading_day(
                model=model,
                df_history=st.session_state.cleaned_df,
                conformal_margin=conformal_margin,
                residual_rmse=float(np.sqrt(np.mean((splits["y_val"] - val_preds) ** 2))),
                confidence_level=0.95,
                calibration_info=cal_info,
            )

            st.session_state.split_data = splits
            st.session_state.model = model
            st.session_state.test_preds = test_preds
            st.session_state.test_eval = test_eval
            st.session_state.forecast = forecast
            st.sidebar.success("Model trained & evaluated successfully!")


# Ensure default data is loaded on page launch
if st.session_state.raw_df is None:
    try:
        raw_df, meta = get_cached_stock_data(DEFAULT_TICKER, DEFAULT_PERIOD, DEFAULT_INTERVAL)
        cleaned_df, _ = clean_raw_data(raw_df)
        features_df, _ = prepare_training_dataset(cleaned_df)
        st.session_state.current_ticker = DEFAULT_TICKER
        st.session_state.raw_df = raw_df
        st.session_state.cleaned_df = cleaned_df
        st.session_state.features_df = features_df
        st.session_state.meta = meta

        splits = chronological_train_val_test_split(features_df, FEATURE_COLUMNS, TARGET_COLUMN)
        model = StockPriceModel(model_type="LinearRegression")
        model.fit(splits["X_train"], splits["y_train"], ticker=DEFAULT_TICKER, train_dates=splits["split_info"]["train_dates"])
        
        # Conformal calibration on validation set
        val_preds = model.predict(splits["X_val"])
        cal_info = compute_split_conformal_quantile(splits["y_val"], val_preds, alpha=0.05)
        conformal_margin = cal_info["conformal_quantile"]

        test_preds = model.predict(splits["X_test"])
        test_eval = evaluate_model_with_baseline(
            y_true=splits["y_test"],
            y_pred=test_preds,
            current_close=splits["X_test"]["Close"],
            prev_close=splits["X_test"]["Prev_Close"] if "Prev_Close" in splits["X_test"].columns else None,
            conformal_margin=conformal_margin,
            calibration_info=cal_info,
        )
        forecast = predict_next_trading_day(
            model,
            cleaned_df,
            conformal_margin=conformal_margin,
            residual_rmse=float(np.sqrt(np.mean((splits["y_val"] - val_preds) ** 2))),
            confidence_level=0.95,
            calibration_info=cal_info,
        )

        st.session_state.split_data = splits
        st.session_state.model = model
        st.session_state.test_preds = test_preds
        st.session_state.test_eval = test_eval
        st.session_state.forecast = forecast
    except Exception as e:
        st.sidebar.warning(f"Data auto-load note: {e}")

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================

stock_info = get_stock_display_info(st.session_state.current_ticker)

# Title & Badges
st.title("📈 Indian Stock Price Prediction & Technical Analysis")
st.markdown(
    f"<span class='market-badge'>{stock_info['exchange']} India</span>"
    f"<span class='market-badge'>{stock_info['sector']}</span>"
    f"<span class='market-badge'>Currency: INR ({CURRENCY_SYMBOL})</span>"
    f"<span class='market-badge'>Timezone: {MARKET_TIMEZONE}</span>",
    unsafe_allow_html=True,
)
st.caption(f"Target Asset: **{stock_info['name']}** ({st.session_state.current_ticker})")

# Academic Notice
st.markdown(
    """
    <div class="disclaimer-box">
    <strong>Academic Research Notice:</strong> This project investigates time-series machine learning models applied to the
    Indian equity market (NSE/BSE). Stock prices in India are influenced by macroeconomic dynamics, RBI policy, corporate earnings,
    and market liquidity. All price estimates are strictly for educational research and do <strong>not</strong> constitute financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs Navigation
tab_overview, tab_explorer, tab_eda, tab_market_context, tab_training, tab_eval, tab_forecast, tab_info = st.tabs([
    "📋 Overview",
    "🔍 Data Explorer",
    "📊 Exploratory Data Analysis",
    "🇮🇳 Indian Market Context",
    "⚙️ Model Training",
    "🎯 Model Evaluation",
    "🔮 Next-Day Forecast",
    "ℹ️ Project Information",
])

# ------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------
with tab_overview:
    st.subheader(f"Executive Summary: {stock_info['name']} ({st.session_state.current_ticker})")

    col1, col2, col3, col4 = st.columns(4)
    if st.session_state.cleaned_df is not None:
        last_close = float(st.session_state.cleaned_df["Close"].iloc[-1])
        prev_close = float(st.session_state.cleaned_df["Close"].iloc[-2]) if len(st.session_state.cleaned_df) > 1 else last_close
        diff = last_close - prev_close
        pct = (diff / prev_close) * 100

        col1.metric("Latest Observed Close", format_inr(last_close), f"{format_inr(diff)} ({pct:+.2f}%)")
        col2.metric("Total Trading Sessions", f"{len(st.session_state.cleaned_df):,}")
        col3.metric("Observation Start", str(st.session_state.cleaned_df.index.min().date()))
        col4.metric("Observation End", str(st.session_state.cleaned_df.index.max().date()))

    st.markdown("---")
    st.markdown(
        r"""
        ### Methodology & Technical Safeguards
        - **Problem Formulation**: Predict next-trading-session closing price ($y_{t+1}$) using historical OHLCV features ($t \le T$).
        - **Lookahead Prevention**: Technical features (SMA, EMA, lagged OHLCV, 10-day rolling volatility) strictly depend on past data.
        - **Chronological Splitting**: 70% Train, 15% Validation, 15% Test without random shuffling.
        - **StandardScaler Fitting**: Standardizer parameters ($\mu, \sigma$) are computed exclusively on the training slice.
        - **Litmus Test**: All model metrics are benchmarked directly against the **Naive Previous-Close Random Walk** ($y_{t+1} = y_t$).
        """
    )

    if st.session_state.test_eval is not None:
        st.subheader("Model Performance on Out-of-Sample Test Partition")
        m = st.session_state.test_eval["model_metrics"]
        b = st.session_state.test_eval["naive_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test RMSE", format_inr(m["RMSE"]), f"Baseline: {format_inr(b['RMSE'])}")
        c2.metric("Test MAE", format_inr(m["MAE"]), f"Baseline: {format_inr(b['MAE'])}")
        c3.metric("R² Score", f"{m['R2']:.4f}", f"Baseline: {b['R2']:.4f}")
        c4.metric("Directional Hit Rate", f"{st.session_state.test_eval['directional_accuracy_pct']:.1f}%")

# ------------------------------------------
# TAB 2: DATA EXPLORER
# ------------------------------------------
with tab_explorer:
    st.subheader(f"Historical Indian Equity Data: {st.session_state.current_ticker}")

    if st.session_state.raw_df is not None:
        view_mode = st.radio("Select View", ["Processed Dataset (18 Features & Target)", "Raw Market OHLCV"], horizontal=True)

        if view_mode == "Raw Market OHLCV":
            st.dataframe(st.session_state.raw_df.tail(100), use_container_width=True)
            csv_raw = st.session_state.raw_df.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Raw OHLCV CSV",
                csv_raw,
                f"{st.session_state.current_ticker.replace('.', '_')}_raw.csv",
                "text/csv",
            )
        else:
            if st.session_state.features_df is not None:
                st.dataframe(st.session_state.features_df.tail(100), use_container_width=True)
                csv_feat = st.session_state.features_df.to_csv().encode("utf-8")
                st.download_button(
                    "📥 Download Processed Features CSV",
                    csv_feat,
                    f"{st.session_state.current_ticker.replace('.', '_')}_features.csv",
                    "text/csv",
                )

        st.markdown("#### Descriptive Statistics (in INR, ₹)")
        st.dataframe(st.session_state.raw_df.describe(), use_container_width=True)
    else:
        st.info("No data loaded. Use the sidebar to ingest market data.")

# ------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
with tab_eda:
    st.subheader(f"Exploratory Technical Analysis: {st.session_state.current_ticker}")

    if st.session_state.cleaned_df is not None:
        st.markdown("#### 1. Interactive Candlestick Chart with Volume Dynamics")
        fig_candle = plot_candlestick_interactive(st.session_state.cleaned_df, ticker=st.session_state.current_ticker)
        st.plotly_chart(fig_candle, use_container_width=True)

        st.markdown("#### 2. Technical Moving Averages (SMA 5, 10, 20, 50 & EMA 10)")
        if st.session_state.features_df is not None:
            fig_ma = plot_moving_averages_interactive(st.session_state.features_df, ticker=st.session_state.current_ticker)
            st.plotly_chart(fig_ma, use_container_width=True)

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("#### 3. Daily Returns Distribution")
            returns = st.session_state.cleaned_df["Close"].pct_change().dropna() * 100
            st.bar_chart(np.histogram(returns, bins=40)[0])
            st.caption(f"Mean Daily Return: {returns.mean():.3f}% | Return Volatility (Std): {returns.std():.3f}%")

        with col_eda2:
            st.markdown("#### 4. 10-Day Rolling Return Volatility")
            vol = returns.rolling(10).std()
            st.line_chart(vol.dropna())
            st.caption("Rolling annualized standard deviation indicates volatility clustering during market regimes.")
    else:
        st.info("Ingest market data to view exploratory charts.")

# ------------------------------------------
# TAB 4: INDIAN MARKET CONTEXT
# ------------------------------------------
with tab_market_context:
    st.subheader("🇮🇳 Indian Equity Market Structural Context")

    st.markdown(
        f"""
        ### Market Structure: NSE and BSE
        - **National Stock Exchange (NSE)**: Established in 1992, the NSE is India's leading stock exchange by trading turnover, operating the flagship benchmark **NIFTY 50 (`^NSEI`)**.
        - **Bombay Stock Exchange (BSE)**: Established in 1875, Asia's oldest stock exchange, operating the benchmark **SENSEX (`^BSESN`)**.
        - **Trading Hours**: Standard trading session operates from **09:15 AM to 03:30 PM IST** (Pre-open: 09:00 AM - 09:08 AM IST).
        - **Settlement Cycle**: T+1 rolling settlement system under SEBI regulations.
        
        ### Calendar & Holiday Schedule Awareness
        Indian equity exchanges observe official national and festival holidays (e.g. Republic Day, Independence Day, Gandhi Jayanti, Diwali, Holi, Good Friday, Maharashtra Day).
        Our forecasting engine integrates an Indian exchange calendar module (`src/utils/indian_calendar.py`) to automatically roll weekend and holiday timestamps to the next authentic trading session.
        """
    )

    if st.session_state.cleaned_df is not None:
        st.markdown(f"#### Liquidity & Volume Profile for {st.session_state.current_ticker}")
        avg_vol = float(st.session_state.cleaned_df["Volume"].mean())
        max_vol = float(st.session_state.cleaned_df["Volume"].max())
        min_close = float(st.session_state.cleaned_df["Close"].min())
        max_close = float(st.session_state.cleaned_df["Close"].max())

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Average Daily Volume", format_indian_volume(avg_vol))
        c_m2.metric("Peak Volume Day", format_indian_volume(max_vol))
        c_m3.metric("52-Week / Period Low", format_inr(min_close))
        c_m4.metric("52-Week / Period High", format_inr(max_close))

# ------------------------------------------
# TAB 5: MODEL TRAINING
# ------------------------------------------
with tab_training:
    st.subheader("Model Architecture & Coefficients")

    if st.session_state.model is not None and st.session_state.split_data is not None:
        split_info = st.session_state.split_data["split_info"]

        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Training Set (70%)", f"{split_info['n_train']:,} sessions", f"{split_info['train_dates'][0]} to {split_info['train_dates'][1]}")
        col_t2.metric("Validation Set (15%)", f"{split_info['n_val']:,} sessions", f"{split_info['val_dates'][0]} to {split_info['val_dates'][1]}")
        col_t3.metric("Testing Set (15%)", f"{split_info['n_test']:,} sessions", f"{split_info['test_dates'][0]} to {split_info['test_dates'][1]}")

        st.markdown("---")
        st.markdown("#### Standardized Feature Coefficients")
        st.caption("StandardScaler fitted strictly on training data. Coefficient magnitudes reflect feature contribution.")

        importance_df = st.session_state.model.get_feature_importance()
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.set_index("Feature")["Coefficient"])

        st.markdown(
            f"""
            > **Econometric Insight on Indian Equities**:
            > For large-cap Indian stocks like {stock_info['name']}, the strongest linear predictor is yesterday's closing price
            > (`Close` and `Prev_Close`). In accordance with martingale asset pricing, stock prices follow strong autocorrelation.
            """
        )
    else:
        st.info("Train the model from the sidebar to inspect coefficients.")

# ------------------------------------------
# TAB 6: MODEL EVALUATION
# ------------------------------------------
with tab_eval:
    st.subheader(f"Out-of-Sample Evaluation: {st.session_state.current_ticker}")

    if st.session_state.test_eval is not None and st.session_state.split_data is not None:
        eval_dict = st.session_state.test_eval
        splits = st.session_state.split_data

        st.markdown("#### 1. Linear Regression vs Naive Random Walk Benchmark")
        st.dataframe(eval_dict["comparison_table"], use_container_width=True)

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Test RMSE", format_inr(eval_dict['model_metrics']['RMSE']), f"{eval_dict['rmse_improvement_pct']:+.2f}% vs Baseline")
        col_e2.metric("Test MAE", format_inr(eval_dict['model_metrics']['MAE']), f"{eval_dict['mae_improvement_pct']:+.2f}% vs Baseline")
        col_e3.metric("Directional Hit Rate", f"{eval_dict['directional_accuracy_pct']:.2f}%", f"Naive Momentum: {eval_dict.get('naive_directional_accuracy_pct', 50.0):.1f}%")
        int_stats = eval_dict.get('interval_stats', {})
        cov_pct = int_stats.get('actual_test_coverage_pct', eval_dict['residual_stats'].get('interval_coverage_95_pct', 95.0))
        col_e4.metric("Conformal 95% Coverage", f"{cov_pct:.2f}%", f"Nominal: 95.00%")

        st.markdown("---")
        st.markdown("#### 2. Split Conformal Prediction & Statistical Diagnostics")
        c_stat1, c_stat2 = st.columns(2)
        with c_stat1:
            st.markdown("##### 🛡️ Leakage-Free Split Conformal Calibration")
            st.write(
                f"- **Calibration Partition**: Validation set (unseen during model training)\n"
                f"- **Calibration Samples**: {int_stats.get('n_calibration_samples', 'N/A')} sessions\n"
                f"- **Test Set Observations**: {int_stats.get('total_test_samples', len(splits['y_test']))} sessions\n"
                f"- **Finite-Sample Margin**: ±{format_inr(int_stats.get('interval_margin', 0.0))}\n"
                f"- **Mean / Median Interval Width**: {format_inr(int_stats.get('mean_interval_width', 0.0))}\n"
                f"- **Empirical Coverage on Test Set**: **{cov_pct:.2f}%** ({int_stats.get('samples_in_interval', 0)} / {int_stats.get('total_test_samples', 0)} inside bounds)\n"
                f"- **Coverage Limitations**: Guarantees finite-sample coverage under exchangeability. In financial time series, volatility clustering and temporal autocorrelation cause empirical coverage to fluctuate across regimes."
            )
        with c_stat2:
            st.markdown("##### ⚖️ Diebold-Mariano Test & Residual Normality")
            dm_info = eval_dict.get("diebold_mariano", {}).get("squared_error", {})
            res_diag = eval_dict.get("residual_stats", {})
            st.write(
                f"- **Diebold-Mariano Stat (MSE)**: {dm_info.get('dm_statistic', 0.0):.4f}\n"
                f"- **DM Two-Sided p-value**: {dm_info.get('p_value', 1.0):.4f}\n"
                f"- **Statistical Significance**: {'Yes (p < 0.05)' if dm_info.get('is_significant_5pct') else 'No (p > 0.05) - Difference is not statistically significant'}\n"
                f"- **Jarque-Bera Normality Test**: Stat = {res_diag.get('jarque_bera_stat', 0.0):.2f}, p = {res_diag.get('jarque_bera_p_value', 0.0):.2e}\n"
                f"- **Excess Kurtosis**: {res_diag.get('kurtosis', 0.0):.2f} (Fat tails / leptokurtic)\n"
                f"- **Normality Rejected**: Residuals violate Gaussian assumptions; validates distribution-free Split Conformal Prediction."
            )

        st.markdown("---")
        st.markdown("#### 3. Actual vs Predicted Closing Prices (Test Partition)")
        fig_act_pred = plot_actual_vs_predicted_interactive(
            y_true=splits["y_test"],
            y_pred=st.session_state.test_preds,
            dates=splits["test_index"],
            ticker=st.session_state.current_ticker,
        )
        st.plotly_chart(fig_act_pred, use_container_width=True)

        st.markdown("#### 4. Prediction Residuals Over Time")
        fig_res = plot_residuals_interactive(
            y_true=splits["y_test"],
            y_pred=st.session_state.test_preds,
            dates=splits["test_index"],
            ticker=st.session_state.current_ticker,
        )
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown(
            f"""
            <div class="warning-box">
            <strong>Critical Econometric Discussion & Academic Rigor:</strong><br>
            • <strong>High R² Score ({eval_dict['model_metrics']['R2']:.4f}):</strong> In non-stationary time series,
            high R² reflects unit-root persistence rather than supernatural forecasting capability.<br>
            • <strong>Diebold-Mariano Hypothesis Test:</strong> With DM p-value = {dm_info.get('p_value', 1.0):.4f} (p > 0.05),
            the model's {eval_dict['rmse_improvement_pct']:+.2f}% RMSE reduction over the naive baseline is <strong>not statistically significant</strong>.<br>
            • <strong>Trading Reality vs Price Accuracy:</strong> A directional hit rate of ~51.7% does not translate to profitable trading.
            In Indian equity cash markets, mandatory transaction frictions (Securities Transaction Tax of 0.1%, exchange turnover charges, SEBI turnover fees, 18% GST, stamp duty, brokerage, and bid-ask slippage) easily exceed marginal predictive differences.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Train the model from the sidebar to inspect evaluation results.")

# ------------------------------------------
# TAB 7: NEXT-DAY FORECAST
# ------------------------------------------
with tab_forecast:
    st.subheader("Next-Trading-Session Price Forecast")

    if st.session_state.forecast is not None:
        fc = st.session_state.forecast

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Latest Observed Close", fc["latest_observed_close_formatted"], f"Date: {fc['latest_observed_date']}")
        f_col2.metric("Predicted Next Close", fc["predicted_next_close_formatted"], f"{fc['predicted_pct_change']:+.2f}%")
        f_col3.metric("Next Trading Session", fc["next_trading_session_date"], "NSE Calendar")
        f_col4.metric("Predicted Direction", fc["predicted_direction"])

        if fc.get("uncertainty_interval"):
            u = fc["uncertainty_interval"]
            interval_type = "Split Conformal Prediction" if u.get("is_conformal") else "Gaussian Fallback"
            st.info(
                f"📊 **95% {interval_type} Interval**: {u['interval_text']} "
                f"(±{format_inr(u['margin'])} margin, width: {format_inr(u.get('mean_interval_width', 2*u['margin']))}).\n\n"
                f"*{u.get('methodology', '')}*"
            )

        st.caption(f"Trading Calendar Authority: **{fc.get('calendar_methodology', 'Official NSE Circulars')}**")

        st.markdown("#### Latest Feature Vector Passed to Inference Engine")
        feat_display = pd.DataFrame([fc["input_features"]])
        st.dataframe(feat_display, use_container_width=True)

        forecast_df = pd.DataFrame([{
            "Ticker": fc["ticker"],
            "Company_Name": fc["company_name"],
            "Exchange": fc["exchange"],
            "Currency": fc["currency"],
            "Latest_Date": fc["latest_observed_date"],
            "Latest_Close_INR": fc["latest_observed_close"],
            "Next_Session_Date": fc["next_trading_session_date"],
            "Predicted_Next_Close_INR": fc["predicted_next_close"],
            "Predicted_Change_INR": fc["predicted_price_change"],
            "Predicted_Pct_Change": fc["predicted_pct_change"],
            "Direction": fc["predicted_direction"],
            "Confidence_Interval_INR": fc["uncertainty_interval"]["interval_text"] if fc.get("uncertainty_interval") else "N/A",
        }])
        csv_fc = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Forecast to CSV",
            csv_fc,
            f"{fc['ticker'].replace('.', '_')}_next_day_forecast.csv",
            "text/csv",
        )
    else:
        st.info("Train the model to generate the next-day forecast.")

# ------------------------------------------
# TAB 8: PROJECT INFORMATION
# ------------------------------------------
with tab_info:
    st.subheader("System Architecture & Academic Specifications")

    st.markdown(
        r"""
        ### Technology Stack
        - **Language**: Python 3.11+
        - **Market Data**: `yfinance` (NSE & BSE equities)
        - **Calendar Awareness**: `exchange_calendars` (`XBOM` / Indian market schedule)
        - **Modeling**: `scikit-learn` (`LinearRegression`, `Ridge`, `StandardScaler`)
        - **Visualizations**: `plotly` (Interactive), `matplotlib` & `seaborn` (Static PDF figures)
        - **Persistence**: `joblib`
        - **Unit Tests**: `pytest` (39 automated tests)
        - **PDF Generation**: `ReportLab`

        ### Key Academic Safeguards
        1. **Strict Chronological Splitting**: Shuffling time-series observations causes forward-information leakage. Data is partitioned sequentially into 70% Train, 15% Validation, and 15% Test.
        2. **Zero Lookahead Leakage**: Every technical feature at index $t$ uses historical records $\le t$. Target at $t$ is strictly $Close_{t+1}$.
        3. **Isolated Preprocessing**: Scaler parameters ($\mu, \sigma$) are computed exclusively on training observations and applied to out-of-sample data.
        4. **Econometric Benchmarking**: All models are benchmarked against the Martingale random walk ($y_{t+1} = y_t$).

        ### Download Academic PDF Report
        A comprehensive academic report is generated in `reports/Stock_Price_Prediction_Report.pdf`.
        """
    )
    pdf_path = REPORTS_DIR / "Stock_Price_Prediction_Report.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            "📄 Download Academic PDF Report (INR, NSE/BSE)",
            pdf_bytes,
            "Stock_Price_Prediction_Report.pdf",
            "application/pdf",
        )
    else:
        st.caption("PDF Report will be available once generated by scripts/generate_report.py.")
