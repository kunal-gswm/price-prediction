"""Visualization module for the Indian Stock Price Prediction system.

Generates:
1. High-resolution Matplotlib/Seaborn static figures for academic PDF reports (saved to reports/figures/).
2. Interactive Plotly figures for the Streamlit dashboard application.
All currency labels, axes, and annotations strictly use Indian Rupees (INR, ₹).
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.utils.config import FIGURES_DIR, CURRENCY_SYMBOL
from src.utils.currency import format_inr, format_indian_volume
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configure academic visual style for Matplotlib/Seaborn
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ==========================================
# 1. MATPLOTLIB STATIC FIGURES (FOR PDF)
# ==========================================

def plot_historical_price_matplotlib(
    df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots historical closing price line chart with volume subplot in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_price_history.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Price Line in INR
    ax1.plot(df.index, df["Close"], label="Closing Price", color="#1f77b4", linewidth=1.5)
    ax1.set_title(f"Historical Daily Closing Price - {ticker} (NSE/BSE)", fontweight="bold")
    ax1.set_ylabel(f"Price ({CURRENCY_SYMBOL} INR)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Volume Bars (in Lakhs / Crores proxy: Millions)
    ax2.bar(df.index, df["Volume"] / 1e5, color="#aec7e8", width=1.0, label="Volume (Lakh shares)")
    ax2.set_ylabel("Vol (Lakhs)")
    ax2.set_xlabel("Trading Date")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved historical price plot to {save_path}")
    return save_path


def plot_moving_averages_matplotlib(
    df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots Close price overlaid with SMA 5, 10, 20, 50, and EMA 10 in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_moving_averages.png"

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(df.index, df["Close"], label="Close Price", color="#333333", alpha=0.6, linewidth=1.2)
    if "SMA_5" in df.columns:
        ax.plot(df.index, df["SMA_5"], label="SMA 5", color="#1f77b4", linewidth=1.0)
    if "SMA_10" in df.columns:
        ax.plot(df.index, df["SMA_10"], label="SMA 10", color="#ff7f0e", linewidth=1.0)
    if "SMA_20" in df.columns:
        ax.plot(df.index, df["SMA_20"], label="SMA 20", color="#2ca02c", linewidth=1.2)
    if "SMA_50" in df.columns:
        ax.plot(df.index, df["SMA_50"], label="SMA 50", color="#d62728", linewidth=1.5)
    if "EMA_10" in df.columns:
        ax.plot(df.index, df["EMA_10"], label="EMA 10", color="#9467bd", linestyle="--", linewidth=1.2)

    ax.set_title(f"Technical Moving Averages Overlay - {ticker} (NSE/BSE)", fontweight="bold")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel(f"Price ({CURRENCY_SYMBOL} INR)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_returns_distribution_matplotlib(
    df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots daily returns histogram with KDE and normal curve comparison."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_returns_distribution.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if "Daily_Return" in df.columns:
        returns = df["Daily_Return"].dropna() * 100.0
    else:
        returns = df["Close"].pct_change().dropna() * 100.0

    sns.histplot(returns, kde=True, ax=ax, color="#1f77b4", stat="density", bins=50, alpha=0.6)

    mean_ret = returns.mean()
    std_ret = returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    p = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ret) / std_ret) ** 2)
    ax.plot(x, p, "r--", linewidth=1.5, label=f"Normal Fit (μ={mean_ret:.2f}%, σ={std_ret:.2f}%)")

    ax.set_title(f"Daily Returns Distribution - {ticker} (NSE)", fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_correlation_heatmap_matplotlib(
    df: pd.DataFrame,
    feature_cols: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """Plots correlation heatmap matrix for engineered features and target."""
    if save_path is None:
        save_path = FIGURES_DIR / "feature_correlation_heatmap.png"

    cols = [c for c in feature_cols if c in df.columns]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Indian Equity Feature & Target Pearson Correlation Matrix", fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_rolling_volatility_matplotlib(
    df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots 10-day rolling volatility over time for the Indian equity."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_rolling_volatility.png"

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if "Rolling_Volatility_10" in df.columns:
        vol = df["Rolling_Volatility_10"] * 100.0
    else:
        vol = df["Close"].pct_change().rolling(10).std() * 100.0

    ax.plot(df.index, vol, color="#e377c2", linewidth=1.2, label="10-Day Rolling Return Volatility")
    ax.axhline(vol.mean(), color="gray", linestyle="--", label=f"Mean Volatility ({vol.mean():.2f}%)")

    ax.set_title(f"Rolling Volatility Dynamics - {ticker} (NSE)", fontweight="bold")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_actual_vs_predicted_matplotlib(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: pd.Index,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots actual vs predicted stock prices on the untouched test partition in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_actual_vs_predicted.png"

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(dates, y_true, label=f"Actual Closing Price ({CURRENCY_SYMBOL})", color="#1f77b4", linewidth=1.8)
    ax.plot(dates, y_pred, label=f"Predicted Price (Linear Regression)", color="#ff7f0e", linestyle="--", linewidth=1.5)

    ax.set_title(f"Model Predictions vs Actual Test Prices - {ticker} (NSE)", fontweight="bold")
    ax.set_xlabel("Date (Untouched Chronological Test Partition)")
    ax.set_ylabel(f"Closing Price ({CURRENCY_SYMBOL} INR)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_predicted_vs_actual_scatter_matplotlib(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots scatter of predicted vs actual with ideal 45-degree identity line in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_scatter_prediction.png"

    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.scatter(y_true, y_pred, alpha=0.6, color="#2ca02c", edgecolors="k", s=30)
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Line (y = x)")

    ax.set_title(f"Predicted vs Actual Scatter - {ticker}", fontweight="bold")
    ax.set_xlabel(f"Actual Next-Session Close ({CURRENCY_SYMBOL} INR)")
    ax.set_ylabel(f"Predicted Next-Session Close ({CURRENCY_SYMBOL} INR)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_residuals_matplotlib(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: pd.Index,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots residual distribution and residual time-series on test set in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_residuals_analysis.png"

    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Time series of residuals
    ax1.plot(dates, residuals, color="#d62728", linewidth=1.0)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_title("Residuals Over Time (y_true - y_pred)", fontweight="bold")
    ax1.set_xlabel("Trading Date")
    ax1.set_ylabel(f"Residual Error ({CURRENCY_SYMBOL} INR)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2, color="#d62728", bins=30)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_title(f"Residual Distribution (Mean={CURRENCY_SYMBOL}{np.mean(residuals):.2f})", fontweight="bold")
    ax2.set_xlabel(f"Residual Error ({CURRENCY_SYMBOL} INR)")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_baseline_comparison_matplotlib(
    model_metrics: Dict[str, float],
    naive_metrics: Dict[str, float],
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Bar chart comparing Linear Regression vs Naive Baseline on RMSE and MAE in INR."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_baseline_comparison.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    categories = [f"RMSE ({CURRENCY_SYMBOL})", f"MAE ({CURRENCY_SYMBOL})"]
    model_vals = [model_metrics["RMSE"], model_metrics["MAE"]]
    naive_vals = [naive_metrics["RMSE"], naive_metrics["MAE"]]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, model_vals, width, label="Linear Regression", color="#1f77b4")
    ax.bar(x + width/2, naive_vals, width, label="Naive Baseline (y_{t+1}=y_t)", color="#ff7f0e")

    ax.set_ylabel(f"Error ({CURRENCY_SYMBOL} INR)")
    ax.set_title(f"Model Error vs Naive Baseline - {ticker}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate values with ₹
    for i, (m, n) in enumerate(zip(model_vals, naive_vals)):
        ax.annotate(f"{CURRENCY_SYMBOL}{m:.2f}", (x[i] - width/2, m), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{CURRENCY_SYMBOL}{n:.2f}", (x[i] + width/2, n), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_feature_importance_matplotlib(
    importance_df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    save_path: Optional[Path] = None,
) -> Path:
    """Plots feature coefficients ordered by magnitude."""
    safe_t = ticker.replace(".", "_")
    if save_path is None:
        save_path = FIGURES_DIR / f"{safe_t}_feature_coefficients.png"

    fig, ax = plt.subplots(figsize=(9, 5))

    df_sorted = importance_df.sort_values(by="Coefficient", ascending=True)
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in df_sorted["Coefficient"]]

    ax.barh(df_sorted["Feature"], df_sorted["Coefficient"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_title(f"Standardized Feature Coefficients - {ticker} (NSE)", fontweight="bold")
    ax.set_xlabel("Standardized Coefficient Magnitude")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ==========================================
# 2. PLOTLY INTERACTIVE CHARTS (FOR STREAMLIT)
# ==========================================

def plot_candlestick_interactive(df: pd.DataFrame, ticker: str = "RELIANCE.NS") -> go.Figure:
    """Creates interactive Plotly OHLC Candlestick chart with volume subplot in INR."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Candlestick Price Chart (INR, ₹)", "Trading Volume (Shares)")
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1
    )

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.8,
        ),
        row=2, col=1
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_moving_averages_interactive(df: pd.DataFrame, ticker: str = "RELIANCE.NS") -> go.Figure:
    """Creates interactive Plotly line chart with Moving Averages in INR."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="#333333", width=1.5)))

    ma_configs = [
        ("SMA_5", "#1f77b4", 1.0, "solid"),
        ("SMA_10", "#ff7f0e", 1.0, "solid"),
        ("SMA_20", "#2ca02c", 1.2, "solid"),
        ("SMA_50", "#d62728", 1.5, "solid"),
        ("EMA_10", "#9467bd", 1.2, "dash"),
    ]

    for col, color, width, dash in ma_configs:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=width, dash=dash)
                )
            )

    fig.update_layout(
        title=f"Technical Moving Averages - {ticker} (₹ INR)",
        xaxis_title="Date",
        yaxis_title=f"Price ({CURRENCY_SYMBOL} INR)",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_actual_vs_predicted_interactive(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: pd.Index,
    ticker: str = "RELIANCE.NS",
) -> go.Figure:
    """Creates interactive Plotly line chart comparing actual test prices vs predictions in INR."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_true,
            mode="lines",
            name=f"Actual Close ({CURRENCY_SYMBOL})",
            line=dict(color="#1f77b4", width=2.0)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_pred,
            mode="lines",
            name="Predicted Close (Linear Regression)",
            line=dict(color="#ff7f0e", width=2.0, dash="dot")
        )
    )

    fig.update_layout(
        title=f"Actual vs Predicted Closing Prices - {ticker} (₹ INR)",
        xaxis_title="Date",
        yaxis_title=f"Price ({CURRENCY_SYMBOL} INR)",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_residuals_interactive(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: pd.Index,
    ticker: str = "RELIANCE.NS",
) -> go.Figure:
    """Creates interactive Plotly residual chart in INR."""
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=residuals,
            mode="lines+markers",
            name="Residual Error (y_true - y_pred)",
            line=dict(color="#d62728", width=1.2),
            marker=dict(size=4)
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=f"Prediction Residuals Over Time - {ticker} ({CURRENCY_SYMBOL} INR)",
        xaxis_title="Date",
        yaxis_title=f"Residual Error ({CURRENCY_SYMBOL} INR)",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
