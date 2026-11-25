# Indian Stock Price Prediction & Technical Analysis (NSE / BSE)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Market: India](https://img.shields.io/badge/Market-NSE%20%7C%20BSE%20India-orange.svg)](https://www.nseindia.com/)
[![Currency: INR](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-green.svg)](https://www.rbi.org.in/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-39%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Academic PDF](https://img.shields.io/badge/Report-Academic%20PDF-purple.svg)](reports/Stock_Price_Prediction_Report.pdf)

An end-to-end machine learning system and interactive econometric dashboard engineered to forecast next-trading-session closing prices ($y_{t+1}$) for Indian equities traded on the **National Stock Exchange of India (NSE)** and the **Bombay Stock Exchange (BSE)**. Denominated entirely in **Indian Rupees (INR, ₹)**, the project incorporates zero-leakage technical feature engineering, Indian exchange holiday and calendar awareness (Asia/Kolkata timezone), chronological cross-validation, out-of-sample Naive Random Walk benchmarking, and automated academic PDF report generation.

---

## 📑 Table of Contents
1. [Project Overview & Market Focus](#-project-overview--market-focus)
2. [Indian Market Architecture](#-indian-market-architecture)
3. [Methodology & Econometric Safeguards](#-methodology--econometric-safeguards)
4. [Empirical Evaluation Results (RELIANCE.NS)](#-empirical-evaluation-results-reliancens)
5. [Installation & Setup Guide](#-installation--setup-guide)
6. [Execution Commands](#-execution-commands)
7. [Streamlit Dashboard Features](#-streamlit-dashboard-features)
8. [Automated Academic PDF Report](#-automated-academic-pdf-report)
9. [Deployment Guide](#-deployment-guide)
10. [Academic Disclaimers & Limitations](#-academic-disclaimers--limitations)

---

## 🇮🇳 Project Overview & Market Focus

The Indian stock market represents one of the fastest-growing and most liquid equity ecosystems in the world. Operating under the regulatory oversight of the Securities and Exchange Board of India (SEBI), Indian equities trade between 09:15 AM and 03:30 PM IST with T+1 settlement.

This project delivers:
- **Primary Market Configuration**: Centered on the National Stock Exchange of India (NSE) with support for Bombay Stock Exchange (BSE) symbols.
- **Default Asset**: **Reliance Industries Limited (`RELIANCE.NS`)**, India's benchmark large-cap market leader.
- **Supported Ticker Formats**:
  - NSE Equities: `TICKER.NS` (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `SBIN.NS`, `ITC.NS`, `LT.NS`, `BHARTIARTL.NS`, `TATAMOTORS.NS`)
  - BSE Equities: `TICKER.BO` (e.g. `RELIANCE.BO`, `TCS.BO`, `INFY.BO`)
  - Benchmark Indices: `^NSEI` (NIFTY 50), `^BSESN` (SENSEX)
- **Indian Numbering & Currency**: Formatted with the ₹ symbol, Indian comma grouping (e.g. `₹1,25,000.00`), and volume in Lakhs (`L`) and Crores (`Cr`).
- **Indian Exchange Calendar**: Powered by `src/utils/indian_calendar.py` and `exchange_calendars` (`XBOM`), automatically rolling weekend sessions and official SEBI holidays (Republic Day, Independence Day, Gandhi Jayanti, Diwali, Holi, Good Friday, Maharashtra Day).
- **Academic Benchmark**: Direct out-of-sample comparison against the **Naive Previous-Close Random Walk** ($\hat{y}_{t+1} = y_t$).

---

## 🏗 Indian Market Architecture

```text
price-prediction/
│
├── app.py                      # Streamlit interactive dashboard (NSE/BSE focus)
├── requirements.txt            # Pinned dependencies
├── runtime.txt                 # Cloud deployment runtime (Python 3.11.9)
├── README.md                   # Full Indian equity project documentation
├── .gitignore                  # Git ignore rules
├── .env.example                # Indian market environment configuration
│
├── src/
│   ├── data/
│   │   ├── collector.py        # yfinance Indian equity ingestion & CSV validation
│   │   └── preprocessing.py    # IST timezone handling, splitting, scaling
│   ├── features/
│   │   └── engineering.py      # Technical indicators & target generation
│   ├── models/
│   │   ├── linear_regression.py# LinearRegression & Ridge regressors in INR
│   │   ├── evaluation.py       # RMSE, MAE, R², naive baseline comparison in INR
│   │   └── forecasting.py      # Indian exchange session roll & prediction interval
│   ├── visualization/
│   │   └── charts.py           # Matplotlib/Seaborn & Plotly chart generators in INR
│   └── utils/
│       ├── config.py           # Centralized Indian market configuration
│       ├── currency.py         # Indian numbering (Lakhs, Crores) and INR formatting
│       ├── indian_calendar.py  # NSE/BSE trading session and holiday resolution
│       └── logger.py           # Logging utility
│
├── scripts/
│   ├── train.py                # Full CLI training & evaluation pipeline
│   └── generate_report.py      # ReportLab academic PDF generator in INR
│
├── tests/
│   ├── conftest.py             # Deterministic synthetic test fixtures
│   ├── test_collector.py       # NSE/BSE ingestion unit tests
│   ├── test_preprocessing.py   # Leak-free splitting tests
│   ├── test_features.py        # Technical indicator tests
│   ├── test_models.py          # Model fitting and persistence tests
│   ├── test_evaluation.py      # Metrics & baseline benchmarking tests
│   ├── test_forecasting.py     # Indian calendar roll and prediction tests
│   └── test_indian_market.py   # Currency, Lakh/Crore, and holiday tests
│
├── data/
│   ├── raw/                    # Raw market CSVs (e.g. RELIANCE_NS_raw.csv)
│   └── processed/              # Cleaned feature matrices
│
├── models/                     # Saved joblib models (e.g. RELIANCE_NS_linearregression.joblib)
│
├── reports/
│   ├── figures/                # 10 static academic figures in INR (PNG)
│   ├── RELIANCE_NS_latest_results.json # Verified metrics & metadata
│   └── Stock_Price_Prediction_Report.pdf # Final academic PDF report (10 pages)
│
└── assets/                     # Media and static assets
```

---

## 🔬 Methodology & Econometric Safeguards

### 1. Ingestion & Indian Provenance
- Ingestion fetches daily OHLCV records via Yahoo Finance (`yfinance`) with fallback protection.
- Dates are converted to Indian Standard Time (`Asia/Kolkata`) and stored as clean daily timestamps.
- Raw CSV files and metadata JSON are persisted under `data/raw/`.

### 2. Zero Lookahead Leakage
- Every predictor $X_{t, j}$ is computed strictly using observations at or prior to session $t$.
- The prediction target is created by shifting Close by -1: $y_t = Close_{t+1}$.
- Rows with unobserved rolling warmup values (initial 50 rows for SMA 50) and the final row (where $t+1$ has not yet occurred) are systematically removed for training.

### 3. Chronological Train / Validation / Test Splitting
- **Strictly Temporal**: No random shuffling.
- Partition slices (for 5 years of `RELIANCE.NS`):
  - **Train (70%)**: $828$ trading sessions (2022-02-03 to 2025-06-13)
  - **Validation (15%)**: $177$ trading sessions (2025-06-16 to 2026-02-27)
  - **Untouched Test (15%)**: $179$ trading sessions (2026-03-02 to 2026-11-23)

### 4. Leakage-Free Feature Scaling
- `StandardScaler` ($\mu, \sigma$) is fitted exclusively on $X_{\text{train}}$.
- $X_{\text{val}}$ and $X_{\text{test}}$ are transformed using the parameters learned from $X_{\text{train}}$.

---

## 📊 Empirical Evaluation Results (RELIANCE.NS)

Actual measured performance on the untouched out-of-sample test partition for Reliance Industries Limited (`RELIANCE.NS`):

| Evaluation Metric | Linear Regression | Naive Random Walk Benchmark ($y_{t+1} = y_t$) | Relative Edge |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **₹20.60** | **₹20.76** | +0.76% |
| **Mean Absolute Error (MAE)** | **₹14.70** | **₹14.66** | -0.27% |
| **Coefficient of Determination ($R^2$)**| **0.9027** | **0.9012** | +0.0015 |
| **Mean Absolute Pct Error (MAPE)** | **1.12%** | **1.12%** | Baseline Parity |
| **Directional Hit Rate** | **51.69%** | 50.00% (No Edge) | +1.69% |

### 💡 Econometric Interpretation
1. **The High $R^2$ Illusion**: An $R^2$ of $0.9027$ is driven by strong unit-root autocorrelation in raw equity prices. Yesterday's price is the dominant predictor of today's price.
2. **The Naive Benchmark Litmus Test**: Comparing against the trivial naive model reveals that predicting today's close as tomorrow's close yields an RMSE of ₹20.76, nearly matching the Linear Regression model (₹20.60).
3. **Market Efficiency Confirmation**: In accordance with the Weak-Form Efficient Market Hypothesis, standard linear regression on raw OHLCV does not produce excess arbitrage returns once STT (Securities Transaction Tax), exchange turnover charges, and slippage are incorporated.

---

## 💻 Installation & Setup Guide

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone Repository
```bash
git clone https://github.com/kunal-gswm/price-prediction.git
cd price-prediction
```

### 2. Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS (Bash):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Commands

### 1. Run Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Opens in your default browser at `http://localhost:8501`.

### 2. Execute Training Pipeline via CLI
Train Linear Regression on 5 years of daily RELIANCE.NS data, compute out-of-sample metrics, generate static figures, and save models:
```bash
python scripts/train.py --ticker RELIANCE.NS --period 5y
```
*Train on other Indian equities:*
```bash
python scripts/train.py --ticker TCS.NS --period 5y
python scripts/train.py --ticker INFY.BO --period 5y
```

### 3. Compile Academic PDF Report
Generates the publication-ready report at `reports/Stock_Price_Prediction_Report.pdf`:
```bash
python scripts/generate_report.py
```

### 4. Execute Automated Test Suite
Run all 39 unit and integration tests:
```bash
python -m pytest -v
```

---

## 🖥 Streamlit Dashboard Features

The application provides an academic-grade web interface across eight distinct tabs:

1. **📋 Overview**: Executive summary, latest observed close in ₹, total data samples, and performance snapshot.
2. **🔍 Data Explorer**: Interactive tables of raw OHLCV and processed feature matrices with CSV download buttons.
3. **📊 Exploratory Data Analysis (EDA)**: Interactive Plotly Candlestick charts with volume subplots, moving average overlays, returns distribution, and 10-day rolling volatility.
4. **🇮🇳 Indian Market Context**: Structural overview of NSE and BSE, trading session hours, T+1 settlement, and volume/liquidity profiles.
5. **⚙️ Model Training**: Chronological sample counts, standardized feature coefficients bar chart, and mathematical formulas.
6. **🎯 Model Evaluation**: Side-by-side error metrics table against the Naive Baseline in ₹, interactive actual vs predicted closing prices, and prediction residual scatter plots.
7. **🔮 Next-Day Forecast**: Point forecast for the next Indian trading session (aware of weekends and NSE holidays), percentage delta, and 95% statistical prediction uncertainty interval ($\hat{y} \pm 1.96 \cdot \text{RMSE}$).
8. **ℹ️ Project Information**: Architecture specifications, safeguards checklist, and a direct download button for the compiled academic PDF report.

---

## 📄 Automated Academic PDF Report

The generated PDF report (`reports/Stock_Price_Prediction_Report.pdf`) is a 10-page academic document featuring:
- Formal academic cover page with Indian stock metadata and Executive Abstract.
- High-resolution (300 DPI) static figures:
  - Price & Volume history line chart (₹ INR)
  - Candlestick & moving average overlays (SMA 5, 10, 20, 50, EMA 10)
  - Daily returns distribution with Gaussian fit
  - 10-day rolling return volatility dynamics
  - Pearson feature correlation heatmap
  - Standardized linear regression coefficient magnitudes
  - Out-of-sample actual vs predicted price curves
  - Predicted vs actual scatter plot with 45-degree reference line
  - Model prediction residuals time-series and histogram
  - Error comparison bar chart against the Naive Baseline
- Formatted tables with verified out-of-sample metrics in INR.
- Next-trading-session point forecast with confidence intervals.
- In-depth econometric discussion on random walks and non-stationarity.
- Complete source code listings in the appendix.

---

## 🌐 Deployment Guide

### Deploying to Streamlit Community Cloud (Free)
1. Push this repository to GitHub: `https://github.com/kunal-gswm/price-prediction`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select your repository, branch (`master` or `main`), and set **Main file path** to `app.py`.
4. Click **"Deploy"**. The app will automatically build using `requirements.txt` and `runtime.txt`.

---

## ⚠️ Academic Disclaimers & Limitations

> **IMPORTANT NOTICE:**
> 1. **Educational Experiment Only**: This software is built strictly for educational, academic, and scientific purposes. It does not constitute investment advice, financial planning, or trading recommendations.
> 2. **Financial Market Complexity**: Real-world asset prices are driven by complex macroeconomic factors, RBI monetary policy announcements, Union Budget decisions, institutional FII/DII liquidity, and global commodity shocks.
> 3. **Model Limitations**: Ordinary Least Squares Linear Regression assumes linearity, homoscedasticity, and absence of regime shifts—assumptions frequently violated during market crashes or high volatility clusters.
> 4. **No Financial Warranty**: Neither the authors nor the contributors assume liability for any financial losses incurred from relying on the model's outputs.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
