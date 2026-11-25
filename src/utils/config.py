"""Centralized Market Configuration Module for the Indian Equity Market.

Configures:
- Primary Exchange: National Stock Exchange of India (NSE)
- Secondary Exchange: Bombay Stock Exchange (BSE)
- Currency: Indian Rupee (INR, ₹)
- Timezone: Asia/Kolkata (IST)
- Primary Instrument: Reliance Industries Limited (RELIANCE.NS)
- Historical Periods, Technical Feature List, and Splitting Constants
"""

from pathlib import Path
import os
from typing import Dict, Any, List

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models and Reports paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Ensure essential directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, ASSETS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Market Configuration Constants
PRIMARY_MARKET = "NSE India"
PRIMARY_EXCHANGE = "National Stock Exchange of India (NSE)"
SECONDARY_EXCHANGE = "Bombay Stock Exchange (BSE)"
MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"

# Ticker Configuration
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "RELIANCE.NS")
DEFAULT_TICKER_NAME = "Reliance Industries Limited"

# Popular Indian Large-Cap Equities (NSE & BSE)
INDIAN_STOCKS_METADATA: Dict[str, Dict[str, str]] = {
    "RELIANCE.NS": {"name": "Reliance Industries Limited", "exchange": "NSE", "sector": "Energy & Retail"},
    "TCS.NS": {"name": "Tata Consultancy Services Limited", "exchange": "NSE", "sector": "Information Technology"},
    "INFY.NS": {"name": "Infosys Limited", "exchange": "NSE", "sector": "Information Technology"},
    "HDFCBANK.NS": {"name": "HDFC Bank Limited", "exchange": "NSE", "sector": "Banking & Financial Services"},
    "ICICIBANK.NS": {"name": "ICICI Bank Limited", "exchange": "NSE", "sector": "Banking & Financial Services"},
    "SBIN.NS": {"name": "State Bank of India", "exchange": "NSE", "sector": "Public Sector Banking"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel Limited", "exchange": "NSE", "sector": "Telecommunications"},
    "ITC.NS": {"name": "ITC Limited", "exchange": "NSE", "sector": "FMCG & Conglomerate"},
    "LT.NS": {"name": "Larsen & Toubro Limited", "exchange": "NSE", "sector": "Infrastructure & Engineering"},
    "TATAMOTORS.NS": {"name": "Tata Motors Limited", "exchange": "NSE", "sector": "Automotive"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever Limited", "exchange": "NSE", "sector": "FMCG"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance Limited", "exchange": "NSE", "sector": "Non-Banking Financial"},
    # BSE Alternatives
    "RELIANCE.BO": {"name": "Reliance Industries Limited (BSE)", "exchange": "BSE", "sector": "Energy & Retail"},
    "TCS.BO": {"name": "Tata Consultancy Services (BSE)", "exchange": "BSE", "sector": "Information Technology"},
    "INFY.BO": {"name": "Infosys Limited (BSE)", "exchange": "BSE", "sector": "Information Technology"},
}

POPULAR_TICKERS: List[str] = list(INDIAN_STOCKS_METADATA.keys())

# Contextual Market Indices (Not for direct stock modeling)
CONTEXTUAL_INDICES = {
    "^NSEI": "NIFTY 50 Benchmark Index (NSE)",
    "^BSESN": "BSE SENSEX Benchmark Index (BSE)",
}

# Historical data defaults
DEFAULT_PERIOD = os.getenv("DEFAULT_PERIOD", "5y")
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1d")

# Chronological split ratios (No shuffling)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Core Feature Definitions
FEATURE_COLUMNS = [
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_10",
    "Daily_Return",
    "Prev_Close",
    "Prev_Open",
    "Prev_High",
    "Prev_Low",
    "Prev_Volume",
    "Rolling_Volatility_10",
    "Volume_Change",
]

TARGET_COLUMN = "Target_Next_Close"

# Deterministic random seed
RANDOM_SEED = 42


def is_indian_stock_symbol(ticker: str) -> bool:
    """Verifies whether a ticker symbol represents an Indian exchange equity or index."""
    t = str(ticker).strip().upper()
    return t.endswith(".NS") or t.endswith(".BO") or t.startswith("^NSE") or t.startswith("^BSE")


def get_stock_display_info(ticker: str) -> Dict[str, str]:
    """Retrieves full descriptive information for an Indian equity."""
    t = str(ticker).strip().upper()
    if t in INDIAN_STOCKS_METADATA:
        return INDIAN_STOCKS_METADATA[t]

    exchange = "NSE" if t.endswith(".NS") else ("BSE" if t.endswith(".BO") else "Custom/Other")
    return {
        "name": t,
        "exchange": exchange,
        "sector": "Indian Equities" if is_indian_stock_symbol(t) else "Equities",
    }
