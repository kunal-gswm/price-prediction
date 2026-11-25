"""Specialized tests for Indian market conventions, currency formatting, and metadata integrity."""

import pytest
import pandas as pd
from pathlib import Path
import json

from src.utils.currency import (
    format_indian_number,
    format_inr,
    format_inr_compact,
    format_indian_volume,
)
from src.utils.config import (
    is_indian_stock_symbol,
    get_stock_display_info,
    REPORTS_DIR,
    DEFAULT_TICKER,
    PRIMARY_EXCHANGE,
    MARKET_CURRENCY,
    CURRENCY_SYMBOL,
)


def test_format_indian_number():
    """Tests Indian number formatting with 3-digit then 2-digit grouping."""
    assert format_indian_number(1250.50) == "1,250.50"
    assert format_indian_number(25000) == "25,000.00"
    assert format_indian_number(125000) == "1,25,000.00"
    assert format_indian_number(10000000) == "1,00,00,000.00"
    assert format_indian_number(987654321.5) == "98,76,54,321.50"


def test_format_inr():
    """Tests INR symbol prefix and Indian comma formatting."""
    assert format_inr(1250.50) == "₹1,250.50"
    assert format_inr(25000) == "₹25,000.00"
    assert format_inr(125000) == "₹1,25,000.00"
    assert format_inr(10000000) == "₹1,00,00,000.00"


def test_format_inr_negative_number():
    """Regression test: verifies that negative currency places the minus sign before ₹."""
    assert format_inr(-1250.50) == "-₹1,250.50"
    assert format_inr(-25000) == "-₹25,000.00"
    assert format_inr_compact(-15000000) == "-₹1.50 Cr"
    assert format_inr_compact(-250000) == "-₹2.50 L"


def test_format_inr_compact_lakhs_crores():
    """Tests compact currency representation using Lakhs and Crores."""
    assert format_inr_compact(15000000) == "₹1.50 Cr"
    assert format_inr_compact(250000) == "₹2.50 L"
    assert format_inr_compact(1250.50) == "₹1,250.50"


def test_format_indian_volume():
    """Tests volume formatting in Lakh and Crore shares."""
    assert format_indian_volume(18500000) == "1.85 Cr shares"
    assert format_indian_volume(450000) == "4.50 L shares"
    assert format_indian_volume(50000) == "50,000 shares"


def test_indian_ticker_validation():
    """Tests NSE and BSE ticker detection."""
    assert is_indian_stock_symbol("RELIANCE.NS")
    assert is_indian_stock_symbol("TCS.BO")
    assert is_indian_stock_symbol("^NSEI")
    assert is_indian_stock_symbol("^BSESN")
    assert not is_indian_stock_symbol("MSFT")
    assert not is_indian_stock_symbol("GOOGL")


def test_stock_display_info():
    """Tests metadata resolution for Reliance Industries and TCS."""
    rel = get_stock_display_info("RELIANCE.NS")
    assert rel["name"] == "Reliance Industries Limited"
    assert rel["exchange"] == "NSE"

    tcs = get_stock_display_info("TCS.BO")
    assert tcs["exchange"] == "BSE"


def test_market_configuration_defaults():
    """Verifies that primary defaults reflect Indian stock market."""
    assert DEFAULT_TICKER == "RELIANCE.NS"
    assert "NSE" in PRIMARY_EXCHANGE
    assert MARKET_CURRENCY == "INR"
    assert CURRENCY_SYMBOL == "₹"


def test_latest_results_no_usd_or_aapl():
    """Audits latest run results to verify that no USD or AAPL values remain in active results."""
    results_file = REPORTS_DIR / "RELIANCE_NS_latest_results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "AAPL" not in content
        assert "USD" not in content
        assert "Reliance Industries Limited" in content
        assert "INR" in content


def test_muhurat_trading_and_floating_holidays():
    """Regression test: verifies that floating exchange holidays and Muhurat sessions are identified."""
    from src.utils.indian_calendar import is_indian_market_trading_day, get_special_session_notes

    # Guru Nanak Jayanti 2026
    assert not is_indian_market_trading_day("2026-11-24")
    # Holi 2026
    assert not is_indian_market_trading_day("2026-03-03")
    # Muhurat Trading date description
    muhurat_note = get_special_session_notes("2024-11-01")
    assert muhurat_note is not None
    assert "Muhurat Trading" in muhurat_note


def test_directional_accuracy_identical_valid_observations():
    """Regression test: verifies directional accuracy is evaluated on identical valid non-zero sessions."""
    from src.models.evaluation import evaluate_model_with_baseline
    import numpy as np

    y_true = np.array([105.0, 95.0, 100.0, 102.0])
    current_close = np.array([100.0, 100.0, 100.0, 100.0])  # Day 3 is flat (100 -> 100)
    prev_close = np.array([98.0, 102.0, 99.0, 101.0])
    y_pred = np.array([103.0, 97.0, 101.0, 99.0])  # Day 1: up (correct), Day 2: down (correct), Day 4: down (incorrect)

    res = evaluate_model_with_baseline(y_true, y_pred, current_close, prev_close)
    # Valid sessions: indices 0, 1, 3 (Day 2 has zero move: 100 -> 100)
    assert res["valid_directional_sessions"] == 3
    # Model: correct on 0 and 1, wrong on 3 -> 2/3 = 66.67%
    assert np.isclose(res["directional_accuracy_pct"], 2.0 / 3.0 * 100.0)
    # Naive momentum direction:
    # 0: current(100) > prev(98) [up] -> actual true(105) > 100 [up] (Hit)
    # 1: current(100) < prev(102) [down] -> actual true(95) < 100 [down] (Hit)
    # 3: current(100) < prev(101) [down] -> actual true(102) > 100 [up] (Miss)
    # Hit rate = 2/3 = 66.67%
    assert np.isclose(res["naive_directional_accuracy_pct"], 2.0 / 3.0 * 100.0)
    assert "Directional Accuracy (%)" in res["comparison_table"].columns


def test_prediction_interval_coverage_on_held_out_data():
    """Regression test: verifies empirical coverage calculation and calibration."""
    from src.models.evaluation import evaluate_model_with_baseline
    import numpy as np

    y_true = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
    y_pred = np.array([100.0, 101.0, 99.0, 102.0, 98.0])  # Exact match
    current_close = np.array([99.0, 100.0, 100.0, 101.0, 99.0])

    res = evaluate_model_with_baseline(y_true, y_pred, current_close)
    # With zero error, all observations are within interval
    assert res["residual_stats"]["interval_coverage_95_pct"] == 100.0


def test_calendar_provenance_and_xense_distinction():
    """Regression test: verifies distinction between official NSE authority and XBOM BSE proxy."""
    from src.utils.indian_calendar import get_calendar_provenance

    provenance = get_calendar_provenance()
    assert "Official National Stock Exchange of India" in provenance["primary_authority"]
    assert "XBOM" in provenance["secondary_proxy"]
    assert provenance["xense_in_exchange_calendars"] is False
    assert provenance["timezone"] == "Asia/Kolkata"


def test_future_unverified_session_conservative_label():
    """Regression test: verifies that dates beyond verified circulars are handled conservatively."""
    from src.utils.indian_calendar import get_next_indian_trading_session

    future_date = pd.Timestamp("2029-01-01")
    next_session, method = get_next_indian_trading_session(future_date)
    assert "Estimated Next Session" in method
    assert "Subject to official NSE circular" in method
    assert next_session > future_date
