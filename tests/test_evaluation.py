"""Unit tests for metrics computation, split conformal prediction, and statistical tests."""

import pytest
import numpy as np
import pandas as pd

from src.models.evaluation import (
    calculate_regression_metrics,
    evaluate_model_with_baseline,
    compute_split_conformal_quantile,
    compute_diebold_mariano_test,
)


def test_calculate_regression_metrics_exact():
    """Tests metrics calculation with perfect predictions."""
    y_true = np.array([100.0, 105.0, 110.0])
    y_pred = np.array([100.0, 105.0, 110.0])

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["RMSE"], 0.0)
    assert np.isclose(metrics["MAE"], 0.0)
    assert np.isclose(metrics["R2"], 1.0)
    assert np.isclose(metrics["MAPE"], 0.0)


def test_calculate_regression_metrics_known_error():
    """Tests metrics with known error deviations."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])

    metrics = calculate_regression_metrics(y_true, y_pred)
    # Errors: [2, -2, 0] -> MAE = 4/3 = 1.3333, MSE = 8/3 = 2.6667, RMSE = sqrt(8/3) = 1.6330
    assert np.isclose(metrics["MAE"], 4.0 / 3.0)
    assert np.isclose(metrics["RMSE"], np.sqrt(8.0 / 3.0))


def test_evaluate_model_with_baseline():
    """Tests comparative evaluation against naive baseline."""
    y_true = np.array([105.0, 110.0, 115.0])
    current_close = np.array([100.0, 105.0, 110.0])  # naive baseline predicts 100, 105, 110 (error = 5 each)
    y_pred = np.array([104.0, 109.0, 114.0])  # model predicts closer (error = 1 each)

    results = evaluate_model_with_baseline(y_true, y_pred, current_close)

    assert results["model_metrics"]["RMSE"] < results["naive_metrics"]["RMSE"]
    assert results["rmse_improvement_pct"] > 0
    assert "comparison_table" in results
    assert "directional_accuracy_pct" in results
    assert "diebold_mariano" in results


def test_compute_split_conformal_quantile_finite_sample():
    """Tests exact finite-sample quantile calculation for split conformal prediction."""
    # 10 calibration residuals: 1.0, 2.0, 3.0, ..., 10.0
    y_cal = np.arange(1, 11, dtype=float)
    y_cal_pred = np.zeros(10, dtype=float)  # residuals = 1.0 to 10.0

    # For n=10, alpha=0.10 (90% coverage):
    # k = ceil((10 + 1) * 0.90) = ceil(9.9) = 10
    # The 10th value is 10.0
    cal_info = compute_split_conformal_quantile(y_cal, y_cal_pred, alpha=0.10)
    assert cal_info["finite_sample_k"] == 10
    assert np.isclose(cal_info["conformal_quantile"], 10.0)
    assert np.isclose(cal_info["interval_width"], 20.0)
    assert cal_info["nominal_coverage_pct"] == 90.0

    # For n=10, alpha=0.20 (80% coverage):
    # k = ceil((10 + 1) * 0.80) = ceil(8.8) = 9
    # The 9th value is 9.0
    cal_info_80 = compute_split_conformal_quantile(y_cal, y_cal_pred, alpha=0.20)
    assert cal_info_80["finite_sample_k"] == 9
    assert np.isclose(cal_info_80["conformal_quantile"], 9.0)


def test_evaluate_model_with_baseline_conformal_no_leakage():
    """Regression test: verifies that validation conformal margin evaluates test set without test RMSE leakage."""
    # Calibration partition
    y_cal = np.array([100.0, 102.0, 98.0, 105.0])
    y_cal_pred = np.array([100.0, 101.0, 99.0, 103.0])  # absolute errors: [0, 1, 1, 2]
    cal_info = compute_split_conformal_quantile(y_cal, y_cal_pred, alpha=0.25)
    margin = cal_info["conformal_quantile"]

    # Test partition with intentional high errors
    y_test = np.array([100.0, 105.0, 110.0])
    y_test_pred = np.array([100.0, 101.0, 102.0])  # errors: [0, 4, 8]
    current_close = np.array([99.0, 104.0, 109.0])

    res = evaluate_model_with_baseline(
        y_true=y_test,
        y_pred=y_test_pred,
        current_close=current_close,
        conformal_margin=margin,
        calibration_info=cal_info,
    )

    int_stats = res["interval_stats"]
    assert int_stats["is_leakage_free"] is True
    assert int_stats["interval_margin"] == margin
    # Only index 0 error (0) <= margin (1 or 2); index 1 (4) and 2 (8) exceed margin
    assert int_stats["samples_in_interval"] < len(y_test)
    assert int_stats["mean_interval_width"] == 2.0 * margin
    assert int_stats["median_interval_width"] == 2.0 * margin


def test_diebold_mariano_test_statistic():
    """Tests Diebold-Mariano hypothesis test on model vs baseline."""
    # When model is strictly superior with varying loss differentials
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    y_pred_good = y_true + np.array([0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.2])
    y_pred_bad = y_true + np.array([4.0, -5.0, 6.0, -3.0, 5.0, -4.0, 6.0, -5.0])

    dm_res = compute_diebold_mariano_test(y_true, y_pred_good, y_pred_bad, loss="squared")
    assert dm_res["dm_statistic"] < 0  # good model has lower loss (d = e_good^2 - e_bad^2 < 0)
    assert dm_res["p_value"] < 0.05
    assert dm_res["is_significant_5pct"] is True


def test_directional_accuracy_zero_movement_exclusion():
    """Tests directional accuracy behavior on flat / zero movement sessions."""
    y_true = np.array([105.0, 95.0, 100.0, 100.0])  # Day 4 is flat (100 -> 100)
    current_close = np.array([100.0, 100.0, 100.0, 100.0])
    prev_close = np.array([99.0, 101.0, 99.0, 99.0])
    y_pred = np.array([104.0, 96.0, 102.0, 102.0])

    res = evaluate_model_with_baseline(y_true, y_pred, current_close, prev_close)
    # Day 3 and 4: actual diffs are [5, -5, 0, 0] -> 2 valid non-zero sessions
    assert res["valid_directional_sessions"] == 2
    assert res["zero_movement_sessions"] == 2
    # Model hit rate on valid non-zero sessions: Day 1 (up/up) hit, Day 2 (down/down) hit -> 100%
    assert res["directional_accuracy_pct"] == 100.0
    # No-change baseline is 0.00%
    assert res["no_change_directional_accuracy_pct"] == 0.00
