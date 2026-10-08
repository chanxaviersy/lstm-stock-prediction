"""测试 backtest.py"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_direction_accuracy_perfect():
    """完美预测时方向准确率 = 1.0"""
    from backtest import direction_accuracy

    y_true = np.array([100, 101, 102, 103, 104])
    y_pred = np.array([100, 101, 102, 103, 104])
    assert direction_accuracy(y_true, y_pred) == 1.0


def test_direction_accuracy_random():
    """完全反向预测时方向准确率 = 0.0"""
    from backtest import direction_accuracy

    y_true = np.array([100, 101, 102, 103, 104])
    y_pred = np.array([104, 103, 102, 101, 100])
    assert direction_accuracy(y_true, y_pred) == 0.0


def test_simple_backtest_runs():
    """simple_backtest 应能跑通并返回 DataFrame"""
    from backtest import simple_backtest

    prices = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    predictions = np.array([100, 102, 103, 105, 106, 107], dtype=float)
    result = simple_backtest(prices, predictions, initial_cash=10000.0)
    assert isinstance(result, pd.DataFrame)
    assert "portfolio_value" in result.columns
    assert "position" in result.columns
    assert "signal" in result.columns
    assert len(result) == len(predictions) - 1


def test_compute_metrics_keys():
    """compute_metrics 应返回包含 5 个指标的字典"""
    from backtest import compute_metrics, simple_backtest

    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=float)
    predictions = np.array([100, 102, 103, 105, 106, 107, 108, 109], dtype=float)
    backtest_df = simple_backtest(prices, predictions)
    metrics = compute_metrics(backtest_df)
    assert "total_return" in metrics
    assert "annual_return" in metrics
    assert "annual_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    # 最大回撤应为负数或 0
    assert metrics["max_drawdown"] <= 0
