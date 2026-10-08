"""测试 features.py 技术指标计算"""
from __future__ import annotations

import pandas as pd
import pytest


def test_add_moving_averages_default_windows(sample_stock_data):
    """默认窗口应生成 MA_5, MA_10, MA_20, MA_60 四列"""
    from features import add_moving_averages

    result = add_moving_averages(sample_stock_data)
    assert "MA_5" in result.columns
    assert "MA_10" in result.columns
    assert "MA_20" in result.columns
    assert "MA_60" in result.columns
    # MA_5 的前 4 行应为 NaN
    assert result["MA_5"].iloc[:4].isna().all()


def test_add_moving_averages_custom_windows(sample_stock_data):
    """自定义窗口应只生成指定列"""
    from features import add_moving_averages

    result = add_moving_averages(sample_stock_data, windows=[7, 14])
    assert "MA_7" in result.columns
    assert "MA_14" in result.columns
    assert "MA_5" not in result.columns


def test_add_moving_averages_value_correctness(sample_stock_data):
    """MA_5 第 5 行 = 前 5 行 Close 平均"""
    from features import add_moving_averages

    result = add_moving_averages(sample_stock_data, windows=[5])
    expected = sample_stock_data["Close"].iloc[:5].mean()
    assert result["MA_5"].iloc[4] == pytest.approx(expected)


def test_add_rsi_range(sample_stock_data):
    """RSI 应在 0-100 之间"""
    from features import add_rsi

    result = add_rsi(sample_stock_data)
    rsi_values = result["RSI"].dropna()
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()


def test_add_rsi_custom_period(sample_stock_data):
    """自定义 period 应生效"""
    from features import add_rsi

    result = add_rsi(sample_stock_data, period=7)
    assert "RSI" in result.columns
    assert result["RSI"].iloc[:6].isna().all()  # 前 6 行 NaN


def test_add_macd_columns(sample_stock_data):
    """MACD 应生成三列：MACD, MACD_Signal, MACD_Hist"""
    from features import add_macd

    result = add_macd(sample_stock_data)
    assert "MACD" in result.columns
    assert "MACD_Signal" in result.columns
    assert "MACD_Hist" in result.columns


def test_add_macd_hist_is_difference(sample_stock_data):
    """MACD_Hist 应等于 MACD - MACD_Signal"""
    from features import add_macd

    result = add_macd(sample_stock_data)
    diff = result["MACD"] - result["MACD_Signal"]
    assert (result["MACD_Hist"].dropna() == diff.dropna()).all()


def test_add_bollinger_bands_columns(sample_stock_data):
    """布林带应生成三列"""
    from features import add_bollinger_bands

    result = add_bollinger_bands(sample_stock_data)
    assert "BB_Upper" in result.columns
    assert "BB_Lower" in result.columns
    assert "BB_Width" in result.columns


def test_bollinger_upper_greater_than_lower(sample_stock_data):
    """BB_Upper 应始终 >= BB_Lower"""
    from features import add_bollinger_bands

    result = add_bollinger_bands(sample_stock_data)
    valid = result.dropna()
    assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()


def test_add_returns_columns(sample_stock_data):
    """Returns 应生成三列"""
    from features import add_returns

    result = add_returns(sample_stock_data)
    assert "Return_1d" in result.columns
    assert "Return_5d" in result.columns
    assert "Log_Return" in result.columns


def test_build_features_complete_pipeline(sample_stock_data):
    """build_features 应生成完整特征矩阵并 dropna"""
    from features import build_features

    result = build_features(sample_stock_data)
    # Target 列应存在
    assert "Target" in result.columns
    # 不应包含 NaN
    assert not result.isna().any().any()


def test_get_feature_columns_excludes_target(sample_stock_data):
    """get_feature_columns 应排除目标列与价格列"""
    from features import build_features, get_feature_columns

    df = build_features(sample_stock_data)
    feature_cols = get_feature_columns(df)
    assert "Target" not in feature_cols
    assert "Close" not in feature_cols
    assert "Open" not in feature_cols
    assert len(feature_cols) > 0
