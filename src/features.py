"""特征工程模块。
- 计算技术指标（MA、RSI、MACD、布林带等）
- 输出可用于 LSTM 的特征矩阵
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """添加移动平均线特征。"""
    if windows is None:
        windows = [5, 10, 20, 60]
    df = df.copy()
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 RSI（相对强弱指数）。"""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标。"""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """计算布林带。"""
    df = df.copy()
    ma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = ma + num_std * std
    df["BB_Lower"] = ma - num_std * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (ma + 1e-10)
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """添加收益率特征。"""
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建完整的特征矩阵。"""
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)

    # 目标变量：次日收盘价
    df["Target"] = df["Close"].shift(-1)

    # 丢掉因 rolling/shift 产生的 NaN 行
    df = df.dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """获取用于建模的特征列（排除目标）。"""
    exclude = {"Target", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    return [c for c in df.columns if c not in exclude]