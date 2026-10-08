"""数据获取与预处理模块。
- 使用 yfinance 拉取历史 OHLCV 数据
- 处理缺失值、异常值
- 输出标准化后的特征矩阵
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker: str = "AAPL",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """从 Yahoo Finance 拉取历史行情数据。

    参数:
        ticker: 股票代码，例如 "AAPL"、"600519.SS"
        start: 起始日期
        end: 结束日期

    返回:
        包含 OHLCV 的 DataFrame
    """
    print(f"[INFO] 正在拉取 {ticker} 从 {start} 到 {end} 的数据...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"未能下载 {ticker} 的数据，请检查 ticker 或日期范围")
    print(f"[INFO] 数据形状: {df.shape}，时间区间: {df.index[0]} ~ {df.index[-1]}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数据：处理缺失值、异常值。

    - 前向填充缺失值
    - 用 IQR 方法剔除极端异常值
    """
    df = df.copy()
    # 缺失值：先前向填充，再后向填充兜底
    df = df.ffill().bfill()

    # 异常值：用 IQR 替换
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集/测试集（避免数据泄漏）。"""
    n = len(df)
    train_size = int(n * train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    print(f"[INFO] 训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")
    return train_df, test_df