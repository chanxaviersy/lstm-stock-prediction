"""pytest 共享 fixtures"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 把 src/ 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """生成示例股票数据（200 天，固定 seed 保证可重复）"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.5,
            "High": close + abs(np.random.randn(n)),
            "Low": close - abs(np.random.randn(n)),
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def sample_features() -> np.ndarray:
    """示例特征矩阵 (100, 5)"""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float32)


@pytest.fixture
def sample_target() -> np.ndarray:
    """示例目标向量 (100,)"""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float32) * 10 + 100
