"""回测与策略评估模块。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算方向准确率：预测涨跌方向的实际命中率。"""
    # 由于 LSTM 预测的是价格而非收益，这里计算相邻样本的价格变化方向
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct = np.sum(np.sign(true_diff) == np.sign(pred_diff))
    return correct / len(true_diff)


def simple_backtest(
    prices: np.ndarray, predictions: np.ndarray, initial_cash: float = 10000.0
) -> pd.DataFrame:
    """简单回测策略：
    - 当预测明日上涨 → 满仓买入
    - 当预测明日下跌 → 清仓
    """
    cash = initial_cash
    position = 0  # 持仓股数
    history = []

    for i in range(len(predictions) - 1):
        pred_today = predictions[i]
        pred_next = predictions[i + 1]
        price_today = prices[i]

        signal = 1 if pred_next > pred_today else 0  # 1=买入, 0=卖出

        if signal == 1 and cash > 0:
            # 满仓买入
            position = cash // price_today
            cash -= position * price_today
        elif signal == 0 and position > 0:
            # 清仓
            cash += position * price_today
            position = 0

        portfolio_value = cash + position * price_today
        history.append(
            {
                "day": i,
                "price": price_today,
                "signal": signal,
                "position": position,
                "cash": cash,
                "portfolio_value": portfolio_value,
            }
        )

    df = pd.DataFrame(history)
    return df


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """计算回测评估指标。"""
    returns = df["portfolio_value"].pct_change().dropna()
    total_return = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / (annual_vol + 1e-10)

    # 最大回撤
    cummax = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }