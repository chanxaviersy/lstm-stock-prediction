<!-- 徽章 -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/chanxaviersy/01-lstm-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/chanxaviersy/01-lstm-stock-prediction/actions)
[![Last Commit](https://img.shields.io/github/last-commit/chanxaviersy/01-lstm-stock-prediction)](https://github.com/chanxaviersy/01-lstm-stock-prediction)

---

# LSTM 股票市场趋势预测

> 基于深度学习时序预测的股票市场动态评估模型

本项目基于简历中「LSTM-Based Stock Market Trend Prediction」研究/课程项目（2024.9 - 2025.6）整理而来，
研究贡献已申请中国大陆专利。本仓库是其工程化实现版本。

## 项目目标

- 搭建端到端的 LSTM 时间序列预测流水线
- 完成数据采集、清洗、特征工程、模型训练、回测等全流程
- 评估模型在金融时序数据上的预测能力（含方向准确率、夏普比率等指标）

## 技术栈

- **语言**：Python 3.10+
- **核心库**：PyTorch / NumPy / Pandas / scikit-learn
- **可视化**：Matplotlib / Plotly
- **数据源**：yfinance（开源财经 API）

## 目录结构

```
01-lstm-stock-prediction/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖列表
├── data/                     # 数据目录（运行时自动下载）
├── models/                   # 模型保存目录
├── notebooks/                # 实验 Notebook
│   └── exploration.ipynb
├── src/
│   ├── data_loader.py        # 数据获取与预处理
│   ├── features.py           # 特征工程（MA、RSI、MACD 等）
│   ├── dataset.py            # PyTorch 数据集封装
│   ├── model.py              # 模型定义（LSTM）
│   ├── train.py              # 训练脚本
│   ├── backtest.py           # 回测与策略评估
│   └── utils.py              # 工具函数
└── run_demo.py               # 一键运行 Demo 入口
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行 Demo（自动拉取 AAPL 历史数据 + 训练 + 回测）
python run_demo.py
```

## 模型架构

```
输入特征 (窗口=60) → LSTM(128, return_sequences) → Dropout(0.2)
       → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
```

- 损失函数：Huber Loss（对金融噪声更鲁棒）
- 优化器：Adam（初始 lr=1e-3，含 ReduceLROnPlateau 调度）
- 正则化：Dropout + Early Stopping

## 评估指标

- **回归指标**：RMSE、MAE、R²
- **方向准确率**：预测涨跌方向的实际准确率
- **回测指标**：累计收益、年化波动率、夏普比率、最大回撤

## 实验结果（示例）

> 以下为示例运行结果（实际数字会随下载的数据区间变化）：

| 指标              | 数值       |
|------------------|-----------|
| RMSE             | ~12.3     |
| MAE              | ~9.1      |
| 方向准确率         | ~54.7%    |
| 累计收益（回测）    | 视区间而定 |
| 夏普比率          | 视区间而定 |

## 注意事项

- 本项目**仅用于学习与研究**，不构成任何投资建议。
- 金融时序预测受市场噪声影响极大，深度学习模型在大多数学术 benchmark 上
  难以稳定跑赢简单的趋势/均值策略。
- 实盘中需考虑交易成本、滑点、流动性等因素。

## 后续可扩展方向

- 加入 Transformer、Informer 等更先进的时序模型
- 多资产组合预测（portfolio-level forecasting）
- 加入宏观因子（利率、汇率、CPI 等）做多模态融合
- 接入实时行情，做日内信号生成

## License

MIT
## 📚 更多文档

- [项目架构](docs/architecture.md)
- [使用指南](docs/usage.md)
- [开发笔记](docs/dev-notes.md)
