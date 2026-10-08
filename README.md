<!-- ============= 顶部徽章 ============= -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="https://github.com/chanxaviersy/lstm-stock-prediction/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/chanxaviersy/lstm-stock-prediction/test.yml?label=CI&style=flat-square" alt="CI"/>
  </a>
  <a href="https://github.com/chanxaviersy/lstm-stock-prediction">
    <img src="https://img.shields.io/github/last-commit/chanxaviersy/lstm-stock-prediction?style=flat-square" alt="Last Commit"/>
  </a>
  <a href="https://github.com/chanxaviersy/lstm-stock-prediction/stargazers">
    <img src="https://img.shields.io/github/stars/chanxaviersy/lstm-stock-prediction?style=flat-square" alt="Stars"/>
  </a>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"/>
</p>

<!-- ============= 标题区 ============= -->
<br/>
<div align="center">

# 📈 LSTM 股票市场趋势预测

### 基于 Attention-LSTM 的端到端时序预测与回测系统

[🚀 快速开始](#-快速开始) · [📖 文档](docs/architecture.md) · [🐛 报告 Bug](https://github.com/chanxaviersy/lstm-stock-prediction/issues) · [💡 提出新特性](https://github.com/chanxaviersy/lstm-stock-prediction/issues)

</div>

<!-- ============= 项目亮点卡片 ============= -->
<p align="center">
  <table>
    <tr>
      <td align="center" width="200">
        <h3>🎯</h3>
        <b>方向准确率</b><br/>
        <sub><code>~58%</code></sub>
      </td>
      <td align="center" width="200">
        <h3>📊</h3>
        <b>13+ 技术指标</b><br/>
        <sub><code>MA / RSI / MACD</code></sub>
      </td>
      <td align="center" width="200">
        <h3>⚡</h3>
        <b>Huber Loss</b><br/>
        <sub><code>金融噪声鲁棒</code></sub>
      </td>
      <td align="center" width="200">
        <h3>🔁</h3>
        <b>端到端回测</b><br/>
        <sub><code>夏普 / 最大回撤</code></sub>
      </td>
    </tr>
  </table>
</p>

---

<!-- ============= 目录 ============= -->
## 📑 目录

- [🎯 项目目标](#-项目目标)
- [🛠 技术栈](#-技术栈)
- [📂 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [🏗 模型架构](#-模型架构)
- [📊 评估指标](#-评估指标)
- [📈 实验结果](#-实验结果)
- [⚠️ 注意事项](#️-注意事项)
- [🚀 后续可扩展方向](#-后续可扩展方向)
- [📚 更多文档](#-更多文档)
- [📄 License](#-license)

---

## 🎯 项目目标

- 搭建端到端的 **LSTM 时间序列预测流水线**
- 完成数据采集、清洗、特征工程、模型训练、回测等全流程
- 评估模型在金融时序数据上的预测能力（含方向准确率、夏普比率等指标）

> 本项目基于简历中「LSTM-Based Stock Market Trend Prediction」研究/课程项目（2024.9 - 2025.6）整理而来，研究贡献已申请专利。本仓库是其工程化实现版本。

---

## 🛠 技术栈

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,numpy,pandas,sklearn,matplotlib,plotly,git,github,vscode" alt="Tech Stack"/>
</p>

| 类别       | 技术                                              |
| ---------- | ------------------------------------------------- |
| **语言**   | Python 3.10+                                       |
| **深度学习** | PyTorch 2.0+                                      |
| **数据处理** | NumPy · Pandas · scikit-learn                     |
| **可视化**  | Matplotlib · Plotly                               |
| **数据源**  | yfinance（开源财经 API）                          |

---

## 📂 目录结构

```
01-lstm-stock-prediction/
├── 📄 README.md                 # 项目说明
├── 📋 requirements.txt          # 依赖列表
├── 📂 data/                     # 数据目录（运行时自动下载）
├── 📂 models/                   # 模型保存目录
├── 📂 notebooks/                # 实验 Notebook
│   └── exploration.ipynb
├── 🐍 src/
│   ├── data_loader.py           # 数据获取与预处理
│   ├── features.py              # 特征工程（MA、RSI、MACD 等）
│   ├── dataset.py               # PyTorch 数据集封装
│   ├── model.py                 # 模型定义（LSTM）
│   ├── train.py                 # 训练脚本
│   ├── backtest.py              # 回测与策略评估
│   └── utils.py                 # 工具函数
├── 🧪 tests/                    # 单元测试
├── 📚 docs/                     # 详细文档
│   ├── architecture.md
│   ├── usage.md
│   └── dev-notes.md
└── 🎬 run_demo.py               # 一键运行 Demo 入口
```

---

## 🚀 快速开始

### 前置要求

- Python 3.10 或更高版本
- pip 包管理器
- 推荐 8GB+ 内存

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/chanxaviersy/lstm-stock-prediction.git
cd lstm-stock-prediction

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 一键运行 Demo（自动拉取 AAPL 历史数据 + 训练 + 回测）
python run_demo.py
```

> ⏱️ 首次运行约需 2-3 分钟（拉数据 + 训练 50 epochs）

---

## 🏗 模型架构

```
输入特征 (窗口=60) → LSTM(128, return_sequences) → Dropout(0.2)
       → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
```

| 组件       | 选型              | 原因                                |
| ---------- | ----------------- | ----------------------------------- |
| 损失函数   | **Huber Loss**    | 对金融噪声更鲁棒（异常值不爆炸）   |
| 优化器     | Adam              | 自适应学习率，收敛快                 |
| 学习率     | 1e-3 + ReduceLROnPlateau | 训练中自动降速               |
| 正则化     | Dropout 0.2 + EarlyStopping | 防过拟合                     |

---

## 📊 评估指标

- **回归指标**：RMSE、MAE、R²
- **方向准确率**：预测涨跌方向的实际命中率
- **回测指标**：累计收益、年化波动率、夏普比率、最大回撤

---

## 📈 实验结果

> 以下为示例运行结果（实际数字会随下载的数据区间变化）：

| 指标              | 数值          |
| ----------------- | ------------- |
| RMSE              | ~12.3         |
| MAE               | ~9.1          |
| **方向准确率**    | **~58%**      |
| 累计收益（回测）  | 视区间而定    |
| 夏普比率          | 视区间而定    |

### 📸 可视化

> 截图待补充：运行 `run_demo.py` 后会生成结果图表，保存到 [`assets/`](assets/) 后展示在这里。

<details>
<summary>📊 点击展开：预期生成的图表</summary>

- `training_curves.png` — 训练 loss 曲线
- `predictions_vs_actual.png` — 测试集预测 vs 真实
- `backtest_equity_curve.png` — 回测资金曲线

</details>

---

## ⚠️ 注意事项

- 本项目**仅用于学习与研究**，不构成任何投资建议。
- 金融时序预测受市场噪声影响极大，深度学习模型在大多数学术 benchmark 上难以稳定跑赢简单的趋势/均值策略。
- 实盘中需考虑交易成本、滑点、流动性等因素。

---

## 🚀 后续可扩展方向

- 加入 Transformer、Informer 等更先进的时序模型
- 多资产组合预测（portfolio-level forecasting）
- 加入宏观因子（利率、汇率、CPI 等）做多模态融合
- 接入实时行情，做日内信号生成

---

## 📚 更多文档

| 文档 | 说明 |
|------|------|
| [📐 项目架构](docs/architecture.md) | 整体设计、模块关系、数据流 |
| [📖 使用指南](docs/usage.md) | 详细安装、配置、自定义训练 |
| [🔧 开发笔记](docs/dev-notes.md) | 踩过的坑、性能优化、学术参考 |
| [📝 CHANGELOG](CHANGELOG.md) | 版本变更记录 |
| [🤝 CONTRIBUTING](CONTRIBUTING.md) | 如何参与贡献 |

---

## 📄 License

本项目基于 [MIT](LICENSE) 协议开源。

---

<div align="center">

**[⬆ 回到顶部](#-lstm-股票市场趋势预测)**

Made with ❤️ by [Xavier Chen](https://github.com/chanxaviersy)

</div>
