# 详细使用指南

## 环境要求

- Python 3.10+
- pip / conda
- 推荐 8GB+ 内存（GPU 可选）

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/chanxaviersy/lstm-stock-prediction.git
cd lstm-stock-prediction

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 配置（可选）

如需自定义股票代码或时间范围：

```bash
cp .env.example .env
# 编辑 .env，修改 TICKER / START_DATE / END_DATE
```

## 运行

### 一键 Demo（推荐新手）

```bash
python run_demo.py
```

输出：
- 训练过程的 loss 曲线
- 测试集上的预测 vs 真实
- 方向准确率、RMSE
- 简单回测的累计收益曲线

### 自定义训练

```python
from src.data_loader import load_stock_data
from src.features import build_features
from src.model import LSTMModel
from src.train import train_model

# 1. 加载数据
df = load_stock_data("TSLA", "2018-01-01", "2024-12-31")

# 2. 特征工程
features = build_features(df)

# 3. 训练
model = train_model(features, epochs=100)
```

### 评估 & 回测

```python
from src.backtest import simple_backtest, compute_metrics

# 假设 y_pred / y_true 已有
backtest_df = simple_backtest(prices, predictions)
metrics = compute_metrics(backtest_df)
print(metrics)
```

## 常见问题

**Q: 报错 `yfinance failed`？**
A: yfinance 偶尔会因为 Yahoo 限流失败，重试或换 `TUSHARE` 等其他数据源。

**Q: 训练很慢？**
A: 检查 PyTorch 是否识别 GPU。`python -c "import torch; print(torch.cuda.is_available())"`。

**Q: 方向准确率只有 50% 左右？**
A: 正常水平。学术界对股票方向的预测准确率很难显著超过 55-60%。

**Q: 能在 A 股上跑吗？**
A: 可以。修改 `data_loader.py` 接入 Tushare / AKShare，或用 yfinance 的 `.SS`/`.SZ` 后缀。

## 进阶

- 想换 Transformer？见 `src/model.py`，已留扩展空间
- 想加更多特征？扩展 `src/features.py`
- 想做多资产组合？把循环套到多只股票上
