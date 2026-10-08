"""一键运行 Demo：自动拉取数据 → 训练 → 回测 → 打印结果。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 将 src/ 加入路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import compute_metrics, direction_accuracy, simple_backtest  # noqa: E402
from data_loader import clean_data, fetch_stock_data, split_train_test  # noqa: E402
from dataset import StockSequenceDataset  # noqa: E402
from features import build_features, get_feature_columns  # noqa: E402
from model import StockLSTM  # noqa: E402
from utils import get_device, set_seed  # noqa: E402


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"[DEMO] 使用设备: {device}")

    # 1. 拉取 & 处理
    df = clean_data(fetch_stock_data("AAPL", "2015-01-01", "2024-12-31"))
    df = build_features(df)
    feat_cols = get_feature_columns(df)

    train_df, test_df = split_train_test(df, train_ratio=0.8)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(train_df[feat_cols].values)
    y_train = scaler_y.fit_transform(train_df[["Target"]].values).flatten()
    X_test = scaler_X.transform(test_df[feat_cols].values)
    y_test = scaler_y.transform(test_df[["Target"]].values).flatten()

    # 2. 训练（精简版，只跑 20 个 epoch 便于快速演示）
    window = 60
    train_ds = StockSequenceDataset(X_train, y_train, window_size=window)
    test_ds = StockSequenceDataset(X_test, y_test, window_size=window)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    model = StockLSTM(input_dim=X_train.shape[1], hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[DEMO] 开始训练（20 个 epoch 精简版）...")
    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"  epoch {epoch:2d}/20 done")

    # 3. 评估
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).cpu().numpy().flatten()
            preds.append(pred)
            targets.append(y.numpy().flatten())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 反标准化回原始价格
    preds_price = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_price = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(targets_price, preds_price))
    mae = mean_absolute_error(targets_price, preds_price)
    r2 = r2_score(targets_price, preds_price)
    dir_acc = direction_accuracy(targets_price, preds_price)

    print("\n========== 评估结果 ==========")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  R²            : {r2:.4f}")
    print(f"  方向准确率     : {dir_acc:.2%}")

    # 4. 简单回测
    bt = simple_backtest(targets_price, preds_price)
    metrics = compute_metrics(bt)
    print("\n========== 回测结果 ==========")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()