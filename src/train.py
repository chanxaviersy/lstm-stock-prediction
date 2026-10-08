"""训练脚本：包含训练循环、验证、Early Stopping、模型保存。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import StockSequenceDataset
from features import build_features, get_feature_columns
from data_loader import clean_data, fetch_stock_data, split_train_test
from model import StockLSTM


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        # 梯度裁剪，防止 LSTM 训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total += loss.item() * len(x)
            preds.append(pred.cpu().numpy().flatten())
            targets.append(y.cpu().numpy().flatten())
    return total / len(loader.dataset), np.concatenate(preds), np.concatenate(targets)


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    # 1. 拉取 & 清洗数据
    raw = fetch_stock_data(args.ticker, args.start, args.end)
    df = clean_data(raw)
    df = build_features(df)

    feature_cols = get_feature_columns(df)
    print(f"[INFO] 特征列 ({len(feature_cols)}): {feature_cols}")

    # 2. 切分训练集 / 测试集
    train_df, test_df = split_train_test(df, train_ratio=args.train_ratio)

    # 3. 标准化（只在训练集上 fit，避免数据泄漏）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(train_df[feature_cols].values)
    y_train = scaler_y.fit_transform(train_df[["Target"]].values).flatten()
    X_test = scaler_X.transform(test_df[feature_cols].values)
    y_test = scaler_y.transform(test_df[["Target"]].values).flatten()

    # 4. 构建数据集
    train_ds = StockSequenceDataset(X_train, y_train, window_size=args.window)
    test_ds = StockSequenceDataset(X_test, y_test, window_size=args.window)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 5. 初始化模型
    model = StockLSTM(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 6. 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = Path("models/best_lstm.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "feature_cols": feature_cols,
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    print(f"[INFO] 最佳验证损失: {best_val_loss:.6f}，模型已保存到 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM 股票预测训练脚本")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    main(parser.parse_args())