"""PyTorch 数据集封装：将时序数据切分为滑动窗口样本。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """股票时序滑动窗口数据集。

    每个样本：
        X = 过去 window_size 天的特征
        y = 第 window_size+1 天的目标（次日收盘价）
    """

    def __init__(self, features: np.ndarray, target: np.ndarray, window_size: int = 60):
        assert len(features) == len(target), "features 与 target 长度必须一致"
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.target[idx + self.window_size]
        return torch.from_numpy(x), torch.from_numpy(np.array([y], dtype=np.float32))