"""测试 dataset.py"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 未安装，跳过 dataset 测试")


def test_dataset_length(sample_features, sample_target):
    """Dataset 长度 = samples - window_size"""
    from dataset import StockSequenceDataset

    window_size = 10
    ds = StockSequenceDataset(sample_features, sample_target, window_size=window_size)
    assert len(ds) == len(sample_features) - window_size


def test_dataset_item_shape(sample_features, sample_target):
    """每个样本的 X shape = (window, features), y shape = (1,)"""
    from dataset import StockSequenceDataset

    window_size = 10
    n_features = sample_features.shape[1]
    ds = StockSequenceDataset(sample_features, sample_target, window_size=window_size)
    x, y = ds[0]
    assert x.shape == (window_size, n_features)
    assert y.shape == (1,)


def test_dataset_values_correct(sample_features, sample_target):
    """第 0 个样本的 X 应为 features[0:window]"""
    from dataset import StockSequenceDataset

    window_size = 10
    ds = StockSequenceDataset(sample_features, sample_target, window_size=window_size)
    x, y = ds[0]
    expected_x = torch.from_numpy(sample_features[:window_size])
    assert torch.allclose(x, expected_x)
    # y 应为 target[window_size]
    assert y.item() == pytest.approx(float(sample_target[window_size]))


def test_dataset_length_mismatch_raises(sample_features, sample_target):
    """features 与 target 长度不一致应抛 AssertionError"""
    from dataset import StockSequenceDataset

    with pytest.raises(AssertionError):
        StockSequenceDataset(sample_features, sample_target[:50], window_size=10)
