"""测试 utils.py"""
from __future__ import annotations

import random

import numpy as np
import pytest

# torch 是 utils.py 的依赖，但 torch 在 CI 中可能不可用
torch = pytest.importorskip("torch", reason="torch 未安装，跳过 utils 测试")


def test_set_seed_random():
    """set_seed 后 random 应可复现"""
    from utils import set_seed

    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b


def test_set_seed_numpy():
    """set_seed 后 numpy 应可复现"""
    from utils import set_seed

    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_get_device_returns_string():
    """get_device 应返回字符串"""
    from utils import get_device

    device = get_device()
    assert isinstance(device, str)
    assert device in ("cuda", "mps", "cpu")
