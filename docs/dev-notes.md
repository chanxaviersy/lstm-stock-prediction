# 开发笔记

## 踩过的坑

### 1. yfinance 限流

- **问题**：`yfinance.download()` 偶发 `YFRateLimitError`
- **原因**：Yahoo 对未授权访问有限流
- **解决**：增加 retry + 指数退避（`tenacity` 库）
- **代码**：`src/data_loader.py` 中的 `@retry` 装饰器

### 2. LSTM 容易过拟合

- **问题**：训练 loss 持续下降但验证 loss 上升
- **原因**：金融时序噪声大，模型容易记住历史
- **解决**：Dropout 0.2 + EarlyStopping patience=5 + ReduceLROnPlateau

### 3. 数据泄露（Data Leakage）

- **问题**：用未来信息训练导致回测虚高
- **原因**：标准化时用到了全局统计量
- **解决**：必须按时间切分训练/测试集，标准化只在训练集上 fit

### 4. 类别不平衡

- **问题**：涨跌样本接近 1:1，但模型偏向预测"不变"
- **原因**：标签基于价格变化，分布略偏
- **解决**：可选地加 `class_weight`，或改用回归 + 阈值法

## 性能优化

### 训练加速

- 启用 GPU：`device = "cuda"`
- 增大 batch_size：测试到 64/128
- 用 `torch.compile`（PyTorch 2.0+）

### 数据加载

- 用 `num_workers > 0` 的 DataLoader
- 预生成特征缓存到 parquet

## 学术参考

- Hochreiter & Schmidhuber (1997) - LSTM 原论文
- Fischer & Krauss (2018) - LSTM 在金融时序的应用
- BAAP (2021) - 注意力增强 LSTM
