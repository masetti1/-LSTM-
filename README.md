# -LSTM-
融合新闻情绪的 LSTM 模型
本项目用于复现与对比两种模型：
- **baseline**：仅使用技术面特征（OHLCV 等）预测收益率（return）
- **fusion**：在 baseline 基础上融合财经新闻情绪特征（sentiment）进行预测

项目当前以 **AMZN 2023 年**为实验区间，输出包括：
- k 折交叉验证（CV）指标与汇总
- 最终 holdout 测试集指标
- 测试集预测结果 CSV（含日期与预测值）
- --安装依赖
-  Python >= 3.9（推荐 3.10）
- PyTorch（CPU/GPU 均可）
- 其余依赖见 `requirements.txt`

安装依赖：
```bash
pip install -r requirements.txt
