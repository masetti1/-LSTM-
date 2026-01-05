import numpy as np
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences_return(
    df,
    feature_cols,
    lookback: int,
    horizon: int,
    date_col: str = "Date",
    close_col: str = "Close",
    high_col: str = "High",
    news_count_col: str = "news_count",
):
    """
    用窗口最后一天 t 的信息预测 t+horizon 的收益率：
      ret_close = Close[t+h]/Close[t] - 1
      ret_high  = High[t+h]/High[t] - 1

    返回：
      X: [N, lookback, F]
      y: [N, 2]  (Close_ret, High_ret)
      label_dates: [N]
      base_close: [N] (窗口最后一天 Close[t]) 用于还原价格
      base_high : [N] (窗口最后一天 High[t])  用于还原价格
      news_mask : [N] 窗口内 news_count 总和>=1 的样本标记（体现情绪作用用）
    """
    feat = df[feature_cols].values.astype(np.float32)
    close = df[close_col].values.astype(np.float32)
    high = df[high_col].values.astype(np.float32)
    dates = df[date_col].values

    # news mask（不管 baseline 还是 fusion 都能算，因为 df 里有该列）
    if news_count_col in df.columns:
        news_count = df[news_count_col].values.astype(np.float32)
    else:
        news_count = np.zeros((len(df),), dtype=np.float32)

    X_list, y_list, d_list = [], [], []
    base_close_list, base_high_list = [], []
    mask_list = []

    # i 是窗口最后一天索引 t，label 是 t+horizon
    for i in range(lookback - 1, len(df) - horizon):
        t = i
        y_idx = i + horizon

        X_win = feat[t - lookback + 1: t + 1]  # [lookback, F]

        # return targets
        bc = close[t]
        bh = high[t]
        # 防止除 0
        if bc == 0:
            ret_c = 0.0
        else:
            ret_c = (close[y_idx] / bc) - 1.0

        if bh == 0:
            ret_h = 0.0
        else:
            ret_h = (high[y_idx] / bh) - 1.0

        # news mask：窗口内新闻条数总和>=1
        win_news_sum = float(np.sum(news_count[t - lookback + 1: t + 1]))
        has_news_window = (win_news_sum >= 1.0)

        X_list.append(X_win)
        y_list.append([ret_c, ret_h])
        d_list.append(dates[y_idx])
        base_close_list.append(bc)
        base_high_list.append(bh)
        mask_list.append(has_news_window)

    if len(X_list) == 0:
        X = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        y = np.empty((0, 2), dtype=np.float32)
        return X, y, np.array([]), np.array([]), np.array([]), np.array([])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    label_dates = np.asarray(d_list)
    base_close = np.asarray(base_close_list, dtype=np.float32)
    base_high = np.asarray(base_high_list, dtype=np.float32)
    news_mask = np.asarray(mask_list, dtype=bool)

    return X, y, label_dates, base_close, base_high, news_mask
