from dataclasses import dataclass

@dataclass
class Config:
    # data
    stock_path: str = r"data/amzn_us_d.csv"
    news_path: str = r"data/news/polygon_news_sample.json"
    ticker: str = "AMZN"
    out_dir: str = "outputs"

    # task
    lookback: int = 10
    horizon: int = 1  # 预测 t+1
    target_cols = ("Close", "High")

    # training
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 60
    patience: int = 10

    # model
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    # split
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # sentiment aggregation
    use_ewm: bool = True
    ewm_alpha: float = 0.3

    # ===== 关键：只选“有新闻”的序列样本 =====
    use_news_only_sequences: bool = True
    news_filter_mode: str = "window"  # "window" or "label"
    min_news_in_window: float = 1.0   # 窗口内 news_count 总和 >= 1
