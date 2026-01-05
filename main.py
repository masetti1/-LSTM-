import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.data_loader import load_stock_txt
from src.news_parsers import explode_polygon_news_to_ticker_daily, build_daily_sentiment_features
from src.features import add_technical_features, merge_stock_with_sentiment
from src.model_lstm import LSTMRegressor
from src.train_lstm import fit, set_seed
from src.metrics import metrics_return_and_price


def make_sequences_return_label_news(
    df: pd.DataFrame,
    feature_cols,
    lookback: int,
    horizon: int,
    min_news_on_label_day: float = 1.0,
):
    """
    预测 t+horizon 的收益率:
      ret_close = Close[t+h]/Close[t] - 1
      ret_high  = High[t+h]/High[t] - 1

    news_mask 定义为：label day（t+horizon 那天）是否有新闻 (news_count>=threshold)
    """
    feat = df[feature_cols].values.astype(np.float32)
    close = df["Close"].values.astype(np.float32)
    high = df["High"].values.astype(np.float32)
    dates = df["Date"].values

    if "news_count" in df.columns:
        news_count = df["news_count"].values.astype(np.float32)
    else:
        news_count = np.zeros(len(df), dtype=np.float32)

    X_list, y_list, d_list = [], [], []
    base_close_list, base_high_list = [], []
    mask_list = []

    for i in range(lookback - 1, len(df) - horizon):
        t = i
        y_idx = i + horizon

        X_win = feat[t - lookback + 1: t + 1]  # [T,F]

        bc = close[t]
        bh = high[t]
        ret_c = (close[y_idx] / (bc + 1e-8)) - 1.0
        ret_h = (high[y_idx] / (bh + 1e-8)) - 1.0

        has_news_label = (news_count[y_idx] >= float(min_news_on_label_day))

        X_list.append(X_win)
        y_list.append([ret_c, ret_h])
        d_list.append(dates[y_idx])
        base_close_list.append(bc)
        base_high_list.append(bh)
        mask_list.append(has_news_label)

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


def eval_split(y_true, y_pred, base_close, base_high, news_mask):
    """输出 all-days 与 news-only 两套指标（Close/High）。"""
    out = {}

    out["Close_All"] = metrics_return_and_price(y_true[:, 0], y_pred[:, 0], base_close)
    out["High_All"]  = metrics_return_and_price(y_true[:, 1], y_pred[:, 1], base_high)

    if np.any(news_mask):
        out["Close_News"] = metrics_return_and_price(y_true[news_mask, 0], y_pred[news_mask, 0], base_close[news_mask])
        out["High_News"]  = metrics_return_and_price(y_true[news_mask, 1], y_pred[news_mask, 1], base_high[news_mask])
        out["news_n"] = int(np.sum(news_mask))
    else:
        out["Close_News"] = None
        out["High_News"] = None
        out["news_n"] = 0

    out["n"] = int(len(y_true))
    return out


def train_and_predict(cfg, Xtr, ytr, Xva, yva, Xte, yte):
    """
    训练：train/val（早停）
    推理：返回 test 上 y_pred/y_true（都是 return 的原始尺度）
    """
    # scaler（fit only on train）
    X_scaler = MinMaxScaler()
    y_scaler = StandardScaler()

    X_scaler.fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr_s = X_scaler.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xva_s = X_scaler.transform(Xva.reshape(-1, Xva.shape[-1])).reshape(Xva.shape)
    Xte_s = X_scaler.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)

    y_scaler.fit(ytr)
    ytr_s = y_scaler.transform(ytr)
    yva_s = y_scaler.transform(yva)
    yte_s = y_scaler.transform(yte)

    from src.dataset import SeqDataset
    train_ds = SeqDataset(Xtr_s, ytr_s)
    val_ds = SeqDataset(Xva_s, yva_s)
    test_ds = SeqDataset(Xte_s, yte_s)

    model = LSTMRegressor(
        input_size=Xtr.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        out_dim=2
    ).to(cfg.device)

    # ckpt path（避免不同fold覆盖）
    os.makedirs(cfg.out_dir, exist_ok=True)
    model = fit(model, train_ds, val_ds, cfg, ckpt_path=cfg._ckpt_path)

    # predict on test split
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
            trues.append(yb.numpy())

    y_pred_s = np.concatenate(preds, axis=0)
    y_true_s = np.concatenate(trues, axis=0)

    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_scaler.inverse_transform(y_true_s)
    return y_true, y_pred


def make_walk_forward_folds(n_trainval: int, k: int, min_train: int = 60):
    """
    Walk-forward folds:
      Fold i: train [0:train_end), val [train_end:val_end)
    val chunk 大小自动均分；保证 train 至少 min_train。
    """
    # 把 trainval 切成 (k+1) 份：前 i 份训练，第 i+1 份验证
    chunk = n_trainval // (k + 1)
    folds = []
    for i in range(1, k + 1):
        train_end = i * chunk
        val_end = (i + 1) * chunk
        if train_end < min_train:
            continue
        if val_end <= train_end:
            continue
        folds.append((train_end, val_end))
    return folds


def flatten_metrics(tag, split_name, fold_id, metrics_dict):
    """
    metrics_dict 结构：Close_All/High_All/Close_News/High_News + n/news_n
    展平成一行，便于写CSV。
    """
    row = {
        "tag": tag,
        "split": split_name,
        "fold": fold_id,
        "n": metrics_dict["n"],
        "news_n": metrics_dict["news_n"],
    }
    for key in ["Close_All", "High_All"]:
        for m, v in metrics_dict[key].items():
            row[f"{key}_{m}"] = v
    if metrics_dict["Close_News"] is not None:
        for m, v in metrics_dict["Close_News"].items():
            row[f"Close_News_{m}"] = v
        for m, v in metrics_dict["High_News"].items():
            row[f"High_News_{m}"] = v
    return row


def summarize_cv(df_rows: pd.DataFrame):
    """
    对 K 折结果做均值/标准差汇总。
    """
    num_cols = [c for c in df_rows.columns if c not in ("tag", "split", "fold")]
    g = df_rows.groupby(["tag", "split"])
    mean_df = g[num_cols].mean().add_suffix("_mean")
    std_df = g[num_cols].std(ddof=0).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    return out


def build_sequences_for_tag(cfg, merged, use_sent: bool):
    df = add_technical_features(merged)

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    tech_cols = ["ret_close_1", "ret_high_1", "range_hl", "oc", "ma5", "ma10", "std5", "log_vol"]

    sent_cols = ["sent_mean", "sent_std", "news_count", "pos_ratio", "neg_ratio", "has_news", "sent_ewm"]
    extra_sent_cols = [c for c in ["sent_shock", "news_z", "sent_abs"] if c in df.columns]

    feature_cols = base_cols + tech_cols + (sent_cols + extra_sent_cols if use_sent else [])

    X, y, dates, bc, bh, mask = make_sequences_return_label_news(
        df,
        feature_cols=feature_cols,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        min_news_on_label_day=cfg.min_news_in_window,
    )
    return X, y, dates, bc, bh, mask


def main():
    cfg = Config()

    # ====== 固定为 2023 小样本更合适的设置（你可自行改）======
    cfg.lookback = 10
    cfg.horizon = 1
    cfg.min_news_in_window = 1.0      # label day news_count>=1
    cfg.cv_k = 5                      # K折
    cfg.holdout_test_ratio = 0.20     # 最后 20% 真实测试集
    cfg.seed = 42

    # training超参（小样本建议多一些epochs，patience略大）
    cfg.epochs = 60
    cfg.patience = 10
    cfg.batch_size = 64

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # 1) load stock/news & merge
    stock_df = load_stock_txt(cfg.stock_path)
    ticker_daily = explode_polygon_news_to_ticker_daily(cfg.news_path)
    sent_daily = build_daily_sentiment_features(ticker_daily, use_ewm=cfg.use_ewm, ewm_alpha=cfg.ewm_alpha)
    merged = merge_stock_with_sentiment(stock_df, sent_daily, cfg.ticker)

    # 2) filter 2023
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    merged = merged[(merged["Date"] >= "2023-01-01") & (merged["Date"] <= "2023-12-31")].copy()
    merged = merged.sort_values("Date").reset_index(drop=True)

    print("[INFO] after filter 2023, merged shape:", merged.shape)
    print("[INFO] stock date range:", merged["Date"].min(), "~", merged["Date"].max())
    news_days = merged.loc[merged.get("has_news", 0) > 0, "Date"]
    print("[INFO] 2023 news unique days:", int(news_days.nunique()) if len(news_days) > 0 else 0)

    # 3) build sequences for baseline/fusion
    results_rows = []
    holdout_rows = []

    for tag, use_sent in [("baseline", False), ("fusion", True)]:
        print(f"\n===== BUILD SEQUENCES: {tag} =====")
        X, y, dates, bc, bh, mask = build_sequences_for_tag(cfg, merged, use_sent=use_sent)

        n = len(X)
        if n < 80:
            raise ValueError(f"样本太少 n={n}，建议 lookback 更小（如5）或不要只用2023。")

        # 4) holdout test: last 20%
        test_n = int(n * cfg.holdout_test_ratio)
        trainval_n = n - test_n
        if test_n < 20:
            print("[WARN] holdout test 太小，建议调大样本或调大test比例。")

        X_tv, y_tv, d_tv, bc_tv, bh_tv, m_tv = X[:trainval_n], y[:trainval_n], dates[:trainval_n], bc[:trainval_n], bh[:trainval_n], mask[:trainval_n]
        X_ho, y_ho, d_ho, bc_ho, bh_ho, m_ho = X[trainval_n:], y[trainval_n:], dates[trainval_n:], bc[trainval_n:], bh[trainval_n:], mask[trainval_n:]

        # 5) walk-forward CV on trainval
        folds = make_walk_forward_folds(trainval_n, k=cfg.cv_k, min_train=60)
        if len(folds) == 0:
            raise ValueError("无法生成CV folds：trainval太短或min_train过大。")

        print(f"[INFO] {tag} trainval_n={trainval_n}, holdout_n={test_n}, folds={len(folds)}")

        for fi, (train_end, val_end) in enumerate(folds, start=1):
            # split
            Xtr, ytr = X_tv[:train_end], y_tv[:train_end]
            Xva, yva = X_tv[train_end:val_end], y_tv[train_end:val_end]

            # 这里的“test”就是这一折的val段（用于评估）
            Xte, yte = Xva, yva
            bc_te, bh_te = bc_tv[train_end:val_end], bh_tv[train_end:val_end]
            m_te = m_tv[train_end:val_end]

            # fold seed（可复现）
            set_seed(cfg.seed + fi)

            # 每折单独ckpt
            cfg._ckpt_path = os.path.join(cfg.out_dir, f"best_{cfg.ticker}_{tag}_fold{fi}.pt")

            y_true, y_pred = train_and_predict(cfg, Xtr, ytr, Xva, yva, Xte, yte)
            met = eval_split(y_true, y_pred, bc_te, bh_te, m_te)

            results_rows.append(flatten_metrics(tag, "cv_val", fi, met))

        # 6) final train on full trainval, eval on holdout test
        # 训练集=前 trainval_n；验证集=最后一段 trainval 的 20%（用于早停）
        # （为了不泄漏，val 仍然来自 trainval 的尾部，不用 holdout）
        tv_val_n = max(20, int(trainval_n * 0.20))
        tv_train_n = trainval_n - tv_val_n

        Xtr, ytr = X_tv[:tv_train_n], y_tv[:tv_train_n]
        Xva, yva = X_tv[tv_train_n:], y_tv[tv_train_n:]
        Xte, yte = X_ho, y_ho

        bc_te, bh_te = bc_ho, bh_ho
        m_te = m_ho

        set_seed(cfg.seed + 999)
        cfg._ckpt_path = os.path.join(cfg.out_dir, f"best_{cfg.ticker}_{tag}_holdout.pt")

        y_true, y_pred = train_and_predict(cfg, Xtr, ytr, Xva, yva, Xte, yte)
        met = eval_split(y_true, y_pred, bc_te, bh_te, m_te)
        holdout_rows.append(flatten_metrics(tag, "holdout_test", 0, met))

        # 输出 holdout 预测csv（方便画图/写论文）
        close_true_price = bc_te * (1.0 + y_true[:, 0])
        close_pred_price = bc_te * (1.0 + y_pred[:, 0])
        high_true_price = bh_te * (1.0 + y_true[:, 1])
        high_pred_price = bh_te * (1.0 + y_pred[:, 1])

        out_csv = os.path.join(cfg.out_dir, f"test_predictions_{cfg.ticker}_2023_{tag}.csv")
        pd.DataFrame({
            "Date": pd.to_datetime(d_ho),
            "label_day_has_news": m_te.astype(int),

            "ret_close_true": y_true[:, 0],
            "ret_close_pred": y_pred[:, 0],
            "Close_base": bc_te,
            "Close_true": close_true_price,
            "Close_pred": close_pred_price,

            "ret_high_true": y_true[:, 1],
            "ret_high_pred": y_pred[:, 1],
            "High_base": bh_te,
            "High_true": high_true_price,
            "High_pred": high_pred_price,
        }).to_csv(out_csv, index=False)
        print(f"[INFO] Saved holdout predictions: {out_csv}")

    # 7) save CV + holdout metrics
    df_cv = pd.DataFrame(results_rows)
    cv_csv = os.path.join(cfg.out_dir, f"cv_metrics_{cfg.ticker}_2023.csv")
    df_cv.to_csv(cv_csv, index=False)

    df_cv_sum = summarize_cv(df_cv)
    cv_sum_csv = os.path.join(cfg.out_dir, f"cv_metrics_summary_{cfg.ticker}_2023.csv")
    df_cv_sum.to_csv(cv_sum_csv, index=False)

    df_ho = pd.DataFrame(holdout_rows)
    ho_csv = os.path.join(cfg.out_dir, f"holdout_metrics_{cfg.ticker}_2023.csv")
    df_ho.to_csv(ho_csv, index=False)

    print("\n===== SAVED =====")
    print("CV folds:", cv_csv)
    print("CV summary (mean±std):", cv_sum_csv)
    print("Holdout test:", ho_csv)

    print("\n===== CV SUMMARY (mean±std) =====")
    print(df_cv_sum)

    print("\n===== HOLDOUT TEST =====")
    print(df_ho)


if __name__ == "__main__":
    main()
