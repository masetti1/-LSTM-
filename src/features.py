import numpy as np
import pandas as pd

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # returns
    out["ret_close_1"] = out["Close"].pct_change()
    out["ret_high_1"] = out["High"].pct_change()

    # range/oc
    out["range_hl"] = (out["High"] - out["Low"]) / (out["Close"].replace(0, np.nan))
    out["oc"] = (out["Close"] - out["Open"]) / (out["Open"].replace(0, np.nan))

    # ma/std
    out["ma5"] = out["Close"].rolling(5).mean()
    out["ma10"] = out["Close"].rolling(10).mean()
    out["std5"] = out["ret_close_1"].rolling(5).std()

    # log volume（避免量级太大）
    out["log_vol"] = np.log1p(out["Volume"].clip(lower=0))

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def merge_stock_with_sentiment(stock_df: pd.DataFrame, sent_daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    s = stock_df.copy()
    s["ticker"] = ticker
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()

    t = sent_daily.copy()
    if len(t) > 0:
        t["Date"] = pd.to_datetime(t["Date"]).dt.normalize()

    merged = s.merge(t, on=["ticker","Date"], how="left")

    sent_cols = ["sent_mean","sent_std","news_count","pos_ratio","neg_ratio","has_news","sent_ewm"]
    for c in sent_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged
