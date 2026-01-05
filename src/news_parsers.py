import json
import pandas as pd
import numpy as np

_SENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def explode_polygon_news_to_ticker_daily(news_path: str) -> pd.DataFrame:
    """
    输出: [ticker, Date, sent_score]（每条 insight 一行）
    Polygon 样例里 sentiment 在 insights 下：insights[{ticker,sentiment,...}]
    """
    with open(news_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for art in data:
        pub = art.get("published_utc", None)
        if not pub:
            continue

        # 统一成“无时区日期”，避免 datetime64 vs datetime64[UTC] merge 报错
        d = pd.to_datetime(pub, utc=True).tz_convert(None).normalize()

        insights = art.get("insights", []) or []
        for ins in insights:
            tkr = ins.get("ticker", None)
            s = ins.get("sentiment", None)
            if tkr is None or s is None:
                continue
            score = _SENT_MAP.get(str(s).lower(), 0.0)
            rows.append((tkr, d, score))

    if not rows:
        return pd.DataFrame(columns=["ticker", "Date", "sent_score"])

    return pd.DataFrame(rows, columns=["ticker", "Date", "sent_score"])


import numpy as np
import pandas as pd

def build_daily_sentiment_features(ticker_daily: pd.DataFrame, use_ewm=True, ewm_alpha=0.3) -> pd.DataFrame:
    """
    输入: [ticker, Date, sent_score]
    输出: [ticker, Date, sent_mean, sent_std, news_count, pos_ratio, neg_ratio, has_news, sent_ewm]
    """
    cols = ["ticker","Date","sent_mean","sent_std","news_count","pos_ratio","neg_ratio","has_news","sent_ewm"]

    if ticker_daily is None or len(ticker_daily) == 0:
        return pd.DataFrame(columns=cols)

    df = ticker_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["sent_score"] = pd.to_numeric(df["sent_score"], errors="coerce").fillna(0.0)

    # 直接做布尔列，避免 apply 产生列名不一致
    df["is_pos"] = (df["sent_score"] > 0).astype(float)
    df["is_neg"] = (df["sent_score"] < 0).astype(float)

    agg = df.groupby(["ticker","Date"]).agg(
        sent_mean=("sent_score", "mean"),
        sent_std=("sent_score", "std"),
        news_count=("sent_score", "size"),
        pos_ratio=("is_pos", "mean"),
        neg_ratio=("is_neg", "mean"),
    ).reset_index()

    agg["sent_std"] = agg["sent_std"].fillna(0.0)
    agg["has_news"] = (agg["news_count"] > 0).astype(float)

    if use_ewm:
        agg = agg.sort_values(["ticker","Date"]).reset_index(drop=True)
        agg["sent_ewm"] = agg.groupby("ticker")["sent_mean"].transform(
            lambda s: s.ewm(alpha=ewm_alpha, adjust=False).mean()
        )
    else:
        agg["sent_ewm"] = 0.0

    return agg[cols]

