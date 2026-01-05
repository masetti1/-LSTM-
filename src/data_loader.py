import pandas as pd

def load_stock_txt(stock_path: str) -> pd.DataFrame:
    # txt 实际是 csv
    df = pd.read_csv(stock_path)
    # Date,Open,High,Low,Close,Volume,OpenInt
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    return df
