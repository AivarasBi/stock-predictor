
from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_prices(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data via yfinance and return a clean DataFrame."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df
