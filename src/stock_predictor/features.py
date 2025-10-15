
from __future__ import annotations
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, lookback: int = 20):
    """Create rolling features and a next-day target from a price DataFrame."""
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in price data.")
    X = pd.DataFrame(index=df.index)
    X["return_1d"] = df["close"].pct_change()
    X["sma"] = df["close"].rolling(lookback).mean()
    X["sma_ratio"] = df["close"] / X["sma"] - 1.0
    X["volatility"] = X["return_1d"].rolling(lookback).std()
    X = X.dropna()

    # Target: next-day return
    y = df["close"].pct_change().shift(-1).reindex(X.index)
    mask = y.notna()
    return X[mask], y[mask]
