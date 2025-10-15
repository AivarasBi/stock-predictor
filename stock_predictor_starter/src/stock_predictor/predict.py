
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import make_features

def forecast(model, df: pd.DataFrame, days_ahead: int = 5, lookback: int = 20) -> pd.DataFrame:
    """Naive multi-step forecast: roll forward using last known features and compound returns."""
    hist = df.copy()
    close = hist["close"].astype(float).copy()
    preds = []
    current_df = hist.copy()
    for step in range(days_ahead):
        X, _ = make_features(current_df, lookback=lookback)
        x_last = X.iloc[-1:]
        next_return = float(model.predict(x_last)[0])
        next_price = close.iloc[-1] * (1.0 + next_return)
        # Append a synthetic next day
        next_date = close.index[-1] + pd.tseries.offsets.BDay(1)
        new_row = current_df.iloc[-1].copy()
        new_row["close"] = next_price
        new_df = pd.DataFrame([new_row.values], columns=current_df.columns, index=[next_date])
        current_df = pd.concat([current_df, new_df])
        close = current_df["close"]
        preds.append({"date": next_date, "pred_close": next_price, "pred_return": next_return})
    return pd.DataFrame(preds).set_index("date")
