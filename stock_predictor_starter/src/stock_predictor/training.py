
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 3):
    """Simple time-series CV with a RandomForest baseline."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        scores.append(r2_score(y_te, pred))
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)
    info = {"cv_r2_scores": scores, "cv_r2_mean": float(np.mean(scores))}
    return model, info
