
import argparse
from .data import load_prices
from .features import make_features
from .training import train_model
from .predict import forecast

def main():
    p = argparse.ArgumentParser(description="Stock predictor CLI")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--period", type=str, default="5y")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--lookback", type=int, default=20)
    args = p.parse_args()

    df = load_prices(args.ticker, period=args.period)
    X, y = make_features(df, lookback=args.lookback)
    model, info = train_model(X, y)
    print(f"CV R2 scores: {info['cv_r2_scores']} (mean={info['cv_r2_mean']:.4f})")
    pred = forecast(model, df, days_ahead=args.days, lookback=args.lookback)
    print(pred.tail(args.days))
