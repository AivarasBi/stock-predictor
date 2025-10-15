
# stock-predictor

Educational AI tool to forecast **closing prices** for a given stock using a simple ML baseline.
Not financial advice.

## Quickstart (Colab)
```python
!pip install -q git+https://github.com/USERNAME/stock-predictor@main
from stock_predictor import load_prices, make_features, train_model, forecast

df = load_prices("AAPL", period="5y")
X, y = make_features(df, lookback=20)
model, info = train_model(X, y)
pred = forecast(model, df, days_ahead=5, lookback=20)
pred
```
Replace `USERNAME` with your GitHub username once you upload this repo.
