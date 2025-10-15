
# Run this in Colab after pushing to GitHub:
# !pip install -q git+https://github.com/USERNAME/stock-predictor@main
# from stock_predictor import load_prices, make_features, train_model, forecast
# df = load_prices("AAPL", period="5y")
# X, y = make_features(df, lookback=20)
# model, info = train_model(X, y)
# pred = forecast(model, df, days_ahead=5, lookback=20)
# print(pred.tail())
