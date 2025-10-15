
import streamlit as st
from stock_predictor import load_prices, make_features, train_model, forecast
import matplotlib.pyplot as plt

st.title("Stock Predictor (Educational)")
ticker = st.text_input("Ticker", "AAPL")
period = st.selectbox("History period", ["2y", "5y", "10y"], index=1)
lookback = st.slider("Lookback (days)", 5, 60, 20)
days = st.slider("Days ahead", 1, 20, 5)

if st.button("Run"):
    with st.spinner("Training model..."):
        df = load_prices(ticker, period=period)
        X, y = make_features(df, lookback=lookback)
        model, info = train_model(X, y)
        pred = forecast(model, df, days_ahead=days, lookback=lookback)
        st.write(f"CV R2 mean: {info['cv_r2_mean']:.4f}")
        # Plot
        fig = plt.figure()
        df["close"].plot(label="Historical")
        pred["pred_close"].plot(label="Forecast")
        plt.legend()
        st.pyplot(fig)
