
def test_imports():
    import stock_predictor
    assert hasattr(stock_predictor, "load_prices")
