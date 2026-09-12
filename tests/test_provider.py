import inspect

from stock_forecaster.data.provider import MarketDataProvider


def test_market_data_provider_is_abstract() -> None:
    assert inspect.isabstract(MarketDataProvider)
