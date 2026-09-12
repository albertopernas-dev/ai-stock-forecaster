import importlib
import os
from unittest.mock import Mock


def _configure_yahoo_http():
    module = importlib.import_module("stock_forecaster.data.http")
    return module.configure_yahoo_http


def test_disabled_system_trust_does_not_set_yfinance_backend(monkeypatch) -> None:
    monkeypatch.delenv("YF_DISABLE_CURL_CFFI", raising=False)

    _configure_yahoo_http()(use_system_trust=False)

    assert "YF_DISABLE_CURL_CFFI" not in os.environ


def test_disabled_system_trust_does_not_inject_truststore(monkeypatch) -> None:
    inject_mock = Mock()
    monkeypatch.setattr("truststore.inject_into_ssl", inject_mock)

    _configure_yahoo_http()(use_system_trust=False)

    inject_mock.assert_not_called()


def test_enabled_system_trust_selects_requests_fallback(monkeypatch) -> None:
    monkeypatch.delenv("YF_DISABLE_CURL_CFFI", raising=False)
    monkeypatch.setattr("truststore.inject_into_ssl", Mock())

    _configure_yahoo_http()(use_system_trust=True)

    assert os.environ["YF_DISABLE_CURL_CFFI"] == "1"


def test_enabled_system_trust_injects_truststore(monkeypatch) -> None:
    inject_mock = Mock()
    monkeypatch.setattr("truststore.inject_into_ssl", inject_mock)

    _configure_yahoo_http()(use_system_trust=True)

    inject_mock.assert_called_once_with()
