"""HTTP configuration for Yahoo Finance access."""

import os

import truststore


def configure_yahoo_http(use_system_trust: bool) -> None:
    """Configure yfinance to use requests with the operating-system trust store."""
    if not use_system_trust:
        return

    os.environ["YF_DISABLE_CURL_CFFI"] = "1"
    truststore.inject_into_ssl()
