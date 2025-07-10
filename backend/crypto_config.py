#!/usr/bin/env python3
"""Configuration for cryptocurrency trading module.

This module contains all configuration data for the crypto trading system,
including common cryptocurrencies, sentiment words, and API settings.

Example Generic Markdown Table Format:
=====================================

| Asset Name | Symbol | Price | Change (24h) | Market Cap | Volume |
|------------|--------|-------|--------------|------------|--------|
| [ASSET_NAME_1] | [SYMBOL_1] | [PRICE_1] | [CHANGE_1] | [MARKET_CAP_1] | [VOLUME_1] |
| [ASSET_NAME_2] | [SYMBOL_2] | [PRICE_2] | [CHANGE_2] | [MARKET_CAP_2] | [VOLUME_2] |
| [ASSET_NAME_3] | [SYMBOL_3] | [PRICE_3] | [CHANGE_3] | [MARKET_CAP_3] | [VOLUME_3] |

Generic Entity Examples:
- [CRYPTO_NAME]: Bitcoin, Ethereum, Cardano
- [SYMBOL]: BTC, ETH, ADA
- [PRICE_VALUE]: $45,123.45, $2,987.23
- [PERCENTAGE_CHANGE]: +5.24%, -2.18%
- [MARKET_CAP_VALUE]: $850.2B, $356.7M
- [VOLUME_VALUE]: $12.4B, $876.3M
"""

# Common cryptocurrency mappings
CRYPTO_MAPPINGS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binance": "BNB",
    "cardano": "ADA",
    "solana": "SOL",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "polygon": "MATIC",
    "avalanche": "AVAX",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "uniswap": "UNI",
    "litecoin": "LTC",
    "cosmos": "ATOM",
    "stellar": "XLM",
    "vechain": "VET",
    "theta": "THETA",
    "filecoin": "FIL",
    "tron": "TRX",
    "monero": "XMR",
}

# Common stock symbols for the search function
COMMON_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. Class A",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "MA": "Mastercard Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "The Home Depot Inc.",
    "DIS": "The Walt Disney Company",
    "BAC": "Bank of America Corp.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce Inc.",
    "NFLX": "Netflix Inc.",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "INTC": "Intel Corporation",
    "CSCO": "Cisco Systems Inc.",
    "VZ": "Verizon Communications Inc.",
    "NKE": "Nike Inc.",
    "IBM": "International Business Machines",
    "ORCL": "Oracle Corporation",
    "AMD": "Advanced Micro Devices Inc.",
    "QCOM": "QUALCOMM Inc.",
}

# Sentiment analysis word lists
SENTIMENT_WORDS = {
    "positive": [
        "surge", "bull", "rally", "gain", "up", "rise", "soar", "pump",
        "green", "bullish", "moon", "rocket", "breakout", "boom", "growth",
        "optimistic", "positive", "strong", "recovery", "rebound"
    ],
    "negative": [
        "crash", "bear", "fall", "down", "drop", "dump", "red", "bearish",
        "decline", "plunge", "collapse", "slump", "correction", "weak",
        "pessimistic", "negative", "downturn", "selloff", "bloodbath"
    ]
}

# API Configuration
COINTELEGRAPH_API_CONFIG = {
    "base_url": "https://api.apify.com/v2/acts/dadhalfdev~cointelegraph-scraper-crypto-news/run-sync-get-dataset-items",
    "timeout": 60,
    "memory": 128,
    "default_limit": 20,
    "cache_duration": 300,  # 5 minutes
}

# News categories
NEWS_CATEGORIES = [
    "all",
    "bitcoin",
    "ethereum",
    "altcoins",
    "blockchain",
    "defi",
    "nft",
    "regulation",
    "analysis",
    "technology",
    "business",
    "markets"
]

# Exchange mappings
EXCHANGE_MAPPINGS = {
    "binance": "Binance",
    "coinbase": "Coinbase",
    "kraken": "Kraken",
    "ftx": "FTX",
    "kucoin": "KuCoin",
    "huobi": "Huobi",
    "okex": "OKEx",
    "gemini": "Gemini",
    "bitstamp": "Bitstamp",
    "bitfinex": "Bitfinex"
}
