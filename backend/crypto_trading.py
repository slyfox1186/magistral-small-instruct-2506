#!/usr/bin/env python3
"""Cryptocurrency market data retrieval using CoinGecko API.

This module provides comprehensive cryptocurrency market data including:
- Real-time cryptocurrency prices and market data
- Market analysis and trends
- Global market statistics
- Trending cryptocurrencies

Dependencies:
    pip install pycoingecko requests pandas numpy

Author: Claude Code
License: MIT
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pycoingecko import CoinGeckoAPI

UTC = UTC

# Import configuration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CryptoNews:
    """Data class for cryptocurrency news article (placeholder for compatibility)."""

    title: str
    url: str
    published_date: str
    author: str | None = None
    summary: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    content_preview: str | None = None


@dataclass
class CryptoQuote:
    """Data class for cryptocurrency quote information."""

    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    price_change_24h: float
    price_change_percentage_24h: float
    volume_24h: float
    circulating_supply: float | None = None
    total_supply: float | None = None
    max_supply: float | None = None
    ath: float | None = None
    ath_change_percentage: float | None = None
    ath_date: str | None = None
    atl: float | None = None
    atl_change_percentage: float | None = None
    atl_date: str | None = None


@dataclass
class MarketStats:
    """Data class for global market statistics."""

    total_market_cap: float
    total_volume_24h: float
    market_cap_change_24h: float
    active_cryptocurrencies: int
    markets: int
    market_cap_percentage: dict[str, float]
    updated_at: str


@dataclass
class TrendingCoin:
    """Data class for trending cryptocurrency."""

    id: str
    symbol: str
    name: str
    market_cap_rank: int
    price_btc: float
    score: int
    thumb: str
    large: str


class CryptoTrading:
    """Comprehensive cryptocurrency market data retrieval class using CoinGecko API.
    
    This class provides methods to fetch cryptocurrency prices, market data,
    global statistics, and trending cryptocurrencies from CoinGecko.
    """

    def __init__(self):
        """Initialize the CryptoTrading class with CoinGecko API."""
        # Initialize CoinGecko API client
        api_key = os.getenv('COINGECKO_API_KEY')
        if api_key:
            self.cg = CoinGeckoAPI(api_key=api_key)
            logger.info("CryptoTrading initialized with CoinGecko API (authenticated)")
        else:
            self.cg = CoinGeckoAPI()
            logger.info("CryptoTrading initialized with CoinGecko API (public)")

        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        logger.info("CryptoTrading initialized with CoinGecko API")

    def clear_cache(self):
        """Clear all cached data to force fresh API calls."""
        self.cache.clear()
        logger.info("Crypto data cache cleared - all subsequent requests will fetch fresh data")

    def _get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs):
        """Helper method to implement caching for API calls."""
        now = time.time()

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_duration:
                logger.debug(f"Returning cached data for {cache_key}")
                return cached_data

        try:
            data = fetch_func(*args, **kwargs)
            self.cache[cache_key] = (data, now)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {cache_key}: {e!s}")
            raise

    def get_crypto_quote(self, coin_id: str, include_market_data: bool = True) -> CryptoQuote | None:
        """Get detailed cryptocurrency quote information.
        
        Args:
            coin_id (str): CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            include_market_data (bool): Include additional market data
        
        Returns:
            Optional[CryptoQuote]: Cryptocurrency quote data
        """
        try:
            cache_key = f"quote_{coin_id}_{include_market_data}"

            def fetch():
                data = self.cg.get_coin_by_id(
                    id=coin_id,
                    localization=False,
                    tickers=False,
                    market_data=include_market_data,
                    community_data=False,
                    developer_data=False,
                    sparkline=False
                )
                return data

            coin_data = self._get_cached_or_fetch(cache_key, fetch)

            if not coin_data or 'market_data' not in coin_data:
                logger.warning(f"No market data available for {coin_id}")
                return None

            market_data = coin_data['market_data']

            quote = CryptoQuote(
                id=coin_data.get('id', coin_id),
                symbol=coin_data.get('symbol', '').upper(),
                name=coin_data.get('name', ''),
                current_price=market_data.get('current_price', {}).get('usd', 0.0),
                market_cap=market_data.get('market_cap', {}).get('usd', 0.0),
                market_cap_rank=market_data.get('market_cap_rank', 0),
                price_change_24h=market_data.get('price_change_24h', 0.0),
                price_change_percentage_24h=market_data.get('price_change_percentage_24h', 0.0),
                volume_24h=market_data.get('total_volume', {}).get('usd', 0.0),
                circulating_supply=market_data.get('circulating_supply'),
                total_supply=market_data.get('total_supply'),
                max_supply=market_data.get('max_supply'),
                ath=market_data.get('ath', {}).get('usd'),
                ath_change_percentage=market_data.get('ath_change_percentage', {}).get('usd'),
                ath_date=market_data.get('ath_date', {}).get('usd'),
                atl=market_data.get('atl', {}).get('usd'),
                atl_change_percentage=market_data.get('atl_change_percentage', {}).get('usd'),
                atl_date=market_data.get('atl_date', {}).get('usd'),
            )

            logger.info(f"Retrieved quote for {coin_id}: ${quote.current_price:,.2f}")
            return quote

        except Exception as e:
            logger.error(f"Error retrieving crypto quote for {coin_id}: {e!s}")
            return None

    def get_multiple_crypto_quotes(self, coin_ids: list[str], force_refresh: bool = False, use_scraping: bool = True) -> list[CryptoQuote]:
        """Get multiple cryptocurrency quotes efficiently.
        
        Args:
            coin_ids (list[str]): List of CoinGecko coin IDs
            force_refresh (bool): Force refresh cached data
            use_scraping (bool): Unused parameter for compatibility
        
        Returns:
            list[CryptoQuote]: List of cryptocurrency quotes
        """
        try:
            if force_refresh:
                # Clear cache for specific coins
                for coin_id in coin_ids:
                    cache_keys_to_remove = [k for k in self.cache.keys() if coin_id in k]
                    for key in cache_keys_to_remove:
                        del self.cache[key]

            cache_key = f"multi_quotes_{'_'.join(sorted(coin_ids))}"

            def fetch():
                # Use CoinGecko's coins/markets endpoint for efficient bulk fetching
                data = self.cg.get_coins_markets(
                    vs_currency='usd',
                    ids=','.join(coin_ids),
                    order='market_cap_desc',
                    per_page=len(coin_ids),
                    page=1,
                    sparkline=False,
                    price_change_percentage='24h'
                )
                return data

            market_data = self._get_cached_or_fetch(cache_key, fetch)

            quotes = []
            for coin in market_data:
                quote = CryptoQuote(
                    id=coin.get('id', ''),
                    symbol=coin.get('symbol', '').upper(),
                    name=coin.get('name', ''),
                    current_price=coin.get('current_price', 0.0),
                    market_cap=coin.get('market_cap', 0.0),
                    market_cap_rank=coin.get('market_cap_rank', 0),
                    price_change_24h=coin.get('price_change_24h', 0.0),
                    price_change_percentage_24h=coin.get('price_change_percentage_24h', 0.0),
                    volume_24h=coin.get('total_volume', 0.0),
                    circulating_supply=coin.get('circulating_supply'),
                    total_supply=coin.get('total_supply'),
                    max_supply=coin.get('max_supply'),
                    ath=coin.get('ath'),
                    ath_change_percentage=coin.get('ath_change_percentage'),
                    ath_date=coin.get('ath_date'),
                    atl=coin.get('atl'),
                    atl_change_percentage=coin.get('atl_change_percentage'),
                    atl_date=coin.get('atl_date'),
                )
                quotes.append(quote)

            logger.info(f"Retrieved {len(quotes)} cryptocurrency quotes")
            return quotes

        except Exception as e:
            logger.error(f"Error retrieving multiple crypto quotes: {e!s}")
            return []

    def get_global_market_data(self) -> MarketStats | None:
        """Get global cryptocurrency market statistics.
        
        Returns:
            Optional[MarketStats]: Global market statistics
        """
        try:
            cache_key = "global_market_data"

            def fetch():
                return self.cg.get_global()

            global_data = self._get_cached_or_fetch(cache_key, fetch)

            if not global_data or 'data' not in global_data:
                logger.warning("No global market data available")
                return None

            data = global_data['data']

            stats = MarketStats(
                total_market_cap=data.get('total_market_cap', {}).get('usd', 0.0),
                total_volume_24h=data.get('total_volume', {}).get('usd', 0.0),
                market_cap_change_24h=data.get('market_cap_change_percentage_24h_usd', 0.0),
                active_cryptocurrencies=data.get('active_cryptocurrencies', 0),
                markets=data.get('markets', 0),
                market_cap_percentage=data.get('market_cap_percentage', {}),
                updated_at=datetime.now(UTC).isoformat(),
            )

            logger.info(f"Retrieved global market data: ${stats.total_market_cap:,.0f} total market cap")
            return stats

        except Exception as e:
            logger.error(f"Error retrieving global market data: {e!s}")
            return None

    def get_trending_coins(self) -> list[TrendingCoin]:
        """Get trending cryptocurrencies from CoinGecko.
        
        Returns:
            list[TrendingCoin]: List of trending cryptocurrencies
        """
        try:
            cache_key = "trending_coins"

            def fetch():
                return self.cg.get_search_trending()

            trending_data = self._get_cached_or_fetch(cache_key, fetch)

            trending_coins = []
            if trending_data and 'coins' in trending_data:
                for item in trending_data['coins']:
                    coin = item.get('item', {})
                    trending_coin = TrendingCoin(
                        id=coin.get('id', ''),
                        symbol=coin.get('symbol', ''),
                        name=coin.get('name', ''),
                        market_cap_rank=coin.get('market_cap_rank', 0),
                        price_btc=coin.get('price_btc', 0.0),
                        score=coin.get('score', 0),
                        thumb=coin.get('thumb', ''),
                        large=coin.get('large', ''),
                    )
                    trending_coins.append(trending_coin)

            logger.info(f"Retrieved {len(trending_coins)} trending cryptocurrencies")
            return trending_coins

        except Exception as e:
            logger.error(f"Error retrieving trending coins: {e!s}")
            return []

    def format_crypto_data_with_sources(self, coin_ids: list[str]) -> tuple[str, list[dict[str, str]]]:
        """Format cryptocurrency data with proper markdown table formatting.
        
        Args:
            coin_ids (list[str]): List of coin IDs to get data for
        
        Returns:
            tuple[str, list[dict[str, str]]]: Formatted markdown table and sources
        """
        quotes = self.get_multiple_crypto_quotes(coin_ids)

        if not quotes:
            return "No cryptocurrency data available.", []

        # Build properly formatted markdown table with enhanced styling
        table = "## Cryptocurrency Market Data\n\n"
        table += "| Cryptocurrency | Price (USD) | 24h Change | Market Cap Rank | Market Cap | Volume (24h) |\n"
        table += "|----------------|-------------|------------|-----------------|------------|---------------|\n"

        sources = []

        for quote in quotes:
            # Enhanced formatting with proper alignment and styling
            price_change_emoji = "🟢" if quote.price_change_percentage_24h >= 0 else "🔴"
            change_sign = "+" if quote.price_change_percentage_24h >= 0 else ""

            # Format large numbers with proper abbreviations
            market_cap_formatted = self._format_large_number(quote.market_cap)
            volume_formatted = self._format_large_number(quote.volume_24h)

            table += f"| **{quote.name}** ({quote.symbol}) | `${quote.current_price:,.4f}` | {price_change_emoji} **{change_sign}{quote.price_change_percentage_24h:.2f}%** | #{quote.market_cap_rank} | ${market_cap_formatted} | ${volume_formatted} |\n"

            # Create source entry
            sources.append({
                "title": f"{quote.name} ({quote.symbol}) Market Data",
                "url": f"https://www.coingecko.com/en/coins/{quote.id}",
                "date": datetime.now(UTC).isoformat(),
                "source": "CoinGecko API",
                "category": "Cryptocurrency Market Data",
            })

        # Add data timestamp
        table += f"\n*Data retrieved: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n"

        return table, sources

    def _format_large_number(self, number: float) -> str:
        """Format large numbers with appropriate abbreviations (K, M, B, T)."""
        if number >= 1_000_000_000_000:  # Trillion
            return f"{number/1_000_000_000_000:.2f}T"
        elif number >= 1_000_000_000:  # Billion
            return f"{number/1_000_000_000:.2f}B"
        elif number >= 1_000_000:  # Million
            return f"{number/1_000_000:.2f}M"
        elif number >= 1_000:  # Thousand
            return f"{number/1_000:.2f}K"
        else:
            return f"{number:,.2f}"

    # Compatibility methods for news functionality
    def get_crypto_news(self, limit: int = 20, category: str | None = None) -> list[CryptoNews]:
        """Placeholder for news functionality - not available with CoinGecko."""
        logger.warning("News functionality not available with CoinGecko API. Use price data instead.")
        return []

    def search_crypto_news(self, query: str, limit: int = 10) -> list[CryptoNews]:
        """Placeholder for news search functionality."""
        logger.warning("News search not available with CoinGecko API.")
        return []

    def get_market_sentiment(self) -> dict[str, Any]:
        """Get market sentiment based on price movements and trends."""
        try:
            # Get top 50 cryptocurrencies by market cap
            market_data = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=50,
                page=1,
                sparkline=False,
                price_change_percentage='24h'
            )

            if not market_data:
                return {}

            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for coin in market_data:
                change_24h = coin.get('price_change_percentage_24h', 0)
                if change_24h > 1:
                    positive_count += 1
                elif change_24h < -1:
                    negative_count += 1
                else:
                    neutral_count += 1

            total = len(market_data)
            sentiment_score = (positive_count - negative_count) / total if total > 0 else 0

            return {
                "total_coins": total,
                "positive_sentiment": positive_count,
                "negative_sentiment": negative_count,
                "neutral_sentiment": neutral_count,
                "sentiment_score": sentiment_score,
                "overall_sentiment": (
                    "Bullish" if sentiment_score > 0.1
                    else "Bearish" if sentiment_score < -0.1
                    else "Neutral"
                ),
                "updated_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e!s}")
            return {}


# Utility functions for easy access
def get_crypto_quote(coin_id: str) -> CryptoQuote | None:
    """Quick function to get cryptocurrency quote."""
    trader = CryptoTrading()
    return trader.get_crypto_quote(coin_id)


def get_global_crypto_market() -> MarketStats | None:
    """Quick function to get global market stats."""
    trader = CryptoTrading()
    return trader.get_global_market_data()


def get_trending_cryptos() -> list[TrendingCoin]:
    """Quick function to get trending cryptocurrencies."""
    trader = CryptoTrading()
    return trader.get_trending_coins()


def get_market_sentiment() -> dict[str, Any]:
    """Quick function to get market sentiment."""
    trader = CryptoTrading()
    return trader.get_market_sentiment()


# Example usage and testing
if __name__ == "__main__":
    # Initialize the crypto trading module
    crypto_trader = CryptoTrading()

    print("\n=== Testing CoinGecko API ===")

    # Get Bitcoin quote
    print("\n=== Bitcoin Quote ===")
    bitcoin = crypto_trader.get_crypto_quote("bitcoin")
    if bitcoin:
        print(f"Bitcoin (BTC): ${bitcoin.current_price:,.2f}")
        print(f"24h Change: {bitcoin.price_change_percentage_24h:.2f}%")
        print(f"Market Cap: ${bitcoin.market_cap:,.0f}")
        print(f"Volume: ${bitcoin.volume_24h:,.0f}")

    # Get multiple quotes
    print("\n=== Top Cryptocurrencies ===")
    top_coins = crypto_trader.get_multiple_crypto_quotes(["bitcoin", "ethereum", "cardano"])
    for coin in top_coins:
        print(f"{coin.name} ({coin.symbol}): ${coin.current_price:,.2f} ({coin.price_change_percentage_24h:+.2f}%)")

    # Get global market data
    print("\n=== Global Market Data ===")
    global_data = crypto_trader.get_global_market_data()
    if global_data:
        print(f"Total Market Cap: ${global_data.total_market_cap:,.0f}")
        print(f"Total Volume (24h): ${global_data.total_volume_24h:,.0f}")
        print(f"Market Cap Change (24h): {global_data.market_cap_change_24h:.2f}%")
        print(f"Active Cryptocurrencies: {global_data.active_cryptocurrencies:,}")

    # Get trending coins
    print("\n=== Trending Cryptocurrencies ===")
    trending = crypto_trader.get_trending_coins()
    for i, coin in enumerate(trending[:5], 1):
        print(f"{i}. {coin.name} ({coin.symbol}) - Rank #{coin.market_cap_rank}")

    # Get market sentiment
    print("\n=== Market Sentiment ===")
    sentiment = crypto_trader.get_market_sentiment()
    if sentiment:
        print(f"Overall Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        print(f"Positive: {sentiment.get('positive_sentiment', 0)} coins")
        print(f"Negative: {sentiment.get('negative_sentiment', 0)} coins")
        print(f"Neutral: {sentiment.get('neutral_sentiment', 0)} coins")
