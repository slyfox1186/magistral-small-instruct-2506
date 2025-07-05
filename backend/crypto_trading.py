#!/usr/bin/env python3
"""Cryptocurrency news and market data retrieval using Cointelegraph scraper API.

This module provides comprehensive cryptocurrency news and market insights including:
- Real-time cryptocurrency news from Cointelegraph
- Market analysis and trends
- Breaking crypto news and updates
- Industry insights and developments

Dependencies:
    pip install requests pandas numpy

Author: Claude Code
License: MIT
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
UTC = timezone.utc
from typing import Any

import requests

# Import configuration
from crypto_config import (
    CRYPTO_MAPPINGS,
    SENTIMENT_WORDS,
    COINTELEGRAPH_API_CONFIG,
    NEWS_CATEGORIES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CryptoNews:
    """Data class for cryptocurrency news article."""

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
    """Data class for cryptocurrency quote information (compatibility with existing interface)."""

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
    """Comprehensive cryptocurrency news and market data retrieval class using Cointelegraph.

    This class provides methods to fetch cryptocurrency news, market analysis,
    and trending topics from Cointelegraph via Apify scraper.
    """

    def __init__(self):
        """Initialize the CryptoTrading class with Cointelegraph API."""
        self.apify_token = None  # Will be set from environment if available
        self.cache = {}
        self.cache_duration = COINTELEGRAPH_API_CONFIG["cache_duration"]
        self.base_url = COINTELEGRAPH_API_CONFIG["base_url"]
        self.timeout = COINTELEGRAPH_API_CONFIG["timeout"]
        self.memory = COINTELEGRAPH_API_CONFIG["memory"]
        logger.info("CryptoTrading initialized with Cointelegraph scraper")

    def clear_cache(self):
        """Clear all cached data to force fresh API calls."""
        self.cache.clear()
        logger.info("Crypto news cache cleared - all subsequent requests will fetch fresh data")

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

    def get_crypto_news(self, limit: int = 20, category: str | None = None) -> list[CryptoNews]:
        """Get latest cryptocurrency news from Cointelegraph.

        Args:
            limit (int): Number of articles to retrieve
            category (str): News category filter (optional)

        Returns:
            List[CryptoNews]: List of cryptocurrency news articles
        """
        try:
            cache_key = f"crypto_news_{limit}_{category}"

            def fetch():
                # Apify API parameters for Cointelegraph scraper
                params = {
                    "token": self.apify_token,
                    "timeout": self.timeout,
                    "memory": self.memory
                }

                # Input for the scraper
                scraper_input = {
                    "maxArticles": limit,
                    "category": category or "all",
                    "includeContent": True,
                }

                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; CryptoNewsBot/1.0)",
                }

                # Make request to Apify
                response = requests.post(
                    self.base_url,
                    json=scraper_input,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Apify API error: {response.status_code} - {response.text}")
                    return []

            news_data = self._get_cached_or_fetch(cache_key, fetch)

            news_articles = []
            if isinstance(news_data, list):
                for article in news_data:
                    news_article = CryptoNews(
                        title=article.get("title", ""),
                        url=article.get("url", ""),
                        published_date=article.get("publishedDate", ""),
                        author=article.get("author"),
                        summary=article.get("summary"),
                        category=article.get("category"),
                        tags=article.get("tags", []),
                        content_preview=article.get("contentPreview"),
                    )
                    news_articles.append(news_article)

            logger.info(f"Retrieved {len(news_articles)} cryptocurrency news articles")
            return news_articles

        except Exception as e:
            logger.error(f"Error retrieving crypto news: {e!s}")
            return []

    def format_crypto_news_table(
        self, news_articles: list[CryptoNews]
    ) -> tuple[str, list[dict[str, str]]]:
        """Format cryptocurrency news into a markdown table.

        Args:
            news_articles (List[CryptoNews]): List of news articles

        Returns:
            Tuple[str, List[Dict[str, str]]]: Formatted table and sources
        """
        if not news_articles:
            return "No cryptocurrency news available.", []

        # Build markdown table
        table = "| Title | Date | Category | Author |\n"
        table += "|-------|------|----------|--------|\n"

        sources = []

        for article in news_articles:
            # Truncate title if too long
            title = article.title[:60] + "..." if len(article.title) > 60 else article.title
            date = (
                article.published_date.split("T")[0]
                if "T" in article.published_date
                else article.published_date
            )
            category = article.category or "General"
            author = article.author or "Cointelegraph"

            table += f"| [{title}]({article.url}) | {date} | {category} | {author} |\n"

            # Create source entry
            sources.append(
                {
                    "title": article.title,
                    "url": article.url,
                    "date": article.published_date,
                    "source": "Cointelegraph",
                    "category": category,
                }
            )

        return table, sources

    # Compatibility methods to maintain existing interface
    def get_crypto_quote(
        self, coin_id: str, include_market_data: bool = True
    ) -> CryptoQuote | None:
        """Compatibility method - returns mock data since we're now focused on news.
        For real price data, consider using the news sentiment analysis.
        """
        logger.warning("Price data not available with news-focused module. Returning mock data.")
        return CryptoQuote(
            id=coin_id,
            symbol=coin_id.upper()[:4],
            name=coin_id.replace("-", " ").title(),
            current_price=0.0,
            market_cap=0.0,
            market_cap_rank=0,
            price_change_24h=0.0,
            price_change_percentage_24h=0.0,
            volume_24h=0.0,
        )

    def get_multiple_crypto_quotes(
        self, coin_ids: list[str], force_refresh: bool = False, use_scraping: bool = True
    ) -> list[CryptoQuote]:
        """Compatibility method - returns news-based market insights instead of price data."""
        logger.info("Fetching crypto market insights from news instead of direct price data")
        quotes = []
        for coin_id in coin_ids:
            quotes.append(self.get_crypto_quote(coin_id))
        return quotes

    def get_global_market_data(self) -> MarketStats | None:
        """Compatibility method - returns mock market stats.
        Real market sentiment can be derived from news analysis.
        """
        logger.warning(
            "Global market data not available with news-focused module. Check news sentiment instead."
        )
        return MarketStats(
            total_market_cap=0.0,
            total_volume_24h=0.0,
            market_cap_change_24h=0.0,
            active_cryptocurrencies=0,
            markets=0,
            market_cap_percentage={},
            updated_at=datetime.now(UTC).isoformat(),
        )

    def get_trending_coins(self) -> list[TrendingCoin]:
        """Get trending cryptocurrencies based on news coverage."""
        try:
            # Get recent news and analyze trending topics
            news = self.get_crypto_news(limit=50)

            # Simple trending analysis based on news mentions
            coin_mentions = {}
            for article in news:
                title_lower = article.title.lower()
                content_lower = (article.content_preview or "").lower()

                # Use cryptocurrency mappings from config
                for name, symbol in CRYPTO_MAPPINGS.items():
                    if name in title_lower or name in content_lower:
                        if symbol not in coin_mentions:
                            coin_mentions[symbol] = 0
                        coin_mentions[symbol] += 1

            # Convert to TrendingCoin objects
            trending = []
            for i, (symbol, mentions) in enumerate(
                sorted(coin_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
            ):
                trending_coin = TrendingCoin(
                    id=symbol.lower(),
                    symbol=symbol,
                    name=symbol,
                    market_cap_rank=i + 1,
                    price_btc=0.0,
                    score=mentions,
                    thumb="",
                    large="",
                )
                trending.append(trending_coin)

            logger.info(f"Identified {len(trending)} trending cryptocurrencies from news")
            return trending

        except Exception as e:
            logger.error(f"Error identifying trending coins: {e!s}")
            return []

    def format_crypto_data_with_sources(
        self, coin_ids: list[str]
    ) -> tuple[str, list[dict[str, str]]]:
        """Format cryptocurrency news data with proper source URLs for citations.

        Args:
            coin_ids (List[str]): List of coin IDs to get news for

        Returns:
            Tuple[str, List[Dict[str, str]]]: Formatted news data and sources
        """
        # Get crypto news instead of price data
        news = self.get_crypto_news(limit=10)

        if not news:
            return "No cryptocurrency news available.", []

        # Filter news relevant to requested coins if specific coins mentioned
        if coin_ids:
            relevant_news = []
            for article in news:
                article_text = (article.title + " " + (article.content_preview or "")).lower()
                for coin_id in coin_ids:
                    if coin_id.lower() in article_text or coin_id.upper() in article_text:
                        relevant_news.append(article)
                        break

            if relevant_news:
                news = relevant_news

        return self.format_crypto_news_table(news)

    def search_crypto_news(self, query: str, limit: int = 10) -> list[CryptoNews]:
        """Search for cryptocurrency news articles containing specific terms.

        Args:
            query (str): Search query
            limit (int): Maximum number of results

        Returns:
            List[CryptoNews]: List of matching news articles
        """
        try:
            # Get recent news
            all_news = self.get_crypto_news(limit=100)

            # Filter by query
            query_lower = query.lower()
            matching_news = []

            for article in all_news:
                article_text = (article.title + " " + (article.content_preview or "")).lower()
                if query_lower in article_text:
                    matching_news.append(article)
                    if len(matching_news) >= limit:
                        break

            logger.info(f"Found {len(matching_news)} articles matching '{query}'")
            return matching_news

        except Exception as e:
            logger.error(f"Error searching crypto news: {e!s}")
            return []

    def get_market_sentiment(self) -> dict[str, Any]:
        """Analyze market sentiment based on recent news headlines.

        Returns:
            Dict[str, Any]: Market sentiment analysis
        """
        try:
            news = self.get_crypto_news(limit=50)

            positive_words = SENTIMENT_WORDS["positive"]
            negative_words = SENTIMENT_WORDS["negative"]

            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for article in news:
                title_lower = article.title.lower()
                has_positive = any(word in title_lower for word in positive_words)
                has_negative = any(word in title_lower for word in negative_words)

                if has_positive and not has_negative:
                    positive_count += 1
                elif has_negative and not has_positive:
                    negative_count += 1
                else:
                    neutral_count += 1

            total = len(news)
            sentiment_score = (positive_count - negative_count) / total if total > 0 else 0

            return {
                "total_articles": total,
                "positive_sentiment": positive_count,
                "negative_sentiment": negative_count,
                "neutral_sentiment": neutral_count,
                "sentiment_score": sentiment_score,
                "overall_sentiment": (
                    "Bullish"
                    if sentiment_score > 0.1
                    else "Bearish"
                    if sentiment_score < -0.1
                    else "Neutral"
                ),
                "updated_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e!s}")
            return {}


# Utility functions for easy access (maintaining compatibility)
def get_crypto_news(limit: int = 10) -> list[CryptoNews]:
    """Quick function to get cryptocurrency news."""
    trader = CryptoTrading()
    return trader.get_crypto_news(limit)


def get_market_sentiment() -> dict[str, Any]:
    """Quick function to get market sentiment from news."""
    trader = CryptoTrading()
    return trader.get_market_sentiment()


def search_crypto_news(query: str, limit: int = 10) -> list[CryptoNews]:
    """Quick function to search cryptocurrency news."""
    trader = CryptoTrading()
    return trader.search_crypto_news(query, limit)


def get_trending_cryptos() -> list[TrendingCoin]:
    """Quick function to get trending cryptocurrencies from news."""
    trader = CryptoTrading()
    return trader.get_trending_coins()


# Compatibility functions (return news-based data instead of price data)
def get_crypto_quote(coin_id: str) -> CryptoQuote | None:
    """Compatibility function - returns mock quote."""
    trader = CryptoTrading()
    return trader.get_crypto_quote(coin_id)


def get_global_crypto_market() -> MarketStats | None:
    """Compatibility function - returns mock market stats."""
    trader = CryptoTrading()
    return trader.get_global_market_data()


# Example usage and testing
if __name__ == "__main__":
    # Initialize the crypto trading module
    crypto_trader = CryptoTrading()

    print("\n=== Testing Cointelegraph News Scraper ===")

    # Get latest crypto news
    print("\n=== Latest Cryptocurrency News ===")
    news = crypto_trader.get_crypto_news(limit=5)
    for i, article in enumerate(news, 1):
        print(f"{i}. {article.title}")
        print(f"   Published: {article.published_date}")
        print(f"   URL: {article.url}")
        if article.summary:
            print(f"   Summary: {article.summary[:100]}...")
        print()

    # Get market sentiment
    print("\n=== Market Sentiment Analysis ===")
    sentiment = crypto_trader.get_market_sentiment()
    if sentiment:
        print(f"Overall Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        print(f"Sentiment Score: {sentiment.get('sentiment_score', 0):.3f}")
        print(f"Positive Articles: {sentiment.get('positive_sentiment', 0)}")
        print(f"Negative Articles: {sentiment.get('negative_sentiment', 0)}")
        print(f"Neutral Articles: {sentiment.get('neutral_sentiment', 0)}")

    # Get trending cryptocurrencies from news
    print("\n=== Trending Cryptocurrencies (from news) ===")
    trending = crypto_trader.get_trending_coins()
    for i, coin in enumerate(trending[:5], 1):
        print(f"{i}. {coin.name} ({coin.symbol}) - Mentioned {coin.score} times")

    # Search for specific crypto news
    print("\n=== Bitcoin News Search ===")
    bitcoin_news = crypto_trader.search_crypto_news("bitcoin", limit=3)
    for i, article in enumerate(bitcoin_news, 1):
        print(f"{i}. {article.title}")
        print(f"   URL: {article.url}")
        print()
