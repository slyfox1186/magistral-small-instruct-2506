#!/usr/bin/env python3
"""Trading API endpoints for crypto and stock data."""

import asyncio
import logging

from fastapi import FastAPI, HTTPException

# Import models and globals
from .models import (
    CryptoQuoteRequest,
    CryptoDataResponse,
    StockQuoteRequest,
    StockDataResponse,
)
from .globals import app_state

logger = logging.getLogger(__name__)


def setup_trading_routes(app: FastAPI):
    """Setup trading API routes."""

    # ===================== Trading API Endpoints =====================
    @app.post("/api/crypto/quotes", response_model=CryptoDataResponse)
    async def get_crypto_quotes(request: CryptoQuoteRequest) -> CryptoDataResponse:
        """Get cryptocurrency quotes for specified coin IDs."""
        if not app_state.crypto_trader:
            raise HTTPException(status_code=503, detail="Cryptocurrency trading service not available")

        try:
            # Get cryptocurrency data with sources
            formatted_data, sources = await asyncio.to_thread(
                app_state.crypto_trader.format_crypto_data_with_sources, request.coin_ids
            )

            return CryptoDataResponse(success=True, data=formatted_data, sources=sources)
        except Exception as e:
            logger.error(f"Error fetching crypto quotes: {e}", exc_info=True)
            return CryptoDataResponse(
                success=False, error=f"Failed to fetch cryptocurrency data: {e!s}"
            )

    @app.post("/api/stocks/quotes", response_model=StockDataResponse)
    async def get_stock_quotes(request: StockQuoteRequest) -> StockDataResponse:
        """Get stock quotes for specified ticker symbols."""
        if not app_state.stock_searcher:
            raise HTTPException(status_code=503, detail="Stock search service not available")

        try:
            # Get stock data with sources
            formatted_data, sources = await asyncio.to_thread(
                app_state.stock_searcher.format_stock_data_with_sources, request.symbols
            )

            return StockDataResponse(success=True, data=formatted_data, sources=sources)
        except Exception as e:
            logger.error(f"Error fetching stock quotes: {e}", exc_info=True)
            return StockDataResponse(success=False, error=f"Failed to fetch stock data: {e!s}")

    @app.get("/api/crypto/trending", response_model=CryptoDataResponse)
    async def get_trending_cryptos() -> CryptoDataResponse:
        """Get trending cryptocurrencies."""
        if not app_state.crypto_trader:
            raise HTTPException(status_code=503, detail="Cryptocurrency trading service not available")

        try:
            trending_coins = await asyncio.to_thread(app_state.crypto_trader.get_trending_coins)

            if not trending_coins:
                return CryptoDataResponse(
                    success=True, data="No trending cryptocurrencies found.", sources=[]
                )

            # Format trending data
            table = "| Rank | Symbol | Name | Market Cap Rank | Score |\n"
            table += "|------|--------|------|----------------|-------|\n"

            sources = []
            for i, coin in enumerate(trending_coins[:10], 1):
                table += (
                    f"| {i} | {coin.symbol} | {coin.name} | #{coin.market_cap_rank} | {coin.score} |\n"
                )
                sources.append(
                    {
                        "id": coin.id,
                        "symbol": coin.symbol,
                        "name": coin.name,
                        "url": f"https://www.coingecko.com/en/coins/{coin.id}",
                        "title": f"{coin.name} ({coin.symbol}) Trending Data",
                        "source": "CoinGecko",
                    }
                )

            return CryptoDataResponse(success=True, data=table, sources=sources)
        except Exception as e:
            logger.error(f"Error fetching trending cryptos: {e}", exc_info=True)
            return CryptoDataResponse(
                success=False, error=f"Failed to fetch trending cryptocurrencies: {e!s}"
            )

    @app.get("/api/crypto/global", response_model=CryptoDataResponse)
    async def get_global_crypto_market() -> CryptoDataResponse:
        """Get global cryptocurrency market data."""
        if not app_state.crypto_trader:
            raise HTTPException(status_code=503, detail="Cryptocurrency trading service not available")

        try:
            global_stats = await asyncio.to_thread(app_state.crypto_trader.get_global_market_data)

            if not global_stats:
                return CryptoDataResponse(
                    success=True, data="Global cryptocurrency market data not available.", sources=[]
                )

            # Format global market data
            table = "| Metric | Value |\n"
            table += "|-----------|-------|\n"
            table += f"| Total Market Cap | ${global_stats.total_market_cap:,.0f} |\n"
            table += f"| 24h Volume | ${global_stats.total_volume_24h:,.0f} |\n"
            table += f"| 24h Change | {global_stats.market_cap_change_24h:+.2f}% |\n"
            table += f"| Active Cryptocurrencies | {global_stats.active_cryptocurrencies:,} |\n"
            table += f"| Markets | {global_stats.markets:,} |\n"

            sources = [
                {
                    "id": "global",
                    "name": "Global Cryptocurrency Market",
                    "url": "https://www.coingecko.com/en/global_charts",
                    "title": "Global Cryptocurrency Market Statistics",
                    "source": "CoinGecko",
                }
            ]

            return CryptoDataResponse(success=True, data=table, sources=sources)
        except Exception as e:
            logger.error(f"Error fetching global crypto market: {e}", exc_info=True)
            return CryptoDataResponse(
                success=False, error=f"Failed to fetch global cryptocurrency market data: {e!s}"
            )