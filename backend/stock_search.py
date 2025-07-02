#!/usr/bin/env python3
"""High-quality stock market data retrieval using yfinance.

This module provides comprehensive stock market data functions including:
- Real-time stock quotes and historical data
- Financial statements and ratios
- Market analysis and recommendations
- Insider trading and institutional holdings
- Market trends and sector analysis

Dependencies:
    pip install yfinance pandas numpy

Author: Claude Code
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """Data class for stock quote information."""

    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None


@dataclass
class MarketSummary:
    """Data class for market summary information."""

    symbol: str
    name: str
    current_price: float
    day_change: float
    day_change_percent: float
    volume: int
    avg_volume: int | None = None
    beta: float | None = None
    market_cap: float | None = None


class StockSearch:
    """Comprehensive stock market data retrieval and analysis class.

    This class provides methods to fetch various types of stock market data
    including quotes, historical data, financial statements, and market analysis.
    """

    def __init__(self):
        """Initialize the StockSearch class."""
        self.cache = {}
        logger.info("StockSearch initialized")

    def get_stock_quote(self, symbol: str) -> StockQuote | None:
        """Get current stock quote for a symbol.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')

        Returns:
            Optional[StockQuote]: Stock quote data or None if error
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            if not info or "currentPrice" not in info:
                logger.warning(f"No data found for symbol: {symbol}")
                return None

            # Calculate change and change percent
            current_price = info.get("currentPrice", 0)
            previous_close = info.get("previousClose", current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0

            quote = StockQuote(
                symbol=symbol.upper(),
                name=info.get("longName", info.get("shortName", symbol)),
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=info.get("volume", 0),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                dividend_yield=info.get("dividendYield"),
                day_high=info.get("dayHigh"),
                day_low=info.get("dayLow"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            )

            logger.info(f"Retrieved quote for {symbol}: ${current_price:.2f}")
            return quote

        except Exception as e:
            logger.error(f"Error retrieving quote for {symbol}: {e!s}")
            return None

    def get_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """Get historical stock price data.

        Args:
            symbol (str): Stock ticker symbol
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d,
                           1wk, 1mo, 3mo)

        Returns:
            Optional[pd.DataFrame]: Historical price data or None if error
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None

            # Add technical indicators
            data["Returns"] = data["Close"].pct_change()
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["Volatility"] = data["Returns"].rolling(window=20).std() * np.sqrt(252)

            logger.info(f"Retrieved {len(data)} historical records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e!s}")
            return None

    def get_financial_data(self, symbol: str) -> dict[str, Any]:
        """Get comprehensive financial data for a stock.

        Args:
            symbol (str): Stock ticker symbol

        Returns:
            Dict[str, Any]: Financial data including statements and ratios
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            financial_data = {
                "symbol": symbol.upper(),
                "info": ticker.info,
                "income_statement": None,
                "balance_sheet": None,
                "cash_flow": None,
                "earnings": None,
                "dividends": None,
                "splits": None,
            }

            # Get financial statements
            try:
                financial_data["income_statement"] = (
                    ticker.financials.to_dict() if not ticker.financials.empty else None
                )
            except Exception:
                pass

            try:
                financial_data["balance_sheet"] = (
                    ticker.balance_sheet.to_dict() if not ticker.balance_sheet.empty else None
                )
            except Exception:
                pass

            try:
                financial_data["cash_flow"] = (
                    ticker.cashflow.to_dict() if not ticker.cashflow.empty else None
                )
            except Exception:
                pass

            try:
                financial_data["earnings"] = (
                    ticker.earnings.to_dict() if not ticker.earnings.empty else None
                )
            except Exception:
                pass

            try:
                dividends = ticker.dividends
                if not dividends.empty:
                    financial_data["dividends"] = dividends.tail(10).to_dict()
            except Exception:
                pass

            try:
                splits = ticker.splits
                if not splits.empty:
                    financial_data["splits"] = splits.to_dict()
            except Exception:
                pass

            logger.info(f"Retrieved financial data for {symbol}")
            return financial_data

        except Exception as e:
            logger.error(f"Error retrieving financial data for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_analyst_data(self, symbol: str) -> dict[str, Any]:
        """Get analyst recommendations and price targets.

        Args:
            symbol (str): Stock ticker symbol

        Returns:
            Dict[str, Any]: Analyst data including recommendations and targets
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            analyst_data = {
                "symbol": symbol.upper(),
                "recommendations": None,
                "price_targets": None,
                "upgrades_downgrades": None,
                "earnings_estimate": None,
            }

            # Get recommendations
            try:
                recommendations = ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    analyst_data["recommendations"] = recommendations.tail(10).to_dict("records")
            except Exception:
                pass

            # Get analyst price targets (if available)
            try:
                if hasattr(ticker, "analyst_price_targets"):
                    analyst_data["price_targets"] = ticker.analyst_price_targets
            except Exception:
                pass

            # Get upgrades/downgrades
            try:
                if hasattr(ticker, "upgrades_downgrades"):
                    upgrades = ticker.upgrades_downgrades
                    if upgrades is not None and not upgrades.empty:
                        analyst_data["upgrades_downgrades"] = upgrades.tail(10).to_dict("records")
            except Exception:
                pass

            logger.info(f"Retrieved analyst data for {symbol}")
            return analyst_data

        except Exception as e:
            logger.error(f"Error retrieving analyst data for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_insider_data(self, symbol: str) -> dict[str, Any]:
        """Get insider trading and institutional holdings data.

        Args:
            symbol (str): Stock ticker symbol

        Returns:
            Dict[str, Any]: Insider and institutional data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            insider_data = {
                "symbol": symbol.upper(),
                "institutional_holders": None,
                "major_holders": None,
                "insider_transactions": None,
            }

            # Get institutional holders
            try:
                institutional = ticker.institutional_holders
                if institutional is not None and not institutional.empty:
                    insider_data["institutional_holders"] = institutional.to_dict("records")
            except Exception:
                pass

            # Get major holders
            try:
                major = ticker.major_holders
                if major is not None and not major.empty:
                    insider_data["major_holders"] = major.to_dict("records")
            except Exception:
                pass

            # Get insider transactions
            try:
                if hasattr(ticker, "insider_transactions"):
                    transactions = ticker.insider_transactions
                    if transactions is not None and not transactions.empty:
                        insider_data["insider_transactions"] = transactions.tail(20).to_dict(
                            "records"
                        )
            except Exception:
                pass

            logger.info(f"Retrieved insider data for {symbol}")
            return insider_data

        except Exception as e:
            logger.error(f"Error retrieving insider data for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_market_data(self, symbols: list[str]) -> list[MarketSummary]:
        """Get market summary data for multiple symbols.

        Args:
            symbols (List[str]): List of stock ticker symbols

        Returns:
            List[MarketSummary]: List of market summary data
        """
        summaries = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol.upper())
                info = ticker.info

                if not info or "currentPrice" not in info:
                    continue

                current_price = info.get("currentPrice", 0)
                previous_close = info.get("previousClose", current_price)
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close != 0 else 0

                summary = MarketSummary(
                    symbol=symbol.upper(),
                    name=info.get("longName", info.get("shortName", symbol)),
                    current_price=current_price,
                    day_change=change,
                    day_change_percent=change_percent,
                    volume=info.get("volume", 0),
                    avg_volume=info.get("averageVolume"),
                    beta=info.get("beta"),
                    market_cap=info.get("marketCap"),
                )

                summaries.append(summary)

            except Exception as e:
                logger.error(f"Error retrieving market data for {symbol}: {e!s}")
                continue

        logger.info(f"Retrieved market data for {len(summaries)} symbols")
        return summaries

    def search_stocks(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Search for stocks by name or symbol (basic implementation).

        Args:
            query (str): Search query
            limit (int): Maximum number of results

        Returns:
            List[Dict[str, str]]: List of matching stocks
        """
        # This is a basic implementation - for production, you might want to use
        # a dedicated stock search API or maintain a symbol database
        common_symbols = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "NFLX", "name": "Netflix Inc."},
            {"symbol": "BABA", "name": "Alibaba Group"},
            {"symbol": "V", "name": "Visa Inc."},
        ]

        query_lower = query.lower()
        results = []

        for stock in common_symbols:
            if query_lower in stock["symbol"].lower() or query_lower in stock["name"].lower():
                results.append(stock)
                if len(results) >= limit:
                    break

        return results

    def get_dividends_and_splits(self, symbol: str, period: str = "5y") -> dict[str, Any]:
        """Get dividend and stock split history.

        Args:
            symbol (str): Stock ticker symbol
            period (str): Period for historical data

        Returns:
            Dict[str, Any]: Dividend and split data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            result = {
                "symbol": symbol.upper(),
                "dividends": None,
                "splits": None,
                "actions": None,
                "dividend_yield": None,
                "dividend_rate": None,
                "ex_dividend_date": None,
            }

            # Get dividend info from info dict
            info = ticker.info
            if info:
                result["dividend_yield"] = info.get("dividendYield")
                result["dividend_rate"] = info.get("dividendRate")
                result["ex_dividend_date"] = info.get("exDividendDate")

            # Get dividend history
            try:
                dividends = ticker.dividends
                if dividends is not None and not dividends.empty:
                    # Get last 10 dividends
                    recent_dividends = dividends.tail(10)
                    result["dividends"] = [
                        {"date": date.strftime("%Y-%m-%d"), "amount": float(amount)}
                        for date, amount in recent_dividends.items()
                    ]
            except Exception:
                pass

            # Get split history
            try:
                splits = ticker.splits
                if splits is not None and not splits.empty:
                    result["splits"] = [
                        {"date": date.strftime("%Y-%m-%d"), "ratio": float(ratio)}
                        for date, ratio in splits.items()
                    ]
            except Exception:
                pass

            # Get combined actions
            try:
                actions = ticker.actions
                if actions is not None and not actions.empty:
                    result["actions"] = actions.tail(20).to_dict("records")
            except Exception:
                pass

            logger.info(f"Retrieved dividend/split data for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving dividend/split data for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_company_info(self, symbol: str, fast: bool = False) -> dict[str, Any]:
        """Get comprehensive company information.

        Args:
            symbol (str): Stock ticker symbol
            fast (bool): Use fast_info for quicker access

        Returns:
            Dict[str, Any]: Company information
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            if fast and hasattr(ticker, "fast_info"):
                # Use fast_info for quick access
                fast_info = ticker.fast_info
                return {"symbol": symbol.upper(), "fast_info": True, "data": fast_info}
            else:
                # Get comprehensive info
                info = ticker.info

                # Extract key information
                company_data = {
                    "symbol": symbol.upper(),
                    "name": info.get("longName", info.get("shortName", symbol)),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "website": info.get("website"),
                    "description": info.get("longBusinessSummary"),
                    "employees": info.get("fullTimeEmployees"),
                    "headquarters": {
                        "city": info.get("city"),
                        "state": info.get("state"),
                        "country": info.get("country"),
                        "address": info.get("address1"),
                    },
                    "officers": info.get("companyOfficers", []),
                    "market_data": {
                        "market_cap": info.get("marketCap"),
                        "enterprise_value": info.get("enterpriseValue"),
                        "shares_outstanding": info.get("sharesOutstanding"),
                        "float_shares": info.get("floatShares"),
                        "beta": info.get("beta"),
                        "52_week_high": info.get("fiftyTwoWeekHigh"),
                        "52_week_low": info.get("fiftyTwoWeekLow"),
                    },
                    "financials": {
                        "revenue": info.get("totalRevenue"),
                        "gross_profit": info.get("grossProfits"),
                        "ebitda": info.get("ebitda"),
                        "net_income": info.get("netIncomeToCommon"),
                        "profit_margin": info.get("profitMargins"),
                        "operating_margin": info.get("operatingMargins"),
                    },
                    "ratios": {
                        "pe_ratio": info.get("trailingPE"),
                        "forward_pe": info.get("forwardPE"),
                        "peg_ratio": info.get("pegRatio"),
                        "price_to_book": info.get("priceToBook"),
                        "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                        "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
                    },
                }

                logger.info(f"Retrieved company info for {symbol}")
                return company_data

        except Exception as e:
            logger.error(f"Error retrieving company info for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_news(self, symbol: str, count: int = 10) -> list[dict[str, Any]]:
        """Get recent news for a stock.

        Args:
            symbol (str): Stock ticker symbol
            count (int): Number of news items to retrieve

        Returns:
            List[Dict[str, Any]]: List of news articles
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            # Get news
            news = ticker.news

            if news:
                news_items = []
                for article in news[:count]:
                    news_item = {
                        "title": article.get("title"),
                        "publisher": article.get("publisher"),
                        "link": article.get("link"),
                        "published": article.get("providerPublishTime"),
                        "type": article.get("type", "news"),
                        "thumbnail": (
                            article.get("thumbnail", {}).get("resolutions", [{}])[0].get("url")
                            if article.get("thumbnail")
                            else None
                        ),
                        "related_tickers": article.get("relatedTickers", []),
                    }

                    # Convert timestamp to readable format
                    if news_item["published"]:
                        from datetime import datetime

                        news_item["published_date"] = datetime.fromtimestamp(
                            news_item["published"]
                        ).strftime("%Y-%m-%d %H:%M:%S")

                    news_items.append(news_item)

                logger.info(f"Retrieved {len(news_items)} news items for {symbol}")
                return news_items
            else:
                logger.warning(f"No news found for {symbol}")
                return []

        except Exception as e:
            logger.error(f"Error retrieving news for {symbol}: {e!s}")
            return []

    def get_shares_history(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> dict[str, Any]:
        """Get historical share count data.

        Args:
            symbol (str): Stock ticker symbol
            start (str): Start date (YYYY-MM-DD)
            end (str): End date (YYYY-MM-DD)

        Returns:
            Dict[str, Any]: Share count history
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            result = {
                "symbol": symbol.upper(),
                "shares_outstanding": None,
                "float_shares": None,
                "shares_history": None,
            }

            # Get current shares from info
            info = ticker.info
            if info:
                result["shares_outstanding"] = info.get("sharesOutstanding")
                result["float_shares"] = info.get("floatShares")

            # Get shares history if available
            try:
                if hasattr(ticker, "get_shares_full"):
                    shares_history = ticker.get_shares_full(start=start, end=end)
                    if shares_history is not None and not shares_history.empty:
                        result["shares_history"] = [
                            {"date": date.strftime("%Y-%m-%d"), "shares": int(shares)}
                            for date, shares in shares_history.items()
                        ]
            except Exception:
                pass

            logger.info(f"Retrieved shares history for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving shares history for {symbol}: {e!s}")
            return {"symbol": symbol.upper(), "error": str(e)}

    def get_technical_indicators(self, symbol: str, period: str = "6mo") -> dict[str, Any]:
        """Calculate technical indicators for a stock.

        Args:
            symbol (str): Stock ticker symbol
            period (str): Data period for calculation

        Returns:
            Dict[str, Any]: Technical indicators
        """
        try:
            data = self.get_historical_data(symbol, period=period)
            if data is None or data.empty:
                return {"symbol": symbol, "error": "No data available"}

            # Calculate additional technical indicators
            close_prices = data["Close"]

            # RSI calculation
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            # MACD calculation
            def calculate_macd(prices, fast=12, slow=26, signal=9):
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal).mean()
                histogram = macd_line - signal_line
                return macd_line, signal_line, histogram

            # Bollinger Bands
            def calculate_bollinger_bands(prices, window=20, num_std=2):
                sma = prices.rolling(window=window).mean()
                std = prices.rolling(window=window).std()
                upper_band = sma + (std * num_std)
                lower_band = sma - (std * num_std)
                return upper_band, lower_band, sma

            # Calculate indicators
            rsi = calculate_rsi(close_prices)
            macd_line, signal_line, histogram = calculate_macd(close_prices)
            upper_bb, lower_bb, middle_bb = calculate_bollinger_bands(close_prices)

            # Get latest values
            latest_data = {
                "symbol": symbol.upper(),
                "timestamp": data.index[-1].isoformat(),
                "current_price": float(close_prices.iloc[-1]),
                "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                "macd": {
                    "macd_line": (
                        float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
                    ),
                    "signal_line": (
                        float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
                    ),
                    "histogram": (
                        float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                    ),
                },
                "bollinger_bands": {
                    "upper": float(upper_bb.iloc[-1]) if not pd.isna(upper_bb.iloc[-1]) else None,
                    "middle": (
                        float(middle_bb.iloc[-1]) if not pd.isna(middle_bb.iloc[-1]) else None
                    ),
                    "lower": float(lower_bb.iloc[-1]) if not pd.isna(lower_bb.iloc[-1]) else None,
                },
                "moving_averages": {
                    "sma_20": (
                        float(data["SMA_20"].iloc[-1])
                        if not pd.isna(data["SMA_20"].iloc[-1])
                        else None
                    ),
                    "sma_50": (
                        float(data["SMA_50"].iloc[-1])
                        if not pd.isna(data["SMA_50"].iloc[-1])
                        else None
                    ),
                },
                "volatility": (
                    float(data["Volatility"].iloc[-1])
                    if not pd.isna(data["Volatility"].iloc[-1])
                    else None
                ),
            }

            logger.info(f"Calculated technical indicators for {symbol}")
            return latest_data

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e!s}")
            return {"symbol": symbol, "error": str(e)}

    def get_yahoo_finance_url(self, symbol: str) -> str:
        """Generate Yahoo Finance URL for a given stock symbol.

        Args:
            symbol (str): Stock ticker symbol

        Returns:
            str: Yahoo Finance URL for the symbol
        """
        # Clean up symbol (remove special characters for indices)
        symbol.replace("^", "")
        return f"https://finance.yahoo.com/quote/{symbol}"

    def get_yahoo_finance_urls(self, symbols: list[str]) -> dict[str, str]:
        """Generate Yahoo Finance URLs for multiple symbols.

        Args:
            symbols (List[str]): List of stock ticker symbols

        Returns:
            Dict[str, str]: Dictionary mapping symbols to their Yahoo Finance URLs
        """
        urls = {}
        for symbol in symbols:
            urls[symbol] = self.get_yahoo_finance_url(symbol)
        return urls

    def format_stock_data_with_sources(
        self, symbols: list[str]
    ) -> tuple[str, list[dict[str, str]]]:
        """Format stock data with proper source URLs for citations.

        Args:
            symbols (List[str]): List of stock symbols to format

        Returns:
            Tuple[str, List[Dict[str, str]]]: Formatted data and list of sources
        """
        quotes = []
        sources = []

        for symbol in symbols:
            quote = self.get_stock_quote(symbol)
            if quote:
                quotes.append(quote)
                # Create source entry
                sources.append(
                    {
                        "symbol": symbol,
                        "name": quote.name,
                        "url": self.get_yahoo_finance_url(symbol),
                        "title": f"{quote.name} ({symbol}) Stock Quote",
                        "source": "Yahoo Finance",
                    }
                )

        # Format the data
        if not quotes:
            return "No stock data available.", []

        # Build markdown table
        table = "| Symbol | Company | Price | Change | Volume | Market Cap |\n"
        table += "|--------|---------|-------|--------|--------|------------|\n"

        for quote in quotes:
            price = f"${quote.price:.2f}"
            change = f"{quote.change:+.2f} ({quote.change_percent:+.1f}%)"

            # Format volume
            if quote.volume >= 1_000_000_000:
                volume = f"{quote.volume / 1_000_000_000:.1f}B"
            elif quote.volume >= 1_000_000:
                volume = f"{quote.volume / 1_000_000:.1f}M"
            else:
                volume = f"{quote.volume:,}"

            # Format market cap
            mcap = "N/A"
            if quote.market_cap:
                if quote.market_cap >= 1_000_000_000_000:
                    mcap = f"${quote.market_cap / 1_000_000_000_000:.2f}T"
                elif quote.market_cap >= 1_000_000_000:
                    mcap = f"${quote.market_cap / 1_000_000_000:.1f}B"
                else:
                    mcap = f"${quote.market_cap / 1_000_000:.1f}M"

            table += f"| {quote.symbol} | {quote.name} | {price} | {change} | {volume} | {mcap} |\n"

        return table, sources


# Utility functions for easy access
def get_quote(symbol: str) -> StockQuote | None:
    """Quick function to get a stock quote."""
    searcher = StockSearch()
    return searcher.get_stock_quote(symbol)


def get_history(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """Quick function to get historical data."""
    searcher = StockSearch()
    return searcher.get_historical_data(symbol, period)


def get_financials(symbol: str) -> dict[str, Any]:
    """Quick function to get financial data."""
    searcher = StockSearch()
    return searcher.get_financial_data(symbol)


def get_technicals(symbol: str) -> dict[str, Any]:
    """Quick function to get technical indicators."""
    searcher = StockSearch()
    return searcher.get_technical_indicators(symbol)


def get_dividends(symbol: str) -> dict[str, Any]:
    """Quick function to get dividend and split data."""
    searcher = StockSearch()
    return searcher.get_dividends_and_splits(symbol)


def get_company(symbol: str) -> dict[str, Any]:
    """Quick function to get company information."""
    searcher = StockSearch()
    return searcher.get_company_info(symbol)


def get_stock_news(symbol: str, count: int = 5) -> list[dict[str, Any]]:
    """Quick function to get stock news."""
    searcher = StockSearch()
    return searcher.get_news(symbol, count)


def get_stock_url(symbol: str) -> str:
    """Quick function to get Yahoo Finance URL for a stock."""
    searcher = StockSearch()
    return searcher.get_yahoo_finance_url(symbol)


def get_stock_urls(symbols: list[str]) -> dict[str, str]:
    """Quick function to get Yahoo Finance URLs for multiple stocks."""
    searcher = StockSearch()
    return searcher.get_yahoo_finance_urls(symbols)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the stock search
    stock_search = StockSearch()

    # Test with Apple stock
    symbol = "AAPL"
    print(f"\n=== Testing Stock Search with {symbol} ===")

    # Get quote
    quote = stock_search.get_stock_quote(symbol)
    if quote:
        print(f"\nQuote: {quote.name} ({quote.symbol})")
        print(f"Price: ${quote.price:.2f} ({quote.change:+.2f}, {quote.change_percent:+.2f}%)")
        print(f"Volume: {quote.volume:,}")
        if quote.market_cap:
            print(f"Market Cap: ${quote.market_cap:,.0f}")

    # Get technical indicators
    print(f"\n=== Technical Indicators for {symbol} ===")
    tech_data = stock_search.get_technical_indicators(symbol)
    if "error" not in tech_data:
        print(f"RSI: {tech_data.get('rsi', 'N/A')}")
        print(f"Current Price: ${tech_data.get('current_price', 0):.2f}")
        if tech_data.get("moving_averages"):
            ma = tech_data["moving_averages"]
            print(f"SMA 20: ${ma.get('sma_20', 0):.2f}")
            print(f"SMA 50: ${ma.get('sma_50', 0):.2f}")

    # Test market data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    print(f"\n=== Market Summary for {', '.join(symbols)} ===")
    market_data = stock_search.get_market_data(symbols)
    for summary in market_data:
        print(
            f"{summary.symbol}: ${summary.current_price:.2f} ({summary.day_change_percent:+.2f}%)"
        )
