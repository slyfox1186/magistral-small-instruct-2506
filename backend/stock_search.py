#!/usr/bin/env python3
"""Enhanced stock market data retrieval using yfinance.

This module provides comprehensive stock market data functions with improved
error handling, caching, and better use of yfinance features.

Features:
- Real-time stock quotes with caching
- Bulk data downloading for multiple tickers
- Enhanced financial data retrieval
- Better error handling and logging
- Proper use of yfinance Tickers class
- Async-ready architecture

Dependencies:
    pip install yfinance==0.2.28 pandas numpy

Author: Claude Code
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Import ColorCodes for terminal colorization
from colored_logging import ColorCodes

# Configure logging with better format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DURATION = timedelta(minutes=5)
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


@dataclass
class StockQuote:
    """Enhanced data class for stock quote information."""
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
    last_updated: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        if self.last_updated:
            data["last_updated"] = self.last_updated.isoformat()
        return data


@dataclass
class FinancialSummary:
    """Summary of key financial metrics."""
    revenue: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None


class EnhancedStockSearch:
    """Enhanced stock market data retrieval with better caching and error handling."""

    def __init__(self, cache_duration: timedelta = CACHE_DURATION):
        """Initialize with configurable cache duration."""
        self.cache_duration = cache_duration
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._tickers_cache: dict[str, yf.Ticker] = {}
        logger.info("EnhancedStockSearch initialized")

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get cached ticker object."""
        symbol = symbol.upper()
        if symbol not in self._tickers_cache:
            self._tickers_cache[symbol] = yf.Ticker(symbol)
        return self._tickers_cache[symbol]

    def _get_from_cache(self, key: str) -> Any | None:
        """Get data from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set data in cache with timestamp."""
        self._cache[key] = (data, datetime.now())

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear cache for specific symbol or all."""
        if symbol:
            symbol = symbol.upper()
            keys_to_delete = [k for k in self._cache.keys() if symbol in k]
            for key in keys_to_delete:
                del self._cache[key]
            if symbol in self._tickers_cache:
                del self._tickers_cache[symbol]
        else:
            self._cache.clear()
            self._tickers_cache.clear()

    async def get_stock_quote_async(self, symbol: str) -> StockQuote | None:
        """Async wrapper for get_stock_quote."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_quote, symbol)

    def get_stock_quote(self, symbol: str, use_fast_info: bool = True) -> StockQuote | None:
        """Get current stock quote with improved error handling and caching.
        
        Args:
            symbol: Stock ticker symbol
            use_fast_info: Use fast_info for quicker access (when available)
            
        Returns:
            StockQuote object or None if error
        """
        try:
            symbol = symbol.upper()
            cache_key = f"quote_{symbol}"

            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"Returning cached quote for {symbol}")
                return cached_data

            ticker = self._get_ticker(symbol)

            # Try fast_info first if available and requested
            if use_fast_info and hasattr(ticker, "fast_info"):
                try:
                    fast = ticker.fast_info
                    quote = StockQuote(
                        symbol=symbol,
                        name=symbol,  # fast_info doesn't have name
                        price=fast.get("lastPrice", 0),
                        change=fast.get("lastPrice", 0) - fast.get("previousClose", 0),
                        change_percent=((fast.get("lastPrice", 0) - fast.get("previousClose", 0)) /
                                      fast.get("previousClose", 1) * 100),
                        volume=fast.get("lastVolume", 0),
                        market_cap=fast.get("marketCap"),
                        last_updated=datetime.now()
                    )
                    self._set_cache(cache_key, quote)
                    logger.info(f"Retrieved fast quote for {symbol}: ${quote.price:.2f}")
                    return quote
                except Exception as e:
                    logger.warning(f"Fast info failed for {symbol}, falling back to regular info: {e}")

            # Fall back to regular info
            info = ticker.info
            if not info or "currentPrice" not in info:
                # Try to get price from history as last resort
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                    volume = int(hist["Volume"].iloc[-1])
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
                    return None
            else:
                current_price = info.get("currentPrice", 0)
                volume = info.get("volume", 0)

            previous_close = info.get("previousClose", current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0

            quote = StockQuote(
                symbol=symbol,
                name=info.get("longName", info.get("shortName", symbol)),
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                dividend_yield=info.get("dividendYield"),
                day_high=info.get("dayHigh"),
                day_low=info.get("dayLow"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                last_updated=datetime.now()
            )

            self._set_cache(cache_key, quote)
            logger.info(f"Retrieved quote for {symbol}: ${current_price:.2f}")
            return quote

        except Exception as e:
            logger.error(f"Error retrieving quote for {symbol}: {e!s}")
            return None

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, StockQuote]:
        """Get quotes for multiple symbols efficiently using Tickers class.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to StockQuote objects
        """
        try:
            symbols = [s.upper() for s in symbols]
            tickers = yf.Tickers(" ".join(symbols))
            quotes = {}

            for symbol in symbols:
                if symbol in tickers.tickers:
                    ticker = tickers.tickers[symbol]
                    quote = self.get_stock_quote(symbol)
                    if quote:
                        quotes[symbol] = quote

            logger.info(f"Retrieved quotes for {len(quotes)} out of {len(symbols)} symbols")
            return quotes

        except Exception as e:
            logger.error(f"Error retrieving multiple quotes: {e!s}")
            return {}

    def download_historical_data(
        self,
        symbols: str | list[str],
        start: str | None = None,
        end: str | None = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        group_by: str = "ticker",
        auto_adjust: bool = True,
        prepost: bool = False,
        threads: bool = True,
        progress: bool = False
    ) -> pd.DataFrame | None:
        """Download historical data for one or more symbols using yf.download.
        
        This is more efficient than individual ticker.history() calls for multiple symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            group_by: Group by 'ticker' or 'column'
            auto_adjust: Adjust all OHLC automatically
            prepost: Include pre and post market data
            threads: Use threads for mass downloading
            progress: Show progress bar
            
        Returns:
            DataFrame with historical data or None if error
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            symbols = [s.upper() for s in symbols]

            cache_key = f"hist_{'_'.join(symbols)}_{period}_{interval}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Returning cached historical data for {symbols}")
                return cached_data

            # Use yf.download for efficient bulk downloading
            data = yf.download(
                tickers=symbols if len(symbols) > 1 else symbols[0],
                start=start,
                end=end,
                period=period,
                interval=interval,
                group_by=group_by,
                auto_adjust=auto_adjust,
                prepost=prepost,
                threads=threads,
                progress=progress
            )

            if data.empty:
                logger.warning(f"No historical data found for {symbols}")
                return None

            # Add technical indicators if daily data
            if interval in ["1d", "5d", "1wk", "1mo"]:
                data = self._add_technical_indicators(data, symbols)

            self._set_cache(cache_key, data)
            logger.info(f"Downloaded {len(data)} records for {symbols}")
            return data

        except Exception as e:
            logger.error(f"Error downloading historical data: {e!s}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        """Add technical indicators to historical data."""
        try:
            if len(symbols) == 1:
                # Single symbol - simple column access
                if "Close" in data.columns:
                    close = data["Close"]
                    data["Returns"] = close.pct_change()
                    data["SMA_20"] = close.rolling(window=20).mean()
                    data["SMA_50"] = close.rolling(window=50).mean()
                    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
                    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()
                    data["MACD"] = data["EMA_12"] - data["EMA_26"]
                    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
                    data["Volatility"] = data["Returns"].rolling(window=20).std() * np.sqrt(252)
            else:
                # Multiple symbols - multi-level columns
                for symbol in symbols:
                    if ("Close", symbol) in data.columns:
                        close = data["Close"][symbol]
                        data[("Returns", symbol)] = close.pct_change()
                        data[("SMA_20", symbol)] = close.rolling(window=20).mean()
                        data[("SMA_50", symbol)] = close.rolling(window=50).mean()

        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")

        return data

    def get_financial_summary(self, symbol: str) -> FinancialSummary:
        """Get summarized financial metrics for easy consumption.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            FinancialSummary object with key metrics
        """
        try:
            symbol = symbol.upper()
            ticker = self._get_ticker(symbol)
            info = ticker.info

            summary = FinancialSummary(
                revenue=info.get("totalRevenue"),
                revenue_growth=info.get("revenueGrowth"),
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins"),
                roe=info.get("returnOnEquity"),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio")
            )

            logger.info(f"Retrieved financial summary for {symbol}")
            return summary

        except Exception as e:
            logger.error(f"Error retrieving financial summary for {symbol}: {e!s}")
            return FinancialSummary()

    def get_earnings_calendar(self, symbol: str) -> pd.DataFrame | None:
        """Get earnings calendar and estimates.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with earnings dates and estimates
        """
        try:
            symbol = symbol.upper()
            ticker = self._get_ticker(symbol)

            # Get earnings dates
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                logger.info(f"Retrieved {len(earnings_dates)} earnings dates for {symbol}")
                return earnings_dates

            return None

        except Exception as e:
            logger.error(f"Error retrieving earnings calendar for {symbol}: {e!s}")
            return None

    def get_options_chain(self, symbol: str, date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Get options chain data.
        
        Args:
            symbol: Stock ticker symbol
            date: Specific expiration date (optional)
            
        Returns:
            Tuple of (calls, puts) DataFrames or None
        """
        try:
            symbol = symbol.upper()
            ticker = self._get_ticker(symbol)

            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return None

            # Use specified date or first available
            exp_date = date if date in expirations else expirations[0]

            # Get options chain
            opt = ticker.option_chain(exp_date)

            logger.info(f"Retrieved options chain for {symbol} expiring {exp_date}")
            return (opt.calls, opt.puts)

        except Exception as e:
            logger.error(f"Error retrieving options chain for {symbol}: {e!s}")
            return None

    def get_institutional_holders(self, symbol: str) -> pd.DataFrame | None:
        """Get institutional holders with proper error handling.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with institutional holders or None
        """
        try:
            symbol = symbol.upper()
            cache_key = f"inst_holders_{symbol}"

            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

            ticker = self._get_ticker(symbol)
            inst_holders = ticker.institutional_holders

            if inst_holders is not None and not inst_holders.empty:
                self._set_cache(cache_key, inst_holders)
                logger.info(f"Retrieved {len(inst_holders)} institutional holders for {symbol}")
                return inst_holders

            return None

        except Exception as e:
            logger.error(f"Error retrieving institutional holders for {symbol}: {e!s}")
            return None

    def search_symbols(self, query: str, first: int = 10) -> list[dict[str, Any]]:
        """Search for symbols using direct validation - no regex patterns or truncation.
        
        Args:
            query: Search query (should be exact ticker symbol from LLM)
            first: Maximum number of results
            
        Returns:
            List of matching symbols with basic info
        """
        query = query.upper().strip()
        results = []

        # ONLY try the exact query - no truncation, no regex patterns
        # The LLM should provide the correct symbol
        if query:
            try:
                ticker_obj = self._get_ticker(query)
                info = ticker_obj.info

                # Check if we got valid data
                if info and 'symbol' in info:
                    results.append({
                        "symbol": info.get('symbol', query),
                        "name": info.get('longName', info.get('shortName', query)),
                        "type": "stock",
                        "exchange": info.get('exchange', 'Unknown')
                    })
                    logger.info(f"✅ Found valid symbol: {query} - {info.get('longName', 'Unknown')}")
                else:
                    logger.warning(f"❌ Symbol {query} returned no valid data")

            except Exception as e:
                logger.warning(f"❌ Symbol {query} validation failed: {e}")

        logger.info(f"Found {len(results)} results for query: {query}")
        return results

    def validate_symbols(self, symbols: list[str]) -> dict[str, bool]:
        """Validate if symbols exist and have data.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbols to validity status
        """
        results = {}

        for symbol in symbols:
            try:
                symbol_upper = symbol.upper()
                logger.debug(f"Validating symbol: {symbol_upper}")
                ticker = self._get_ticker(symbol_upper)
                info = ticker.info

                # Check if we got valid data
                is_valid = bool(info and len(info) > 1 and 'symbol' in info)
                results[symbol] = is_valid

                if is_valid:
                    logger.info(f"✅ Symbol {symbol_upper} is valid: {info.get('longName', info.get('shortName', 'Unknown'))}")
                else:
                    logger.warning(f"❌ Symbol {symbol_upper} returned no valid data")

            except Exception as e:
                logger.warning(f"❌ Symbol {symbol} validation failed: {e}")
                results[symbol] = False

        valid_count = sum(results.values())
        logger.info(f"Validated {valid_count} out of {len(symbols)} symbols: {results}")
        return results

    def format_stock_data_with_sources(self, symbols: list[str]) -> tuple[str, list[dict[str, str]]]:
        """Format stock data as a colorized markdown table with sources.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Tuple of (formatted_markdown_table, sources_list)
        """
        try:
            logger.info(f"📊 Formatting stock data for symbols: {symbols}")

            # Get quotes for all symbols
            quotes = self.get_multiple_quotes(symbols)

            logger.info(f"📊 Retrieved quotes for {len(quotes)} symbols: {list(quotes.keys())}")

            if not quotes:
                return "No stock data available for the requested symbols.", []

            # Build colorized markdown table with headers
            table_lines = [
                f"| {ColorCodes.BOLD}{ColorCodes.BLUE}Symbol{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}Name{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}Price{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}Change{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}% Change{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}Volume{ColorCodes.RESET} | {ColorCodes.BOLD}{ColorCodes.BLUE}Market Cap{ColorCodes.RESET} |",
                "|:---------|:---------------|:------------|:-----------|:-------------|:----------------|:---------------|"
            ]

            sources = []

            for symbol, quote in quotes.items():
                if quote:
                    # Color symbol in bright yellow
                    colored_symbol = f"{ColorCodes.BRIGHT_YELLOW}{symbol}{ColorCodes.RESET}"

                    # Color price in bold white
                    price_str = f"{ColorCodes.BOLD}{ColorCodes.WHITE}${quote.price:.2f}{ColorCodes.RESET}"

                    # Color change based on positive/negative
                    if quote.change > 0:
                        change_str = f"{ColorCodes.GREEN}+${quote.change:.2f}{ColorCodes.RESET}"
                        pct_str = f"{ColorCodes.GREEN}+{quote.change_percent:.2f}%{ColorCodes.RESET}"
                    elif quote.change < 0:
                        change_str = f"{ColorCodes.RED}${quote.change:.2f}{ColorCodes.RESET}"
                        pct_str = f"{ColorCodes.RED}{quote.change_percent:.2f}%{ColorCodes.RESET}"
                    else:
                        change_str = f"{ColorCodes.BRIGHT_BLACK}$0.00{ColorCodes.RESET}"
                        pct_str = f"{ColorCodes.BRIGHT_BLACK}0.00%{ColorCodes.RESET}"

                    # Color volume in cyan
                    if quote.volume:
                        volume_str = f"{ColorCodes.CYAN}{quote.volume:,}{ColorCodes.RESET}"
                    else:
                        volume_str = f"{ColorCodes.BRIGHT_BLACK}N/A{ColorCodes.RESET}"

                    # Format and color market cap in magenta
                    if quote.market_cap:
                        if quote.market_cap >= 1e12:
                            mcap_str = f"{ColorCodes.MAGENTA}${quote.market_cap/1e12:.2f} Trillion{ColorCodes.RESET}"
                        elif quote.market_cap >= 1e9:
                            mcap_str = f"{ColorCodes.MAGENTA}${quote.market_cap/1e9:.2f} Billion{ColorCodes.RESET}"
                        elif quote.market_cap >= 1e6:
                            mcap_str = f"{ColorCodes.MAGENTA}${quote.market_cap/1e6:.2f} Million{ColorCodes.RESET}"
                        else:
                            mcap_str = f"{ColorCodes.MAGENTA}${quote.market_cap:,.0f}{ColorCodes.RESET}"
                    else:
                        mcap_str = f"{ColorCodes.BRIGHT_BLACK}N/A{ColorCodes.RESET}"

                    # Add table row with proper alignment
                    table_lines.append(
                        f"| {colored_symbol}     | {quote.name[:30]} | {price_str} | "
                        f"{change_str}   | {pct_str}     | {volume_str}    | {mcap_str}|"
                    )

                    # Add source
                    sources.append({
                        "symbol": symbol,
                        "name": quote.name,
                        "url": f"https://finance.yahoo.com/quote/{symbol}",
                        "title": f"{quote.name} ({symbol}) Stock Quote",
                        "source": "Yahoo Finance"
                    })

            formatted_data = "\n".join(table_lines)

            # Add colorized summary statistics
            if len(quotes) > 1:
                avg_change = sum(q.change_percent for q in quotes.values() if q) / len(quotes)
                if avg_change > 0:
                    avg_color = ColorCodes.GREEN
                    avg_prefix = "+"
                elif avg_change < 0:
                    avg_color = ColorCodes.RED
                    avg_prefix = ""
                else:
                    avg_color = ColorCodes.BRIGHT_BLACK
                    avg_prefix = ""

                formatted_data += f"\n\n**Average Change**: {avg_color}{avg_prefix}{avg_change:.2f}%{ColorCodes.RESET}"

            return formatted_data, sources

        except Exception as e:
            logger.error(f"Error formatting stock data: {e}")
            return f"Error formatting stock data: {e!s}", []


# Singleton instance for easy import
stock_search = EnhancedStockSearch()

# Convenience functions using the singleton
def get_quote(symbol: str) -> StockQuote | None:
    """Get stock quote using singleton instance."""
    return stock_search.get_stock_quote(symbol)

def get_quotes(symbols: list[str]) -> dict[str, StockQuote]:
    """Get multiple stock quotes efficiently."""
    return stock_search.get_multiple_quotes(symbols)

def download_data(
    symbols: str | list[str],
    period: str = "1mo",
    interval: str = "1d"
) -> pd.DataFrame | None:
    """Download historical data for symbols."""
    return stock_search.download_historical_data(symbols, period=period, interval=interval)

def get_financials(symbol: str) -> FinancialSummary:
    """Get financial summary for a symbol."""
    return stock_search.get_financial_summary(symbol)

def search_stocks(query: str) -> list[dict[str, Any]]:
    """Search for stock symbols."""
    return stock_search.search_symbols(query)

def validate_symbol(symbol: str) -> bool:
    """Check if a symbol is valid."""
    result = stock_search.validate_symbols([symbol])
    return result.get(symbol, False)

def format_stock_data_with_sources(symbols: list[str]) -> tuple[str, list[dict[str, str]]]:
    """Format stock data with colorized sources for API responses.
    
    Args:
        symbols: List of stock ticker symbols
        
    Returns:
        Tuple of (colorized_formatted_data_string, sources_list)
    """
    return stock_search.format_stock_data_with_sources(symbols)


# Example usage
if __name__ == "__main__":
    # Test the enhanced functionality
    print("=== Enhanced Stock Search Test ===\n")

    # Test single quote
    quote = get_quote("AAPL")
    if quote:
        print(f"Apple Quote: ${quote.price:.2f} ({quote.change_percent:+.2f}%)")
        print(f"Last Updated: {quote.last_updated}\n")

    # Test multiple quotes
    symbols = ["MSFT", "GOOGL", "TSLA"]
    quotes = get_quotes(symbols)
    print(f"Multiple Quotes ({len(quotes)} results):")
    for symbol, quote in quotes.items():
        print(f"  {symbol}: ${quote.price:.2f}")

    # Test bulk download
    print("\n=== Testing Bulk Download ===")
    data = download_data(["AAPL", "MSFT"], period="1mo", interval="1d")
    if data is not None:
        print(f"Downloaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns[:5])}...")

    # Test search
    print("\n=== Testing Search ===")
    results = search_stocks("apple")
    for result in results[:3]:
        print(f"  {result['symbol']}: {result['name']}")
