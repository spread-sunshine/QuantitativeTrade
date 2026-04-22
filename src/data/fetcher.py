# Data fetcher module for acquiring market data from various sources
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import logging

from ..utils.logger import setup_logger
from ..utils.cache import CacheManager
from ..utils.retry import retry, NETWORK_RETRY
from config.settings import (
    YAHOO_FETCH_ENABLED,
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_ENABLED,
    USE_CACHE,
    CACHE_EXPIRY_DAYS,
)

logger = setup_logger(__name__)


class DataFetcher:
    """Fetches market data from various sources."""

    def __init__(self, cache_enabled: bool = USE_CACHE):
        """Initialize data fetcher.

        Args:
            cache_enabled: Whether to use caching for fetched data.
        """
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache: Optional[CacheManager] = CacheManager(expiry_days=CACHE_EXPIRY_DAYS)
        else:
            self.cache: Optional[CacheManager] = None

    @retry(
        max_attempts=NETWORK_RETRY.max_attempts,
        delay=NETWORK_RETRY.delay,
        exceptions=NETWORK_RETRY.exceptions,
    )
    def fetch_yahoo(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL').
            start: Start date in 'YYYY-MM-DD' format. If None, defaults to 1 year ago.
            end: End date in 'YYYY-MM-DD' format. If None, defaults to today.
            interval: Data interval ('1d', '1h', '1m', etc.).
            auto_adjust: Whether to adjust for dividends and splits.

        Returns:
            DataFrame with OHLCV data.

        Raises:
            ValueError: If Yahoo Finance fetching is disabled or symbol is invalid.
        """
        if not YAHOO_FETCH_ENABLED:
            raise ValueError("Yahoo Finance fetching is disabled in configuration.")

        # Set default dates if not provided
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        cache_key = f"yahoo_{symbol}_{start}_{end}_{interval}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved {symbol} data from cache")
                return cached_data

        logger.info(f"Fetching {symbol} data from Yahoo Finance ({start} to {end})")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start, end=end, interval=interval, auto_adjust=auto_adjust
            )

            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            # Rename columns to standard format
            df.columns = [col.lower() for col in df.columns]

            # Add symbol column
            df["symbol"] = symbol

            # Reset index to make date a column
            df = df.reset_index()

            # Cache the data
            if self.cache_enabled:
                self.cache.set(cache_key, df)

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of stock symbols.
            start: Start date.
            end: End date.
            interval: Data interval.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_yahoo(symbol, start, end, interval)
                data[symbol] = df
                # Avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        return data

    @retry(
        max_attempts=NETWORK_RETRY.max_attempts,
        delay=NETWORK_RETRY.delay,
        exceptions=NETWORK_RETRY.exceptions,
    )
    def fetch_alpha_vantage(
        self,
        symbol: str,
        function: str = "TIME_SERIES_DAILY",
        output_size: str = "full",
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage.

        Args:
            symbol: Stock symbol.
            function: API function (TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, etc.).
            output_size: 'compact' (latest 100 data points) or 'full' (full-length).

        Returns:
            DataFrame with OHLCV data.

        Raises:
            ValueError: If Alpha Vantage is disabled or API key is missing.
        """
        if not ALPHA_VANTAGE_ENABLED:
            raise ValueError("Alpha Vantage fetching is disabled.")

        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("Alpha Vantage API key is not configured.")

        cache_key = f"av_{symbol}_{function}_{output_size}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved {symbol} data from cache (Alpha Vantage)")
                return cached_data

        logger.info(f"Fetching {symbol} data from Alpha Vantage")

        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": output_size,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "datatype": "json",
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse the response based on function
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.sort_index()
            elif "Time Series (60min)" in data:
                time_series = data["Time Series (60min)"]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.sort_index()
            else:
                raise ValueError(f"Unexpected response format: {list(data.keys())}")

            # Add symbol column
            df["symbol"] = symbol

            # Reset index to make date a column
            df = df.reset_index().rename(columns={"index": "date"})

            # Cache the data
            if self.cache_enabled:
                self.cache.set(cache_key, df)

            logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Alpha Vantage")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            raise

    def get_available_symbols(self, source: str = "yahoo") -> List[str]:
        """Get list of available symbols from a data source.

        Args:
            source: Data source ('yahoo' or 'alpha_vantage').

        Returns:
            List of available symbols.
        """
        # This is a simplified implementation
        # In practice, you would fetch from a known list or API
        from config.settings import DEFAULT_SYMBOLS

        if source == "yahoo":
            return DEFAULT_SYMBOLS
        elif source == "alpha_vantage":
            # Alpha Vantage doesn't provide a symbol list API for free tier
            return DEFAULT_SYMBOLS
        else:
            raise ValueError(f"Unknown source: {source}")

    def validate_symbol(self, symbol: str, source: str = "yahoo") -> bool:
        """Validate if a symbol exists in the data source.

        Args:
            symbol: Symbol to validate.
            source: Data source.

        Returns:
            True if symbol is valid, False otherwise.
        """
        try:
            if source == "yahoo":
                ticker = yf.Ticker(symbol)
                # Try to get some basic info
                info = ticker.info
                return info is not None and len(info) > 0
            elif source == "alpha_vantage":
                # Try to fetch a small amount of data
                df = self.fetch_alpha_vantage(symbol, output_size="compact")
                return not df.empty
            else:
                return False
        except Exception:
            return False