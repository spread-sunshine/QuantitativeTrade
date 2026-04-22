# Data processing and feature engineering module
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    """Processes market data and creates features for trading strategies."""

    def __init__(self):
        """Initialize data processor."""
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw market data.

        Args:
            df: Raw DataFrame with market data.

        Returns:
            Cleaned DataFrame.
        """
        if df.empty:
            return df

        df_clean = df.copy()

        # Ensure index is datetime
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            if "date" in df_clean.columns:
                df_clean.set_index("date", inplace=True)
            elif "Date" in df_clean.columns:
                df_clean.set_index("Date", inplace=True)
            else:
                raise ValueError("DataFrame must have a date column or index")

        # Sort by date
        df_clean.sort_index(inplace=True)

        # Handle missing values
        # Forward fill for OHLC, zero for volume
        ohlc_columns = ["open", "high", "low", "close", "adj_close"]
        for col in ohlc_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill()

        if "volume" in df_clean.columns:
            df_clean["volume"] = df_clean["volume"].fillna(0)

        # Remove rows with any remaining NaN in OHLC
        df_clean = df_clean.dropna(subset=ohlc_columns)

        # Remove duplicate dates (keep last)
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

        logger.info(f"Cleaned data: {len(df_clean)} rows after cleaning")
        return df_clean

    def add_returns(self, df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """Add returns to DataFrame.

        Args:
            df: DataFrame with price data.
            column: Column to calculate returns from.

        Returns:
            DataFrame with added returns columns.
        """
        if df.empty:
            return df

        df_result = df.copy()

        # Simple returns
        df_result["returns"] = df_result[column].pct_change()

        # Log returns
        df_result["log_returns"] = np.log(df_result[column] / df_result[column].shift(1))

        # Cumulative returns
        df_result["cumulative_returns"] = (1 + df_result["returns"]).cumprod() - 1

        return df_result

    def add_moving_averages(
        self, df: pd.DataFrame, windows: List[int] = None, column: str = "close"
    ) -> pd.DataFrame:
        """Add moving averages to DataFrame.

        Args:
            df: DataFrame with price data.
            windows: List of window sizes for moving averages.
            column: Column to calculate moving averages on.

        Returns:
            DataFrame with added moving average columns.
        """
        if df.empty:
            return df

        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]

        df_result = df.copy()

        for window in windows:
            col_name = f"ma_{window}"
            df_result[col_name] = df_result[column].rolling(window=window).mean()

        return df_result

    def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volatility measures to DataFrame.

        Args:
            df: DataFrame with returns data.
            window: Rolling window for volatility calculation.

        Returns:
            DataFrame with added volatility columns.
        """
        if df.empty:
            return df

        df_result = df.copy()

        # Ensure returns column exists
        if "returns" not in df_result.columns:
            df_result = self.add_returns(df_result)

        # Rolling standard deviation (historical volatility)
        df_result[f"volatility_{window}"] = (
            df_result["returns"].rolling(window=window).std() * np.sqrt(252)
        )

        # Rolling average true range (ATR)
        if all(col in df_result.columns for col in ["high", "low", "close"]):
            df_result["tr"] = np.maximum(
                df_result["high"] - df_result["low"],
                np.maximum(
                    abs(df_result["high"] - df_result["close"].shift(1)),
                    abs(df_result["low"] - df_result["close"].shift(1)),
                ),
            )
            df_result[f"atr_{window}"] = df_result["tr"].rolling(window=window).mean()

        return df_result

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with added technical indicators.
        """
        if df.empty:
            return df

        df_result = df.copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df_result.columns]
        if missing_cols:
            logger.warning(f"Missing columns for technical indicators: {missing_cols}")
            return df_result

        # Relative Strength Index (RSI)
        delta = df_result["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_result["rsi"] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        ema_12 = df_result["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_result["close"].ewm(span=26, adjust=False).mean()
        df_result["macd"] = ema_12 - ema_26
        df_result["macd_signal"] = df_result["macd"].ewm(span=9, adjust=False).mean()
        df_result["macd_hist"] = df_result["macd"] - df_result["macd_signal"]

        # Bollinger Bands
        df_result["bb_middle"] = df_result["close"].rolling(window=20).mean()
        bb_std = df_result["close"].rolling(window=20).std()
        df_result["bb_upper"] = df_result["bb_middle"] + (bb_std * 2)
        df_result["bb_lower"] = df_result["bb_middle"] - (bb_std * 2)
        df_result["bb_width"] = df_result["bb_upper"] - df_result["bb_lower"]
        df_result["bb_position"] = (df_result["close"] - df_result["bb_lower"]) / (
            df_result["bb_upper"] - df_result["bb_lower"]
        )

        # On-Balance Volume (OBV)
        df_result["obv"] = 0
        df_result.loc[df_result["close"] > df_result["close"].shift(1), "obv"] = df_result["volume"]
        df_result.loc[df_result["close"] < df_result["close"].shift(1), "obv"] = -df_result["volume"]
        df_result["obv"] = df_result["obv"].cumsum()

        # Price channels
        df_result["high_20"] = df_result["high"].rolling(window=20).max()
        df_result["low_20"] = df_result["low"].rolling(window=20).min()

        # Average Directional Index (ADX) approximation
        df_result["tr"] = np.maximum(
            df_result["high"] - df_result["low"],
            np.maximum(
                abs(df_result["high"] - df_result["close"].shift(1)),
                abs(df_result["low"] - df_result["close"].shift(1)),
            ),
        )
        df_result["atr"] = df_result["tr"].rolling(window=14).mean()

        return df_result

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to DataFrame.

        Args:
            df: DataFrame with datetime index.

        Returns:
            DataFrame with added time features.
        """
        if df.empty:
            return df

        df_result = df.copy()

        if not isinstance(df_result.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, cannot add time features")
            return df_result

        # Basic time features
        df_result["year"] = df_result.index.year
        df_result["month"] = df_result.index.month
        df_result["day"] = df_result.index.day
        df_result["day_of_week"] = df_result.index.dayofweek
        df_result["day_of_year"] = df_result.index.dayofyear
        df_result["week_of_year"] = df_result.index.isocalendar().week

        # Quarter
        df_result["quarter"] = df_result.index.quarter

        # Is month end/start
        df_result["is_month_end"] = df_result.index.is_month_end.astype(int)
        df_result["is_month_start"] = df_result.index.is_month_start.astype(int)
        df_result["is_quarter_end"] = df_result.index.is_quarter_end.astype(int)
        df_result["is_quarter_start"] = df_result.index.is_quarter_start.astype(int)
        df_result["is_year_end"] = df_result.index.is_year_end.astype(int)
        df_result["is_year_start"] = df_result.index.is_year_start.astype(int)

        # Time of day (for intraday data)
        if df_result.index.hour.any():
            df_result["hour"] = df_result.index.hour
            df_result["minute"] = df_result.index.minute

        return df_result

    def prepare_features(
        self, df: pd.DataFrame, target_col: str = "returns", lookahead: int = 1
    ) -> pd.DataFrame:
        """Prepare features for machine learning.

        Args:
            df: DataFrame with market data.
            target_col: Column to use as target.
            lookahead: Number of periods ahead for target.

        Returns:
            DataFrame with features and target.
        """
        if df.empty:
            return df

        df_result = df.copy()

        # Clean data
        df_result = self.clean_data(df_result)

        # Add returns if not present
        if "returns" not in df_result.columns:
            df_result = self.add_returns(df_result)

        # Add moving averages
        df_result = self.add_moving_averages(df_result)

        # Add volatility
        df_result = self.add_volatility(df_result)

        # Add technical indicators
        df_result = self.add_technical_indicators(df_result)

        # Add time features
        df_result = self.add_time_features(df_result)

        # Create target
        if target_col == "returns":
            df_result["target"] = df_result["returns"].shift(-lookahead)
        elif target_col in df_result.columns:
            df_result["target"] = df_result[target_col].shift(-lookahead)
        else:
            logger.warning(f"Target column {target_col} not found, using returns")
            df_result["target"] = df_result["returns"].shift(-lookahead)

        # Drop rows with NaN (from rolling calculations and target shift)
        df_result = df_result.dropna()

        logger.info(f"Prepared features: {len(df_result)} rows, {len(df_result.columns)} columns")
        return df_result

    def resample_data(self, df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        """Resample data to different frequency.

        Args:
            df: DataFrame with datetime index.
            freq: Resampling frequency ('D', 'W', 'M', 'Q', 'Y').

        Returns:
            Resampled DataFrame.
        """
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")

        # Define aggregation rules
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Add any additional columns
        for col in df.columns:
            if col not in ohlc_dict:
                ohlc_dict[col] = "last"

        # Resample
        df_resampled = df.resample(freq).apply(ohlc_dict)

        return df_resampled

    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Normalize specified columns.

        Args:
            df: DataFrame to normalize.
            columns: List of columns to normalize. If None, normalize all numeric columns.

        Returns:
            DataFrame with normalized columns.
        """
        if df.empty:
            return df

        df_result = df.copy()

        if columns is None:
            # Select numeric columns
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if col != "target"]

        for col in columns:
            if col in df_result.columns:
                # Z-score normalization
                mean = df_result[col].mean()
                std = df_result[col].std()
                if std > 0:
                    df_result[f"{col}_norm"] = (df_result[col] - mean) / std
                else:
                    df_result[f"{col}_norm"] = 0

        return df_result

    def split_train_test(
        self, df: pd.DataFrame, test_size: float = 0.2, date_cutoff: str = None
    ) -> tuple:
        """Split data into training and testing sets.

        Args:
            df: DataFrame to split.
            test_size: Proportion of data to use for testing (if date_cutoff is None).
            date_cutoff: Date string to split on (e.g., '2022-01-01').

        Returns:
            Tuple of (train_df, test_df).
        """
        if df.empty:
            return df.copy(), df.copy()

        if date_cutoff:
            # Split by date
            cutoff_date = pd.to_datetime(date_cutoff)
            train_df = df[df.index < cutoff_date].copy()
            test_df = df[df.index >= cutoff_date].copy()
        else:
            # Split by proportion
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df