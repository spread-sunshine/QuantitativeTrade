# 均值回归策略
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseStrategy
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MeanReversion(BaseStrategy):
    """基于布林带的均值回归策略。

    此策略假设价格倾向于回归均值。
    当价格跌破下布林带时买入（超卖）。
    当价格突破上布林带时卖出（超买）。
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        name: str = "MeanReversion",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """初始化均值回归策略。

        Args:
            window: 移动平均和标准差的窗口。
            num_std: 布林带的标准差数量。
            rsi_period: RSI确认周期。
            rsi_oversold: RSI超卖阈值。
            rsi_overbought: RSI超买阈值。
            name: 策略名称。
            initial_capital: 初始资金。
            commission: 佣金率。
            slippage: 滑点比例。
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        self.window = window
        self.num_std = num_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Validate parameters
        if window <= 0:
            raise ValueError("Window must be positive")
        if num_std <= 0:
            raise ValueError("Number of standard deviations must be positive")
        if rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if rsi_oversold >= rsi_overbought:
            raise ValueError("RSI oversold must be less than overbought")
        
        logger.info(
            f"Initialized {name} with window={window}, num_std={num_std}, "
            f"rsi_period={rsi_period}"
        )

    def calculate_bollinger_bands(self, prices: pd.Series):
        """Calculate Bollinger Bands.

        Args:
            prices: Price series.

        Returns:
            Tuple of (middle_band, upper_band, lower_band, bandwidth, position).
        """
        middle_band = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        bandwidth = (upper_band - lower_band) / middle_band * 100
        position = (prices - lower_band) / (upper_band - lower_band) * 100
        
        return middle_band, upper_band, lower_band, bandwidth, position

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index (RSI).

        Args:
            prices: Price series.

        Returns:
            RSI series.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on mean reversion.

        Args:
            data: DataFrame with market data.

        Returns:
            DataFrame with signals and indicators.
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        df = data.copy()
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band, bandwidth, position = self.calculate_bollinger_bands(df['close'])
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'])
        
        # Generate signals
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Buy signal: price crosses below lower band AND RSI < oversold
        buy_condition = (df['close'] < lower_band) & (rsi < self.rsi_oversold)
        buy_signals = buy_condition & ~buy_condition.shift(1).fillna(False)
        signals.loc[buy_signals, 'signal'] = 1
        
        # Sell signal: price crosses above upper band AND RSI > overbought
        sell_condition = (df['close'] > upper_band) & (rsi > self.rsi_overbought)
        sell_signals = sell_condition & ~sell_condition.shift(1).fillna(False)
        signals.loc[sell_signals, 'signal'] = -1
        
        # Exit signal: price returns to middle band
        exit_long = (df['close'] >= middle_band) & (signals['signal'].shift(1) == 1)
        exit_short = (df['close'] <= middle_band) & (signals['signal'].shift(1) == -1)
        
        # Clear signals on exit
        signals.loc[exit_long, 'signal'] = 0
        signals.loc[exit_short, 'signal'] = 0
        
        # Add indicators to signals dataframe
        signals['close'] = df['close']
        signals['middle_band'] = middle_band
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['bandwidth'] = bandwidth
        signals['bb_position'] = position
        signals['rsi'] = rsi
        
        # Add signal type descriptions
        signals['signal_type'] = 'hold'
        signals.loc[signals['signal'] == 1, 'signal_type'] = 'buy'
        signals.loc[signals['signal'] == -1, 'signal_type'] = 'sell'
        signals.loc[exit_long, 'signal_type'] = 'exit_long'
        signals.loc[exit_short, 'signal_type'] = 'exit_short'
        
        # Add position
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Add mean reversion score (higher score = stronger mean reversion signal)
        signals['mr_score'] = 0
        signals.loc[buy_condition, 'mr_score'] = (lower_band - df['close']) / lower_band * 100
        signals.loc[sell_condition, 'mr_score'] = (df['close'] - upper_band) / upper_band * 100
        
        logger.info(f"Generated {signals['signal'].abs().sum()} signals")
        logger.info(f"Buy signals: {(signals['signal'] == 1).sum()}")
        logger.info(f"Sell signals: {(signals['signal'] == -1).sum()}")
        
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary with strategy parameters.
        """
        base_params = super().get_parameters()
        base_params.update({
            "window": self.window,
            "num_std": self.num_std,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "strategy_type": "MeanReversion",
        })
        return base_params


class PairsTrading(BaseStrategy):
    """Pairs trading strategy (statistical arbitrage).

    This strategy trades pairs of correlated assets.
    Go long the underperforming asset and short the outperforming asset
    when the spread diverges from its mean.
    """

    def __init__(
        self,
        lookback_period: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_holding_period: int = 20,
        name: str = "PairsTrading",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """Initialize pairs trading strategy.

        Args:
            lookback_period: Period for calculating spread statistics.
            entry_zscore: Z-score threshold for entry.
            exit_zscore: Z-score threshold for exit.
            max_holding_period: Maximum holding period in days.
            name: Strategy name.
            initial_capital: Initial capital.
            commission: Commission rate.
            slippage: Slippage fraction.
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_holding_period = max_holding_period
        
        # Validate parameters
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive")
        if entry_zscore <= exit_zscore:
            raise ValueError("Entry z-score must be greater than exit z-score")
        if max_holding_period <= 0:
            raise ValueError("Max holding period must be positive")
        
        # Pairs data
        self.asset_a: Optional[str] = None
        self.asset_b: Optional[str] = None
        self.spread_mean: float = 0
        self.spread_std: float = 1
        
        logger.info(
            f"Initialized {name} with lookback={lookback_period}, "
            f"entry_z={entry_zscore}, exit_z={exit_zscore}"
        )

    def set_pair(self, asset_a: str, asset_b: str):
        """Set the trading pair.

        Args:
            asset_a: First asset symbol.
            asset_b: Second asset symbol.
        """
        self.asset_a = asset_a
        self.asset_b = asset_b
        logger.info(f"Set trading pair: {asset_a}/{asset_b}")

    def calculate_spread(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """Calculate spread between two price series.

        Args:
            price_a: First price series.
            price_b: Second price series.

        Returns:
            Spread series.
        """
        # Simple price difference spread
        spread = price_a - price_b
        
        # Alternatively, could use ratio or log ratio
        # spread = np.log(price_a) - np.log(price_b)
        
        return spread

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals.

        Args:
            data: DataFrame with market data for both assets.
                Must contain columns: 'close_a', 'close_b' or similar.

        Returns:
            DataFrame with signals and spread information.
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        # Determine price columns
        if 'close_a' in data.columns and 'close_b' in data.columns:
            price_a = data['close_a']
            price_b = data['close_b']
        elif 'close' in data.columns and self.asset_a and self.asset_b:
            # Assuming data is for a single asset, need separate data for pairs
            raise ValueError("For pairs trading, need data for both assets")
        else:
            raise ValueError("Data must contain 'close_a' and 'close_b' columns")
        
        # Calculate spread
        spread = self.calculate_spread(price_a, price_b)
        
        # Calculate rolling statistics
        spread_mean = spread.rolling(window=self.lookback_period).mean()
        spread_std = spread.rolling(window=self.lookback_period).std()
        
        # Calculate z-score
        zscore = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position_a'] = 0  # Position for asset A
        signals['position_b'] = 0  # Position for asset B
        
        # Entry signals
        # When spread is too wide (zscore > entry_zscore):
        # - Short asset A (overpriced), Long asset B (underpriced)
        # When spread is too narrow (zscore < -entry_zscore):
        # - Long asset A (underpriced), Short asset B (overpriced)
        
        short_a_long_b = zscore > self.entry_zscore
        long_a_short_b = zscore < -self.entry_zscore
        
        signals.loc[short_a_long_b, 'signal'] = -1  # Negative signal for spread contraction
        signals.loc[long_a_short_b, 'signal'] = 1   # Positive signal for spread expansion
        
        # Set positions
        signals.loc[short_a_long_b, 'position_a'] = -1
        signals.loc[short_a_long_b, 'position_b'] = 1
        
        signals.loc[long_a_short_b, 'position_a'] = 1
        signals.loc[long_a_short_b, 'position_b'] = -1
        
        # Exit signals (when spread returns to mean)
        exit_condition = abs(zscore) < self.exit_zscore
        
        # Clear signals on exit
        signals.loc[exit_condition, 'signal'] = 0
        signals.loc[exit_condition, 'position_a'] = 0
        signals.loc[exit_condition, 'position_b'] = 0
        
        # Add indicators to signals dataframe
        signals['price_a'] = price_a
        signals['price_b'] = price_b
        signals['spread'] = spread
        signals['spread_mean'] = spread_mean
        signals['spread_std'] = spread_std
        signals['zscore'] = zscore
        
        # Add signal type descriptions
        signals['signal_type'] = 'hold'
        signals.loc[short_a_long_b & ~exit_condition, 'signal_type'] = 'short_a_long_b'
        signals.loc[long_a_short_b & ~exit_condition, 'signal_type'] = 'long_a_short_b'
        signals.loc[exit_condition, 'signal_type'] = 'exit'
        
        # Track holding period
        signals['holding_days'] = 0
        in_trade = False
        entry_day = 0
        
        for i in range(len(signals)):
            if signals.iloc[i]['signal'] != 0 and not in_trade:
                in_trade = True
                entry_day = i
            elif signals.iloc[i]['signal'] == 0 and in_trade:
                in_trade = False
            
            if in_trade:
                holding_days = i - entry_day
                signals.iloc[i, signals.columns.get_loc('holding_days')] = holding_days
                
                # Force exit if holding too long
                if holding_days >= self.max_holding_period:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    signals.iloc[i, signals.columns.get_loc('position_a')] = 0
                    signals.iloc[i, signals.columns.get_loc('position_b')] = 0
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'force_exit'
                    in_trade = False
        
        logger.info(f"Generated {signals['signal'].abs().sum()} signals")
        logger.info(f"Short A/Long B signals: {(signals['signal'] == -1).sum()}")
        logger.info(f"Long A/Short B signals: {(signals['signal'] == 1).sum()}")
        
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary with strategy parameters.
        """
        base_params = super().get_parameters()
        base_params.update({
            "lookback_period": self.lookback_period,
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "max_holding_period": self.max_holding_period,
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "strategy_type": "PairsTrading",
        })
        return base_params