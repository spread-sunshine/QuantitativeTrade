# 突破策略
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseStrategy
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """基于价格通道的突破策略。

    此策略识别盘整形态的突破。
    当价格突破阻力位时买入（看涨突破）。
    当价格跌破支撑位时卖出（看跌突破）。
    """

    def __init__(
        self,
        lookback_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        consolidation_period: int = 10,
        min_consolidation_range: float = 0.02,
        name: str = "BreakoutStrategy",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """初始化突破策略。

        Args:
            lookback_period: 计算支撑/阻力的周期。
            atr_period: 平均真实范围计算周期。
            atr_multiplier: 基于ATR的止损乘数。
            consolidation_period: 最小盘整周期天数。
            min_consolidation_range: 盘整最小价格范围（比例）。
            name: 策略名称。
            initial_capital: 初始资金。
            commission: 佣金率。
            slippage: 滑点比例。
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.consolidation_period = consolidation_period
        self.min_consolidation_range = min_consolidation_range
        
        # 验证参数
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive")
        if atr_period <= 0:
            raise ValueError("ATR period must be positive")
        if atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        if consolidation_period <= 0:
            raise ValueError("Consolidation period must be positive")
        if min_consolidation_range <= 0:
            raise ValueError("Minimum consolidation range must be positive")
        
        logger.info(
            f"Initialized {name} with lookback={lookback_period}, "
            f"atr_period={atr_period}, atr_multiplier={atr_multiplier}"
        )

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算平均真实范围（ATR）。

        Args:
            high: 最高价。
            low: 最低价。
            close: 收盘价。

        Returns:
            ATR序列。
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr

    def calculate_support_resistance(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> tuple:
        """计算支撑和阻力位。

        Args:
            high: 最高价。
            low: 最低价。
            close: 收盘价。

        Returns:
            元组（阻力位, 支撑位, 是否盘整, 盘整范围）。
        """
        # 滚动最高价和最低价
        rolling_high = high.rolling(window=self.lookback_period).max()
        rolling_low = low.rolling(window=self.lookback_period).min()
        
        # 识别盘整周期
        price_range = (rolling_high - rolling_low) / rolling_low
        is_consolidating = price_range < self.min_consolidation_range
        
        # 检查盘整是否持续最小周期
        consolidation_count = is_consolidating.rolling(window=self.consolidation_period).sum()
        valid_consolidation = consolidation_count >= self.consolidation_period
        
        return rolling_high, rolling_low, valid_consolidation, price_range

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成突破交易信号。

        Args:
            data: 包含市场数据的DataFrame。

        Returns:
            包含信号和指标的DataFrame。
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")
        
        df = data.copy()
        
        # 计算ATR
        atr = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # 计算支撑/阻力位
        resistance, support, is_consolidating, consolidation_range = self.calculate_support_resistance(
            df['high'], df['low'], df['close']
        )
        
        # 生成信号
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        # Bullish breakout: close above resistance after consolidation
        bullish_breakout = (df['close'] > resistance.shift(1)) & is_consolidating.shift(1)
        signals.loc[bullish_breakout, 'signal'] = 1
        
        # Bearish breakout: close below support after consolidation
        bearish_breakout = (df['close'] < support.shift(1)) & is_consolidating.shift(1)
        signals.loc[bearish_breakout, 'signal'] = -1
        
        # Calculate stop loss and take profit levels
        # For long positions: stop loss = entry - (ATR * multiplier)
        # For short positions: stop loss = entry + (ATR * multiplier)
        entry_prices = df['close'].where(signals['signal'] != 0)
        
        long_stop = entry_prices - (atr * self.atr_multiplier)
        short_stop = entry_prices + (atr * self.atr_multiplier)
        
        signals.loc[signals['signal'] == 1, 'stop_loss'] = long_stop
        signals.loc[signals['signal'] == -1, 'stop_loss'] = short_stop
        
        # Take profit at 2:1 risk/reward ratio
        signals.loc[signals['signal'] == 1, 'take_profit'] = entry_prices + (2 * (entry_prices - long_stop))
        signals.loc[signals['signal'] == -1, 'take_profit'] = entry_prices - (2 * (short_stop - entry_prices))
        
        # Exit signals (stop loss or take profit hit)
        exit_long_stop = (df['low'] < signals['stop_loss']) & (signals['signal'].shift(1) == 1)
        exit_long_profit = (df['high'] > signals['take_profit']) & (signals['signal'].shift(1) == 1)
        
        exit_short_stop = (df['high'] > signals['stop_loss']) & (signals['signal'].shift(1) == -1)
        exit_short_profit = (df['low'] < signals['take_profit']) & (signals['signal'].shift(1) == -1)
        
        exit_signals = exit_long_stop | exit_long_profit | exit_short_stop | exit_short_profit
        
        # Clear signals on exit
        signals.loc[exit_signals, 'signal'] = 0
        signals.loc[exit_signals, 'stop_loss'] = 0
        signals.loc[exit_signals, 'take_profit'] = 0
        
        # Add indicators to signals dataframe
        signals['close'] = df['close']
        signals['high'] = df['high']
        signals['low'] = df['low']
        signals['resistance'] = resistance
        signals['support'] = support
        signals['atr'] = atr
        signals['atr_pct'] = atr / df['close'] * 100
        signals['is_consolidating'] = is_consolidating
        signals['consolidation_range'] = consolidation_range
        signals['consolidation_strength'] = 1 / (consolidation_range + 0.001)  # Inverse of range
        
        # Add signal type descriptions
        signals['signal_type'] = 'hold'
        signals.loc[bullish_breakout & ~exit_signals, 'signal_type'] = 'bullish_breakout'
        signals.loc[bearish_breakout & ~exit_signals, 'signal_type'] = 'bearish_breakout'
        signals.loc[exit_long_stop, 'signal_type'] = 'stop_loss_long'
        signals.loc[exit_long_profit, 'signal_type'] = 'take_profit_long'
        signals.loc[exit_short_stop, 'signal_type'] = 'stop_loss_short'
        signals.loc[exit_short_profit, 'signal_type'] = 'take_profit_short'
        
        # Add position
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Add breakout strength (distance from breakout level)
        signals['breakout_strength'] = 0
        signals.loc[bullish_breakout, 'breakout_strength'] = (df['close'] - resistance.shift(1)) / resistance.shift(1) * 100
        signals.loc[bearish_breakout, 'breakout_strength'] = (support.shift(1) - df['close']) / support.shift(1) * 100
        
        # Add volume confirmation (if volume data available)
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'] / avg_volume
            signals['volume_ratio'] = volume_ratio
            signals.loc[bullish_breakout, 'volume_confirmation'] = volume_ratio > 1.5
            signals.loc[bearish_breakout, 'volume_confirmation'] = volume_ratio > 1.5
        
        logger.info(f"Generated {signals['signal'].abs().sum()} signals")
        logger.info(f"Bullish breakouts: {(signals['signal'] == 1).sum()}")
        logger.info(f"Bearish breakouts: {(signals['signal'] == -1).sum()}")
        
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary with strategy parameters.
        """
        base_params = super().get_parameters()
        base_params.update({
            "lookback_period": self.lookback_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "consolidation_period": self.consolidation_period,
            "min_consolidation_range": self.min_consolidation_range,
            "strategy_type": "BreakoutStrategy",
        })
        return base_params


class DonchianChannelBreakout(BaseStrategy):
    """Donchian Channel breakout strategy.

    This strategy uses Donchian Channels (N-period high/low) for breakout signals.
    Popularized by Richard Dennis' Turtle Traders.
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        atr_multiplier: float = 2.0,
        name: str = "DonchianChannelBreakout",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """Initialize Donchian Channel breakout strategy.

        Args:
            entry_period: Period for entry channel (N-period high/low).
            exit_period: Period for exit channel.
            atr_period: Period for ATR calculation.
            atr_multiplier: ATR multiplier for position sizing.
            name: Strategy name.
            initial_capital: Initial capital.
            commission: Commission rate.
            slippage: Slippage fraction.
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # 验证参数
        if entry_period <= 0:
            raise ValueError("Entry period must be positive")
        if exit_period <= 0:
            raise ValueError("Exit period must be positive")
        if atr_period <= 0:
            raise ValueError("ATR period must be positive")
        if atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        
        logger.info(
            f"Initialized {name} with entry_period={entry_period}, "
            f"exit_period={exit_period}, atr_multiplier={atr_multiplier}"
        )

    def calculate_donchian_channels(
        self, high: pd.Series, low: pd.Series
    ) -> tuple:
        """Calculate Donchian Channels.

        Args:
            high: High prices.
            low: Low prices.

        Returns:
            Tuple of (upper_channel, lower_channel, middle_channel, channel_width).
        """
        upper_channel = high.rolling(window=self.entry_period).max()
        lower_channel = low.rolling(window=self.entry_period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        channel_width = (upper_channel - lower_channel) / middle_channel * 100
        
        return upper_channel, lower_channel, middle_channel, channel_width

    def calculate_exit_channels(
        self, high: pd.Series, low: pd.Series
    ) -> tuple:
        """Calculate exit channels.

        Args:
            high: High prices.
            low: Low prices.

        Returns:
            Tuple of (exit_upper, exit_lower).
        """
        exit_upper = high.rolling(window=self.exit_period).max()
        exit_lower = low.rolling(window=self.exit_period).min()
        
        return exit_upper, exit_lower

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Donchian Channel breakout signals.

        Args:
            data: DataFrame with market data.

        Returns:
            DataFrame with signals and indicators.
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")
        
        df = data.copy()
        
        # Calculate Donchian Channels
        upper_channel, lower_channel, middle_channel, channel_width = self.calculate_donchian_channels(
            df['high'], df['low']
        )
        
        # Calculate exit channels
        exit_upper, exit_lower = self.calculate_exit_channels(df['high'], df['low'])
        
        # 计算ATR for position sizing
        atr = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # 生成信号
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0  # Position size as fraction of capital
        
        # Entry signals
        # Long: price breaks above entry period high
        long_entry = df['close'] > upper_channel.shift(1)
        
        # Short: price breaks below entry period low
        short_entry = df['close'] < lower_channel.shift(1)
        
        signals.loc[long_entry, 'signal'] = 1
        signals.loc[short_entry, 'signal'] = -1
        
        # Exit signals
        # Exit long: price breaks below exit period low
        exit_long = df['close'] < exit_lower.shift(1)
        
        # Exit short: price breaks above exit period high
        exit_short = df['close'] > exit_upper.shift(1)
        
        # Clear signals on exit
        signals.loc[exit_long & (signals['signal'].shift(1) == 1), 'signal'] = 0
        signals.loc[exit_short & (signals['signal'].shift(1) == -1), 'signal'] = 0
        
        # Calculate position size based on ATR (Turtle style)
        # Position size = (1% of capital) / (ATR * multiplier)
        dollar_risk = self.initial_capital * 0.01  # Risk 1% per trade
        signals['position_size'] = dollar_risk / (atr * self.atr_multiplier)
        signals['position_size'] = signals['position_size'].clip(upper=self.initial_capital * 0.1)  # Max 10% of capital
        
        # Add indicators to signals dataframe
        signals['close'] = df['close']
        signals['high'] = df['high']
        signals['low'] = df['low']
        signals['upper_channel'] = upper_channel
        signals['lower_channel'] = lower_channel
        signals['middle_channel'] = middle_channel
        signals['channel_width'] = channel_width
        signals['exit_upper'] = exit_upper
        signals['exit_lower'] = exit_lower
        signals['atr'] = atr
        
        # Add signal type descriptions
        signals['signal_type'] = 'hold'
        signals.loc[long_entry & ~exit_long, 'signal_type'] = 'long_entry'
        signals.loc[short_entry & ~exit_short, 'signal_type'] = 'short_entry'
        signals.loc[exit_long, 'signal_type'] = 'exit_long'
        signals.loc[exit_short, 'signal_type'] = 'exit_short'
        
        # Add position
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Add breakout information
        signals['breakout_distance'] = 0
        signals.loc[long_entry, 'breakout_distance'] = (df['close'] - upper_channel.shift(1)) / upper_channel.shift(1) * 100
        signals.loc[short_entry, 'breakout_distance'] = (lower_channel.shift(1) - df['close']) / lower_channel.shift(1) * 100
        
        # Add channel position (where price is within channel)
        signals['channel_position'] = (df['close'] - lower_channel) / (upper_channel - lower_channel) * 100
        
        logger.info(f"Generated {signals['signal'].abs().sum()} signals")
        logger.info(f"Long entries: {(signals['signal'] == 1).sum()}")
        logger.info(f"Short entries: {(signals['signal'] == -1).sum()}")
        
        return signals

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算平均真实范围（ATR）。

        Args:
            high: 最高价。
            low: 最低价。
            close: 收盘价。

        Returns:
            ATR序列。
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary with strategy parameters.
        """
        base_params = super().get_parameters()
        base_params.update({
            "entry_period": self.entry_period,
            "exit_period": self.exit_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "strategy_type": "DonchianChannelBreakout",
        })
        return base_params