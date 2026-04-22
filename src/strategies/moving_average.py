# 移动平均线交叉策略
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseStrategy
from .utils import (
    calculate_crossover_signals,
    create_signals_dataframe,
    add_signal_descriptions,
    calculate_position_from_signals,
    log_signal_statistics,
    validate_strategy_parameters,
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MovingAverageCrossover(BaseStrategy):
    """移动平均线交叉策略。

    此策略基于两条移动平均线的交叉生成信号：
    - 当短期MA上穿长期MA时买入（金叉）
    - 当短期MA下穿长期MA时卖出（死叉）
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        name: str = "MovingAverageCrossover",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """初始化移动平均线交叉策略。

        Args:
            short_window: 短期移动平均线窗口。
            long_window: 长期移动平均线窗口。
            name: 策略名称。
            initial_capital: 回测初始资金。
            commission: 每笔交易佣金率。
            slippage: 价格滑点比例。
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        self.short_window = short_window
        self.long_window = long_window
        
        # Validate parameters
        params = {
            'short_window': short_window,
            'long_window': long_window,
            'initial_capital': initial_capital,
            'commission': commission,
            'slippage': slippage
        }
        validate_strategy_parameters(params)
        
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        
        logger.info(
            f"Initialized {name} with short_window={short_window}, "
            f"long_window={long_window}"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on moving average crossover.

        Args:
            data: DataFrame with market data (must contain 'close' column).

        Returns:
            DataFrame with signals, moving averages, and additional metrics.

        Raises:
            ValueError: If data is empty or missing required columns.
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        df = data.copy()
        
        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        
        # Use utility function to calculate crossover signals
        signals_series = calculate_crossover_signals(df['short_ma'], df['long_ma'])
        
        # Create standardized signals DataFrame
        signals = create_signals_dataframe(df.index)
        signals['signal'] = signals_series
        
        # Add signal descriptions and position
        signals = add_signal_descriptions(signals)
        signals['position'] = calculate_position_from_signals(signals['signal'])
        
        # Add moving averages and price data
        signals['short_ma'] = df['short_ma']
        signals['long_ma'] = df['long_ma']
        signals['close'] = df['close']
        
        # Add crossover metrics
        signals['ma_diff'] = df['short_ma'] - df['long_ma']
        signals['ma_diff_pct'] = signals['ma_diff'] / df['long_ma'].replace(0, np.nan) * 100
        
        # Log signal statistics
        log_signal_statistics(signals['signal'], self.name, logger)
        
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary with strategy parameters.
        """
        base_params = super().get_parameters()
        base_params.update({
            "short_window": self.short_window,
            "long_window": self.long_window,
            "strategy_type": "MovingAverageCrossover",
        })
        return base_params


class MovingAverageRibbon(BaseStrategy):
    """Moving average ribbon strategy.

    This strategy uses multiple moving averages to identify trends.
    Buy when price is above all moving averages (bullish trend).
    Sell when price is below all moving averages (bearish trend).
    """

    def __init__(
        self,
        windows: Optional[list] = None,
        name: str = "MovingAverageRibbon",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """Initialize moving average ribbon strategy.

        Args:
            windows: List of moving average windows. Defaults to [10, 20, 30, 50, 100, 200].
            name: Strategy name.
            initial_capital: Initial capital.
            commission: Commission rate.
            slippage: Slippage fraction.
        """
        super().__init__(name, initial_capital, commission, slippage)
        
        if windows is None:
            windows = [10, 20, 30, 50, 100, 200]
        
        self.windows = sorted(windows)
        
        if len(self.windows) < 2:
            raise ValueError("At least two moving average windows required")
        
        logger.info(f"Initialized {name} with windows={windows}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on moving average ribbon.

        Args:
            data: DataFrame with market data.

        Returns:
            DataFrame with signals and moving averages.
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        df = data.copy()
        
        # Calculate all moving averages
        for window in self.windows:
            col_name = f'ma_{window}'
            df[col_name] = df['close'].rolling(window=window).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Check if price is above all moving averages (bullish)
        above_all = (df['close'] > df[f'ma_{self.windows[0]}'])
        for window in self.windows[1:]:
            above_all = above_all & (df['close'] > df[f'ma_{window}'])
        
        # Check if price is below all moving averages (bearish)
        below_all = (df['close'] < df[f'ma_{self.windows[0]}'])
        for window in self.windows[1:]:
            below_all = below_all & (df['close'] < df[f'ma_{window}'])
        
        # Buy when price moves from not above all to above all
        buy_signals = above_all & ~above_all.shift(1).fillna(False)
        signals.loc[buy_signals, 'signal'] = 1
        
        # Sell when price moves from not below all to below all
        sell_signals = below_all & ~below_all.shift(1).fillna(False)
        signals.loc[sell_signals, 'signal'] = -1
        
        # Add moving averages to signals dataframe
        for window in self.windows:
            signals[f'ma_{window}'] = df[f'ma_{window}']
        
        signals['close'] = df['close']
        
        # Add signal type descriptions
        signals['signal_type'] = 'hold'
        signals.loc[signals['signal'] == 1, 'signal_type'] = 'buy'
        signals.loc[signals['signal'] == -1, 'signal_type'] = 'sell'
        
        # Add position
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Calculate ribbon width (difference between fastest and slowest MA)
        fastest_ma = f'ma_{self.windows[0]}'
        slowest_ma = f'ma_{self.windows[-1]}'
        signals['ribbon_width'] = df[fastest_ma] - df[slowest_ma]
        signals['ribbon_width_pct'] = signals['ribbon_width'] / df[slowest_ma] * 100
        
        # Calculate ribbon slope (trend strength)
        signals['ribbon_slope'] = signals['ribbon_width'].diff()
        
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
            "windows": self.windows,
            "strategy_type": "MovingAverageRibbon",
        })
        return base_params