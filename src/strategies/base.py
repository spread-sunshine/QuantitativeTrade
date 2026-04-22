# Base strategy class for trading strategies
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple, TypeAlias
import pandas as pd
import numpy as np
import logging

from ..utils.logger import setup_logger

# Type aliases for better readability
Series: TypeAlias = pd.Series
DataFrame: TypeAlias = pd.DataFrame
Timestamp: TypeAlias = pd.Timestamp
BacktestResult = Dict[str, Any]
SignalDict = Dict[str, Union[str, float, pd.Timestamp, None]]

logger = setup_logger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(
        self,
        name: str = "BaseStrategy",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """Initialize strategy.

        Args:
            name: Strategy name.
            initial_capital: Initial capital for backtesting.
            commission: Commission rate per trade.
            slippage: Slippage as fraction of price.
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # State variables
        self.position: int = 0  # Current position (+1 long, -1 short, 0 flat)
        self.capital: float = initial_capital
        self.equity: float = initial_capital
        self.trades: List[Dict[str, Any]] = []  # List of trade records
        
        # Performance metrics
        self.returns: List[float] = []  # Daily returns
        self.drawdown: float = 0.0  # Current drawdown
        self.max_drawdown: float = 0.0  # Maximum drawdown
        self.sharpe_ratio: float = 0.0  # Sharpe ratio
        self.sortino_ratio: float = 0.0  # Sortino ratio
        self.win_rate: float = 0.0  # Win rate
        
        # Data
        self.data: Optional[DataFrame] = None  # Historical market data
        self.signals: Optional[DataFrame] = None  # Generated trading signals

    @abstractmethod
    def generate_signals(self, data: DataFrame) -> DataFrame:
        """Generate trading signals from market data.

        Subclasses must implement this method to generate trading signals
        based on their specific logic. The method should return a DataFrame
        containing at least a 'signal' column with values:
        - 1: Buy/Long signal
        - -1: Sell/Short signal
        - 0: Hold/No position signal

        Args:
            data: DataFrame with historical market data containing at least
                OHLCV columns ('open', 'high', 'low', 'close', 'volume').

        Returns:
            DataFrame with the same index as input data, containing:
            - 'signal': Trading signals (1, -1, or 0)
            - Additional columns as needed (e.g., 'strength', 'reason')

        Raises:
            ValueError: If input data is missing required columns or is empty.
        """
        pass

    def calculate_returns(self, prices: Series, signals: Series) -> Series:
        """Calculate strategy returns based on signals and prices.

        This method calculates daily returns for the strategy, adjusting for
        transaction costs (commission) and market impact (slippage).
        Signals are shifted by one period to avoid look-ahead bias.

        Args:
            prices: Price series (e.g., close prices).
            signals: Trading signals series (1 for long, -1 for short, 0 for flat).

        Returns:
            Series containing daily strategy returns (including costs).

        Raises:
            ValueError: If prices and signals have different lengths.
            ValueError: If signals contain values other than -1, 0, or 1.
        """
        if len(prices) != len(signals):
            raise ValueError(f"Prices ({len(prices)}) and signals ({len(signals)}) must have same length")
        
        # Validate signal values
        unique_signals = signals.unique()
        invalid_signals = set(unique_signals) - {-1, 0, 1}
        if invalid_signals:
            raise ValueError(f"Signals must be -1, 0, or 1. Found invalid values: {invalid_signals}")
        
        # Shift signals to avoid look-ahead bias (trade at next period's open)
        signals_shifted = signals.shift(1)
        
        # Calculate price returns
        price_returns = prices.pct_change()
        
        # Calculate raw strategy returns (signal * price return)
        strategy_returns = signals_shifted * price_returns
        
        # Adjust for commission (pay commission on entry and exit)
        trade_changes = signals_shifted.diff().fillna(0).abs()
        strategy_returns = strategy_returns - (trade_changes * self.commission)
        
        # Adjust for slippage (cost proportional to position size)
        strategy_returns = strategy_returns - (abs(signals_shifted) * self.slippage)
        
        return strategy_returns

    def calculate_equity_curve(self, returns: Series) -> Series:
        """Calculate equity curve from returns.

        Args:
            returns: Strategy returns series (daily returns).

        Returns:
            Equity curve series (cumulative product of returns).
        """
        equity_curve = (1 + returns).cumprod() * self.initial_capital
        return equity_curve

    def calculate_drawdown(self, equity_curve: Series) -> Series:
        """Calculate drawdown series from equity curve.

        Args:
            equity_curve: Equity curve series.

        Returns:
            Drawdown series (negative values representing drawdowns).
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown

    def calculate_sharpe_ratio(
        self, returns: Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio (risk-adjusted return).

        Args:
            returns: Strategy returns series (daily returns).
            risk_free_rate: Annual risk-free rate (default 0.02 = 2%).

        Returns:
            Sharpe ratio (annualized). Returns 0.0 for insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualize returns and standard deviation
        # Assuming daily returns (252 trading days)
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol == 0:
            return 0.0
        
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        return sharpe

    def calculate_sortino_ratio(
        self, returns: Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return).

        Args:
            returns: Strategy returns series (daily returns).
            risk_free_rate: Annual risk-free rate (default 0.02 = 2%).

        Returns:
            Sortino ratio (annualized). Returns 0.0 for insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        
        # Separate negative returns
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) < 2:
            return 0.0
        
        # Annualize returns and downside deviation
        annualized_return = returns.mean() * 252
        downside_vol = negative_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = (annualized_return - risk_free_rate) / downside_vol
        return sortino

    def calculate_win_rate(self, returns: Series) -> float:
        """Calculate win rate (percentage of profitable periods).

        Args:
            returns: Strategy returns series (daily returns).

        Returns:
            Win rate between 0.0 and 1.0. Returns 0.0 for insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        
        # Count positive returns (assuming each period is a "trade")
        positive_trades = (returns > 0).sum()
        total_trades = len(returns)
        
        if total_trades == 0:
            return 0.0
        
        win_rate = positive_trades / total_trades
        return win_rate

    def calculate_max_drawdown(self, drawdown: Series) -> float:
        """Calculate maximum drawdown from drawdown series.

        Args:
            drawdown: Drawdown series (negative values).

        Returns:
            Maximum drawdown (most negative value, e.g., -0.15 for 15% drawdown).
            Returns 0.0 for empty series.
        """
        if len(drawdown) == 0:
            return 0.0
        
        max_dd = drawdown.min()
        return max_dd

    def run_backtest(self, data: DataFrame) -> BacktestResult:
        """Run backtest on historical data.

        Executes a complete backtest by:
        1. Generating trading signals from the data
        2. Calculating strategy returns (adjusted for costs)
        3. Computing performance metrics (Sharpe, Sortino, drawdown, etc.)
        4. Compiling results into a structured dictionary

        Args:
            data: Historical market data DataFrame with OHLCV columns.

        Returns:
            Dictionary containing comprehensive backtest results.

        Raises:
            ValueError: If data is missing required 'close' column.
            RuntimeError: If signal generation fails.
        """
        logger.info(f"Running backtest for {self.name}")
        
        # Store data
        self.data = data.copy()
        
        # Generate signals
        self.signals = self.generate_signals(data)
        
        # Calculate returns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        returns = self.calculate_returns(data['close'], self.signals['signal'])
        
        # Calculate equity curve
        equity_curve = self.calculate_equity_curve(returns)
        
        # Calculate drawdown
        drawdown = self.calculate_drawdown(equity_curve)
        
        # Calculate performance metrics
        self.sharpe_ratio = self.calculate_sharpe_ratio(returns)
        self.sortino_ratio = self.calculate_sortino_ratio(returns)
        self.win_rate = self.calculate_win_rate(returns)
        self.max_drawdown = self.calculate_max_drawdown(drawdown)
        
        # Store returns
        self.returns = returns
        
        # Prepare results
        results = {
            "strategy_name": self.name,
            "initial_capital": self.initial_capital,
            "final_equity": equity_curve.iloc[-1] if not equity_curve.empty else self.initial_capital,
            "total_return": (equity_curve.iloc[-1] / self.initial_capital - 1) if not equity_curve.empty else 0.0,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.signals['signal'].diff().abs().sum(),
            "returns": returns,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
            "signals": self.signals,
        }
        
        # Calculate annualized return
        if len(returns) > 0:
            annualized_return = returns.mean() * 252
            results["annualized_return"] = annualized_return
        
        logger.info(f"Backtest completed for {self.name}")
        logger.info(f"  Total Return: {results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters.

        Returns:
            Dictionary containing all configurable strategy parameters.
        """
        return {
            "name": self.name,
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "slippage": self.slippage,
        }

    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters.

        Args:
            **kwargs: Parameter key-value pairs.

        Returns:
            None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Parameter {key} not found in strategy")

    def reset(self) -> None:
        """Reset strategy state to initial conditions.

        This method clears all accumulated data and resets the strategy
        to its initial state, allowing for a fresh backtest or simulation.
        """
        self.position = 0
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.trades = []
        self.returns = []
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.win_rate = 0.0
        self.data = None
        self.signals = None
        
        logger.info(f"Strategy {self.name} reset")


class Signal:
    """Represents a trading signal with timestamp, symbol, and metadata."""
    
    def __init__(
        self,
        timestamp: Timestamp,
        symbol: str,
        signal_type: str,  # 'long', 'short', 'exit'
        price: float,
        quantity: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize signal with all relevant information.
        
        Args:
            timestamp: Signal timestamp (datetime when signal was generated).
            symbol: Trading symbol (e.g., 'AAPL', 'SPY').
            signal_type: Signal type ('long', 'short', 'exit').
            price: Entry/exit price at signal time.
            quantity: Position size (positive for long, negative for short).
            reason: Human-readable reason for the signal (optional).
        """
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.price = price
        self.quantity = quantity
        self.reason = reason
        
    def to_dict(self) -> SignalDict:
        """Convert signal to serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "price": self.price,
            "quantity": self.quantity,
            "reason": self.reason,
        }