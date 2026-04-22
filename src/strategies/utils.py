# Utility functions for trading strategies
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_signals_dataframe(
    index: pd.Index, initial_signal: int = 0
) -> pd.DataFrame:
    """Create a standardized signals DataFrame with initial values.

    Args:
        index: Index for the DataFrame (should match data index).
        initial_signal: Initial signal value (default 0 for hold).

    Returns:
        DataFrame with standard signal columns initialized.
    """
    signals = pd.DataFrame(index=index)
    signals['signal'] = initial_signal
    signals['signal_type'] = 'hold'
    signals['position'] = 0.0
    signals['reason'] = None
    return signals


def add_signal_descriptions(signals: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable signal descriptions to signals DataFrame.

    Args:
        signals: DataFrame containing 'signal' column.

    Returns:
        Updated DataFrame with 'signal_type' column.
    """
    signals = signals.copy()
    
    # Map signal values to descriptions
    signal_map = {
        1: 'buy',
        -1: 'sell',
        0: 'hold'
    }
    
    signals['signal_type'] = signals['signal'].map(signal_map).fillna('hold')
    return signals


def calculate_position_from_signals(
    signals: pd.Series, fill_method: str = 'ffill'
) -> pd.Series:
    """Calculate position series from discrete signals.

    Args:
        signals: Series of trading signals (1, -1, 0).
        fill_method: Method to fill hold periods ('ffill' or 'bfill').

    Returns:
        Position series where hold periods are filled with last signal.
    """
    if fill_method not in ['ffill', 'bfill']:
        raise ValueError("fill_method must be 'ffill' or 'bfill'")
    
    # Replace 0 with NaN for forward/backward fill
    position = signals.replace(0, np.nan)
    
    if fill_method == 'ffill':
        position = position.ffill().fillna(0)
    else:
        position = position.bfill().fillna(0)
    
    return position


def validate_strategy_parameters(
    parameters: Dict[str, Any], required_params: Optional[list] = None
) -> None:
    """Validate strategy parameters.

    Args:
        parameters: Dictionary of parameters to validate.
        required_params: List of required parameter names.

    Raises:
        ValueError: If any parameter validation fails.
    """
    if required_params is None:
        required_params = []
    
    # Check required parameters
    for param in required_params:
        if param not in parameters:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Validate parameter values
    for param, value in parameters.items():
        if param.endswith('_window') or param in ['short_window', 'long_window', 'window']:
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{param} must be an integer, got {type(value)}")
            if value <= 0:
                raise ValueError(f"{param} must be positive, got {value}")
        
        elif param in ['commission', 'slippage', 'initial_capital']:
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{param} must be a number, got {type(value)}")
            if value < 0:
                raise ValueError(f"{param} must be non-negative, got {value}")


def calculate_crossover_signals(
    series1: pd.Series, series2: pd.Series, 
    buffer_periods: int = 0
) -> pd.Series:
    """Calculate crossover signals between two series.

    Args:
        series1: First series (e.g., short moving average).
        series2: Second series (e.g., long moving average).
        buffer_periods: Number of periods to require confirmation.

    Returns:
        Series with signals: 1 for series1 above series2, -1 for below, 0 otherwise.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have same length")
    
    # Calculate basic crossover
    signals = pd.Series(0, index=series1.index)
    
    # Series1 crosses above series2
    cross_above = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    # Series1 crosses below series2
    cross_below = (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    # Apply buffer/confirmation if requested
    if buffer_periods > 0:
        for i in range(1, buffer_periods + 1):
            cross_above = cross_above & (series1.shift(-i) > series2.shift(-i))
            cross_below = cross_below & (series1.shift(-i) < series2.shift(-i))
    
    signals[cross_above] = 1
    signals[cross_below] = -1
    
    return signals


def log_signal_statistics(
    signals: pd.Series, strategy_name: str, logger: logging.Logger
) -> None:
    """Log statistics about generated signals.

    Args:
        signals: Series of trading signals.
        strategy_name: Name of the strategy (for logging).
        logger: Logger instance.
    """
    if signals.empty:
        logger.warning(f"No signals generated for {strategy_name}")
        return
    
    total_signals = signals.abs().sum()
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    hold_signals = (signals == 0).sum()
    
    logger.info(f"Signal statistics for {strategy_name}:")
    logger.info(f"  Total signals: {total_signals}")
    logger.info(f"  Buy signals: {buy_signals} ({buy_signals/len(signals):.1%})")
    logger.info(f"  Sell signals: {sell_signals} ({sell_signals/len(signals):.1%})")
    logger.info(f"  Hold periods: {hold_signals} ({hold_signals/len(signals):.1%})")


def calculate_returns_metrics(
    returns: pd.Series, risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate comprehensive returns metrics.

    Args:
        returns: Series of strategy returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Dictionary containing various performance metrics.
    """
    if len(returns) < 2:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
    
    metrics = {}
    
    # Basic statistics
    metrics['mean_return'] = returns.mean()
    metrics['std_return'] = returns.std()
    metrics['min_return'] = returns.min()
    metrics['max_return'] = returns.max()
    
    # Sharpe ratio (annualized)
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = (
        (annualized_return - risk_free_rate) / annualized_vol
        if annualized_vol != 0 else 0.0
    )
    
    # Sortino ratio (using downside deviation)
    negative_returns = returns[returns < 0]
    downside_dev = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 0.0
    metrics['sortino_ratio'] = (
        (annualized_return - risk_free_rate) / downside_dev
        if downside_dev != 0 else 0.0
    )
    
    # Win rate
    metrics['win_rate'] = (returns > 0).sum() / len(returns)
    
    return metrics