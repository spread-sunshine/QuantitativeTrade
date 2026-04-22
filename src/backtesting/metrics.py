"""Performance metrics calculation for trading strategies."""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings


def calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    frequency: str = "daily",
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics.
    
    Args:
        returns: Strategy returns series.
        equity_curve: Equity curve series.
        drawdown: Drawdown series.
        initial_capital: Initial capital.
        risk_free_rate: Annual risk-free rate.
        frequency: Data frequency ('daily', 'weekly', 'monthly').
        
    Returns:
        Dictionary with performance metrics.
    """
    if len(returns) < 2:
        return _empty_metrics()
    
    # Remove NaN values
    returns_clean = returns.dropna()
    equity_clean = equity_curve.dropna()
    drawdown_clean = drawdown.dropna()
    
    if len(returns_clean) < 2:
        return _empty_metrics()
    
    # Get annualization factor
    ann_factor = _get_annualization_factor(frequency)
    
    # Calculate basic metrics
    metrics = {}
    
    # Return metrics
    metrics.update(_calculate_return_metrics(
        returns_clean, equity_clean, initial_capital, ann_factor
    ))
    
    # Risk metrics
    metrics.update(_calculate_risk_metrics(
        returns_clean, equity_clean, drawdown_clean, ann_factor
    ))
    
    # Risk-adjusted return metrics
    metrics.update(_calculate_risk_adjusted_metrics(
        returns_clean, risk_free_rate, ann_factor
    ))
    
    # Statistical metrics
    metrics.update(_calculate_statistical_metrics(returns_clean))
    
    # Trade metrics (if trades data available)
    # These would be calculated separately when trades are available
    
    # Benchmark-related metrics (if benchmark available)
    # These would be calculated separately when benchmark is available
    
    # Validation
    metrics = _validate_metrics(metrics)
    
    return metrics


def _get_annualization_factor(frequency: str) -> float:
    """Get annualization factor based on data frequency.
    
    Args:
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'hourly').
        
    Returns:
        Annualization factor.
    """
    factors = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "hourly": 252 * 24,  # Assuming 24 trading hours per day
        "minute": 252 * 24 * 60,
    }
    
    return factors.get(frequency.lower(), 252)  # Default to daily


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dictionary."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "calmar_ratio": 0.0,
        "omega_ratio": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
        "treynor_ratio": 0.0,
        "ulcer_index": 0.0,
        "gain_loss_ratio": 0.0,
        "tail_ratio": 0.0,
    }


def _calculate_return_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    initial_capital: float,
    ann_factor: float,
) -> Dict[str, float]:
    """Calculate return-related metrics.
    
    Args:
        returns: Clean returns series.
        equity_curve: Clean equity curve.
        initial_capital: Initial capital.
        ann_factor: Annualization factor.
        
    Returns:
        Dictionary with return metrics.
    """
    metrics = {}
    
    # Total return
    if len(equity_curve) > 0:
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    else:
        total_return = 0.0
    metrics["total_return"] = total_return
    
    # Annualized return
    if len(returns) > 0:
        ann_return = returns.mean() * ann_factor
    else:
        ann_return = 0.0
    metrics["annualized_return"] = ann_return
    
    # Cumulative returns
    metrics["cumulative_return"] = total_return
    
    # CAGR (Compound Annual Growth Rate)
    if len(equity_curve) > 1:
        n_periods = len(equity_curve)
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (ann_factor / n_periods) - 1
    else:
        cagr = 0.0
    metrics["cagr"] = cagr
    
    # Positive/Negative returns
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    metrics["avg_positive_return"] = positive_returns.mean() if len(positive_returns) > 0 else 0.0
    metrics["avg_negative_return"] = negative_returns.mean() if len(negative_returns) > 0 else 0.0
    metrics["best_day"] = returns.max() if len(returns) > 0 else 0.0
    metrics["worst_day"] = returns.min() if len(returns) > 0 else 0.0
    
    return metrics


def _calculate_risk_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    ann_factor: float,
) -> Dict[str, float]:
    """Calculate risk-related metrics.
    
    Args:
        returns: Clean returns series.
        equity_curve: Clean equity curve.
        drawdown: Clean drawdown series.
        ann_factor: Annualization factor.
        
    Returns:
        Dictionary with risk metrics.
    """
    metrics = {}
    
    # Volatility (annualized)
    if len(returns) > 1:
        volatility = returns.std() * np.sqrt(ann_factor)
    else:
        volatility = 0.0
    metrics["volatility"] = volatility
    
    # Downside deviation (annualized)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
    else:
        downside_deviation = 0.0
    metrics["downside_deviation"] = downside_deviation
    
    # Maximum drawdown
    if len(drawdown) > 0:
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0
    metrics["max_drawdown"] = max_drawdown
    
    # Average drawdown
    if len(drawdown) > 0:
        avg_drawdown = drawdown[drawdown < 0].mean()
    else:
        avg_drawdown = 0.0
    metrics["avg_drawdown"] = avg_drawdown if not np.isnan(avg_drawdown) else 0.0
    
    # Drawdown duration statistics
    if len(drawdown) > 0:
        underwater = drawdown < 0
        if underwater.any():
            underwater_periods = underwater.astype(int).groupby(
                (~underwater).astype(int).cumsum()
            ).sum()
            metrics["max_drawdown_duration"] = underwater_periods.max()
            metrics["avg_drawdown_duration"] = underwater_periods.mean()
        else:
            metrics["max_drawdown_duration"] = 0.0
            metrics["avg_drawdown_duration"] = 0.0
    
    # Value at Risk (95%)
    if len(returns) > 0:
        var_95 = returns.quantile(0.05)
    else:
        var_95 = 0.0
    metrics["var_95"] = var_95
    
    # Conditional Value at Risk (95%)
    if len(returns) > 0:
        cvar_95 = returns[returns <= var_95].mean()
    else:
        cvar_95 = 0.0
    metrics["cvar_95"] = cvar_95 if not np.isnan(cvar_95) else 0.0
    
    # Ulcer Index
    if len(drawdown) > 0:
        ulcer_index = np.sqrt((drawdown ** 2).mean())
    else:
        ulcer_index = 0.0
    metrics["ulcer_index"] = ulcer_index
    
    return metrics


def _calculate_risk_adjusted_metrics(
    returns: pd.Series,
    risk_free_rate: float,
    ann_factor: float,
) -> Dict[str, float]:
    """Calculate risk-adjusted return metrics.
    
    Args:
        returns: Clean returns series.
        risk_free_rate: Annual risk-free rate.
        ann_factor: Annualization factor.
        
    Returns:
        Dictionary with risk-adjusted metrics.
    """
    metrics = {}
    
    if len(returns) < 2:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "treynor_ratio": 0.0,
            "gain_loss_ratio": 0.0,
            "tail_ratio": 0.0,
        }
    
    # Annualized metrics
    ann_return = returns.mean() * ann_factor
    ann_vol = returns.std() * np.sqrt(ann_factor)
    
    # Sharpe ratio
    if ann_vol > 0:
        sharpe = (ann_return - risk_free_rate) / ann_vol
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = sharpe
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_vol = downside_returns.std() * np.sqrt(ann_factor)
        if downside_vol > 0:
            sortino = (ann_return - risk_free_rate) / downside_vol
        else:
            sortino = 0.0
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = sortino
    
    # Calmar ratio (CAGR / Max Drawdown)
    # Note: We need max_drawdown from other function
    # This will be filled in later
    
    # Omega ratio
    threshold = risk_free_rate / ann_factor  # Daily equivalent
    excess_returns = returns - threshold
    
    positive_excess = excess_returns[excess_returns > 0].sum()
    negative_excess = abs(excess_returns[excess_returns < 0].sum())
    
    if negative_excess > 0:
        omega = positive_excess / negative_excess
    else:
        omega = float('inf') if positive_excess > 0 else 0.0
    metrics["omega_ratio"] = min(omega, 100.0) if omega != float('inf') else 100.0
    
    # Gain/Loss ratio
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) > 0 and negative_returns.mean() != 0:
        gain_loss = positive_returns.mean() / abs(negative_returns.mean())
    else:
        gain_loss = 0.0
    metrics["gain_loss_ratio"] = gain_loss
    
    # Tail ratio (95th percentile / 5th percentile)
    if len(returns) >= 10:
        tail_95 = returns.quantile(0.95)
        tail_5 = returns.quantile(0.05)
        if tail_5 != 0:
            tail_ratio = tail_95 / abs(tail_5)
        else:
            tail_ratio = 0.0
    else:
        tail_ratio = 0.0
    metrics["tail_ratio"] = tail_ratio
    
    # Treynor ratio (requires beta)
    # Will be calculated separately if benchmark available
    
    return metrics


def _calculate_statistical_metrics(returns: pd.Series) -> Dict[str, Any]:
    """Calculate statistical metrics.
    
    Args:
        returns: Clean returns series.
        
    Returns:
        Dictionary with statistical metrics.
    """
    metrics = {}
    
    if len(returns) < 2:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "jarque_bera": 0.0,
            "jarque_bera_pvalue": 1.0,
        }
    
    # Skewness
    skew = returns.skew()
    metrics["skewness"] = 0.0 if np.isnan(skew) else skew
    
    # Kurtosis
    kurt = returns.kurtosis()
    metrics["kurtosis"] = 0.0 if np.isnan(kurt) else kurt
    
    # Jarque-Bera test for normality
    if len(returns) >= 4:
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            metrics["jarque_bera"] = jb_stat
            metrics["jarque_bera_pvalue"] = jb_pvalue
        except Exception:
            metrics["jarque_bera"] = 0.0
            metrics["jarque_bera_pvalue"] = 1.0
    else:
        metrics["jarque_bera"] = 0.0
        metrics["jarque_bera_pvalue"] = 1.0
    
    # Autocorrelation (lag 1)
    if len(returns) >= 3:
        try:
            autocorr = returns.autocorr(lag=1)
            metrics["autocorrelation_lag1"] = 0.0 if np.isnan(autocorr) else autocorr
        except Exception:
            metrics["autocorrelation_lag1"] = 0.0
    else:
        metrics["autocorrelation_lag1"] = 0.0
    
    # Hurst exponent (rough estimate)
    if len(returns) >= 20:
        try:
            hurst = _estimate_hurst(returns)
            metrics["hurst_exponent"] = hurst
        except Exception:
            metrics["hurst_exponent"] = 0.5
    else:
        metrics["hurst_exponent"] = 0.5
    
    return metrics


def _estimate_hurst(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent using R/S analysis.
    
    Args:
        series: Time series.
        max_lag: Maximum lag for R/S calculation.
        
    Returns:
        Estimated Hurst exponent.
    """
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    lag_vec = []
    
    for lag in lags:
        # Divide series into chunks
        chunks = len(series) // lag
        if chunks < 2:
            continue
        
        rs_vals = []
        for i in range(chunks):
            chunk = series.iloc[i*lag:(i+1)*lag].values
            if len(chunk) < 2:
                continue
            
            # Calculate R/S for chunk
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            z = np.cumsum(deviations)
            r = np.max(z) - np.min(z)
            s = np.std(chunk)
            
            if s > 0:
                rs_vals.append(r / s)
        
        if rs_vals:
            tau.append(np.log(np.mean(rs_vals)))
            lag_vec.append(np.log(lag))
    
    if len(tau) < 2:
        return 0.5
    
    # Fit line to log-log plot
    hurst, _ = np.polyfit(lag_vec, tau, 1)
    return hurst


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: float = 252,
) -> Tuple[float, float, Dict[str, float]]:
    """Calculate alpha, beta, and related statistics.
    
    Args:
        strategy_returns: Strategy returns series.
        benchmark_returns: Benchmark returns series.
        risk_free_rate: Annual risk-free rate.
        ann_factor: Annualization factor.
        
    Returns:
        Tuple of (alpha, beta, additional_stats).
    """
    # Align returns
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0, {}
    
    strategy_aligned = aligned.iloc[:, 0]
    benchmark_aligned = aligned.iloc[:, 1]
    
    # Calculate excess returns
    rf_daily = risk_free_rate / ann_factor
    strategy_excess = strategy_aligned - rf_daily
    benchmark_excess = benchmark_aligned - rf_daily
    
    # Linear regression
    x = benchmark_excess.values.reshape(-1, 1)
    y = strategy_excess.values
    
    # Add constant for intercept (alpha)
    x_with_const = np.hstack([np.ones((len(x), 1)), x])
    
    try:
        # Solve using least squares
        params = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        alpha = params[0]
        beta = params[1]
        
        # Calculate residuals
        y_pred = x_with_const.dot(params)
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Tracking error
        tracking_error = np.std(residuals) * np.sqrt(ann_factor)
        
        # Information ratio
        avg_excess_return = np.mean(strategy_excess) * ann_factor
        information_ratio = avg_excess_return / tracking_error if tracking_error > 0 else 0.0
        
        # Treynor ratio
        treynor_ratio = (np.mean(strategy_excess) * ann_factor) / beta if beta != 0 else 0.0
        
        stats = {
            "alpha": alpha * ann_factor,  # Annualized
            "beta": beta,
            "r_squared": r_squared,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "treynor_ratio": treynor_ratio,
            "residual_std": np.std(residuals),
        }
        
        return alpha * ann_factor, beta, stats
        
    except Exception as e:
        warnings.warn(f"Error calculating alpha/beta: {e}")
        return 0.0, 0.0, {}


def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """Calculate metrics based on trades.
    
    Args:
        trades: DataFrame with trade information.
        
    Returns:
        Dictionary with trade metrics.
    """
    if trades.empty:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade": 0.0,
            "total_trades": 0,
            "total_winning_trades": 0,
            "total_losing_trades": 0,
            "avg_holding_period": 0.0,
        }
    
    metrics = {}
    
    # Basic counts
    total_trades = len(trades)
    metrics["total_trades"] = total_trades
    
    # PNL-based metrics
    if 'pnl' in trades.columns:
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_winning = len(winning_trades)
        total_losing = len(losing_trades)
        
        metrics["total_winning_trades"] = float(total_winning)
        metrics["total_losing_trades"] = float(total_losing)
        metrics["win_rate"] = total_winning / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average win/loss
        metrics["avg_win"] = winning_trades['pnl'].mean() if total_winning > 0 else 0.0
        metrics["avg_loss"] = losing_trades['pnl'].mean() if total_losing > 0 else 0.0
        metrics["avg_trade"] = trades['pnl'].mean()
        
        # Largest win/loss
        metrics["largest_win"] = winning_trades['pnl'].max() if total_winning > 0 else 0.0
        metrics["largest_loss"] = losing_trades['pnl'].min() if total_losing > 0 else 0.0
        
        # Expectancy
        avg_win = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]
        win_rate = metrics["win_rate"]
        loss_rate = 1 - win_rate
        
        metrics["expectancy"] = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    
    # Holding period metrics
    if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
        try:
            holding_periods = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / (24 * 3600)
            metrics["avg_holding_period"] = holding_periods.mean()
            metrics["median_holding_period"] = holding_periods.median()
            metrics["max_holding_period"] = holding_periods.max()
            metrics["min_holding_period"] = holding_periods.min()
        except Exception:
            pass
    
    return metrics


def _validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Validate and clean metrics.
    
    Args:
        metrics: Raw metrics dictionary.
        
    Returns:
        Validated metrics dictionary.
    """
    validated = {}
    
    for key, value in metrics.items():
        # Replace NaN and infinite values with 0
        if np.isnan(value) or np.isinf(value):
            validated[key] = 0.0
        else:
            validated[key] = float(value)
    
    # Ensure required metrics exist
    required = [
        "total_return", "annualized_return", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "win_rate",
    ]
    
    for metric in required:
        if metric not in validated:
            validated[metric] = 0.0
    
    # Calculate Calmar ratio
    if "cagr" in validated and "max_drawdown" in validated:
        cagr = validated["cagr"]
        max_dd = abs(validated["max_drawdown"])
        validated["calmar_ratio"] = cagr / max_dd if max_dd > 0 else 0.0
    
    return validated