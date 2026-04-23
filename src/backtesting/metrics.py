"""交易策略性能指标计算。"""

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
    """计算全面的性能指标。
    
    Args:
        returns: 策略收益序列。
        equity_curve: 权益曲线序列。
        drawdown: 回撤序列。
        initial_capital: 初始资金。
        risk_free_rate: 年化无风险利率。
        frequency: 数据频率（'daily', 'weekly', 'monthly'）。
        
    Returns:
        包含性能指标的字典。
    """
    if len(returns) < 2:
        return _empty_metrics()
    
    # 移除NaN值
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
    """根据数据频率获取年化因子。
    
    Args:
        frequency: 数据频率（'daily'、'weekly'、'monthly'、'hourly'）。
        
    Returns:
        年化因子。
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
    """返回空指标字典。"""
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
    """计算收益相关指标。
    
    Args:
        returns: 清洗后的收益序列。
        equity_curve: 清洗后的权益曲线。
        initial_capital: 初始资金。
        ann_factor: 年化因子。
        
    Returns:
        包含收益指标的字典。
    """
    metrics = {}
    
    # 总收益
    if len(equity_curve) > 0:
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    else:
        total_return = 0.0
    metrics["total_return"] = total_return
    
    # 年化收益
    if len(returns) > 0:
        ann_return = returns.mean() * ann_factor
    else:
        ann_return = 0.0
    metrics["annualized_return"] = ann_return
    
    # 累计收益
    metrics["cumulative_return"] = total_return
    
    # CAGR（复合年增长率）
    if len(equity_curve) > 1:
        n_periods = len(equity_curve)
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (ann_factor / n_periods) - 1
    else:
        cagr = 0.0
    metrics["cagr"] = cagr
    
    # 正/负收益
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
    """计算风险相关指标。
    
    Args:
        returns: 清洗后的收益序列。
        equity_curve: 清洗后的权益曲线。
        drawdown: 清洗后的回撤序列。
        ann_factor: 年化因子。
        
    Returns:
        包含风险指标的字典。
    """
    metrics = {}
    
    # 波动率（年化）
    if len(returns) > 1:
        volatility = returns.std() * np.sqrt(ann_factor)
    else:
        volatility = 0.0
    metrics["volatility"] = volatility
    
    # 下行偏差（年化）
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
    else:
        downside_deviation = 0.0
    metrics["downside_deviation"] = downside_deviation
    
    # 最大回撤
    if len(drawdown) > 0:
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0
    metrics["max_drawdown"] = max_drawdown
    
    # 平均回撤
    if len(drawdown) > 0:
        avg_drawdown = drawdown[drawdown < 0].mean()
    else:
        avg_drawdown = 0.0
    metrics["avg_drawdown"] = avg_drawdown if not np.isnan(avg_drawdown) else 0.0
    
    # 回撤持续时间统计
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
    
    # 风险价值（95%）
    if len(returns) > 0:
        var_95 = returns.quantile(0.05)
    else:
        var_95 = 0.0
    metrics["var_95"] = var_95
    
    # 条件风险价值（95%）
    if len(returns) > 0:
        cvar_95 = returns[returns <= var_95].mean()
    else:
        cvar_95 = 0.0
    metrics["cvar_95"] = cvar_95 if not np.isnan(cvar_95) else 0.0
    
    # Ulcer指数
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
    """计算风险调整后的收益指标。
    
    Args:
        returns: 清洗后的收益序列。
        risk_free_rate: 年化无风险利率。
        ann_factor: 年化因子。
        
    Returns:
        包含风险调整后收益指标的字典。
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
    
    # 年化指标
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
    
    # 卡玛比率（CAGR / 最大回撤）
    # 注意：需要从其他函数获取max_drawdown
    # 稍后会填充
    
    # Omega比率
    threshold = risk_free_rate / ann_factor  # Daily equivalent
    excess_returns = returns - threshold
    
    positive_excess = excess_returns[excess_returns > 0].sum()
    negative_excess = abs(excess_returns[excess_returns < 0].sum())
    
    if negative_excess > 0:
        omega = positive_excess / negative_excess
    else:
        omega = float('inf') if positive_excess > 0 else 0.0
    metrics["omega_ratio"] = min(omega, 100.0) if omega != float('inf') else 100.0
    
    # 盈亏比
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) > 0 and negative_returns.mean() != 0:
        gain_loss = positive_returns.mean() / abs(negative_returns.mean())
    else:
        gain_loss = 0.0
    metrics["gain_loss_ratio"] = gain_loss
    
    # 尾部比率（第95百分位 / 第5百分位）
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
    
    # 特雷诺比率（需要beta值）
    # 如果有基准数据，将单独计算
    
    return metrics


def _calculate_statistical_metrics(returns: pd.Series) -> Dict[str, Any]:
    """计算统计指标。
    
    Args:
        returns: 清洗后的收益序列。
        
    Returns:
        包含统计指标的字典。
    """
    metrics = {}
    
    if len(returns) < 2:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "jarque_bera": 0.0,
            "jarque_bera_pvalue": 1.0,
        }
    
    # 偏度
    skew = returns.skew()
    metrics["skewness"] = 0.0 if np.isnan(skew) else skew
    
    # 峰度
    kurt = returns.kurtosis()
    metrics["kurtosis"] = 0.0 if np.isnan(kurt) else kurt
    
    # Jarque-Bera正态性检验
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
    
    # 自相关（滞后1期）
    if len(returns) >= 3:
        try:
            autocorr = returns.autocorr(lag=1)
            metrics["autocorrelation_lag1"] = 0.0 if np.isnan(autocorr) else autocorr
        except Exception:
            metrics["autocorrelation_lag1"] = 0.0
    else:
        metrics["autocorrelation_lag1"] = 0.0
    
    # Hurst指数（粗略估计）
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
    """使用R/S分析估计Hurst指数。
    
    Args:
        series: 时间序列。
        max_lag: R/S计算的最大滞后阶数。
        
    Returns:
        估计的Hurst指数。
    """
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    lag_vec = []
    
    for lag in lags:
        # 将序列分割为块
        chunks = len(series) // lag
        if chunks < 2:
            continue
        
        rs_vals = []
        for i in range(chunks):
            chunk = series.iloc[i*lag:(i+1)*lag].values
            if len(chunk) < 2:
                continue
            
            # 计算块的R/S值
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
    
    # 对双对数图拟合直线
    hurst, _ = np.polyfit(lag_vec, tau, 1)
    return hurst


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: float = 252,
) -> Tuple[float, float, Dict[str, float]]:
    """计算alpha、beta及相关统计信息。
    
    Args:
        strategy_returns: 策略收益序列。
        benchmark_returns: 基准收益序列。
        risk_free_rate: 年化无风险利率。
        ann_factor: 年化因子。
        
    Returns:
        元组（alpha、beta、附加统计信息）。
    """
    # 对齐收益序列
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0, {}
    
    strategy_aligned = aligned.iloc[:, 0]
    benchmark_aligned = aligned.iloc[:, 1]
    
    # 计算超额收益
    rf_daily = risk_free_rate / ann_factor
    strategy_excess = strategy_aligned - rf_daily
    benchmark_excess = benchmark_aligned - rf_daily
    
    # 线性回归
    x = benchmark_excess.values.reshape(-1, 1)
    y = strategy_excess.values
    
    # 添加常数项（截距，即alpha）
    x_with_const = np.hstack([np.ones((len(x), 1)), x])
    
    try:
        # 使用最小二乘法求解
        params = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        alpha = params[0]
        beta = params[1]
        
        # 计算残差
        y_pred = x_with_const.dot(params)
        residuals = y - y_pred
        
        # R平方
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # 跟踪误差
        tracking_error = np.std(residuals) * np.sqrt(ann_factor)
        
        # 信息比率
        avg_excess_return = np.mean(strategy_excess) * ann_factor
        information_ratio = avg_excess_return / tracking_error if tracking_error > 0 else 0.0
        
        # 特雷诺比率
        treynor_ratio = (np.mean(strategy_excess) * ann_factor) / beta if beta != 0 else 0.0
        
        stats = {
            "alpha": alpha * ann_factor,  # 年化
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
    """计算基于交易的指标。
    
    Args:
        trades: 包含交易信息的DataFrame。
        
    Returns:
        包含交易指标的字典。
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
    
    # 基本计数
    total_trades = len(trades)
    metrics["total_trades"] = total_trades
    
    # 基于盈亏的指标
    if 'pnl' in trades.columns:
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_winning = len(winning_trades)
        total_losing = len(losing_trades)
        
        metrics["total_winning_trades"] = float(total_winning)
        metrics["total_losing_trades"] = float(total_losing)
        metrics["win_rate"] = total_winning / total_trades if total_trades > 0 else 0.0
        
        # 盈利因子
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 平均盈亏
        metrics["avg_win"] = winning_trades['pnl'].mean() if total_winning > 0 else 0.0
        metrics["avg_loss"] = losing_trades['pnl'].mean() if total_losing > 0 else 0.0
        metrics["avg_trade"] = trades['pnl'].mean()
        
        # 最大盈/亏
        metrics["largest_win"] = winning_trades['pnl'].max() if total_winning > 0 else 0.0
        metrics["largest_loss"] = losing_trades['pnl'].min() if total_losing > 0 else 0.0
        
        # 期望值
        avg_win = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]
        win_rate = metrics["win_rate"]
        loss_rate = 1 - win_rate
        
        metrics["expectancy"] = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    
    # 持仓期指标
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
    """验证和清理指标。
    
    Args:
        metrics: 原始指标字典。
        
    Returns:
        验证后的指标字典。
    """
    validated = {}
    
    for key, value in metrics.items():
        # 将NaN和无穷大值替换为0
        if np.isnan(value) or np.isinf(value):
            validated[key] = 0.0
        else:
            validated[key] = float(value)
    
    # 确保必需指标存在
    required = [
        "total_return", "annualized_return", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "win_rate",
    ]
    
    for metric in required:
        if metric not in validated:
            validated[metric] = 0.0
    
    # 计算卡玛比率
    if "cagr" in validated and "max_drawdown" in validated:
        cagr = validated["cagr"]
        max_dd = abs(validated["max_drawdown"])
        validated["calmar_ratio"] = cagr / max_dd if max_dd > 0 else 0.0
    
    return validated