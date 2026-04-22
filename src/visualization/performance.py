# 性能分析与比较
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import logging

from .charts import set_plot_style
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """计算综合性能指标。

    参数：
        returns: 策略收益序列。
        risk_free_rate: 年化无风险利率。

    返回：
        包含性能指标的字典。
    """
    if returns.empty or len(returns) < 2:
        return {}
    
    metrics = {}
    
    # 基本收益指标
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = returns.mean() * 252
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    
    # 风险调整后收益
    if metrics['annualized_volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['annualized_volatility']
    else:
        metrics['sharpe_ratio'] = 0
    
    # 索提诺比率（使用下行偏差）
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 1:
        downside_std = negative_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_std
        else:
            metrics['sortino_ratio'] = 0
    else:
        metrics['sortino_ratio'] = 0
    
    # 最大回撤
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # 卡尔玛比率
    if metrics['max_drawdown'] < 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0
    
    # 胜率和盈亏比
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics['profit_factor'] = (
        positive_returns.sum() / abs(negative_returns.sum())
        if negative_returns.sum() < 0 else np.inf
    )
    
    # 平均盈利/亏损
    metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
    metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
    metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] < 0 else np.inf
    
    # 偏度和峰度
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # 风险价值（VaR）和条件风险价值（CVaR）
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    
    # 最佳和最差时期
    rolling_21 = returns.rolling(window=21).sum()  # Monthly
    rolling_252 = returns.rolling(window=252).sum()  # Annual
    
    metrics['best_month'] = rolling_21.max() if not rolling_21.empty else 0
    metrics['worst_month'] = rolling_21.min() if not rolling_21.empty else 0
    metrics['best_year'] = rolling_252.max() if not rolling_252.empty else 0
    metrics['worst_year'] = rolling_252.min() if not rolling_252.empty else 0
    
    # 一致性指标
    monthly_returns = returns.resample('M').sum()
    metrics['positive_month_rate'] = (monthly_returns > 0).mean() if not monthly_returns.empty else 0
    
    # 恢复因子（简化版）
    if metrics['max_drawdown'] < 0:
        # Estimate recovery time as time from max drawdown to new high
        drawdown_dates = drawdown[drawdown == metrics['max_drawdown']]
        if not drawdown_dates.empty:
            max_dd_date = drawdown_dates.index[0]
            recovery_idx = equity_curve[equity_curve.index > max_dd_date]
            if not recovery_idx.empty:
                new_highs = recovery_idx[recovery_idx > equity_curve.loc[max_dd_date]]
                if not new_highs.empty:
                    recovery_date = new_highs.index[0]
                    recovery_days = (recovery_date - max_dd_date).days
                    metrics['recovery_factor'] = abs(metrics['max_drawdown']) / recovery_days if recovery_days > 0 else 0
                else:
                    metrics['recovery_factor'] = 0
            else:
                metrics['recovery_factor'] = 0
        else:
            metrics['recovery_factor'] = 0
    else:
        metrics['recovery_ratio'] = 0
    
    return metrics


def compare_strategies(
    strategy_results: Dict[str, Dict[str, Any]],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """比较多个策略。

    参数：
        strategy_results: 策略名称到结果字典的映射。
        benchmark_returns: 基准收益序列。
        risk_free_rate: 年化无风险利率。

    返回：
        包含比较指标的DataFrame。
    """
    comparison_data = []
    
    # 如果提供了基准，则添加基准
    if benchmark_returns is not None:
        benchmark_metrics = calculate_performance_metrics(benchmark_returns, risk_free_rate)
        benchmark_metrics_with_name = {**benchmark_metrics, 'strategy_name': 'Benchmark'}
        comparison_data.append(benchmark_metrics_with_name)
    
    # 为每个策略计算指标
    for strategy_name, results in strategy_results.items():
        returns = results.get('returns')
        if returns is None or returns.empty:
            logger.warning(f"No returns data for strategy {strategy_name}")
            continue
        
        metrics = calculate_performance_metrics(returns, risk_free_rate)
        # Add additional info from results
        metrics_with_name = {
            **metrics,
            'strategy_name': strategy_name,
            'total_trades': results.get('num_trades', 0),
            'initial_capital': results.get('initial_capital', 0),
            'final_equity': results.get('final_equity', 0),
        }
        
        comparison_data.append(metrics_with_name)
    
    # 创建比较DataFrame
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 将策略名称设置为索引
    comparison_df.set_index('strategy_name', inplace=True)
    
    # 按夏普比率降序排序
    if 'sharpe_ratio' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
    
    return comparison_df


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    theme: str = "dark",
):
    """绘制策略比较图。

    参数：
        comparison_df: 包含策略比较指标的DataFrame。
        metrics: 要绘制的指标列表。如果为None，则使用默认指标。
        title: 图表标题。
        save_path: 保存图片的路径。
        figsize: 图表尺寸。
        theme: 绘图主题。
    """
    if comparison_df.empty:
        logger.warning("No data for strategy comparison")
        return
    
    set_plot_style(theme)
    
    if metrics is None:
        metrics = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
        ]
    
    # 过滤可用的指标
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not available_metrics:
        logger.warning("No available metrics to plot")
        return
    
    # 创建子图
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 创建条形图
        bars = ax.bar(range(len(comparison_df)), comparison_df[metric].values)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right', fontsize=9)
        
        # 根据指标值为条形着色（绿色表示好，红色表示差）
        for i, bar in enumerate(bars):
            value = comparison_df[metric].iloc[i]
            
            # Determine color based on metric
            if metric in ['max_drawdown', 'worst_month', 'worst_year']:
                # For negative metrics, more negative = redder
                color_intensity = min(abs(value) * 5, 1.0) if value < 0 else 0.0
                color = (1.0, 1.0 - color_intensity, 1.0 - color_intensity)
            elif metric in ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'win_rate', 'total_return', 'annualized_return']:
                # For positive metrics, higher = greener
                color_intensity = min(value * 2, 1.0) if value > 0 else 0.0
                color = (1.0 - color_intensity, 1.0, 1.0 - color_intensity)
            else:
                color = '#1f77b4'
            
            bar.set_color(color)
            bar.set_edgecolor('white')
        
        # 在条形顶部添加数值标签
        for i, bar in enumerate(bars):
            value = comparison_df[metric].iloc[i]
            
            # Format value based on metric
            if metric in ['total_return', 'annualized_return', 'max_drawdown', 
                         'best_month', 'worst_month', 'best_year', 'worst_year',
                         'var_95', 'cvar_95', 'avg_win', 'avg_loss']:
                label = f'{value:.2%}'
            elif metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                           'win_loss_ratio', 'recovery_factor']:
                label = f'{value:.2f}'
            elif metric in ['win_rate', 'positive_month_rate']:
                label = f'{value:.1%}'
            elif metric in ['profit_factor']:
                label = f'{value:.1f}'
            else:
                label = f'{value:.2f}'
            
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   label, ha='center', va='bottom', fontsize=8)
    
    # 隐藏未使用的子图
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved strategy comparison to {save_path}")
    
    return fig


def plot_correlation_heatmap(
    strategy_returns: Dict[str, pd.Series],
    title: str = "Strategy Returns Correlation",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    theme: str = "dark",
):
    """绘制策略收益相关性热图。

    参数：
        strategy_returns: 策略名称到收益序列的映射字典。
        title: 图表标题。
        save_path: 保存图片的路径。
        figsize: 图表尺寸。
        theme: 绘图主题。
    """
    if not strategy_returns:
        logger.warning("No strategy returns for correlation heatmap")
        return
    
    set_plot_style(theme)
    
    # 对齐收益序列
    returns_list = []
    strategy_names = []
    
    for name, returns in strategy_returns.items():
        if returns is not None and not returns.empty:
            returns_list.append(returns)
            strategy_names.append(name)
    
    if len(returns_list) < 2:
        logger.warning("Need at least 2 strategies for correlation heatmap")
        return
    
    # 使用对齐的收益创建DataFrame
    returns_df = pd.concat(returns_list, axis=1)
    returns_df.columns = strategy_names
    returns_df = returns_df.dropna()
    
    # 计算相关性矩阵
    corr_matrix = returns_df.corr()
    
    # 创建热图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建上三角掩码
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 绘制热图
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot=True, fmt='.2f', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
    
    return fig


def plot_returns_scatter(
    strategy_returns: Dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Strategy Returns vs Benchmark",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    theme: str = "dark",
):
    """绘制策略收益与基准收益的散点图。

    参数：
        strategy_returns: 策略名称到收益序列的映射字典。
        benchmark_returns: 基准收益序列。
        title: 图表标题。
        save_path: 保存图片的路径。
        figsize: 图表尺寸。
        theme: 绘图主题。
    """
    if not strategy_returns:
        logger.warning("No strategy returns for scatter plot")
        return
    
    if benchmark_returns is None:
        logger.warning("No benchmark returns for scatter plot")
        return
    
    set_plot_style(theme)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 将基准收益与策略收益对齐
    benchmark_aligned = benchmark_returns.copy()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategy_returns)))
    
    for (strategy_name, returns), color in zip(strategy_returns.items(), colors):
        if returns is None or returns.empty:
            continue
        
        # Align returns with benchmark
        aligned_data = pd.concat([returns, benchmark_aligned], axis=1).dropna()
        if len(aligned_data) < 2:
            continue
        
        strategy_returns_aligned = aligned_data.iloc[:, 0]
        benchmark_returns_aligned = aligned_data.iloc[:, 1]
        
        # 计算alpha和beta
        cov = np.cov(strategy_returns_aligned, benchmark_returns_aligned)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
        alpha = strategy_returns_aligned.mean() - beta * benchmark_returns_aligned.mean()
        
        # 绘制散点图
        ax.scatter(benchmark_returns_aligned * 100, strategy_returns_aligned * 100,
                  alpha=0.6, s=50, color=color, label=f'{strategy_name} (β={beta:.2f})')
    
    # 添加回归线
    if len(strategy_returns) == 1:
        # For single strategy, plot regression line
        for strategy_name, returns in strategy_returns.items():
            if returns is None or returns.empty:
                continue
            
            aligned_data = pd.concat([returns, benchmark_aligned], axis=1).dropna()
            if len(aligned_data) < 2:
                continue
            
            x = aligned_data.iloc[:, 1].values * 100
            y = aligned_data.iloc[:, 0].values * 100
            
            # Fit regression line
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            
            # Plot regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = poly(x_line)
            ax.plot(x_line, y_line, '--', color='red', alpha=0.7, linewidth=2)
    
    # 添加对角线（y = x）
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Benchmark Line')
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Benchmark Daily Return (%)', fontsize=12)
    ax.set_ylabel('Strategy Daily Return (%)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved returns scatter plot to {save_path}")
    
    return fig


def create_performance_attribution(
    strategy_returns: pd.Series,
    factor_returns: Dict[str, pd.Series],
    title: str = "Performance Attribution",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    theme: str = "dark",
):
    """创建绩效归因图。

    参数：
        strategy_returns: 策略收益序列。
        factor_returns: 因子名称到收益序列的映射字典。
        title: 图表标题。
        save_path: 保存图片的路径。
        figsize: 图表尺寸。
        theme: 绘图主题。
    """
    if strategy_returns.empty:
        logger.warning("No strategy returns for performance attribution")
        return
    
    if not factor_returns:
        logger.warning("No factor returns for performance attribution")
        return
    
    set_plot_style(theme)
    
    # 对齐所有收益序列
    all_returns = [strategy_returns] + list(factor_returns.values())
    all_names = ['Strategy'] + list(factor_returns.keys())
    
    aligned_returns = pd.concat(all_returns, axis=1).dropna()
    aligned_returns.columns = all_names
    
    if aligned_returns.empty:
        logger.warning("No overlapping data for performance attribution")
        return
    
    # 计算累计收益
    cumulative_returns = (1 + aligned_returns).cumprod() - 1
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制累计收益
    for column in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[column] * 100,
                label=column, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance attribution to {save_path}")
    
    return fig