# Visualization functions for trading results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
from typing import Optional, Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def set_plot_style(theme: str = "dark"):
    """Set matplotlib plot style.

    Args:
        theme: Plot theme ('dark' or 'light').
    """
    if theme == "dark":
        plt.style.use('dark_background')
        sns.set_style("darkgrid", {"axes.facecolor": ".1"})
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_style("whitegrid")
    
    # Set better default colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


def create_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    theme: str = "dark",
) -> plt.Figure:
    """Create equity curve visualization.

    Args:
        equity_curve: Strategy equity curve series.
        benchmark: Benchmark equity curve (e.g., buy and hold).
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot strategy equity curve
    ax.plot(equity_curve.index, equity_curve.values, 
            label='Strategy', linewidth=2, color='#00ff00')
    
    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, 
                label='Benchmark', linewidth=2, color='#ff7f0e', alpha=0.7)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    # Add value at last point
    last_value = equity_curve.iloc[-1]
    ax.annotate(f'${last_value:,.0f}', 
                xy=(equity_curve.index[-1], last_value),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='white' if theme == 'dark' else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve to {save_path}")
    
    return fig


def create_drawdown_chart(
    drawdown: pd.Series,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    theme: str = "dark",
) -> plt.Figure:
    """Create drawdown visualization.

    Args:
        drawdown: Drawdown series (negative values).
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Fill area under drawdown curve
    ax.fill_between(drawdown.index, drawdown.values, 0, 
                    where=drawdown.values < 0,
                    color='red', alpha=0.3, label='Drawdown')
    
    # Plot drawdown line
    ax.plot(drawdown.index, drawdown.values, 
            color='red', linewidth=1, alpha=0.7)
    
    # Add horizontal line at max drawdown
    max_dd = drawdown.min()
    if not np.isnan(max_dd):
        ax.axhline(y=max_dd, color='orange', linestyle='--', 
                  linewidth=1, label=f'Max DD: {max_dd:.1%}')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved drawdown chart to {save_path}")
    
    return fig


def create_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    theme: str = "dark",
) -> plt.Figure:
    """Create returns distribution visualization.

    Args:
        returns: Strategy returns series.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(returns.dropna() * 100, bins=50, edgecolor='black', 
             alpha=0.7, color='#1f77b4')
    ax1.set_title('Returns Histogram', fontsize=14)
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_return = returns.mean() * 100
    median_return = returns.median() * 100
    ax1.axvline(mean_return, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_return:.2f}%')
    ax1.axvline(median_return, color='green', linestyle='--', 
                linewidth=2, label=f'Median: {median_return:.2f}%')
    ax1.legend(fontsize=9)
    
    # Box plot
    ax2.boxplot(returns.dropna() * 100, vert=False, patch_artist=True,
                boxprops=dict(facecolor='#1f77b4'))
    ax2.set_title('Returns Box Plot', fontsize=14)
    ax2.set_xlabel('Daily Return (%)', fontsize=12)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f'Mean: {returns.mean() * 100:.2f}%\n'
        f'Std: {returns.std() * 100:.2f}%\n'
        f'Skew: {returns.skew():.2f}\n'
        f'Kurtosis: {returns.kurtosis():.2f}\n'
        f'Min: {returns.min() * 100:.2f}%\n'
        f'Max: {returns.max() * 100:.2f}%'
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved returns distribution to {save_path}")
    
    return fig


def create_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    title: str = "Rolling Performance Metrics",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    theme: str = "dark",
) -> plt.Figure:
    """Create rolling performance metrics visualization.

    Args:
        returns: Strategy returns series.
        window: Rolling window size (trading days).
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    # Calculate rolling metrics
    rolling_mean = returns.rolling(window=window).mean() * 252 * 100  # Annualized %
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized %
    rolling_sharpe = rolling_mean / rolling_std
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Rolling annualized return
    axes[0].plot(rolling_mean.index, rolling_mean.values, 
                 color='green', linewidth=2)
    axes[0].set_ylabel('Ann. Return (%)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Rolling Annualized Return', fontsize=14)
    
    # Rolling annualized volatility
    axes[1].plot(rolling_std.index, rolling_std.values, 
                 color='blue', linewidth=2)
    axes[1].set_ylabel('Ann. Volatility (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Rolling Annualized Volatility', fontsize=14)
    
    # Rolling Sharpe ratio
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, 
                 color='purple', linewidth=2)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Rolling Sharpe Ratio', fontsize=14)
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    fig.autofmt_xdate()
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved rolling metrics to {save_path}")
    
    return fig


def create_trade_analysis(
    signals: pd.DataFrame,
    prices: pd.Series,
    title: str = "Trade Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    theme: str = "dark",
) -> plt.Figure:
    """Create trade analysis visualization.

    Args:
        signals: DataFrame with trading signals.
        prices: Price series.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Plot price with buy/sell signals
    ax1.plot(prices.index, prices.values, color='white', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Identify buy and sell signals
    if 'signal' in signals.columns:
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, 
                       prices.loc[buy_signals.index], 
                       color='green', s=100, marker='^', 
                       label='Buy', zorder=5)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, 
                       prices.loc[sell_signals.index], 
                       color='red', s=100, marker='v', 
                       label='Sell', zorder=5)
        
        ax1.legend(loc='upper left', fontsize=10)
    
    # Plot position
    if 'position' in signals.columns:
        ax2.fill_between(signals.index, 0, signals['position'], 
                        where=signals['position'] > 0,
                        color='green', alpha=0.3, label='Long')
        ax2.fill_between(signals.index, 0, signals['position'], 
                        where=signals['position'] < 0,
                        color='red', alpha=0.3, label='Short')
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_ylim([-1.5, 1.5])
        ax2.legend(loc='upper left', fontsize=10)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trade analysis to {save_path}")
    
    return fig


def create_interactive_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Interactive Equity Curve",
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create interactive equity curve using Plotly.

    Args:
        equity_curve: Strategy equity curve series.
        benchmark: Benchmark equity curve.
        title: Chart title.
        save_path: Path to save HTML file.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    
    # Add strategy equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#00ff00', width=2),
        hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add benchmark if provided
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='%Y-%m'
        ),
        yaxis=dict(
            title='Equity ($)',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickprefix='$'
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved interactive equity curve to {save_path}")
    
    return fig


def create_performance_dashboard(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    theme: str = "dark",
) -> plt.Figure:
    """Create comprehensive performance dashboard.

    Args:
        results: Backtest results dictionary.
        save_path: Path to save the figure.
        figsize: Figure size.
        theme: Plot theme.

    Returns:
        Matplotlib figure.
    """
    set_plot_style(theme)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    equity_curve = results.get('equity_curve')
    returns = results.get('returns')
    drawdown = results.get('drawdown')
    signals = results.get('signals')
    
    # 1. Equity Curve (top left, spans 2 columns)
    ax1 = plt.subplot(gs[0, :2])
    if equity_curve is not None:
        ax1.plot(equity_curve.index, equity_curve.values, 
                color='green', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14)
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown (top right)
    ax2 = plt.subplot(gs[0, 2])
    if drawdown is not None:
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        where=drawdown.values < 0,
                        color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, 
                color='red', linewidth=1, alpha=0.7)
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 3. Returns Distribution (middle left)
    ax3 = plt.subplot(gs[1, 0])
    if returns is not None:
        ax3.hist(returns.dropna() * 100, bins=30, edgecolor='black', 
                alpha=0.7, color='blue')
        ax3.set_title('Returns Distribution', fontsize=14)
        ax3.set_xlabel('Return (%)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Returns (middle center)
    ax4 = plt.subplot(gs[1, 1])
    if returns is not None:
        cumulative_returns = (1 + returns).cumprod() - 1
        ax4.plot(cumulative_returns.index, cumulative_returns.values * 100,
                color='purple', linewidth=2)
        ax4.set_title('Cumulative Returns', fontsize=14)
        ax4.set_ylabel('Return (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    # 5. Monthly Returns Heatmap (middle right)
    ax5 = plt.subplot(gs[1, 2])
    if returns is not None:
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create monthly returns matrix
        monthly_matrix = monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        ).last().unstack()
        
        # Plot heatmap
        im = ax5.imshow(monthly_matrix.values * 100, cmap='RdYlGn', 
                       aspect='auto', interpolation='nearest')
        ax5.set_title('Monthly Returns Heatmap', fontsize=14)
        ax5.set_xlabel('Month', fontsize=12)
        ax5.set_ylabel('Year', fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax5, label='Return (%)')
    
    # 6. Performance Metrics Table (bottom, spans 3 columns)
    ax6 = plt.subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create metrics table
    metrics_data = [
        ['Total Return', f"{results.get('total_return', 0):.2%}"],
        ['Annualized Return', f"{results.get('annualized_return', 0):.2%}"],
        ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"],
        ['Sortino Ratio', f"{results.get('sortino_ratio', 0):.2f}"],
        ['Max Drawdown', f"{results.get('max_drawdown', 0):.2%}"],
        ['Win Rate', f"{results.get('win_rate', 0):.2%}"],
        ['Number of Trades', f"{results.get('num_trades', 0):.0f}"],
    ]
    
    table = ax6.table(cellText=metrics_data,
                     colLabels=['Metric', 'Value'],
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(metrics_data) + 1):
        table[(i, 0)].set_facecolor('#2E2E2E' if theme == 'dark' else '#F0F0F0')
        table[(i, 1)].set_facecolor('#3E3E3E' if theme == 'dark' else '#F8F8F8')
    
    fig.suptitle(f"Performance Dashboard - {results.get('strategy_name', 'Strategy')}", 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance dashboard to {save_path}")
    
    return fig