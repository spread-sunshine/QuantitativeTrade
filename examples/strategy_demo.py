#!/usr/bin/env python3
"""
Example script demonstrating trading strategies.

This script shows how to:
1. Load market data
2. Create and configure trading strategies
3. Generate trading signals
4. Run backtests
5. Analyze results
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.strategies.moving_average import MovingAverageCrossover
from src.strategies.mean_reversion import MeanReversion
from src.strategies.breakout import BreakoutStrategy
from src.visualization.charts import create_equity_curve, create_drawdown_chart
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("strategy_demo")

def load_sample_data():
    """Load sample market data for demonstration."""
    logger.info("Loading sample market data...")
    
    # Create sample data if needed
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    n = len(dates)
    
    # Generate synthetic price data with trend and noise
    np.random.seed(42)
    trend = np.linspace(100, 200, n)
    noise = np.random.normal(0, 5, n)
    prices = trend + noise + 10 * np.sin(np.linspace(0, 10 * np.pi, n))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    logger.info(f"Created sample data with {len(data)} rows")
    return data

def demo_moving_average_crossover(data):
    """Demonstrate moving average crossover strategy."""
    logger.info("\n" + "="*60)
    logger.info("Moving Average Crossover Strategy Demo")
    logger.info("="*60)
    
    # Create strategy
    strategy = MovingAverageCrossover(
        short_window=20,
        long_window=50,
        name="MA_Crossover_Demo",
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0001
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Run backtest
    results = strategy.run_backtest(data)
    
    # Display results
    logger.info(f"\nStrategy: {results['strategy_name']}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Number of Trades: {results['num_trades']:.0f}")
    
    # Show signal statistics
    signal_stats = signals['signal'].value_counts()
    logger.info(f"\nSignal Statistics:")
    for signal, count in signal_stats.items():
        signal_type = {1: 'Buy', -1: 'Sell', 0: 'Hold'}.get(signal, 'Unknown')
        logger.info(f"  {signal_type}: {count} signals ({count/len(signals):.1%})")
    
    return strategy, signals, results

def demo_mean_reversion(data):
    """Demonstrate mean reversion strategy."""
    logger.info("\n" + "="*60)
    logger.info("Mean Reversion Strategy Demo")
    logger.info("="*60)
    
    # Create strategy
    strategy = MeanReversion(
        window=20,
        num_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        name="MeanReversion_Demo",
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0001
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Run backtest
    results = strategy.run_backtest(data)
    
    # Display results
    logger.info(f"\nStrategy: {results['strategy_name']}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Number of Trades: {results['num_trades']:.0f}")
    
    # Show RSI statistics
    if 'rsi' in signals.columns:
        rsi_stats = signals['rsi'].describe()
        logger.info(f"\nRSI Statistics:")
        logger.info(f"  Mean: {rsi_stats['mean']:.1f}")
        logger.info(f"  Min: {rsi_stats['min']:.1f}")
        logger.info(f"  Max: {rsi_stats['max']:.1f}")
        logger.info(f"  % in oversold (<30): {(signals['rsi'] < 30).mean():.1%}")
        logger.info(f"  % in overbought (>70): {(signals['rsi'] > 70).mean():.1%}")
    
    return strategy, signals, results

def demo_breakout_strategy(data):
    """Demonstrate breakout strategy."""
    logger.info("\n" + "="*60)
    logger.info("Breakout Strategy Demo")
    logger.info("="*60)
    
    # Create strategy
    strategy = BreakoutStrategy(
        lookback_period=20,
        atr_period=14,
        atr_multiplier=2.0,
        consolidation_period=10,
        min_consolidation_range=0.02,
        name="Breakout_Demo",
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0001
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Run backtest
    results = strategy.run_backtest(data)
    
    # Display results
    logger.info(f"\nStrategy: {results['strategy_name']}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Number of Trades: {results['num_trades']:.0f}")
    
    # Show breakout statistics
    if 'is_consolidating' in signals.columns:
        consolidation_stats = signals['is_consolidating'].value_counts()
        logger.info(f"\nConsolidation Statistics:")
        for consolidated, count in consolidation_stats.items():
            status = "Consolidating" if consolidated else "Not Consolidating"
            logger.info(f"  {status}: {count} periods ({count/len(signals):.1%})")
    
    return strategy, signals, results

def compare_strategies(strategy_results):
    """Compare performance of multiple strategies."""
    logger.info("\n" + "="*60)
    logger.info("Strategy Comparison")
    logger.info("="*60)
    
    comparison_data = []
    
    for strategy_name, results in strategy_results.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return': results['total_return'],
            'Annualized Return': results.get('annualized_return', 0),
            'Sharpe Ratio': results['sharpe_ratio'],
            'Sortino Ratio': results['sortino_ratio'],
            'Win Rate': results['win_rate'],
            'Max Drawdown': results['max_drawdown'],
            'Number of Trades': results['num_trades'],
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison
    logger.info("\nPerformance Comparison:")
    for _, row in comparison_df.iterrows():
        logger.info(f"\n{row['Strategy']}:")
        logger.info(f"  Total Return: {row['Total Return']:.2%}")
        logger.info(f"  Sharpe Ratio: {row['Sharpe Ratio']:.2f}")
        logger.info(f"  Max Drawdown: {row['Max Drawdown']:.2%}")
        logger.info(f"  Win Rate: {row['Win Rate']:.2%}")
    
    # Find best strategy by Sharpe ratio
    best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
    logger.info(f"\nBest Strategy by Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
    
    # Find best strategy by total return
    best_return = comparison_df.loc[comparison_df['Total Return'].idxmax()]
    logger.info(f"Best Strategy by Total Return: {best_return['Strategy']} ({best_return['Total Return']:.2%})")
    
    # Find best strategy by max drawdown (least negative)
    best_drawdown = comparison_df.loc[comparison_df['Max Drawdown'].idxmax()]  # idxmax because drawdown is negative
    logger.info(f"Best Strategy by Max Drawdown: {best_drawdown['Strategy']} ({best_drawdown['Max Drawdown']:.2%})")
    
    return comparison_df

def main():
    """Main demonstration function."""
    logger.info("Starting trading strategy demonstration")
    
    # Load or create sample data
    data = load_sample_data()
    
    # Run strategy demos
    strategy_results = {}
    
    # Moving Average Crossover
    ma_strategy, ma_signals, ma_results = demo_moving_average_crossover(data)
    strategy_results['MA Crossover'] = ma_results
    
    # Mean Reversion
    mr_strategy, mr_signals, mr_results = demo_mean_reversion(data)
    strategy_results['Mean Reversion'] = mr_results
    
    # Breakout Strategy
    br_strategy, br_signals, br_results = demo_breakout_strategy(data)
    strategy_results['Breakout'] = br_results
    
    # Compare strategies
    comparison_df = compare_strategies(strategy_results)
    
    # Save results for visualization
    output_dir = project_root / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison table
    comparison_file = output_dir / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nComparison table saved to: {comparison_file}")
    
    # Save individual strategy results
    for strategy_name, results in strategy_results.items():
        # Save equity curve
        if 'equity_curve' in results and not results['equity_curve'].empty:
            equity_file = output_dir / f"{strategy_name.replace(' ', '_')}_equity.csv"
            results['equity_curve'].to_csv(equity_file)
            
        # Save signals
        if 'signals' in results and not results['signals'].empty:
            signals_file = output_dir / f"{strategy_name.replace(' ', '_')}_signals.csv"
            results['signals'].to_csv(signals_file)
    
    logger.info("\nStrategy demonstration completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Display next steps
    logger.info("\n" + "="*60)
    logger.info("Next Steps:")
    logger.info("1. Run real data backtests using examples/data_pipeline.py")
    logger.info("2. Modify strategy parameters for optimization")
    logger.info("3. Create custom strategies by extending BaseStrategy")
    logger.info("4. Use visualization tools to analyze results")
    logger.info("="*60)

if __name__ == "__main__":
    main()