# Report generation for trading results
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging

from .charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_returns_distribution,
    create_rolling_metrics,
    create_trade_analysis,
    create_performance_dashboard,
    create_interactive_equity_curve,
)
from ..utils.logger import setup_logger
from config.settings import REPORTS_DIR

logger = setup_logger(__name__)


class ReportGenerator:
    """Generates comprehensive reports for trading strategies."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports. If None, uses REPORTS_DIR.
        """
        if output_dir is None:
            output_dir = REPORTS_DIR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Report data
        self.results: Dict[str, Any] = {}
        self.strategy_name: str = "Unknown"
        self.report_date: datetime = datetime.now()

    def set_results(self, results: Dict[str, Any]):
        """Set backtest results for reporting.

        Args:
            results: Backtest results dictionary.
        """
        self.results = results
        self.strategy_name = results.get('strategy_name', 'Unknown')
        logger.info(f"Set results for strategy: {self.strategy_name}")

    def calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional performance metrics.

        Returns:
            Dictionary with additional metrics.
        """
        metrics: dict[str, float] = {}
        
        # Extract basic data
        returns = self.results.get('returns')
        equity_curve = self.results.get('equity_curve')
        drawdown = self.results.get('drawdown')
        
        if returns is None or returns.empty:
            logger.warning("No returns data for additional metrics")
            return metrics
        
        # Basic metrics already in results
        metrics.update({
            'total_return': self.results.get('total_return', 0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0),
            'sortino_ratio': self.results.get('sortino_ratio', 0),
            'max_drawdown': self.results.get('max_drawdown', 0),
            'win_rate': self.results.get('win_rate', 0),
            'num_trades': self.results.get('num_trades', 0),
        })
        
        # Calculate additional metrics
        # 1. Annualized metrics
        if len(returns) > 0:
            metrics['annualized_return'] = returns.mean() * 252
            metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # 2. Downside metrics
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_std'] = negative_returns.std() * np.sqrt(252)
            metrics['downside_capture'] = negative_returns.mean() / returns.mean() if returns.mean() != 0 else 0
        
        # 3. Risk-adjusted returns
        if metrics.get('annualized_volatility', 0) > 0:
            metrics['calmar_ratio'] = (
                metrics['annualized_return'] / abs(metrics['max_drawdown'])
                if metrics['max_drawdown'] != 0 else 0
            )
        
        # 4. Trade statistics
        if 'signals' in self.results and self.results['signals'] is not None:
            signals = self.results['signals']
            if 'signal' in signals.columns:
                trade_changes = signals['signal'].diff().fillna(0)
                metrics['num_buy_signals'] = (trade_changes == 1).sum()
                metrics['num_sell_signals'] = (trade_changes == -1).sum()
                metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(signals)
        
        # 5. Profit factor
        if 'returns' in self.results and self.results['returns'] is not None:
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # 6. Recovery factor
        if drawdown is not None and not drawdown.empty:
            metrics['recovery_factor'] = self._calculate_recovery_factor(drawdown)
        
        # 7. Value at Risk (VaR) and Conditional VaR
        if len(returns) > 0:
            metrics['var_95'] = np.percentile(returns, 5)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # 8. Skewness and kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # 9. Best and worst periods
        if len(returns) > 0:
            rolling_21 = returns.rolling(window=21).sum()  # Monthly returns
            metrics['best_month'] = rolling_21.max()
            metrics['worst_month'] = rolling_21.min()
            
            rolling_252 = returns.rolling(window=252).sum()  # Annual returns
            metrics['best_year'] = rolling_252.max()
            metrics['worst_year'] = rolling_252.min()
        
        # 10. Consistency metrics
        if len(returns) > 0:
            positive_months = (returns.resample('M').sum() > 0).mean()
            metrics['positive_month_rate'] = positive_months
            
            # Streaks
            metrics['longest_winning_streak'] = self._calculate_longest_streak(returns > 0)
            metrics['longest_losing_streak'] = self._calculate_longest_streak(returns < 0)
        
        return metrics

    def _calculate_avg_trade_duration(self, signals: pd.DataFrame) -> float:
        """Calculate average trade duration in days.

        Args:
            signals: DataFrame with trading signals.

        Returns:
            Average trade duration in days.
        """
        if 'position' not in signals.columns:
            return 0
        
        position_changes = signals['position'].diff().fillna(0)
        trade_starts = position_changes != 0
        
        if not trade_starts.any():
            return 0
        
        durations = []
        in_trade = False
        start_date = None
        
        for date, is_start in zip(signals.index, trade_starts):
            if is_start and not in_trade:
                # Start of new trade
                in_trade = True
                start_date = date
            elif is_start and in_trade:
                # End of previous trade, start of new
                if start_date:
                    duration = (date - start_date).days
                    durations.append(duration)
                start_date = date
            elif not is_start and not in_trade:
                # Not in trade
                continue
        
        # Handle last trade if still open
        if in_trade and start_date:
            duration = (signals.index[-1] - start_date).days
            durations.append(duration)
        
        return np.mean(durations) if durations else 0

    def _calculate_recovery_factor(self, drawdown: pd.Series) -> float:
        """Calculate recovery factor.

        Args:
            drawdown: Drawdown series.

        Returns:
            Recovery factor.
        """
        if drawdown.empty:
            return 0
        
        # Find major drawdown periods
        major_drawdowns = []
        in_drawdown = False
        drawdown_start = None
        max_dd_depth = 0
        
        for date, dd in zip(drawdown.index, drawdown.values):
            if dd < -0.05:  # Only consider drawdowns > 5%
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = date
                    max_dd_depth = dd
                else:
                    max_dd_depth = min(max_dd_depth, dd)
            elif in_drawdown:
                in_drawdown = False
                if drawdown_start:
                    recovery_days = (date - drawdown_start).days
                    if recovery_days > 0:
                        recovery_factor = abs(max_dd_depth) / recovery_days
                        major_drawdowns.append(recovery_factor)
        
        return np.mean(major_drawdowns) if major_drawdowns else 0

    def _calculate_longest_streak(self, condition: pd.Series) -> int:
        """Calculate longest streak where condition is True.

        Args:
            condition: Boolean series.

        Returns:
            Longest streak length.
        """
        if condition.empty:
            return 0
        
        current_streak = 0
        longest_streak = 0
        
        for value in condition.values:
            if value:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        
        return longest_streak

    def generate_visualizations(self, save_prefix: Optional[str] = None):
        """Generate all visualizations.

        Args:
            save_prefix: Prefix for saved files.
        """
        if save_prefix is None:
            save_prefix = self.strategy_name.replace(' ', '_')
        
        # Extract data
        equity_curve = self.results.get('equity_curve')
        returns = self.results.get('returns')
        drawdown = self.results.get('drawdown')
        signals = self.results.get('signals')
        
        # Generate visualizations
        if equity_curve is not None and not equity_curve.empty:
            # Equity curve
            equity_path = self.output_dir / f"{save_prefix}_equity_curve.png"
            create_equity_curve(
                equity_curve,
                title=f"{self.strategy_name} - Equity Curve",
                save_path=str(equity_path)
            )
            
            # Interactive equity curve (HTML)
            interactive_path = self.output_dir / f"{save_prefix}_interactive_equity.html"
            create_interactive_equity_curve(
                equity_curve,
                title=f"{self.strategy_name} - Equity Curve",
                save_path=str(interactive_path)
            )
        
        if drawdown is not None and not drawdown.empty:
            # Drawdown chart
            drawdown_path = self.output_dir / f"{save_prefix}_drawdown.png"
            create_drawdown_chart(
                drawdown,
                title=f"{self.strategy_name} - Drawdown",
                save_path=str(drawdown_path)
            )
        
        if returns is not None and not returns.empty:
            # Returns distribution
            returns_path = self.output_dir / f"{save_prefix}_returns_dist.png"
            create_returns_distribution(
                returns,
                title=f"{self.strategy_name} - Returns Distribution",
                save_path=str(returns_path)
            )
            
            # Rolling metrics
            rolling_path = self.output_dir / f"{save_prefix}_rolling_metrics.png"
            create_rolling_metrics(
                returns,
                title=f"{self.strategy_name} - Rolling Metrics",
                save_path=str(rolling_path)
            )
        
        if signals is not None and not signals.empty and 'close' in signals.columns:
            # Trade analysis
            trade_path = self.output_dir / f"{save_prefix}_trade_analysis.png"
            create_trade_analysis(
                signals,
                signals['close'],
                title=f"{self.strategy_name} - Trade Analysis",
                save_path=str(trade_path)
            )
        
        # Performance dashboard
        dashboard_path = self.output_dir / f"{save_prefix}_dashboard.png"
        create_performance_dashboard(
            self.results,
            save_path=str(dashboard_path)
        )
        
        logger.info(f"Generated visualizations in {self.output_dir}")

    def generate_text_report(self) -> str:
        """Generate text report.

        Returns:
            Formatted text report.
        """
        # Calculate metrics
        metrics = self.calculate_additional_metrics()
        
        # Create report header
        report = f"{'='*80}\n"
        report += f"STRATEGY PERFORMANCE REPORT\n"
        report += f"{'='*80}\n\n"
        
        report += f"Strategy: {self.strategy_name}\n"
        report += f"Report Date: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Period: {self.results.get('period', 'N/A')}\n"
        report += f"\n{'='*80}\n"
        
        # Performance Summary
        report += "PERFORMANCE SUMMARY\n"
        report += f"{'='*80}\n"
        
        summary_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0):.2%}"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"),
            ('Annualized Volatility', f"{metrics.get('annualized_volatility', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
        ]
        
        for name, value in summary_metrics:
            report += f"{name:30} {value:>15}\n"
        
        report += f"\n{'='*80}\n"
        
        # Risk Metrics
        report += "RISK METRICS\n"
        report += f"{'='*80}\n"
        
        risk_metrics = [
            ('Win Rate', f"{metrics.get('win_rate', 0):.2%}"),
            ('VaR (95%)', f"{metrics.get('var_95', 0):.2%}"),
            ('CVaR (95%)', f"{metrics.get('cvar_95', 0):.2%}"),
            ('Best Month', f"{metrics.get('best_month', 0):.2%}"),
            ('Worst Month', f"{metrics.get('worst_month', 0):.2%}"),
            ('Best Year', f"{metrics.get('best_year', 0):.2%}"),
            ('Worst Year', f"{metrics.get('worst_year', 0):.2%}"),
            ('Skewness', f"{metrics.get('skewness', 0):.2f}"),
            ('Kurtosis', f"{metrics.get('kurtosis', 0):.2f}"),
        ]
        
        for name, value in risk_metrics:
            report += f"{name:30} {value:>15}\n"
        
        report += f"\n{'='*80}\n"
        
        # Trade Statistics
        report += "TRADE STATISTICS\n"
        report += f"{'='*80}\n"
        
        trade_metrics = [
            ('Total Trades', f"{metrics.get('num_trades', 0):.0f}"),
            ('Buy Signals', f"{metrics.get('num_buy_signals', 0):.0f}"),
            ('Sell Signals', f"{metrics.get('num_sell_signals', 0):.0f}"),
            ('Avg Trade Duration', f"{metrics.get('avg_trade_duration', 0):.0f} days"),
            ('Positive Month Rate', f"{metrics.get('positive_month_rate', 0):.2%}"),
            ('Longest Winning Streak', f"{metrics.get('longest_winning_streak', 0):.0f} days"),
            ('Longest Losing Streak', f"{metrics.get('longest_losing_streak', 0):.0f} days"),
        ]
        
        for name, value in trade_metrics:
            report += f"{name:30} {value:>15}\n"
        
        report += f"\n{'='*80}\n"
        
        # Additional Information
        report += "ADDITIONAL INFORMATION\n"
        report += f"{'='*80}\n"
        
        initial_capital = self.results.get('initial_capital', 0)
        final_equity = self.results.get('final_equity', 0)
        
        report += f"Initial Capital: ${initial_capital:,.2f}\n"
        report += f"Final Equity: ${final_equity:,.2f}\n"
        report += f"Net Profit: ${final_equity - initial_capital:,.2f}\n"
        
        # Strategy parameters if available
        if 'parameters' in self.results:
            report += f"\nStrategy Parameters:\n"
            for key, value in self.results['parameters'].items():
                report += f"  {key}: {value}\n"
        
        report += f"\n{'='*80}\n"
        report += "END OF REPORT\n"
        report += f"{'='*80}\n"
        
        return report

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON report.

        Returns:
            Dictionary with report data.
        """
        metrics = self.calculate_additional_metrics()
        
        report = {
            "metadata": {
                "strategy_name": self.strategy_name,
                "report_date": self.report_date.isoformat(),
                "generator": "Quantitative Trading System",
                "version": "1.0",
            },
            "performance_summary": {
                "total_return": metrics.get('total_return', 0),
                "annualized_return": metrics.get('annualized_return', 0),
                "annualized_volatility": metrics.get('annualized_volatility', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "sortino_ratio": metrics.get('sortino_ratio', 0),
                "calmar_ratio": metrics.get('calmar_ratio', 0),
                "max_drawdown": metrics.get('max_drawdown', 0),
                "profit_factor": metrics.get('profit_factor', 0),
            },
            "risk_metrics": {
                "win_rate": metrics.get('win_rate', 0),
                "var_95": metrics.get('var_95', 0),
                "cvar_95": metrics.get('cvar_95', 0),
                "best_month": metrics.get('best_month', 0),
                "worst_month": metrics.get('worst_month', 0),
                "best_year": metrics.get('best_year', 0),
                "worst_year": metrics.get('worst_year', 0),
                "skewness": metrics.get('skewness', 0),
                "kurtosis": metrics.get('kurtosis', 0),
            },
            "trade_statistics": {
                "num_trades": metrics.get('num_trades', 0),
                "num_buy_signals": metrics.get('num_buy_signals', 0),
                "num_sell_signals": metrics.get('num_sell_signals', 0),
                "avg_trade_duration": metrics.get('avg_trade_duration', 0),
                "positive_month_rate": metrics.get('positive_month_rate', 0),
                "longest_winning_streak": metrics.get('longest_winning_streak', 0),
                "longest_losing_streak": metrics.get('longest_losing_streak', 0),
            },
            "capital_info": {
                "initial_capital": self.results.get('initial_capital', 0),
                "final_equity": self.results.get('final_equity', 0),
                "net_profit": self.results.get('final_equity', 0) - self.results.get('initial_capital', 0),
            },
            "strategy_parameters": self.results.get('parameters', {}),
        }
        
        return report

    def save_report(self, format: str = "all"):
        """Save report in specified format.

        Args:
            format: Report format ('text', 'json', 'html', 'all').
        """
        base_name = self.strategy_name.replace(' ', '_')
        
        if format in ['text', 'all']:
            # Save text report
            text_report = self.generate_text_report()
            text_path = self.output_dir / f"{base_name}_report.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_report)
            logger.info(f"Saved text report to {text_path}")
        
        if format in ['json', 'all']:
            # Save JSON report
            json_report = self.generate_json_report()
            json_path = self.output_dir / f"{base_name}_report.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, default=str)
            logger.info(f"Saved JSON report to {json_path}")
        
        # Generate visualizations
        self.generate_visualizations(base_name)
        
        logger.info(f"Report generation complete for {self.strategy_name}")


def generate_report(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
    format: str = "all",
) -> ReportGenerator:
    """Convenience function to generate report.

    Args:
        results: Backtest results dictionary.
        output_dir: Directory to save reports.
        format: Report format.

    Returns:
        ReportGenerator instance.
    """
    generator = ReportGenerator(output_dir)
    generator.set_results(results)
    generator.save_report(format)
    return generator