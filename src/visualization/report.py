# 交易结果报告生成
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
    """为交易策略生成综合报告。"""

    def __init__(self, output_dir: Optional[Path] = None):
        """初始化报告生成器。

        参数：
            output_dir: 保存报告的目录。如果为None，则使用REPORTS_DIR。
        """
        if output_dir is None:
            output_dir = REPORTS_DIR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 报告数据
        self.results: Dict[str, Any] = {}
        self.strategy_name: str = "Unknown"
        self.report_date: datetime = datetime.now()

    def set_results(self, results: Dict[str, Any]):
        """设置回测结果用于报告生成。

        参数：
            results: 回测结果字典。
        """
        self.results = results
        self.strategy_name = results.get('strategy_name', 'Unknown')
        logger.info(f"Set results for strategy: {self.strategy_name}")

    def calculate_additional_metrics(self) -> Dict[str, Any]:
        """计算额外的性能指标。

        返回：
            包含额外指标的字典。
        """
        metrics: dict[str, float] = {}
        
        # 提取基础数据
        returns = self.results.get('returns')
        equity_curve = self.results.get('equity_curve')
        drawdown = self.results.get('drawdown')
        
        if returns is None or returns.empty:
            logger.warning("No returns data for additional metrics")
            return metrics
        
        # 结果中已有的基础指标
        metrics.update({
            'total_return': self.results.get('total_return', 0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0),
            'sortino_ratio': self.results.get('sortino_ratio', 0),
            'max_drawdown': self.results.get('max_drawdown', 0),
            'win_rate': self.results.get('win_rate', 0),
            'num_trades': self.results.get('num_trades', 0),
        })
        
        # 计算额外指标
        # 1. 年化指标
        if len(returns) > 0:
            metrics['annualized_return'] = returns.mean() * 252
            metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # 2. 下行指标
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_std'] = negative_returns.std() * np.sqrt(252)
            metrics['downside_capture'] = negative_returns.mean() / returns.mean() if returns.mean() != 0 else 0
        
        # 3. 风险调整后收益
        if metrics.get('annualized_volatility', 0) > 0:
            metrics['calmar_ratio'] = (
                metrics['annualized_return'] / abs(metrics['max_drawdown'])
                if metrics['max_drawdown'] != 0 else 0
            )
        
        # 4. 交易统计
        if 'signals' in self.results and self.results['signals'] is not None:
            signals = self.results['signals']
            if 'signal' in signals.columns:
                trade_changes = signals['signal'].diff().fillna(0)
                metrics['num_buy_signals'] = (trade_changes == 1).sum()
                metrics['num_sell_signals'] = (trade_changes == -1).sum()
                metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(signals)
        
        # 5. 盈亏比
        if 'returns' in self.results and self.results['returns'] is not None:
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # 6. 恢复因子
        if drawdown is not None and not drawdown.empty:
            metrics['recovery_factor'] = self._calculate_recovery_factor(drawdown)
        
        # 7. 风险价值（VaR）和条件风险价值
        if len(returns) > 0:
            metrics['var_95'] = np.percentile(returns, 5)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # 8. 偏度和峰度
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # 9. 最佳和最差时期
        if len(returns) > 0:
            rolling_21 = returns.rolling(window=21).sum()  # Monthly returns
            metrics['best_month'] = rolling_21.max()
            metrics['worst_month'] = rolling_21.min()
            
            rolling_252 = returns.rolling(window=252).sum()  # Annual returns
            metrics['best_year'] = rolling_252.max()
            metrics['worst_year'] = rolling_252.min()
        
        # 10. 一致性指标
        if len(returns) > 0:
            positive_months = (returns.resample('M').sum() > 0).mean()
            metrics['positive_month_rate'] = positive_months
            
            # 连续期数
            metrics['longest_winning_streak'] = self._calculate_longest_streak(returns > 0)
            metrics['longest_losing_streak'] = self._calculate_longest_streak(returns < 0)
        
        return metrics

    def _calculate_avg_trade_duration(self, signals: pd.DataFrame) -> float:
        """计算平均交易持续时间（天数）。

        参数：
            signals: 包含交易信号的DataFrame。

        返回：
            平均交易持续时间（天数）。
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
                # 新交易开始
                in_trade = True
                start_date = date
            elif is_start and in_trade:
                # 前一个交易结束，新交易开始
                if start_date:
                    duration = (date - start_date).days
                    durations.append(duration)
                start_date = date
            elif not is_start and not in_trade:
                # 未在交易中
                continue
        
        # 处理仍未平仓的最后交易
        if in_trade and start_date:
            duration = (signals.index[-1] - start_date).days
            durations.append(duration)
        
        return np.mean(durations) if durations else 0

    def _calculate_recovery_factor(self, drawdown: pd.Series) -> float:
        """计算恢复因子。

        参数：
            drawdown: 回撤序列。

        返回：
            恢复因子。
        """
        if drawdown.empty:
            return 0
        
        # 寻找主要回撤期
        major_drawdowns = []
        in_drawdown = False
        drawdown_start = None
        max_dd_depth = 0
        
        for date, dd in zip(drawdown.index, drawdown.values):
            if dd < -0.05:  # 仅考虑回撤大于5%的情况
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
        """计算条件为True的最长连续期数。

        参数：
            condition: 布尔序列。

        返回：
            最长连续期数。
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
        """生成所有可视化图表。

        参数：
            save_prefix: 保存文件的前缀。
        """
        if save_prefix is None:
            save_prefix = self.strategy_name.replace(' ', '_')
        
        # 提取数据
        equity_curve = self.results.get('equity_curve')
        returns = self.results.get('returns')
        drawdown = self.results.get('drawdown')
        signals = self.results.get('signals')
        
        # 生成可视化图表
        if equity_curve is not None and not equity_curve.empty:
            # 权益曲线
            equity_path = self.output_dir / f"{save_prefix}_equity_curve.png"
            create_equity_curve(
                equity_curve,
                title=f"{self.strategy_name} - Equity Curve",
                save_path=str(equity_path)
            )
            
            # 交互式权益曲线（HTML）
            interactive_path = self.output_dir / f"{save_prefix}_interactive_equity.html"
            create_interactive_equity_curve(
                equity_curve,
                title=f"{self.strategy_name} - Equity Curve",
                save_path=str(interactive_path)
            )
        
        if drawdown is not None and not drawdown.empty:
            # 回撤图表
            drawdown_path = self.output_dir / f"{save_prefix}_drawdown.png"
            create_drawdown_chart(
                drawdown,
                title=f"{self.strategy_name} - Drawdown",
                save_path=str(drawdown_path)
            )
        
        if returns is not None and not returns.empty:
            # 收益分布
            returns_path = self.output_dir / f"{save_prefix}_returns_dist.png"
            create_returns_distribution(
                returns,
                title=f"{self.strategy_name} - Returns Distribution",
                save_path=str(returns_path)
            )
            
            # 滚动指标
            rolling_path = self.output_dir / f"{save_prefix}_rolling_metrics.png"
            create_rolling_metrics(
                returns,
                title=f"{self.strategy_name} - Rolling Metrics",
                save_path=str(rolling_path)
            )
        
        if signals is not None and not signals.empty and 'close' in signals.columns:
            # 交易分析
            trade_path = self.output_dir / f"{save_prefix}_trade_analysis.png"
            create_trade_analysis(
                signals,
                signals['close'],
                title=f"{self.strategy_name} - Trade Analysis",
                save_path=str(trade_path)
            )
        
        # 绩效仪表盘
        dashboard_path = self.output_dir / f"{save_prefix}_dashboard.png"
        create_performance_dashboard(
            self.results,
            save_path=str(dashboard_path)
        )
        
        logger.info(f"Generated visualizations in {self.output_dir}")

    def generate_text_report(self) -> str:
        """生成文本报告。

        返回：
            格式化的文本报告。
        """
        # 计算指标
        metrics = self.calculate_additional_metrics()
        
        # 创建报告头部
        report = f"{'='*80}\n"
        report += f"STRATEGY PERFORMANCE REPORT\n"
        report += f"{'='*80}\n\n"
        
        report += f"Strategy: {self.strategy_name}\n"
        report += f"Report Date: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Period: {self.results.get('period', 'N/A')}\n"
        report += f"\n{'='*80}\n"
        
        # 绩效摘要
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
        
        # 风险指标
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
        
        # 交易统计
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
        
        # 附加信息
        report += "ADDITIONAL INFORMATION\n"
        report += f"{'='*80}\n"
        
        initial_capital = self.results.get('initial_capital', 0)
        final_equity = self.results.get('final_equity', 0)
        
        report += f"Initial Capital: ${initial_capital:,.2f}\n"
        report += f"Final Equity: ${final_equity:,.2f}\n"
        report += f"Net Profit: ${final_equity - initial_capital:,.2f}\n"
        
        # 如果存在策略参数
        if 'parameters' in self.results:
            report += f"\nStrategy Parameters:\n"
            for key, value in self.results['parameters'].items():
                report += f"  {key}: {value}\n"
        
        report += f"\n{'='*80}\n"
        report += "END OF REPORT\n"
        report += f"{'='*80}\n"
        
        return report

    def generate_json_report(self) -> Dict[str, Any]:
        """生成JSON报告。

        返回：
            包含报告数据的字典。
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
        """以指定格式保存报告。

        参数：
            format: 报告格式（'text'、'json'、'html'、'all'）。
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
        
        # 生成可视化图表
        self.generate_visualizations(base_name)
        
        logger.info(f"Report generation complete for {self.strategy_name}")


def generate_report(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
    format: str = "all",
) -> ReportGenerator:
    """生成报告的便捷函数。

    参数：
        results: 回测结果字典。
        output_dir: 保存报告的目录。
        format: 报告格式。

    返回：
        ReportGenerator实例。
    """
    generator = ReportGenerator(output_dir)
    generator.set_results(results)
    generator.save_report(format)
    return generator